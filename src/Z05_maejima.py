"""Z05_maejima - Maejima 2024: nearest-defender attribution frame-level.

Maejima et al. 2024 ICAART (DOI 10.5220/0013117700003740). Idea: para cada
accion offensive del equipo atacante, atribuir al DEFENSOR MAS CERCANO al
balon (en el frame del evento) un credito defensivo proporcional al valor
SPADL-VAEP de la accion, con signo:

    maejima_value(action, defender) = - atomic_vaep_value(action)
    cuando la accion atacante es exitosa
    maejima_value(action, defender) = + |atomic_vaep_value(action)|
    cuando la accion atacante es fallada (intercepted, blocked, miscontrol).

Asi distinguimos credito por OPOSICION (no solo por accion defensiva
explicita SPADL — que ya cubre vdep_strict). Captura la presencia espacial
del defensor: el que mas cerca esta del balon es responsable parcial de la
contencion del oponente.

Output:
  data/parquet/derived/defensa/maejima/
    per_event.parquet     # per accion offensive con def_player + maejima_value
    per_minute.parquet    # sum maejima per (defender_player, period, minute)

Diseño:
  - Reusa tracking PFF 25Hz alignment via videoTimeMs ≈ event.startTime*1000.
  - Reusa atomic-VAEP applied a WC22 (M08 apply_vaep_to_wc22).
  - Para cada accion offensive (type IN {pass, dribble, cross, shot}):
      1. Match al frame tracking via asof BACKWARD por videoTimeMs.
      2. Ball position en ese frame.
      3. Defender mas cercano = jugador del equipo OPONENT con min dist al balon.
      4. atomic_value = vaep_value de la accion.
      5. maejima_value = signo(success) * |vaep_value| atribuido al nearest_def.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

_REPO    = Path(__file__).resolve().parents[1]
_DERIVED = _REPO / "data" / "parquet" / "derived" / "defensa" / "maejima"

OFF_TYPES = {"pass", "dribble", "cross", "shot"}


def build_per_event(cache: bool = True) -> pl.DataFrame:
    """Por cada accion offensive WC22, identifica nearest defender + asigna value.

    Carga atomic-VAEP applied (de M08), filtra offensive, joina al tracking
    PFF 25Hz por videoTimeMs, computa nearest defender, atribuye signo.
    """
    cache_path = _DERIVED / "per_event.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    import sys
    sys.path.insert(0, str(_REPO / "src"))
    import M08_ataque as atk
    from M01_loader_pff import scan_tracking, load_metadata, load_rosters
    from M03_preprocess import sb_to_pff_match_id

    fit = atk.load_models()
    wc22_atomic = atk.build_wc22_atomic(overwrite=False)
    wc22_with_vaep = atk.apply_vaep_to_wc22(fit, wc22_atomic)

    # Filtrar acciones offensivas + cols utiles
    df_off = pl.from_pandas(wc22_with_vaep[[
        "game_id", "period_id", "time_seconds", "team_id", "player_id",
        "type_name", "vaep_value", "offensive_value",
    ]]).filter(pl.col("type_name").is_in(list(OFF_TYPES)))

    # SB game_id -> PFF match_id
    sb2pff = sb_to_pff_match_id()
    df_off = df_off.with_columns([
        pl.col("game_id").cast(pl.Int64),
        pl.col("game_id").replace_strict(sb2pff, default=None)
            .alias("pff_match_id"),
    ]).filter(pl.col("pff_match_id").is_not_null())

    rows_all: list[dict] = []
    for pff_mid in df_off["pff_match_id"].unique().to_list():
        pff_mid = int(pff_mid)
        md = load_metadata(pff_mid).row(0, named=True)
        home_id = int(md["home_team_id"]); away_id = int(md["away_team_id"])
        ro = load_rosters(pff_mid)
        # jersey -> (team_id, player_id)
        jersey_to_player_home: dict[int, int] = {}
        jersey_to_player_away: dict[int, int] = {}
        for r in ro.iter_rows(named=True):
            if r["shirt_number"] is None:
                continue
            j = int(r["shirt_number"]); p = int(r["player_id"])
            if int(r["team_id"]) == home_id:
                jersey_to_player_home[j] = p
            else:
                jersey_to_player_away[j] = p

        # Tracking del partido: videoTimeMs + ball + jugadores
        tr = scan_tracking(pff_mid).select([
            "videoTimeMs",
            pl.col("homePlayersSmoothed").alias("home"),
            pl.col("awayPlayersSmoothed").alias("away"),
            pl.col("ballsSmoothed").struct.field("x").alias("bx"),
            pl.col("ballsSmoothed").struct.field("y").alias("by"),
        ]).collect().sort("videoTimeMs")

        # Acciones de este partido (atomic-SPADL.time_seconds esta dentro del period:
        # period_id 1 -> [0, 45min] minutos del period). Para alinear con
        # videoTimeMs (segundos absolutos del video), necesitamos period_start.
        # Aproximacion: usar event.startTime DEL EVENT PFF, que coincide con
        # videoTimeMs del tracking. Pero atomic-SPADL no tiene event.startTime.
        # Fallback: time_seconds del SPADL es period-relative. Para mapping a
        # video-time, usariamos metadata.startPeriod1/2/etc + offset. Como
        # simplification, joinear con events via event_uuid no aplica (atomic
        # no preserva uuid). Usamos per-period start estimado del tracking.
        if tr.height == 0:
            continue
        # Per-period videoTimeMs offsets desde tracking
        tr_periods = scan_tracking(pff_mid).select(["period", "videoTimeMs"]) \
            .group_by("period").agg(pl.col("videoTimeMs").min().alias("vstart")).collect()
        period_start = {int(r["period"]): float(r["vstart"])
                        for r in tr_periods.iter_rows(named=True)}

        df_match = df_off.filter(pl.col("pff_match_id") == pff_mid).with_columns(
            (pl.col("time_seconds") * 1000.0
             + pl.col("period_id").cast(pl.Int64).replace_strict(
                 period_start, default=0.0)).alias("vtime_ms")
        ).sort("vtime_ms")

        # asof match a tracking
        matched = df_match.join_asof(
            tr, left_on="vtime_ms", right_on="videoTimeMs",
            strategy="backward",
        )

        for row in matched.iter_rows(named=True):
            if row["bx"] is None:
                continue
            attacker_team = int(row["team_id"])
            opp_side = "away" if attacker_team == home_id else "home"
            opp_team_id = away_id if attacker_team == home_id else home_id
            jersey_map = jersey_to_player_away if attacker_team == home_id \
                          else jersey_to_player_home
            bx, by = float(row["bx"]), float(row["by"])
            best_d = float("inf")
            best_pid = None
            for p in (row[opp_side] or []):
                if p is None or p.get("x") is None or p.get("jerseyNum") is None:
                    continue
                d = float(np.hypot(p["x"] - bx, p["y"] - by))
                if d < best_d:
                    j = int(p["jerseyNum"])
                    pid = jersey_map.get(j)
                    if pid is None:
                        continue
                    best_d = d
                    best_pid = pid
            if best_pid is None:
                continue
            # Signo: vaep_value de la accion atacante. Positivo si atacante
            # genera valor (defensa fallo) → asignar NEGATIVO al defensor.
            # Si vaep_value <= 0 (accion fallada / sin valor), defensor gana
            # positivo proporcional a |valor potencial perdido|.
            v_atk = float(row["vaep_value"] or 0.0)
            mae_value = -v_atk if v_atk > 0 else abs(v_atk)
            rows_all.append({
                "pff_match_id":   pff_mid,
                "period":         int(row["period_id"]),
                "time_seconds":   float(row["time_seconds"]),
                "attacker_id":    int(row["player_id"]) if row["player_id"] else None,
                "type_name":      row["type_name"],
                "vaep_value":     v_atk,
                "pff_player_id":  best_pid,
                "dist_m":         best_d,
                "maejima_value":  mae_value,
            })

    out = pl.DataFrame(rows_all) if rows_all else pl.DataFrame()
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(cache_path, compression="snappy")
    return out


def aggregate_per_minute(per_event: pl.DataFrame, cache: bool = True) -> pl.DataFrame:
    """Suma maejima_value per (defender, period, minute_in_period)."""
    cache_path = _DERIVED / "per_minute.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)
    out = (per_event.with_columns(
        (pl.col("time_seconds") // 60).cast(pl.Int64).alias("minute_in_period")
    ).group_by(["pff_match_id", "pff_player_id", "period", "minute_in_period"])
      .agg([
        pl.col("maejima_value").sum().alias("maejima_value_minute"),
        pl.len().cast(pl.Int64).alias("n_nearest_def_actions"),
        pl.col("dist_m").mean().alias("avg_dist_to_ball_m"),
    ]))
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(cache_path, compression="snappy")
    return out


def compute_all(overwrite: bool = False) -> dict[str, Path]:
    """Pipeline completa Maejima: per_event + per_minute."""
    out_paths = {
        "per_event":  _DERIVED / "per_event.parquet",
        "per_minute": _DERIVED / "per_minute.parquet",
    }
    if not overwrite and all(p.exists() for p in out_paths.values()):
        return out_paths

    print("[Maejima] Building per_event (nearest defender per off action)...")
    pe = build_per_event(cache=not overwrite)
    if overwrite or not out_paths["per_event"].exists():
        out_paths["per_event"].parent.mkdir(parents=True, exist_ok=True)
        pe.write_parquet(out_paths["per_event"], compression="snappy")
    print(f"  per_event: {pe.height:,} actions; "
          f"avg dist_to_ball={pe['dist_m'].mean():.2f}m")
    print(f"  maejima_value range: [{pe['maejima_value'].min():.4f}, "
          f"{pe['maejima_value'].max():.4f}]")

    print("[Maejima] Aggregating per_minute...")
    pm = aggregate_per_minute(pe, cache=False)
    pm.write_parquet(out_paths["per_minute"], compression="snappy")
    print(f"  per_minute: {pm.height:,} (player x match x minute)")
    return out_paths


if __name__ == "__main__":
    paths = compute_all(overwrite=True)
    for k, p in paths.items():
        print(f"  {k:<10} -> {p}  ({p.stat().st_size//1024} KB)")
