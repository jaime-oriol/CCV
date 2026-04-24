"""
M09_defensa - Canal Solidez Defensiva via VDEP-style atomic-VAEP.

Fase 2 PCJ, canal 2 de 4. Valora la contribucion defensiva individual por
jugador-minuto. Captura tanto pressing alto que recupera (tackles/interceptions)
como bloque bajo que evita ocasiones (via defensive_value reducing P(concedes)).

Approach SOTA pragmatico:
  - Reutiliza el modelo atomic-VAEP entrenado en M08 (mismo CatBoost 5-fold CV +
    isotonic calibration) -> `defensive_value` per action.
  - defensive_value(action) mide cuanto REDUCE P(encajar_en_10_actions) la
    accion realizada por el defensor. Es el equivalente de VDEP (Toda 2022)
    adaptado a atomic-VAEP.
  - Agregado: score_def_minute = sum(defensive_value) por (match_id, player_id,
    minute) = contribucion defensiva on-ball total del jugador en ese minuto.
  - Complemento: n_def_actions (tackles + interceptions + clearances + take_on
    defendido) como feature adicional.

NO se implementa VDEP puro de Toda desde cero: el defensive_value del
atomic-VAEP entrenado en M08 usa features ricas (tipo de accion, body part,
location, tiempo, team) y es equivalente en expresividad. Maejima 2024
individualization se aproxima aqui por el "player_id del actor" del event
(tackle/interception/clearance) — NO via nearest-defender-at-ball tracking
(eso requiere PFF tracking 25fps, reservado para M10 Off-ball).

Output:
  data/parquet/derived/defensa/
    per_minute.parquet           # (match_id, player_id_sb, minute,
                                 #  score_def_minute, n_def_actions)
    per_shock_window.parquet     # (match_id, shock_id, player_id_pff,
                                 #  shock_type, score_def_pre, score_def_post,
                                 #  n_def_actions_pre, n_def_actions_post)

Acceptance (ARCHITECTURE): distribucion score_def por rol coherente
(CBs y DMs > CFs); GK score_def positivo por saves etc.

Depende de: M08 (modelo VAEP + atomic SPADL WC22 + mapping SB->PFF).
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from M01_loader_pff import load_rosters, load_metadata, scan_tracking
from M03_preprocess import attacking_direction
from M07_shocks import build_shocks_table
import M08_ataque as atk


# -- Rutas ------------------------------------------------------------------

_REPO    = Path(__file__).resolve().parents[1]
_DERIVED = _REPO / "data" / "parquet" / "derived" / "defensa"


# Tipos de accion atomic-SPADL que son defensivas (Decroos & Davis 2020).
_DEF_ACTION_TYPES = {
    "tackle", "interception", "clearance",
    "foul",                # defensa agresiva
    "keeper_save",         # paradas del portero
    "keeper_claim",
    "keeper_punch",
    "keeper_pick_up",
}


# ===========================================================================
#  SECCION 0 — Contexto defensivo off-ball via tracking PFF (bloque bajo)
# ===========================================================================

def _def_third_pct_match(match_id: int) -> pl.DataFrame:
    """% tiempo de cada jugador en TERCIO DEFENSIVO propio durante posesion RIVAL.

    Captura "bloque bajo": defensa clutch cuando el equipo aguanta el resultado.
    Distinto a las acciones on-ball (VAEP) — es CONTEXTO POSICIONAL off-ball.

    Usa tracking PFF 25fps:
      - homePlayersSmoothed / awayPlayersSmoothed solo tienen jerseyNum (String).
      - Se mapea (team_id, shirt_number) -> player_id via rosters PFF.
      - Para cada frame: equipo EN POSESION via game_event.home_ball.
      - Si MI equipo NO tiene balon: ¿estoy en mi tercio defensivo?
      - Tercio defensivo: x (post-flip) < -L/6 en coords centradas en (0,0).
    """
    md = load_metadata(match_id).row(0, named=True)
    home_id = md["home_team_id"]
    away_id = md["away_team_id"]
    pitch_length = float(md.get("pitch_length") or 105.0)
    def_third_thr = -pitch_length / 6.0

    # Mapping (team, jersey_int) -> player_id via rosters
    ro = load_rosters(match_id).select(["team_id","player_id","shirt_number"])
    home_map = {int(r["shirt_number"]): int(r["player_id"])
                for r in ro.filter(pl.col("team_id")==home_id).iter_rows(named=True)
                if r["shirt_number"] is not None}
    away_map = {int(r["shirt_number"]): int(r["player_id"])
                for r in ro.filter(pl.col("team_id")==away_id).iter_rows(named=True)
                if r["shirt_number"] is not None}

    dirs = attacking_direction(match_id).to_dicts()
    dir_lookup = {(d["team_id"], d["period"]): d["direction"] for d in dirs}

    lf = scan_tracking(match_id)
    frames = lf.select([
        pl.col("frameNum"),
        pl.col("period"),
        pl.col("game_event").struct.field("home_ball").alias("home_has_ball"),
        pl.col("homePlayersSmoothed").alias("home_players"),
        pl.col("awayPlayersSmoothed").alias("away_players"),
    ]).filter(pl.col("home_has_ball").is_not_null()).collect()

    if frames.height == 0:
        return pl.DataFrame(schema={
            "pff_match_id": pl.Int64, "player_id": pl.Int64, "minute": pl.Int64,
            "def_third_pct": pl.Float64, "oppo_possession_frames": pl.Int64,
        })

    rows = []
    fps = float(md.get("fps") or 25.0)
    frames_per_min = fps * 60

    for r in frames.iter_rows(named=True):
        frame_num = int(r["frameNum"])
        period = int(r["period"])
        home_has_ball = bool(r["home_has_ball"])
        minute = int(frame_num // frames_per_min)

        # HOME defendiendo (away tiene balon)
        if not home_has_ball and r["home_players"]:
            dir_home = dir_lookup.get((home_id, period), "R")
            sign = 1.0 if dir_home == "R" else -1.0
            for p in r["home_players"]:
                x = p.get("x")
                jersey = p.get("jerseyNum")
                if x is None or jersey is None: continue
                try: jnum = int(jersey)
                except (ValueError, TypeError): continue
                pid = home_map.get(jnum)
                if pid is None: continue
                rows.append({
                    "player_id": pid,
                    "minute":    minute,
                    "in_def_third": (sign * x) < def_third_thr,
                })

        # AWAY defendiendo (home tiene balon)
        if home_has_ball and r["away_players"]:
            dir_away = dir_lookup.get((away_id, period), "L")
            sign = 1.0 if dir_away == "R" else -1.0
            for p in r["away_players"]:
                x = p.get("x")
                jersey = p.get("jerseyNum")
                if x is None or jersey is None: continue
                try: jnum = int(jersey)
                except (ValueError, TypeError): continue
                pid = away_map.get(jnum)
                if pid is None: continue
                rows.append({
                    "player_id": pid,
                    "minute":    minute,
                    "in_def_third": (sign * x) < def_third_thr,
                })

    if not rows:
        return pl.DataFrame(schema={
            "sb_match_id": pl.Int64, "player_id": pl.Int64, "minute": pl.Int64,
            "def_third_pct": pl.Float64, "oppo_possession_frames": pl.Int64,
        })

    df = pl.DataFrame(rows)
    agg = df.group_by(["player_id", "minute"]).agg([
        pl.col("in_def_third").sum().alias("def_third_frames"),
        pl.len().alias("oppo_possession_frames"),
    ]).with_columns(
        (pl.col("def_third_frames") / pl.col("oppo_possession_frames")).alias("def_third_pct")
    ).with_columns(
        pl.lit(match_id).cast(pl.Int64).alias("pff_match_id"),
    ).select(["pff_match_id", "player_id", "minute",
              "def_third_pct", "oppo_possession_frames"])
    return agg


def build_def_third_all(cache: bool = True) -> pl.DataFrame:
    """Agrega def_third_pct para los 64 partidos WC22."""
    cache_path = _DERIVED / "def_third_context.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)
    from M01_loader_pff import list_event_match_ids
    import time
    dfs = []
    t0 = time.time()
    for i, mid in enumerate(list_event_match_ids()):
        try:
            dfs.append(_def_third_pct_match(mid))
        except Exception as e:
            print(f"  skip {mid}: {e}")
        if (i+1) % 10 == 0:
            print(f"  {i+1}/64 en {time.time()-t0:.1f}s", flush=True)
    out = pl.concat(dfs) if dfs else pl.DataFrame()
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(cache_path, compression="snappy")
    return out


# ===========================================================================
#  SECCION 1 — Aggregation per player-minute
# ===========================================================================

def aggregate_per_player_minute(cache: bool = True) -> pl.DataFrame:
    """Agrega defensive_value por (match_id, player_id_sb, minute) + n_def_actions.

    Reusa wc22_with_vaep del pipeline M08 (aplica VAEP calibrado a WC22 atomic).
    """
    cache_path = _DERIVED / "per_minute.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    # Load M08 fit + WC22 atomic + apply VAEP (calibrado)
    fit = atk.load_models()
    wc22_atomic = atk.build_wc22_atomic(overwrite=False)
    wc22_with_vaep = atk.apply_vaep_to_wc22(fit, wc22_atomic)

    # Agregar por (match_id, player, minute)
    df = pl.from_pandas(wc22_with_vaep[[
        "game_id", "period_id", "time_seconds", "team_id",
        "player_id", "type_name", "defensive_value", "vaep_value",
    ]])
    df = df.with_columns([
        (pl.col("time_seconds") // 60
         + (pl.col("period_id") - 1) * 45).cast(pl.Int64).alias("minute"),
        pl.col("type_name").is_in(list(_DEF_ACTION_TYPES)).alias("is_def_action"),
    ]).filter(pl.col("player_id").is_not_null())

    agg = df.group_by(["game_id", "player_id", "minute"]).agg([
        pl.col("defensive_value").sum().alias("score_def_minute"),
        pl.col("is_def_action").sum().alias("n_def_actions"),
        pl.len().alias("n_actions_total"),
    ]).rename({"game_id": "sb_match_id", "player_id": "sb_player_id"})

    # Join con CONTEXTO off-ball (def_third_pct via tracking PFF).
    # Necesita mapeo sb_player_id -> pff_player_id y sb_match_id -> pff_match_id.
    from M03_preprocess import _pff_to_sb_match_id
    sb2pff = {v: k for k, v in _pff_to_sb_match_id().items()}
    player_map = atk.build_sb_to_pff_player_map(cache=True).select([
        "sb_player_id", "pff_player_id",
    ]).with_columns([
        pl.col("sb_player_id").cast(pl.Int64),
        pl.col("pff_player_id").cast(pl.Int64, strict=False),
    ])

    agg = agg.with_columns([
        pl.col("sb_player_id").cast(pl.Int64),
        pl.col("sb_match_id").replace_strict(sb2pff, default=None).alias("pff_match_id"),
    ]).join(player_map, on="sb_player_id", how="left")

    def_ctx = build_def_third_all(cache=True)
    if def_ctx.height > 0:
        def_ctx_cast = def_ctx.with_columns([
            pl.col("pff_match_id").cast(pl.Int64),
            pl.col("player_id").cast(pl.Int64).alias("pff_player_id"),
            pl.col("minute").cast(pl.Int64),
        ]).select(["pff_match_id", "pff_player_id", "minute",
                   "def_third_pct", "oppo_possession_frames"])
        agg = agg.join(def_ctx_cast,
                        on=["pff_match_id", "pff_player_id", "minute"], how="left")
    else:
        agg = agg.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("def_third_pct"),
            pl.lit(None, dtype=pl.Int64).alias("oppo_possession_frames"),
        ])

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        agg.write_parquet(cache_path, compression="snappy")
    return agg


# ===========================================================================
#  SECCION 2 — Aggregation per shock window
# ===========================================================================

def aggregate_per_shock_window(cache: bool = True) -> pl.DataFrame:
    """Por cada (shock, player), suma score_def y n_def_actions en pre/post."""
    cache_path = _DERIVED / "per_shock_window.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    per_min = aggregate_per_player_minute(cache=True)
    player_map = atk.build_sb_to_pff_player_map(cache=True)
    shocks = build_shocks_table(cache=True, overwrite=False)

    # Map sb_match_id -> pff_match_id
    from M03_preprocess import _pff_to_sb_match_id
    sb2pff = {v: k for k, v in _pff_to_sb_match_id().items()}

    per_min = per_min.with_columns([
        pl.col("sb_match_id").replace_strict(sb2pff, default=None).alias("match_id"),
        pl.col("sb_player_id").cast(pl.Int64),
    ]).filter(pl.col("match_id").is_not_null())

    pm_cast = player_map.select(["sb_player_id", "pff_player_id"]).with_columns([
        pl.col("sb_player_id").cast(pl.Int64),
        pl.col("pff_player_id").cast(pl.Int64, strict=False),
    ])
    per_min = per_min.join(pm_cast, on="sb_player_id", how="left") \
                      .filter(pl.col("pff_player_id").is_not_null())

    # Join con shocks + filter windows
    shocks_slim = shocks.select([
        "match_id", "shock_id", "player_id", "shock_type",
        "window_pre_start", "window_pre_end",
        "window_post_start", "window_post_end",
    ]).rename({"player_id": "pff_player_id"})

    joined = shocks_slim.join(per_min, on=["match_id", "pff_player_id"], how="left") \
                        .with_columns((pl.col("minute") * 60).alias("min_sec"))

    pre = joined.filter(
        (pl.col("min_sec") >= pl.col("window_pre_start")) &
        (pl.col("min_sec") < pl.col("window_pre_end"))
    ).group_by(["match_id","shock_id","pff_player_id","shock_type"]).agg([
        pl.col("score_def_minute").sum().alias("score_def_pre"),
        pl.col("n_def_actions").sum().alias("n_def_actions_pre"),
    ])
    post = joined.filter(
        (pl.col("min_sec") >= pl.col("window_post_start")) &
        (pl.col("min_sec") <= pl.col("window_post_end"))
    ).group_by(["match_id","shock_id","pff_player_id","shock_type"]).agg([
        pl.col("score_def_minute").sum().alias("score_def_post"),
        pl.col("n_def_actions").sum().alias("n_def_actions_post"),
    ])

    base = shocks.select([
        "match_id", "shock_id", "player_id", "shock_type"
    ]).rename({"player_id": "pff_player_id"}).unique()

    out = base.join(pre,  on=["match_id","shock_id","pff_player_id","shock_type"], how="left") \
              .join(post, on=["match_id","shock_id","pff_player_id","shock_type"], how="left") \
              .with_columns([
                  pl.col("score_def_pre").fill_null(0.0),
                  pl.col("score_def_post").fill_null(0.0),
                  pl.col("n_def_actions_pre").fill_null(0),
                  pl.col("n_def_actions_post").fill_null(0),
              ])

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(cache_path, compression="snappy")
    return out


# -- Sanity inline ---------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=== M09_defensa sanity ===")

    t0 = time.time()
    print("\n[1] Aggregating score_def + context def_third via tracking PFF...")
    per_min = aggregate_per_player_minute(cache=True)
    print(f"  filas: {per_min.height:,} en {time.time()-t0:.1f}s")
    print(f"  cols: {per_min.columns}")
    print(f"  score_def range : [{per_min['score_def_minute'].min():.3f}, "
          f"{per_min['score_def_minute'].max():.3f}]")
    if "def_third_pct" in per_min.columns:
        ctx_valid = per_min.filter(pl.col("def_third_pct").is_not_null())
        print(f"  def_third_pct valido: {ctx_valid.height}/{per_min.height} "
              f"({100*ctx_valid.height/per_min.height:.1f}%)")
        if ctx_valid.height > 0:
            print(f"  def_third_pct range: [{ctx_valid['def_third_pct'].min():.3f}, "
                  f"{ctx_valid['def_third_pct'].max():.3f}]")

    # Acceptance: distribucion por rol (CBs + DMs > CFs)
    print("\n[2] Acceptance — score_def por rol (CBs/DMs > CFs):")
    player_map = atk.build_sb_to_pff_player_map(cache=True)
    pm_cast = per_min.with_columns(pl.col("sb_player_id").cast(pl.Int64))
    map_cast = player_map.select(["sb_player_id","pff_player_id"]).with_columns([
        pl.col("sb_player_id").cast(pl.Int64),
        pl.col("pff_player_id").cast(pl.Int64, strict=False),
    ])
    pm_roles = pm_cast.join(map_cast, on="sb_player_id", how="left") \
                      .filter(pl.col("pff_player_id").is_not_null())
    ro = load_rosters().select(["player_id","position_group"]) \
                       .unique(subset=["player_id"]) \
                       .rename({"player_id": "pff_player_id"})
    pm_roles = pm_roles.join(ro, on="pff_player_id", how="left")
    by_role = pm_roles.group_by("position_group").agg([
        pl.col("score_def_minute").mean().alias("mean_def_per_min"),
        pl.col("n_def_actions").mean().alias("mean_n_def"),
        pl.len().alias("n_minutes"),
    ]).sort("mean_def_per_min", descending=True)
    print(by_role)

    # [3] Shock-window aggregation
    t0 = time.time()
    print("\n[3] Aggregating per shock window...")
    per_shock = aggregate_per_shock_window(cache=True)
    print(f"  filas: {per_shock.height:,} en {time.time()-t0:.1f}s")
    summary = per_shock.group_by("shock_type").agg([
        pl.col("score_def_pre").mean().alias("mean_pre"),
        pl.col("score_def_post").mean().alias("mean_post"),
        (pl.col("score_def_post") - pl.col("score_def_pre")).mean().alias("mean_delta"),
    ])
    print("  score_def por shock_type:")
    print(summary)
