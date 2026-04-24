"""
M08_ataque - Canal Empuje Ofensivo via Atomic-VAEP (Decroos & Davis 2020).

Fase 2 PCJ, canal 1 de 4. Valora la produccion ofensiva on-ball por jugador.
Aisla la contribucion individual del rendimiento de sus companeros (atomic
version, mejor que VAEP clasico para DiD within-player).

Training corpus (WC22 EXCLUIDO, sagrado):
  - Euro 2020     (55, 43)  — 51 partidos
  - Euro 2024     (55, 282) — 51 partidos
  - Bundesliga 23 (9, 281)  — 34 partidos
  Total: 136 partidos -> ~60k-80k atomic actions.

Modelo: CatBoost via Z01_vaep.py (2 modelos: P(scores 10 acc), P(concedes 10 acc)).
Features: 148 cols atomic (actiontype_onehot, bodypart, location, polar, movement).

Pipeline:
  1. Load SB events via StatsBombLoader (socceraction nativo).
  2. Convert to SPADL -> Atomic SPADL (convert_to_atomic).
  3. Extract features + labels (compute_features, compute_labels).
  4. Train CatBoost split 80/20 by match.
  5. Apply a WC22 atomic actions -> offensive_value per action.
  6. Aggregate: (match_id, player_id_sb, minute, score_atk_minute, n_actions).
  7. Map player_id_sb -> player_id_pff via nombre+equipo (para join con M07).
  8. Aggregate per shock-window (pre/post -10/+10 min).

Output:
  data/parquet/derived/ataque/
    atomic_training.parquet      # atomic actions training (cached)
    atomic_wc22.parquet          # atomic actions WC22 (cached)
    model/vaep_atk.*.cbm         # modelos CatBoost
    per_minute.parquet           # (match_id, player_id_sb/pff, minute, score_atk_minute)
    per_shock_window.parquet     # (match_id, shock_id, player_id_pff, score_atk_pre, score_atk_post)
    sb_to_pff_player_map.parquet # mapping explicito

Acceptance (ARCHITECTURE): distribucion score_atk por rol coherente (CFs > CBs).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# M01/M07 PFF-side
from M01_loader_pff import load_rosters
from M07_shocks import build_shocks_table
# Z01 VAEP infraestructura (CatBoost + socceraction)
import Z01_vaep as vaep_mod

# socceraction
import socceraction.spadl as spadl
from socceraction.data.statsbomb import StatsBombLoader
from socceraction.atomic.spadl import convert_to_atomic, add_names as atomic_add_names


# -- Rutas y constantes ----------------------------------------------------

_REPO        = Path(__file__).resolve().parents[1]
_DERIVED     = _REPO / "data" / "parquet" / "derived" / "ataque"
_MODEL_DIR   = _DERIVED / "model"
_SB_JSON_DIR = _REPO / "data" / "public" / "statsbomb" / "data"

# SB competition x season combos
TRAINING_COMPS = [
    ("Euro20",   55,  43),
    ("Euro24",   55, 282),
    ("Bundes23",  9, 281),
]
WC22_COMP = ("WC22", 43, 106)

SB_LOADER = StatsBombLoader(root=str(_SB_JSON_DIR), getter="local")


# ===========================================================================
#  SECCION 1 — Build Atomic SPADL (training + WC22)
# ===========================================================================

def _build_atomic_actions(comps: list[tuple], cache_name: str,
                          overwrite: bool = False) -> pd.DataFrame:
    """Load SB events de cada (comp_id, season_id) -> Atomic SPADL. Cache parquet."""
    cache_path = _DERIVED / f"{cache_name}_atomic.parquet"
    if cache_path.exists() and not overwrite:
        return pd.read_parquet(cache_path)

    all_atomic = []
    for alias, cid, sid in comps:
        games = SB_LOADER.games(competition_id=cid, season_id=sid)
        print(f"  [{alias}] {len(games)} partidos...", flush=True)
        for _, g in games.iterrows():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                events = SB_LOADER.events(game_id=int(g.game_id))
            try:
                actions = spadl.statsbomb.convert_to_actions(
                    events, home_team_id=int(g.home_team_id)
                )
                atomic = convert_to_atomic(actions)
                atomic["_competition"] = alias
                all_atomic.append(atomic)
            except Exception as e:
                print(f"    skip game {g.game_id}: {e}")
    df = pd.concat(all_atomic, ignore_index=True)
    df = atomic_add_names(df)
    _DERIVED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def build_training_atomic(overwrite: bool = False) -> pd.DataFrame:
    """Atomic SPADL training (Euro20+Euro24+Bundes23, sin WC22)."""
    return _build_atomic_actions(TRAINING_COMPS, "training", overwrite)


def build_wc22_atomic(overwrite: bool = False) -> pd.DataFrame:
    """Atomic SPADL WC22 (aplicacion)."""
    return _build_atomic_actions([WC22_COMP], "wc22", overwrite)


# ===========================================================================
#  SECCION 2 — Train atomic-VAEP via Z01
# ===========================================================================

def train_vaep_model(atomic_df: pd.DataFrame,
                     holdout_frac: float = 0.2,
                     seed: int = 42) -> dict:
    """Entrena atomic-VAEP via Z01 infraestructura.

    Split por game_id (evita leakage dentro del partido).
    CatBoost con early stopping sobre validation set.
    """
    from sklearn.metrics import roc_auc_score, brier_score_loss

    all_games = atomic_df["game_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(all_games)
    n_val = int(len(all_games) * holdout_frac)
    val_games = set(all_games[:n_val])
    tr_games = set(all_games[n_val:])

    tr_df = atomic_df[atomic_df["game_id"].isin(tr_games)].reset_index(drop=True)
    va_df = atomic_df[atomic_df["game_id"].isin(val_games)].reset_index(drop=True)

    X_tr = vaep_mod.compute_features(tr_df, atomic=True, provider="statsbomb_atk")
    ys_tr, yc_tr = vaep_mod.compute_labels(tr_df, atomic=True, provider="statsbomb_atk")
    X_va = vaep_mod.compute_features(va_df, atomic=True, provider="statsbomb_atk")
    ys_va, yc_va = vaep_mod.compute_labels(va_df, atomic=True, provider="statsbomb_atk")

    model_s, model_c = vaep_mod.train(
        X_tr, ys_tr, yc_tr, X_va, ys_va, yc_va,
        iterations=500, depth=6, learning_rate=0.05, verbose=0,
    )

    # Metricas holdout
    p_s, p_c = vaep_mod.predict(X_va, model_s, model_c)
    metrics = {
        "n_train_actions":   int(len(tr_df)),
        "n_val_actions":     int(len(va_df)),
        "auc_scores":        float(roc_auc_score(ys_va.values.ravel(), p_s)),
        "auc_concedes":      float(roc_auc_score(yc_va.values.ravel(), p_c)),
        "brier_scores":      float(brier_score_loss(ys_va.values.ravel(), p_s)),
        "brier_concedes":    float(brier_score_loss(yc_va.values.ravel(), p_c)),
    }
    return {"model_s": model_s, "model_c": model_c, "metrics": metrics}


def save_models(fit: dict, path_prefix: Path | None = None) -> Path:
    if path_prefix is None:
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path_prefix = _MODEL_DIR / "vaep_atk"
    vaep_mod.save_models(fit["model_s"], fit["model_c"], str(path_prefix))
    return path_prefix


def load_models(path_prefix: Path | None = None) -> tuple:
    if path_prefix is None:
        path_prefix = _MODEL_DIR / "vaep_atk"
    return vaep_mod.load_models(str(path_prefix))


# ===========================================================================
#  SECCION 3 — Apply a WC22 + aggregate per player-minute
# ===========================================================================

def apply_vaep_to_wc22(model_s, model_c,
                       wc22_atomic: pd.DataFrame | None = None) -> pd.DataFrame:
    """Aplica VAEP a acciones WC22. Devuelve DataFrame con vaep values anadidos."""
    if wc22_atomic is None:
        wc22_atomic = build_wc22_atomic(overwrite=False)
    X = vaep_mod.compute_features(wc22_atomic, atomic=True, provider="statsbomb_wc22")
    values = vaep_mod.vaep_values(wc22_atomic, X, model_s, model_c, atomic=True)
    wc22_atomic = wc22_atomic.copy()
    wc22_atomic["offensive_value"]  = values["offensive_value"].values
    wc22_atomic["defensive_value"]  = values["defensive_value"].values
    wc22_atomic["vaep_value"]       = values["vaep_value"].values
    return wc22_atomic


def aggregate_per_player_minute(wc22_with_vaep: pd.DataFrame,
                                 cache: bool = True) -> pl.DataFrame:
    """Agrega atomic-VAEP por (match_id, player_id_sb, minute).

    score_atk_minute = sum(offensive_value) de acciones ON-BALL del jugador.
    """
    cache_path = _DERIVED / "per_minute.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    df = pl.from_pandas(wc22_with_vaep[[
        "game_id", "period_id", "time_seconds", "team_id",
        "player_id", "offensive_value", "vaep_value",
    ]])
    df = df.with_columns([
        (pl.col("time_seconds") // 60
         + (pl.col("period_id") - 1) * 45).cast(pl.Int64).alias("minute"),
    ])
    df = df.filter(pl.col("player_id").is_not_null())

    agg = df.group_by(["game_id", "player_id", "minute"]).agg([
        pl.col("offensive_value").sum().alias("score_atk_minute"),
        pl.col("vaep_value").sum().alias("vaep_minute"),
        pl.len().alias("n_actions"),
    ]).rename({"game_id": "sb_match_id", "player_id": "sb_player_id"})

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        agg.write_parquet(cache_path, compression="snappy")
    return agg


# ===========================================================================
#  SECCION 4 — Mapping SB player_id <-> PFF player_id
# ===========================================================================

def build_sb_to_pff_player_map(cache: bool = True) -> pl.DataFrame:
    """Mapea SB player_id -> PFF player_id via (nombre, equipo) join.

    Para WC22: para cada (sb_player_id, sb_name, sb_team_name), busca
    el (pff_player_id, pff_name, pff_team_name) con mismo nombre exacto
    y mismo equipo (dentro del torneo WC22).
    """
    cache_path = _DERIVED / "sb_to_pff_player_map.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    # SB side: usar loader oficial (player_id, player_name) + M02 matches
    # (socceraction games no trae team_name, pero M02 load_statsbomb_matches si).
    from M02_loader_public import load_statsbomb_matches
    sb_matches = load_statsbomb_matches(comp_id=43, season_id=106)
    home_lookup = {int(r["match_id"]): (r["home_team_id"], r["home_team_name"])
                   for r in sb_matches.iter_rows(named=True)}
    away_lookup = {int(r["match_id"]): (r["away_team_id"], r["away_team_name"])
                   for r in sb_matches.iter_rows(named=True)}

    sb_rows = []
    games = SB_LOADER.games(competition_id=43, season_id=106)
    for _, g in games.iterrows():
        gid = int(g.game_id)
        players = SB_LOADER.players(game_id=gid)
        home_tid, home_tname = home_lookup.get(gid, (None, None))
        away_tid, away_tname = away_lookup.get(gid, (None, None))
        for _, p in players.iterrows():
            pid = int(p.team_id)
            team_name = home_tname if pid == home_tid else away_tname
            sb_rows.append({
                "sb_player_id":   int(p.player_id),
                "sb_player_name": p.player_name,
                "team_id_sb":     pid,
                "team_name":      team_name,
            })
    sb_df = pl.DataFrame(sb_rows).unique(subset=["sb_player_id"])

    # PFF side: rosters tiene (player_id, player_name, team_name)
    pff = load_rosters().select([
        pl.col("player_id").alias("pff_player_id"),
        pl.col("player_name").alias("pff_player_name"),
        pl.col("team_name"),
    ]).unique(subset=["pff_player_id"])

    # Normalizacion de nombres: quitar tildes + lower + trim
    import unicodedata
    def norm(s: str | None) -> str | None:
        if s is None: return None
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        return s.lower().strip()

    sb_df = sb_df.with_columns(
        pl.col("sb_player_name").map_elements(norm, return_dtype=pl.String).alias("name_norm"),
    )
    pff = pff.with_columns(
        pl.col("pff_player_name").map_elements(norm, return_dtype=pl.String).alias("name_norm"),
    )

    # Join exacto por (name_norm, team_name)
    mapping = sb_df.join(
        pff,
        on=["name_norm", "team_name"],
        how="left",
    ).with_columns(
        pl.col("sb_player_id").cast(pl.Int64),
        pl.col("pff_player_id").cast(pl.Int64, strict=False),
    ).select(["sb_player_id", "sb_player_name", "team_name", "pff_player_id"])

    if cache:
        mapping.write_parquet(cache_path, compression="snappy")
    return mapping


# ===========================================================================
#  SECCION 5 — Aggregate per shock window (pre/post ±10 min)
# ===========================================================================

def _sb_to_pff_match_map() -> dict[int, int]:
    """Mapping SB match_id -> PFF match_id (mismo que M03)."""
    from M03_preprocess import _pff_to_sb_match_id
    return {v: k for k, v in _pff_to_sb_match_id().items()}


def aggregate_per_shock_window(per_minute: pl.DataFrame,
                                player_map: pl.DataFrame,
                                shocks: pl.DataFrame | None = None,
                                cache: bool = True) -> pl.DataFrame:
    """Por cada (shock, player), suma score_atk en pre/post windows.

    Args:
        per_minute: (sb_match_id, sb_player_id, minute, score_atk_minute, ...).
        player_map: (sb_player_id, pff_player_id) mapping.
        shocks: tabla de M07 (si None, se carga del cache).

    Returns: (match_id, shock_id, player_id_pff, shock_type,
              score_atk_pre, score_atk_post, n_actions_pre, n_actions_post).
    """
    cache_path = _DERIVED / "per_shock_window.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    if shocks is None:
        shocks = build_shocks_table(cache=True, overwrite=False)

    # Map SB match_id -> PFF match_id
    sb2pff = _sb_to_pff_match_map()
    pff2sb = {v: k for k, v in sb2pff.items()}
    per_min = per_minute.with_columns(
        pl.col("sb_match_id").replace_strict(sb2pff, default=None).alias("match_id")
    ).filter(pl.col("match_id").is_not_null())

    # Map sb_player_id -> pff_player_id (cast to ensure same dtype)
    per_min = per_min.with_columns(pl.col("sb_player_id").cast(pl.Int64))
    pm_cast = player_map.select(["sb_player_id", "pff_player_id"]).with_columns([
        pl.col("sb_player_id").cast(pl.Int64),
        pl.col("pff_player_id").cast(pl.Int64, strict=False),
    ])
    per_min = per_min.join(pm_cast, on="sb_player_id", how="left") \
                      .filter(pl.col("pff_player_id").is_not_null())

    # Join shocks + filter por ventana pre/post:
    # pre: minute in [window_pre_start//60, window_pre_end//60)
    # post: minute in [window_post_start//60, window_post_end//60]
    shocks_slim = shocks.select([
        "match_id", "shock_id", "player_id", "shock_type",
        "window_pre_start", "window_pre_end",
        "window_post_start", "window_post_end",
    ]).rename({"player_id": "pff_player_id"})

    # Join por (match_id, player_id) y luego compute pre/post aggs
    joined = shocks_slim.join(per_min, on=["match_id", "pff_player_id"], how="left")
    # minute en segundos = minute * 60 (approx)
    joined = joined.with_columns(
        (pl.col("minute") * 60).alias("min_sec")
    )
    # Calcular pre/post sums
    pre = joined.filter(
        (pl.col("min_sec") >= pl.col("window_pre_start")) &
        (pl.col("min_sec") < pl.col("window_pre_end"))
    ).group_by(["match_id","shock_id","pff_player_id","shock_type"]).agg([
        pl.col("score_atk_minute").sum().alias("score_atk_pre"),
        pl.col("n_actions").sum().alias("n_actions_pre"),
    ])
    post = joined.filter(
        (pl.col("min_sec") >= pl.col("window_post_start")) &
        (pl.col("min_sec") <= pl.col("window_post_end"))
    ).group_by(["match_id","shock_id","pff_player_id","shock_type"]).agg([
        pl.col("score_atk_minute").sum().alias("score_atk_post"),
        pl.col("n_actions").sum().alias("n_actions_post"),
    ])

    # Full list de (match_id, shock_id, player_id, shock_type) desde shocks
    base = shocks.select([
        "match_id", "shock_id", "player_id", "shock_type"
    ]).rename({"player_id": "pff_player_id"}).unique()

    out = base.join(pre,  on=["match_id","shock_id","pff_player_id","shock_type"], how="left") \
              .join(post, on=["match_id","shock_id","pff_player_id","shock_type"], how="left") \
              .with_columns([
                  pl.col("score_atk_pre").fill_null(0.0),
                  pl.col("score_atk_post").fill_null(0.0),
                  pl.col("n_actions_pre").fill_null(0),
                  pl.col("n_actions_post").fill_null(0),
              ])

    if cache:
        out.write_parquet(cache_path, compression="snappy")
    return out


# -- Sanity inline ---------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=== M08_ataque sanity ===")

    # 1. Build training atomic SPADL (cached)
    t0 = time.time()
    print("\n[1] Building atomic SPADL training (Euro20+Euro24+Bundes23)...")
    train_df = build_training_atomic(overwrite=False)
    print(f"  atomic actions training: {len(train_df):,} en {time.time()-t0:.1f}s")
    print(f"  type_name top 10: {train_df['type_name'].value_counts().head(10).to_dict()}")

    # 2. Build WC22 atomic SPADL (cached)
    t0 = time.time()
    print("\n[2] Building atomic SPADL WC22...")
    wc22_df = build_wc22_atomic(overwrite=False)
    print(f"  atomic actions WC22: {len(wc22_df):,} en {time.time()-t0:.1f}s")

    # 3. Train VAEP model
    model_prefix = _MODEL_DIR / "vaep_atk"
    if (Path(f"{model_prefix}_scores.cbm").exists() and
        Path(f"{model_prefix}_concedes.cbm").exists()):
        model_s, model_c = load_models()
        print("\n[3] VAEP models cargados desde cache")
    else:
        print("\n[3] Training CatBoost atomic-VAEP (80/20 split by match)...")
        t0 = time.time()
        fit = train_vaep_model(train_df, holdout_frac=0.2)
        print(f"  train en {time.time()-t0:.1f}s")
        print(f"  metrics: {fit['metrics']}")
        save_models(fit)
        model_s, model_c = fit["model_s"], fit["model_c"]

    # 4. Apply a WC22
    t0 = time.time()
    print("\n[4] Aplicando VAEP a WC22...")
    wc22_with_vaep = apply_vaep_to_wc22(model_s, model_c, wc22_df)
    print(f"  VAEP applied en {time.time()-t0:.1f}s")
    print(f"  offensive_value range: [{wc22_with_vaep['offensive_value'].min():.3f}, "
          f"{wc22_with_vaep['offensive_value'].max():.3f}]")

    # 5. Aggregate per player-minute
    t0 = time.time()
    print("\n[5] Agregando per player-minute...")
    per_min = aggregate_per_player_minute(wc22_with_vaep, cache=True)
    print(f"  filas: {per_min.height:,} en {time.time()-t0:.1f}s")
    print(f"  cols: {per_min.columns}")
    print(f"  players unicos: {per_min['sb_player_id'].n_unique()}")

    # 6. Mapping SB -> PFF
    print("\n[6] Building SB -> PFF player mapping...")
    mapping = build_sb_to_pff_player_map(cache=True)
    mapped = mapping.filter(pl.col("pff_player_id").is_not_null()).height
    print(f"  mapping rows: {mapping.height}, mapped: {mapped} "
          f"({100*mapped/mapping.height:.1f}%)")

    # 7. Aggregate per shock window
    t0 = time.time()
    print("\n[7] Agregando per shock window...")
    per_shock = aggregate_per_shock_window(per_min, mapping, cache=True)
    print(f"  filas: {per_shock.height:,} en {time.time()-t0:.1f}s")
    print(f"  cols: {per_shock.columns}")
    # Sanity: score_atk_post - score_atk_pre por shock_type
    summary = per_shock.group_by("shock_type").agg([
        pl.col("score_atk_pre").mean().alias("mean_pre"),
        pl.col("score_atk_post").mean().alias("mean_post"),
        (pl.col("score_atk_post") - pl.col("score_atk_pre")).mean().alias("mean_delta"),
    ])
    print("  score_atk por shock_type:")
    print(summary)

    # Sanity acceptance: score_atk por rol (CFs > CBs)
    print("\n[8] Acceptance — distribucion score_atk por rol:")
    pm_cast = per_min.with_columns(pl.col("sb_player_id").cast(pl.Int64))
    map_cast = mapping.select(["sb_player_id","pff_player_id"]).with_columns([
        pl.col("sb_player_id").cast(pl.Int64),
        pl.col("pff_player_id").cast(pl.Int64, strict=False),
    ])
    pm_with_role = pm_cast.join(map_cast, on="sb_player_id", how="left")
    pm_with_role = pm_with_role.filter(pl.col("pff_player_id").is_not_null())
    pm_with_role = pm_with_role.join(
        load_rosters().select(["player_id","position_group"]).unique(subset=["player_id"]).rename({"player_id":"pff_player_id"}),
        on="pff_player_id", how="left",
    )
    by_role = pm_with_role.group_by("position_group").agg([
        pl.col("score_atk_minute").sum().alias("total"),
        pl.col("score_atk_minute").mean().alias("mean_per_minute"),
        pl.len().alias("n_minutes"),
    ]).sort("mean_per_minute", descending=True)
    print(by_role)
