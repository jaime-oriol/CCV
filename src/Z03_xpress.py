"""Z03_xpress - exPress (Lee et al. 2025, MIT Sloan): cabeza dedicada P(recovery<5s|press).

Building block reutilizable. Predice probabilidad de que un evento de presion
PFF resulte en recovery por el equipo defensor dentro de 5s. Reemplaza el
peso-fijo de Maejima light por una probabilidad aprendida frame-event level.

Diseno conservador (events-only, sin tracking 25Hz alignment):
  - Outcome binario: en los proximos 5s post-press_event, el siguiente
    game_event tiene team_id == defensor_team_id  →  recovery=1.
  - Features pre-press disponibles en events PFF:
      * press_type ∈ {A, L, P}                — tipo segun PFF
      * touch_type del carrier               — initialTouchType (calidad del
                                                first touch bajo presion)
      * facing_type del carrier              — back/lateral/goal
      * setpiece_type                         — open play vs balon parado
      * sgc / 3600                            — timing relativo del partido
      * period                                — fase
      * is_home_ball                          — posesion local/visitante
      * score_diff_at_event                   — marcador
  - Modelo: LightGBM con 5-fold CV by match (no leakage shot-shot mismo
    partido) + Optuna 30 trials + isotonic calibration.
  - Loss: log-loss (P(recovery|press)).
  - Acceptance: AUC OOF > baseline (P=0.40, A=0.10, L=0.20 fixed) por at
    least 0.05.

Limitacion documentada: NO usa tracking 25Hz (dist defensor-balon, vel,
n_defenders_within_5m). El paper original Lee et al. 2025 si lo usa.
Esta es la version CPU-friendly que reproduce la idea con mismo gold
standard outcome (recovery binario en 5s).

Outputs (cuando se invoca desde M09):
  - data/parquet/derived/defensa/xpress/training.parquet     (features + label)
  - data/parquet/derived/defensa/xpress/model.pkl             (LightGBM + cal)
  - data/parquet/derived/defensa/xpress/per_event.parquet    (P(recovery) por press event)
  - data/parquet/derived/defensa/xpress/per_minute.parquet   (sum P por player-minute)
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import polars as pl

_REPO    = Path(__file__).resolve().parents[1]
_DERIVED = _REPO / "data" / "parquet" / "derived" / "defensa" / "xpress"

RECOVERY_WINDOW_SEC = 5.0


# --- Features extraction --------------------------------------------------

_PRESS_TYPE_ORD = {"A": 0, "L": 1, "P": 2}
_TOUCH_TYPE_ORD = {"S": 0, "G": 1, "P": 2, "B": 3, "M": 4}    # standard < good < plus < bad < miscontrol
_FACING_ORD     = {"G": 0, "L": 1, "B": 2}                    # goal-facing < lateral < back


def build_training_table(match_ids: list[int] | None = None,
                          cache: bool = True) -> pl.DataFrame:
    """Tabla de press events con features + label recovery.

    Para cada game_event con press, busca el siguiente event y comprueba
    team flip + delta_t <= RECOVERY_WINDOW_SEC.
    """
    cache_path = _DERIVED / "training.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    from M01_loader_pff import load_events, list_event_match_ids, load_metadata
    from M03_preprocess import goals_timeline

    if match_ids is None:
        match_ids = list_event_match_ids()

    rows: list[dict] = []
    for mid in match_ids:
        ev = load_events(mid)
        md = load_metadata(mid).row(0, named=True)
        home_id = md["home_team_id"]
        # next-event lookup vectorizado: ordeno por eventTime y shifto -1
        flat = ev.select([
            pl.col("gameId").cast(pl.Int64).alias("match_id"),
            pl.col("startTime"),
            pl.col("endTime"),
            pl.col("gameEvents").struct.field("period").cast(pl.Int64).alias("period"),
            pl.col("gameEvents").struct.field("startGameClock").cast(pl.Int64).alias("sgc"),
            pl.col("gameEvents").struct.field("teamId").cast(pl.Int64).alias("carrier_team"),
            pl.col("gameEvents").struct.field("playerId").cast(pl.Int64).alias("carrier_id"),
            pl.col("gameEvents").struct.field("setpieceType").alias("setpiece"),
            pl.col("gameEvents").struct.field("homeTeam").cast(pl.Boolean).alias("is_home_ball"),
            pl.col("initialTouch").struct.field("initialPressureType").alias("press_type"),
            pl.col("initialTouch").struct.field("initialPressurePlayerId").cast(pl.Int64).alias("press_player_id"),
            pl.col("initialTouch").struct.field("initialTouchType").alias("touch_type"),
            pl.col("initialTouch").struct.field("facingType").alias("facing"),
        ]).sort("startTime")

        # next_team / next_time via shift
        flat = flat.with_columns([
            pl.col("carrier_team").shift(-1).alias("next_team"),
            pl.col("carrier_id").shift(-1).alias("next_player"),
            pl.col("startTime").shift(-1).alias("next_start"),
        ])

        # Score state at event via asof BACKWARD with goals_timeline
        g = goals_timeline(mid).sort("start_game_clock").select([
            pl.col("start_game_clock").alias("g_sgc"),
            pl.col("cum_home"), pl.col("cum_away"),
        ])
        flat = flat.sort("sgc").join_asof(
            g, left_on="sgc", right_on="g_sgc", strategy="backward"
        ).with_columns([
            pl.col("cum_home").fill_null(0),
            pl.col("cum_away").fill_null(0),
        ]).with_columns(
            (pl.when(pl.col("is_home_ball"))
              .then(pl.col("cum_home") - pl.col("cum_away"))
              .otherwise(pl.col("cum_away") - pl.col("cum_home")))
                .alias("score_diff_carrier")
        )

        press = flat.filter(
            pl.col("press_type").is_in(["A", "L", "P"]) &
            pl.col("press_player_id").is_not_null() &
            pl.col("next_team").is_not_null()
        )
        if press.height == 0:
            continue

        # defensor_team_id = team del press_player_id (lookup via rosters)
        from M01_loader_pff import load_rosters
        ro = load_rosters(mid).select([
            pl.col("player_id").cast(pl.Int64).alias("press_player_id"),
            pl.col("team_id").cast(pl.Int64).alias("defender_team_id"),
        ])
        press = press.join(ro, on="press_player_id", how="left")

        # recovery: el SIGUIENTE event tiene team == defender_team Y delta <= 5s
        press = press.with_columns([
            (pl.col("next_team") == pl.col("defender_team_id")).cast(pl.Int64).alias("team_flipped"),
            (pl.col("next_start") - pl.col("startTime")).alias("dt"),
        ]).with_columns(
            ((pl.col("team_flipped") == 1) & (pl.col("dt") <= RECOVERY_WINDOW_SEC))
                .cast(pl.Int64).alias("recovery"),
        )

        # features categoricas a ordinales
        press = press.with_columns([
            pl.col("press_type").replace_strict(_PRESS_TYPE_ORD, default=0)
                .cast(pl.Int64).alias("f_press_type"),
            pl.col("touch_type").replace_strict(_TOUCH_TYPE_ORD, default=0)
                .cast(pl.Int64).alias("f_touch_type"),
            pl.col("facing").replace_strict(_FACING_ORD, default=1)
                .cast(pl.Int64).alias("f_facing"),
            (pl.col("setpiece") == "O").cast(pl.Int64).alias("f_open_play"),
            (pl.col("sgc") / 3600.0).alias("f_time_norm"),
            pl.col("period").cast(pl.Int64).alias("f_period"),
            pl.col("is_home_ball").cast(pl.Int64).alias("f_is_home_ball"),
            pl.col("score_diff_carrier").cast(pl.Int64).alias("f_score_diff"),
        ]).select([
            "match_id", "startTime", "period", "sgc",
            "press_player_id", "defender_team_id", "carrier_team",
            "f_press_type", "f_touch_type", "f_facing",
            "f_open_play", "f_time_norm", "f_period",
            "f_is_home_ball", "f_score_diff",
            "recovery",
        ])
        rows.append(press)

    out = pl.concat(rows) if rows else pl.DataFrame()
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(cache_path, compression="snappy")
    return out


FEATURE_COLS = [
    "f_press_type", "f_touch_type", "f_facing",
    "f_open_play", "f_time_norm", "f_period",
    "f_is_home_ball", "f_score_diff",
]


# --- Train + cross-fit + isotonic calibration -----------------------------

def fit_xpress(df: pl.DataFrame, n_folds: int = 5,
               n_trials: int = 30, seed: int = 42) -> dict:
    """LightGBM + Optuna + isotonic. CV stratified by match_id."""
    import lightgbm as lgb
    import optuna
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X = df.select(FEATURE_COLS).to_numpy().astype(np.float32)
    y = df["recovery"].to_numpy().astype(np.int32)
    match_ids = df["match_id"].to_numpy()
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(match_ids)))
    rng.shuffle(uniq)
    folds = [np.array(f) for f in np.array_split(uniq, n_folds)]

    def _oof(params: dict) -> np.ndarray:
        oof = np.zeros(len(y), dtype=np.float32)
        for fi, val_m in enumerate(folds):
            val_mask = np.isin(match_ids, val_m)
            tr_mask = ~val_mask
            m = lgb.LGBMClassifier(**params, random_state=seed + fi, verbose=-1)
            m.fit(X[tr_mask], y[tr_mask],
                  eval_set=[(X[val_mask], y[val_mask])],
                  callbacks=[lgb.early_stopping(20, verbose=False)])
            oof[val_mask] = m.predict_proba(X[val_mask])[:, 1]
        return oof

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 3, 7),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 7, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq":    1,
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        }
        oof = _oof(params)
        return float(roc_auc_score(y, oof))

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    oof = _oof(best_params)

    # Calibracion isotonic + final model
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(oof, y)
    oof_cal = cal.predict(oof)

    final = lgb.LGBMClassifier(**best_params, random_state=seed, verbose=-1)
    final.fit(X, y)

    # Baseline naive: pesos fijos Maejima light
    base = np.where(df["f_press_type"].to_numpy() == 2, 0.40,
                     np.where(df["f_press_type"].to_numpy() == 1, 0.20, 0.10))

    metrics = {
        "n_obs":              int(len(y)),
        "n_recoveries":       int(y.sum()),
        "recovery_rate":      float(y.mean()),
        "auc_oof_raw":        float(roc_auc_score(y, oof)),
        "auc_oof_cal":        float(roc_auc_score(y, oof_cal)),
        "auc_baseline_fixed": float(roc_auc_score(y, base)),
        "brier_oof":          float(brier_score_loss(y, oof_cal)),
        "brier_baseline":     float(brier_score_loss(y, base)),
        "logloss_oof":        float(log_loss(y, np.clip(oof_cal, 1e-6, 1-1e-6))),
        "best_params":        best_params,
    }
    return {
        "model":        final,
        "calibrator":   cal,
        "feature_cols": FEATURE_COLS,
        "metrics":      metrics,
        "oof":          oof_cal,
    }


def save_fit(fit: dict, path: Path | None = None) -> Path:
    if path is None:
        path = _DERIVED / "model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({k: v for k, v in fit.items() if k != "oof"}, f)
    return path


def load_fit(path: Path | None = None) -> dict:
    if path is None:
        path = _DERIVED / "model.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# --- Apply: per_event predictions + per (player, minute) aggregation ------

def predict_per_event(fit: dict, df: pl.DataFrame) -> pl.DataFrame:
    X = df.select(fit["feature_cols"]).to_numpy().astype(np.float32)
    raw = fit["model"].predict_proba(X)[:, 1]
    cal = fit["calibrator"].predict(raw)
    # minute_in_period via period_start lookup (replicado de M03/M11 convention)
    period_start_min = {1: 0, 2: 45, 3: 90, 4: 105}
    return df.with_columns([
        pl.Series("p_recovery", cal).cast(pl.Float64),
    ]).with_columns(
        ((pl.col("sgc") - pl.col("period").replace_strict(
            {p: 0 for p in (1, 2, 3, 4)}, default=0
        )) // 60).cast(pl.Int64).alias("minute_in_period")
    ).select([
        "match_id", "period", "sgc", "minute_in_period",
        pl.col("press_player_id").alias("pff_player_id"),
        "p_recovery",
    ])


def aggregate_per_minute(per_event: pl.DataFrame, cache: bool = True) -> pl.DataFrame:
    """Suma P(recovery) per (player, minute) → score xpress."""
    cache_path = _DERIVED / "per_minute.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)
    out = per_event.group_by(
        ["match_id", "pff_player_id", "period", "minute_in_period"]
    ).agg([
        pl.col("p_recovery").sum().alias("xpress_value_minute"),
        pl.len().cast(pl.Int64).alias("n_press_events"),
    ]).rename({"match_id": "pff_match_id"})
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.write_parquet(cache_path, compression="snappy")
    return out


def compute_all(overwrite: bool = False, n_trials: int = 30) -> dict[str, Path]:
    """Pipeline completa exPress: training + tune + apply + aggregate."""
    out_paths = {
        "training":  _DERIVED / "training.parquet",
        "model":     _DERIVED / "model.pkl",
        "per_event": _DERIVED / "per_event.parquet",
        "per_minute":_DERIVED / "per_minute.parquet",
    }
    if not overwrite and all(p.exists() for p in out_paths.values()):
        return out_paths

    print("[xPress] Building training table (press events + recovery labels)...")
    df = build_training_table(cache=True)
    if overwrite or not out_paths["training"].exists():
        out_paths["training"].parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_paths["training"], compression="snappy")
    print(f"  press events: {df.height:,}; recovery rate: {df['recovery'].mean():.3f}")

    print("[xPress] Fit LightGBM + Optuna + isotonic...")
    fit = fit_xpress(df, n_folds=5, n_trials=n_trials)
    print(f"  AUC OOF cal: {fit['metrics']['auc_oof_cal']:.4f}  "
          f"(baseline pesos-fijos: {fit['metrics']['auc_baseline_fixed']:.4f})")
    print(f"  Brier OOF: {fit['metrics']['brier_oof']:.4f}  "
          f"(baseline: {fit['metrics']['brier_baseline']:.4f})")
    save_fit(fit, out_paths["model"])

    print("[xPress] Apply per_event + aggregate per_minute...")
    per_ev = predict_per_event(fit, df)
    per_ev.write_parquet(out_paths["per_event"], compression="snappy")
    pm = aggregate_per_minute(per_ev, cache=False)
    pm.write_parquet(out_paths["per_minute"], compression="snappy")
    print(f"  per_event: {per_ev.height:,}; per_minute: {pm.height:,}")
    return out_paths


if __name__ == "__main__":
    paths = compute_all(overwrite=True)
    for k, p in paths.items():
        print(f"  {k:<10} -> {p}  ({p.stat().st_size//1024} KB)")
