"""Z06_unxpass - un-xPass Robberechts 2023 KDD light: Creative Decision Rating.

Robberechts, Van Roy, Davis (2023) "un-xPass: Measuring Soccer Players'
Creativity". KDD '23 DOI 10.1145/3580305.3599924. Repo Apache-2.0.

Idea: el VAEP atomic mide VALOR de la accion (resultado), pero NO mide LA
DECISION (escogencia entre opciones). Un pase exitoso a un companero
desmarcado es alto VAEP — pero si era el pase OBVIO, no es creatividad.
Un pase exitoso a un companero ESCONDIDO entre defensores es alto VAEP
Y alta creatividad: escogio una opcion improbable + la ejecuto.

un-xPass formaliza esto:
  P(success | pre_pass_state) = "lo esperable dada la situacion"
  unexpected = success_observed - P(success | state)
  unxpass_value = unexpected * vaep_value(action)

Cuando un pase de bajo P(success) sale exitoso (defensa rota, jugada
inesperada), unexpected > 0 → crea valor "sobre lo esperado" + alto VAEP.
Cuando un pase rutinario de alto P(success) sale bien, unexpected ≈ 0 →
el credit creative es bajo (la decision era obvia).

Implementacion light (CPU-friendly, sin Transformer):
  - Features pre-pase (atomic-SPADL): location origen, type, body_part,
    velocity vecinos del balon (proxy de presion), tiempo, score state.
  - P(success | features): LightGBM + Optuna + isotonic + 5-fold by match.
  - Aplica a WC22 pases. Calcula unxpass_value por accion.
  - Aggregate per (player, period, minute_in_period).

Limitacion documentada vs paper original: Robberechts 2023 usa Transformer
sobre tracking 25Hz + grafos espaciales (P(selection) entre N opciones de
pase). La version light captura la idea sin el sequence model.

Outputs:
  data/parquet/derived/ataque/unxpass/
    training.parquet     # features + label
    model.pkl
    per_event.parquet    # P(success) + unxpass_value por pase
    per_minute.parquet   # sum unxpass_value per (player, period, minute)
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import polars as pl

_REPO    = Path(__file__).resolve().parents[1]
_DERIVED = _REPO / "data" / "parquet" / "derived" / "ataque" / "unxpass"

PASS_TYPES = {"pass", "cross"}


def build_training_table(atomic_df, cache: bool = True) -> pl.DataFrame:
    """Para cada pase, extrae features + label success.

    success: en SPADL atomic, un pase es successful si el SIGUIENTE event
    del MISMO equipo recibe el balon (mismos team_id consecutivo).
    Aproximacion: usar columna `result_id` o `success` si existe en atomic;
    si no, derivarlo de "next event same team".
    """
    cache_path = _DERIVED / "training.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    df = pl.from_pandas(atomic_df[[
        "game_id", "period_id", "time_seconds", "team_id", "player_id",
        "type_name", "bodypart_name", "x", "y", "dx", "dy",
    ]])
    # success label: next event mismo team_id (atomic-SPADL convention)
    df = df.sort(["game_id", "period_id", "time_seconds"])
    df = df.with_columns([
        pl.col("team_id").shift(-1).over("game_id").alias("next_team"),
    ])
    # success: el siguiente event (receival/dribble/pass/etc.) es del mismo
    # team — captura que el balon fue recibido por un companero. Convencion
    # atomic-SPADL (socceraction).
    df = df.with_columns(
        (pl.col("next_team") == pl.col("team_id"))
            .cast(pl.Int64).fill_null(0).alias("success")
    )
    df = df.filter(pl.col("type_name").is_in(list(PASS_TYPES)))

    # Features pre-pase
    df = df.with_columns([
        (pl.col("type_name") == "pass").cast(pl.Int64).alias("f_is_pass"),
        (pl.col("type_name") == "cross").cast(pl.Int64).alias("f_is_cross"),
        (pl.col("bodypart_name") == "foot").cast(pl.Int64).alias("f_foot"),
        (pl.col("bodypart_name") == "head").cast(pl.Int64).alias("f_head"),
        (pl.col("time_seconds") / 5400.0).alias("f_time_norm"),
        pl.col("period_id").cast(pl.Int64).alias("f_period"),
        pl.col("x").alias("f_x"),
        pl.col("y").alias("f_y"),
        pl.col("dx").alias("f_dx"),
        pl.col("dy").alias("f_dy"),
        (pl.col("dx").pow(2) + pl.col("dy").pow(2)).sqrt().alias("f_pass_len"),
        # Direccion via componentes normalizados (evita atan2 no soportado por polars Expr)
        (pl.col("dx") / (pl.col("dx").pow(2) + pl.col("dy").pow(2)).sqrt().clip(0.01))
            .alias("f_pass_dir_x"),
        (pl.col("dy") / (pl.col("dx").pow(2) + pl.col("dy").pow(2)).sqrt().clip(0.01))
            .alias("f_pass_dir_y"),
        # x_norm: distancia a porteria rival (atomic SPADL atacando hacia x=105)
        (105.0 - pl.col("x")).alias("f_dist_to_goal"),
    ])
    return df.select([
        "game_id", "period_id", "time_seconds", "team_id", "player_id",
        "type_name",
        "f_is_pass", "f_is_cross", "f_foot", "f_head",
        "f_time_norm", "f_period",
        "f_x", "f_y", "f_dx", "f_dy", "f_pass_len",
    "f_pass_dir_x", "f_pass_dir_y",
        "f_dist_to_goal",
        "success",
    ])


FEATURE_COLS = [
    "f_is_pass", "f_is_cross", "f_foot", "f_head",
    "f_time_norm", "f_period",
    "f_x", "f_y", "f_dx", "f_dy", "f_pass_len",
    "f_pass_dir_x", "f_pass_dir_y",
    "f_dist_to_goal",
]


def fit_unxpass(df: pl.DataFrame, n_folds: int = 5, n_trials: int = 25,
                seed: int = 42) -> dict:
    """LightGBM P(success|features) + Optuna + isotonic + 5-fold CV by match."""
    import lightgbm as lgb
    import optuna
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X = df.select(FEATURE_COLS).to_numpy().astype(np.float32)
    y = df["success"].to_numpy().astype(np.int32)
    match_ids = df["game_id"].to_numpy()
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(match_ids)))
    rng.shuffle(uniq)
    folds = [np.array(f) for f in np.array_split(uniq, n_folds)]

    def _oof(params):
        oof = np.full(len(y), float(y.mean()), dtype=np.float32)
        for fi, val_m in enumerate(folds):
            val_mask = np.isin(match_ids, val_m)
            tr_mask = ~val_mask
            if y[tr_mask].sum() == 0 or y[tr_mask].sum() == tr_mask.sum():
                continue
            m = lgb.LGBMClassifier(**params, random_state=seed + fi, verbose=-1)
            m.fit(X[tr_mask], y[tr_mask],
                  eval_set=[(X[val_mask], y[val_mask])],
                  callbacks=[lgb.early_stopping(20, verbose=False)])
            proba = m.predict_proba(X[val_mask])
            oof[val_mask] = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
        return oof

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 400),
            "max_depth":         trial.suggest_int("max_depth", 4, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq":    1,
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        oof = _oof(params)
        return float(roc_auc_score(y, oof))

    study = optuna.create_study(direction="maximize",
                                  sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    oof = _oof(best)

    cal = IsotonicRegression(out_of_bounds="clip"); cal.fit(oof, y)
    oof_cal = cal.predict(oof)
    final = lgb.LGBMClassifier(**best, random_state=seed, verbose=-1)
    final.fit(X, y)

    metrics = {
        "n_obs":         int(len(y)),
        "success_rate":  float(y.mean()),
        "auc_oof_cal":   float(roc_auc_score(y, oof_cal)),
        "brier_oof":     float(brier_score_loss(y, oof_cal)),
        "best_params":   best,
    }
    return {"model": final, "calibrator": cal, "feature_cols": FEATURE_COLS,
            "metrics": metrics}


def save_fit(fit: dict, path: Path | None = None) -> Path:
    if path is None:
        path = _DERIVED / "model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(fit, f)
    return path


def load_fit(path: Path | None = None) -> dict:
    if path is None:
        path = _DERIVED / "model.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_per_event(fit: dict, atomic_df) -> pl.DataFrame:
    """Aplica P(success) a cada pase WC22 + computa unxpass_value."""
    df = pl.from_pandas(atomic_df[[
        "game_id", "period_id", "time_seconds", "team_id", "player_id",
        "type_name", "bodypart_name", "x", "y", "dx", "dy",
        "vaep_value",
    ]])
    df = df.sort(["game_id", "period_id", "time_seconds"])
    df = df.with_columns(
        pl.col("team_id").shift(-1).over("game_id").alias("next_team")
    )
    df = df.with_columns(
        (pl.col("next_team") == pl.col("team_id"))
            .cast(pl.Int64).fill_null(0).alias("success_obs")
    )
    df = df.filter(pl.col("type_name").is_in(list(PASS_TYPES)))
    df = df.with_columns([
        (pl.col("type_name") == "pass").cast(pl.Int64).alias("f_is_pass"),
        (pl.col("type_name") == "cross").cast(pl.Int64).alias("f_is_cross"),
        (pl.col("bodypart_name") == "foot").cast(pl.Int64).alias("f_foot"),
        (pl.col("bodypart_name") == "head").cast(pl.Int64).alias("f_head"),
        (pl.col("time_seconds") / 5400.0).alias("f_time_norm"),
        pl.col("period_id").cast(pl.Int64).alias("f_period"),
        pl.col("x").alias("f_x"), pl.col("y").alias("f_y"),
        pl.col("dx").alias("f_dx"), pl.col("dy").alias("f_dy"),
        (pl.col("dx").pow(2) + pl.col("dy").pow(2)).sqrt().alias("f_pass_len"),
        (pl.col("dx") / (pl.col("dx").pow(2) + pl.col("dy").pow(2)).sqrt().clip(0.01))
            .alias("f_pass_dir_x"),
        (pl.col("dy") / (pl.col("dx").pow(2) + pl.col("dy").pow(2)).sqrt().clip(0.01))
            .alias("f_pass_dir_y"),
        (105.0 - pl.col("x")).alias("f_dist_to_goal"),
    ])
    X = df.select(fit["feature_cols"]).to_numpy().astype(np.float32)
    p = fit["model"].predict_proba(X)[:, 1]
    p_cal = fit["calibrator"].predict(p)
    df = df.with_columns([
        pl.Series("p_success", p_cal).cast(pl.Float64),
        (pl.col("success_obs") - pl.Series("p_success_tmp", p_cal).cast(pl.Float64))
            .alias("unexpected"),
    ])
    df = df.with_columns(
        (pl.col("unexpected") * pl.col("vaep_value")).alias("unxpass_value")
    )
    return df.select([
        "game_id", "period_id", "time_seconds", "player_id",
        "type_name", "p_success", "success_obs", "unexpected",
        "vaep_value", "unxpass_value",
    ])


def aggregate_per_minute(per_event: pl.DataFrame, cache: bool = True) -> pl.DataFrame:
    """Suma unxpass_value per (sb_player_id, period, minute_in_period). Mapping
    a PFF via M03/M08."""
    cache_path = _DERIVED / "per_minute.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    import sys
    sys.path.insert(0, str(_REPO / "src"))
    from M03_preprocess import sb_to_pff_match_id
    import M08_ataque as atk

    df = per_event.with_columns(
        (pl.col("time_seconds") // 60).cast(pl.Int64).alias("minute_in_period")
    ).group_by(["game_id", "player_id", "period_id", "minute_in_period"]).agg([
        pl.col("unxpass_value").sum().alias("unxpass_value_minute"),
        pl.col("p_success").mean().alias("p_success_mean"),
        pl.col("unexpected").mean().alias("unexpected_mean"),
        pl.len().cast(pl.Int64).alias("n_passes"),
    ]).rename({
        "game_id": "sb_match_id", "player_id": "sb_player_id",
        "period_id": "period",
    })

    sb2pff_match = sb_to_pff_match_id()
    pmap = atk.build_sb_to_pff_player_map(cache=True).select([
        pl.col("sb_player_id").cast(pl.Int64),
        pl.col("pff_player_id").cast(pl.Int64, strict=False),
    ])
    df = df.with_columns([
        pl.col("sb_match_id").cast(pl.Int64),
        pl.col("sb_player_id").cast(pl.Int64),
        pl.col("sb_match_id").replace_strict(sb2pff_match, default=None)
            .alias("pff_match_id"),
    ]).join(pmap, on="sb_player_id", how="left").filter(
        pl.col("pff_match_id").is_not_null() & pl.col("pff_player_id").is_not_null()
    ).select([
        "pff_match_id", "pff_player_id", "period", "minute_in_period",
        "unxpass_value_minute", "p_success_mean", "unexpected_mean", "n_passes",
    ])

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(cache_path, compression="snappy")
    return df


def compute_all(overwrite: bool = False, n_trials: int = 25) -> dict[str, Path]:
    out_paths = {
        "training":   _DERIVED / "training.parquet",
        "model":      _DERIVED / "model.pkl",
        "per_event":  _DERIVED / "per_event.parquet",
        "per_minute": _DERIVED / "per_minute.parquet",
    }
    if not overwrite and all(p.exists() for p in out_paths.values()):
        return out_paths

    import sys
    sys.path.insert(0, str(_REPO / "src"))
    import M08_ataque as atk

    print("[unxPass] Loading atomic SPADL training (Euro20+Euro24+Bundes23)...")
    train_atomic = atk.build_training_atomic(overwrite=False)

    print("[unxPass] Building training table (passes + success label)...")
    df = build_training_table(train_atomic, cache=True)
    if overwrite or not out_paths["training"].exists():
        out_paths["training"].parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_paths["training"], compression="snappy")
    print(f"  passes training: {df.height:,}; success_rate: {df['success'].mean():.3f}")

    print("[unxPass] Fit LightGBM P(success|features) + Optuna + isotonic...")
    fit = fit_unxpass(df, n_folds=5, n_trials=n_trials)
    m = fit["metrics"]
    print(f"  AUC OOF cal: {m['auc_oof_cal']:.4f}  Brier: {m['brier_oof']:.4f}")
    save_fit(fit, out_paths["model"])

    print("[unxPass] Apply WC22 atomic actions...")
    fit_vaep = atk.load_models()
    wc22_atomic = atk.build_wc22_atomic(overwrite=False)
    wc22_with_vaep = atk.apply_vaep_to_wc22(fit_vaep, wc22_atomic)
    pe = predict_per_event(fit, wc22_with_vaep)
    pe.write_parquet(out_paths["per_event"], compression="snappy")
    print(f"  per_event: {pe.height:,} passes WC22")
    print(f"  unxpass_value range: [{pe['unxpass_value'].min():.4f}, "
          f"{pe['unxpass_value'].max():.4f}]")

    pm = aggregate_per_minute(pe, cache=False)
    pm.write_parquet(out_paths["per_minute"], compression="snappy")
    print(f"  per_minute: {pm.height:,} (player x match x minute)")
    return out_paths


if __name__ == "__main__":
    paths = compute_all(overwrite=True)
    for k, p in paths.items():
        print(f"  {k:<10} -> {p}  ({p.stat().st_size//1024} KB)")
