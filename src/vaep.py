"""
vaep - VAEP y Atomic-VAEP baseline con CatBoost.

Extrae features VAEP de acciones SPADL (Opta MA36), computa labels
(P(gol), P(encajar) en 10 acciones), entrena CatBoost y evalua.
Soporta VAEP clasico (142 features) y Atomic-VAEP (148).

Pipeline:
  1. Extraer features (ventana 3 acciones previas)
  2. Computar labels: P(scores 10 acc), P(concedes 10 acc)
  3. Entrenar CatBoost (un modelo por label)
  4. Predecir VAEP values: offensive - defensive
  5. Evaluar: Brier, log-loss, ROC-AUC

Cache granular: parquet por partido en cache/vaep/ para no recalcular.

Uso:
    from src.vaep import (compute_features, compute_labels,
                          train, predict, evaluate, vaep_values,
                          save_models, load_models)

    X = compute_features(actions, atomic=True, provider="opta")
    y_s, y_c = compute_labels(actions, atomic=True, provider="opta")
    model_s, model_c = train(X_train, y_s_train, y_c_train)
    values = vaep_values(actions_test, X_test, model_s, model_c, atomic=True)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# -- Imports VAEP clasico vs Atomic -----------------------------------------
# Se importan ambas ramas; la seleccion se hace via el flag atomic=True/False.

from socceraction.atomic.spadl import (
    convert_to_atomic as _to_atomic,
    add_names as _atomic_add_names,
)

# Features VAEP clasico (11 funciones -> 142 cols con 3 prev)
from socceraction.vaep import features as _vf
from socceraction.vaep import labels as _vl
from socceraction.vaep import formula as _vfm

# Features Atomic-VAEP (10 funciones -> 148 cols con 3 prev)
from socceraction.atomic.vaep import features as _af
from socceraction.atomic.vaep import labels as _al
from socceraction.atomic.vaep import formula as _afm

# -- Constantes -------------------------------------------------------------

_CACHE = Path(__file__).resolve().parents[1] / "cache" / "vaep"

NB_PREV_ACTIONS = 3   # ventana de acciones previas para features
NR_ACTIONS = 10        # horizonte para labels (gol/encajar en 10 acciones)

# Funciones de features para VAEP clasico
_VAEP_FEAT_FNS = [
    _vf.actiontype_onehot, _vf.bodypart_onehot, _vf.result_onehot,
    _vf.goalscore, _vf.startlocation, _vf.endlocation,
    _vf.movement, _vf.space_delta, _vf.time, _vf.time_delta, _vf.team,
]

# Funciones de features para Atomic-VAEP (sin result, sin speed)
_ATOMIC_FEAT_FNS = [
    _af.actiontype_onehot, _af.bodypart_onehot,
    _af.goalscore, _af.location, _af.polar,
    _af.movement_polar, _af.direction, _af.team, _af.time, _af.time_delta,
]

# CatBoost defaults (conservadores, sin tuning agresivo)
_CATBOOST_DEFAULTS = dict(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    eval_metric="Logloss",
    verbose=100,
    task_type="CPU",
    random_seed=42,
)


# -- Helpers privados -------------------------------------------------------

def _feat_fns(atomic: bool) -> list:
    """Devuelve la lista de funciones de features segun el modo."""
    return _ATOMIC_FEAT_FNS if atomic else _VAEP_FEAT_FNS


def _label_mod(atomic: bool):
    """Devuelve el modulo de labels correcto."""
    return _al if atomic else _vl


def _feat_mod(atomic: bool):
    """Devuelve el modulo de features correcto."""
    return _af if atomic else _vf


def _formula_mod(atomic: bool):
    """Devuelve el modulo de formula correcto."""
    return _afm if atomic else _vfm


def _mode_tag(atomic: bool) -> str:
    """Devuelve 'atomic' o 'vaep' para nombres de cache."""
    return "atomic" if atomic else "vaep"


# -- 1. Features -----------------------------------------------------------

def compute_features(
    actions: pd.DataFrame,
    atomic: bool = False,
    nb_prev: int = NB_PREV_ACTIONS,
    provider: str = "default",
) -> pd.DataFrame:
    """Extrae features VAEP de las acciones, partido a partido.

    Procesa por game_id porque gamestates() agrupa por (game_id, period_id).
    Cache: un parquet por partido en cache/vaep/features/{tag}/{game_id}.parquet.

    Args:
        actions  : DataFrame SPADL (con type_name, etc. de add_names).
        atomic   : Si True, usa features Atomic-VAEP.
        nb_prev  : Numero de acciones previas en la ventana (default: 3).
        provider : Etiqueta de provider ("wyscout", "statsbomb") para evitar
                   colisiones de cache entre datasets con game_ids iguales.

    Returns:
        DataFrame con las features (142 cols VAEP, 148 cols Atomic).
        Mismo indice y orden que actions.
    """
    mode = _mode_tag(atomic)
    tag = f"{mode}_{provider}_prev{nb_prev}"
    cache_dir = _CACHE / "features" / tag
    cache_dir.mkdir(parents=True, exist_ok=True)

    fmod = _feat_mod(atomic)
    fns = _feat_fns(atomic)

    parts = []
    for gid, group in actions.groupby("game_id", sort=False):
        cache_path = cache_dir / f"{gid}.parquet"

        if cache_path.exists():
            X = pd.read_parquet(cache_path)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gamestates = fmod.gamestates(group, nb_prev_actions=nb_prev)
                X = pd.concat([fn(gamestates) for fn in fns], axis=1)
            X.to_parquet(cache_path, index=False)

        parts.append(X)

    return pd.concat(parts, ignore_index=True)


# -- 3. Labels --------------------------------------------------------------

def compute_labels(
    actions: pd.DataFrame,
    atomic: bool = False,
    nr_actions: int = NR_ACTIONS,
    provider: str = "default",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computa labels de scores y concedes, partido a partido.

    Cache: un parquet por partido en cache/vaep/labels/{tag}/{game_id}.parquet.

    Args:
        actions    : DataFrame SPADL.
        atomic     : Si True, usa labels Atomic-VAEP.
        nr_actions : Horizonte de acciones (default: 10).
        provider   : Etiqueta de provider para evitar colisiones de cache.

    Returns:
        (y_scores, y_concedes) cada uno DataFrame con 1 columna.
    """
    mode = _mode_tag(atomic)
    tag = f"{mode}_{provider}_h{nr_actions}"
    cache_dir = _CACHE / "labels" / tag
    cache_dir.mkdir(parents=True, exist_ok=True)

    lmod = _label_mod(atomic)

    scores_parts, concedes_parts = [], []
    for gid, group in actions.groupby("game_id", sort=False):
        cache_path = cache_dir / f"{gid}.parquet"

        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            y_s = cached[["scores"]]
            y_c = cached[["concedes"]]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y_s = lmod.scores(group, nr_actions=nr_actions)
                y_c = lmod.concedes(group, nr_actions=nr_actions)
            # Guardar ambos en un solo parquet
            pd.concat([y_s, y_c], axis=1).to_parquet(cache_path, index=False)

        scores_parts.append(y_s)
        concedes_parts.append(y_c)

    return (pd.concat(scores_parts, ignore_index=True),
            pd.concat(concedes_parts, ignore_index=True))


# -- 4. Train ---------------------------------------------------------------

def train(
    X: pd.DataFrame,
    y_scores: pd.DataFrame,
    y_concedes: pd.DataFrame,
    X_val: pd.DataFrame | None = None,
    y_scores_val: pd.DataFrame | None = None,
    y_concedes_val: pd.DataFrame | None = None,
    **catboost_kwargs,
) -> tuple[CatBoostClassifier, CatBoostClassifier]:
    """Entrena dos CatBoost: uno para P(scores), otro para P(concedes).

    Args:
        X              : Features de entrenamiento.
        y_scores       : Labels scores de entrenamiento.
        y_concedes     : Labels concedes de entrenamiento.
        X_val          : Features de validacion (opcional, para early stopping).
        y_scores_val   : Labels scores validacion.
        y_concedes_val : Labels concedes validacion.
        **catboost_kwargs : Sobreescriben _CATBOOST_DEFAULTS.

    Returns:
        (model_scores, model_concedes) ambos CatBoostClassifier entrenados.
    """
    params = {**_CATBOOST_DEFAULTS, **catboost_kwargs}

    model_s = CatBoostClassifier(**params)
    model_c = CatBoostClassifier(**params)

    eval_s = (X_val, y_scores_val.values.ravel()) if X_val is not None else None
    eval_c = (X_val, y_concedes_val.values.ravel()) if X_val is not None else None

    model_s.fit(X, y_scores.values.ravel(),
                eval_set=eval_s, early_stopping_rounds=50 if eval_s else None)
    model_c.fit(X, y_concedes.values.ravel(),
                eval_set=eval_c, early_stopping_rounds=50 if eval_c else None)

    return model_s, model_c


# -- 5. Predict + VAEP values -----------------------------------------------

def predict(
    X: pd.DataFrame,
    model_scores: CatBoostClassifier,
    model_concedes: CatBoostClassifier,
) -> tuple[np.ndarray, np.ndarray]:
    """Predice P(scores) y P(concedes) para cada accion.

    Args:
        X              : Features.
        model_scores   : Modelo entrenado para P(scores).
        model_concedes : Modelo entrenado para P(concedes).

    Returns:
        (p_scores, p_concedes) arrays de probabilidades (N,).
    """
    p_s = model_scores.predict_proba(X)[:, 1]
    p_c = model_concedes.predict_proba(X)[:, 1]
    return p_s, p_c


def vaep_values(
    actions: pd.DataFrame,
    X: pd.DataFrame,
    model_scores: CatBoostClassifier,
    model_concedes: CatBoostClassifier,
    atomic: bool = False,
) -> pd.DataFrame:
    """Calcula VAEP value por accion usando la formula oficial de socceraction.

    Maneja correctamente: cambios de equipo (posesion), goles previos,
    penaltis (prior 0.79), corners (prior 0.047), y acciones muy separadas
    en tiempo (>10s -> reset a 0).

    Args:
        actions        : DataFrame SPADL (necesita team_id, time_seconds,
                         type_name; y result_name si atomic=False).
        X              : Features correspondientes a actions.
        model_scores   : Modelo P(scores).
        model_concedes : Modelo P(concedes).
        atomic         : Si True, usa formula Atomic-VAEP.

    Returns:
        DataFrame con columnas: offensive_value, defensive_value, vaep_value.
    """
    # Reset indices para alinear actions con las Series de probabilidades,
    # ya que socceraction.formula opera con indices pandas internamente.
    actions = actions.reset_index(drop=True)
    X = X.reset_index(drop=True)

    p_s, p_c = predict(X, model_scores, model_concedes)
    formula = _formula_mod(atomic)
    return formula.value(actions, pd.Series(p_s), pd.Series(p_c))


# -- 6. Evaluate -------------------------------------------------------------

def evaluate(
    y_true_scores: pd.DataFrame,
    y_true_concedes: pd.DataFrame,
    p_scores: np.ndarray,
    p_concedes: np.ndarray,
) -> dict:
    """Evalua las predicciones de P(scores) y P(concedes).

    Args:
        y_true_scores   : Labels reales scores (DataFrame 1 col).
        y_true_concedes : Labels reales concedes (DataFrame 1 col).
        p_scores        : Probabilidades predichas P(scores).
        p_concedes      : Probabilidades predichas P(concedes).

    Returns:
        Dict con: brier_scores, brier_concedes, logloss_scores, logloss_concedes,
                  auc_scores, auc_concedes.
    """
    ys = y_true_scores.values.ravel()
    yc = y_true_concedes.values.ravel()

    return {
        "brier_scores":   brier_score_loss(ys, p_scores),
        "brier_concedes": brier_score_loss(yc, p_concedes),
        "logloss_scores":   log_loss(ys, p_scores),
        "logloss_concedes": log_loss(yc, p_concedes),
        "auc_scores":   roc_auc_score(ys, p_scores),
        "auc_concedes": roc_auc_score(yc, p_concedes),
    }


# -- 7. Save / Load modelos --------------------------------------------------

def save_models(
    model_scores: CatBoostClassifier,
    model_concedes: CatBoostClassifier,
    path: str | Path,
) -> Path:
    """Guarda los dos modelos CatBoost en disco.

    Usa el formato nativo de CatBoost (.cbm). Guarda dos ficheros:
    {path}_scores.cbm y {path}_concedes.cbm.

    Args:
        model_scores   : Modelo P(scores).
        model_concedes : Modelo P(concedes).
        path           : Prefijo de ruta (sin extension).

    Returns:
        Path del directorio donde se guardaron.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_scores.save_model(str(path) + "_scores.cbm")
    model_concedes.save_model(str(path) + "_concedes.cbm")
    return path.parent


def load_models(path: str | Path) -> tuple[CatBoostClassifier, CatBoostClassifier]:
    """Carga los dos modelos CatBoost desde disco.

    Args:
        path : Prefijo de ruta (sin extension), mismo que save_models.

    Returns:
        (model_scores, model_concedes).
    """
    model_s = CatBoostClassifier()
    model_c = CatBoostClassifier()
    model_s.load_model(str(path) + "_scores.cbm")
    model_c.load_model(str(path) + "_concedes.cbm")
    return model_s, model_c
