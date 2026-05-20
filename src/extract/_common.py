"""_common - Helpers compartidos por los extractores.

Rutas estandar (data/, data_mundial/), escritura parquet snappy + roundtrip
check lossless (JSON crudo <-> parquet).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import polars as pl

# ---- Rutas ----

_REPO     = Path(__file__).resolve().parents[2]
DATA      = _REPO / "data"
DATA_PUB  = DATA / "public"
DATA_PFF  = _REPO / "data_mundial"
PARQUET   = DATA / "parquet"


def parquet_dir(source: str) -> Path:
    """Devuelve y crea data/parquet/{source}/."""
    p = PARQUET / source
    p.mkdir(parents=True, exist_ok=True)
    return p


def scan_glob(pattern: str) -> "pl.LazyFrame":
    """Scan lazy de parquets per-partido con schemas potencialmente distintos.

    Cada parquet se extrae con su propio schema inferido, asi que cols
    opcionales (e.g. injury_stoppage en StatsBomb) aparecen en unos partidos
    y no en otros. `diagonal_relaxed` une schemas con nulls donde falten.

    Uso:
        df = scan_glob("pff/tracking/*.parquet").filter(
            pl.col("frameNum") < 1000
        ).collect()
    """
    files = sorted(PARQUET.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Sin matches: {PARQUET / pattern}")
    return pl.concat([pl.scan_parquet(f) for f in files], how="diagonal_relaxed")


# ---- Escritura parquet ----

def write_parquet(df: pl.DataFrame, path: Path, overwrite: bool = False) -> Path:
    """Escribe df a parquet snappy. Crea dir padre si no existe."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Ya existe: {path}. Usa overwrite=True.")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="snappy", statistics=True)
    return path


# ---- Roundtrip lossless (JSON <-> parquet) ----

def _normalize(obj: Any) -> Any:
    """Normaliza un valor para comparacion lossless robusta.

    Necesario porque polars unifica el schema entre filas y nuestro JSON
    crudo no tiene esa garantia. Sin normalizar todo daria falsos positivos:
      - dicts: ordena claves + DROP claves con valor None (polars rellena
               las claves vistas en otras filas; null == ausente semantico)
      - listas: normaliza recursivo, mantiene orden
      - float NaN -> None (PFF a veces serializa NaN como null)
      - float entero -> int (polars guarda nullable int como float)
      - strings que codifican enteros -> int (Wyscout mezcla int y str
        numerico en mismas cols; polars unifica a String, value preservado)
    """
    if isinstance(obj, dict):
        return {
            k: _normalize(v)
            for k, v in sorted(obj.items())
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        }
    if isinstance(obj, list):
        return [_normalize(x) for x in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, float) and obj.is_integer():
        return int(obj)
    if isinstance(obj, str):
        s = obj.strip()
        if s and s.lstrip("-").isdigit():
            try:
                return int(s)
            except ValueError:
                pass
    return obj


def deep_equal(a: Any, b: Any) -> bool:
    """Compara dos estructuras JSON lossless ignorando orden de claves."""
    return _normalize(a) == _normalize(b)


# ---- Limpieza de sentinelas Wyscout (compartido wyscout + audit) ----

_NULL_SENTINELS = {"", "null"}


def clean_empty_strings(obj: Any) -> Any:
    """Convierte sentinelas Wyscout de "ausente" a None recursivamente.

    Wyscout mezcla "", "null" (string literal) y None como sinonimos para
    valor ausente. Sin limpiar, polars infiere String en cols que son int
    en 99% de las filas (subEventId, currentTeamId, etc.) y se pierde tipo.
    """
    if isinstance(obj, dict):
        return {k: clean_empty_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_empty_strings(x) for x in obj]
    if isinstance(obj, str) and obj in _NULL_SENTINELS:
        return None
    return obj


def roundtrip_check(original_json_path: Path,
                     parquet_path: Path,
                     n_sample: int | None = None) -> tuple[bool, list[str]]:
    """Verifica que parquet -> dicts == JSON crudo. Si n_sample, compara N filas.

    Returns:
        (ok, errores). ok=True si todo coincide; errores lista hasta 5 diferencias.
    """
    original = json.load(open(original_json_path))
    if n_sample is not None:
        original = original[:n_sample]

    df = pl.read_parquet(parquet_path)
    if n_sample is not None:
        df = df.head(n_sample)
    reconstructed = df.to_dicts()

    if len(original) != len(reconstructed):
        return False, [f"len mismatch: {len(original)} vs {len(reconstructed)}"]

    errors = []
    for i, (a, b) in enumerate(zip(original, reconstructed)):
        if not deep_equal(a, b):
            errors.append(f"row {i} differs")
            if len(errors) >= 5:
                break
    return len(errors) == 0, errors
