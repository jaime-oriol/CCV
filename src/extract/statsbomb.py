"""
extract.statsbomb - Extrae el subset filtrado de StatsBomb open data a parquet.

200 partidos: WC22 (64) + Euro20 (51) + Euro24 (51) + Bundes23/24 (34).

Salida:
  data/parquet/statsbomb/
    competitions.parquet                 # catalogo de competiciones
    matches.parquet                      # union de matches/{comp}/{season}.json
    events/{match_id}.parquet            # 1 fila = 1 evento
    lineups/{match_id}.parquet           # alineaciones por partido
    freeze_frames/{match_id}.parquet     # 360 freeze-frames por evento

LOSSLESS: cada parquet preserva todos los campos del JSON original.
StatsBomb events tienen extras dict heterogeneo -> polars lo maneja como
struct con union de claves de todo el partido.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import polars as pl

from ._common import DATA_PUB, parquet_dir, write_parquet

# -- Rutas ------------------------------------------------------------------

_SB        = DATA_PUB / "statsbomb" / "data"
_EVENTS    = _SB / "events"
_LINEUPS   = _SB / "lineups"
_SIXTY     = _SB / "three-sixty"
_MATCHES   = _SB / "matches"


# -- Listado de partidos ----------------------------------------------------

def list_match_ids() -> list[int]:
    """IDs de los 200 partidos del subset (de la carpeta events/)."""
    return sorted(int(f.stem) for f in _EVENTS.glob("*.json"))


def list_freeze_frame_ids() -> list[int]:
    """IDs con freeze-frames disponibles (no todos los partidos los tienen)."""
    return sorted(int(f.stem) for f in _SIXTY.glob("*.json"))


# -- Competiciones + matches ------------------------------------------------

def extract_competitions(overwrite: bool = False) -> Path:
    """Extrae competitions.json (catalogo)."""
    out = parquet_dir("statsbomb") / "competitions.parquet"
    if out.exists() and not overwrite:
        return out
    rows = json.load(open(_SB / "competitions.json"))
    df = pl.from_dicts(rows, infer_schema_length=None)
    return write_parquet(df, out, overwrite=overwrite)


def extract_matches(overwrite: bool = False) -> Path:
    """Une todos los matches/{comp}/{season}.json en un parquet."""
    out = parquet_dir("statsbomb") / "matches.parquet"
    if out.exists() and not overwrite:
        return out
    rows = []
    for comp_dir in sorted(_MATCHES.iterdir()):
        if not comp_dir.is_dir():
            continue
        for season_f in sorted(comp_dir.glob("*.json")):
            for m in json.load(open(season_f)):
                rows.append(m)
    df = pl.from_dicts(rows, infer_schema_length=None)
    return write_parquet(df, out, overwrite=overwrite)


# -- Events por partido ----------------------------------------------------

def extract_events(
    match_ids: list[int] | None = None,
    overwrite: bool = False,
) -> dict[int, Path]:
    """Convierte events/{match_id}.json a parquet, 1 fichero por partido.

    Cada evento es un dict con sub-dicts (location como list[float],
    type/possession_team/play_pattern como struct, etc.) y tipos union.
    Polars infiere schema con nested types.

    Args:
        match_ids : Subset. None = todos.
        overwrite : Re-escribir si ya existe.

    Returns:
        Dict {match_id: parquet_path}.
    """
    out_dir = parquet_dir("statsbomb/events")
    ids = match_ids or list_match_ids()
    written = {}

    for mid in ids:
        out = out_dir / f"{mid}.parquet"
        if out.exists() and not overwrite:
            written[mid] = out
            continue

        rows = json.load(open(_EVENTS / f"{mid}.json"))
        df = pl.from_dicts(rows, infer_schema_length=None)
        write_parquet(df, out, overwrite=overwrite)
        written[mid] = out

        del rows, df
        gc.collect()

    return written


# -- Lineups por partido ---------------------------------------------------

def extract_lineups(
    match_ids: list[int] | None = None,
    overwrite: bool = False,
) -> dict[int, Path]:
    """Convierte lineups/{match_id}.json a parquet.

    Cada fichero es lista de 2 equipos con sus lineups. Polars guarda
    el lineup como list[struct] dentro de cada fila.
    """
    out_dir = parquet_dir("statsbomb/lineups")
    ids = match_ids or list_match_ids()
    written = {}

    for mid in ids:
        out = out_dir / f"{mid}.parquet"
        if out.exists() and not overwrite:
            written[mid] = out
            continue

        rows = json.load(open(_LINEUPS / f"{mid}.json"))
        df = pl.from_dicts(rows, infer_schema_length=None)
        write_parquet(df, out, overwrite=overwrite)
        written[mid] = out

    return written


# -- Freeze frames por partido ---------------------------------------------

def extract_freeze_frames(
    match_ids: list[int] | None = None,
    overwrite: bool = False,
) -> dict[int, Path]:
    """Convierte three-sixty/{match_id}.json a parquet.

    Cada fichero es lista de freeze-frames (1 por evento etiquetado).
    Cada freeze-frame tiene event_uuid, visible_area (list[float]) y
    freeze_frame (list[struct{teammate, actor, keeper, location}]).
    """
    out_dir = parquet_dir("statsbomb/freeze_frames")
    ids = match_ids or list_freeze_frame_ids()
    written = {}

    for mid in ids:
        out = out_dir / f"{mid}.parquet"
        if out.exists() and not overwrite:
            written[mid] = out
            continue

        src = _SIXTY / f"{mid}.json"
        if not src.exists():
            continue

        rows = json.load(open(src))
        df = pl.from_dicts(rows, infer_schema_length=None)
        write_parquet(df, out, overwrite=overwrite)
        written[mid] = out

    return written


# -- All-in-one ------------------------------------------------------------

def extract_all(overwrite: bool = False) -> dict:
    """Ejecuta todos los extractores StatsBomb en orden seguro de memoria."""
    return {
        "competitions":  extract_competitions(overwrite=overwrite),
        "matches":       extract_matches(overwrite=overwrite),
        "events":        extract_events(overwrite=overwrite),
        "lineups":       extract_lineups(overwrite=overwrite),
        "freeze_frames": extract_freeze_frames(overwrite=overwrite),
    }
