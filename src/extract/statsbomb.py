"""extract.statsbomb - Subset filtrado StatsBomb open data -> parquet.

200 partidos: WC22 (64) + Euro20 (51) + Euro24 (51) + Bundes23/24 (34).

Outputs (data/parquet/statsbomb/):
    competitions.parquet                competiciones (catalogo)
    matches.parquet                     union de matches/{comp}/{season}.json
    events/{match_id}.parquet           1 fila = 1 evento
    lineups/{match_id}.parquet          alineaciones por partido
    freeze_frames/{match_id}.parquet    360 freeze-frames por evento

LOSSLESS: cada parquet preserva todos los campos del JSON original.
StatsBomb events tienen extras dict heterogeneo -> polars infiere Struct
con union de claves de todo el partido.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import polars as pl

from ._common import DATA_PUB, parquet_dir, write_parquet

# ---- Rutas raw ----

_SB        = DATA_PUB / "statsbomb" / "data"
_EVENTS    = _SB / "events"
_LINEUPS   = _SB / "lineups"
_SIXTY     = _SB / "three-sixty"
_MATCHES   = _SB / "matches"


def list_match_ids() -> list[int]:
    """IDs de los 200 partidos del subset (carpeta events/)."""
    return sorted(int(f.stem) for f in _EVENTS.glob("*.json"))


def list_freeze_frame_ids() -> list[int]:
    """IDs con freeze-frames disponibles (no todos los partidos los tienen)."""
    return sorted(int(f.stem) for f in _SIXTY.glob("*.json"))


# ---- Competitions + matches ----

def extract_competitions(overwrite: bool = False) -> Path:
    """Catalogo de competiciones (competitions.json)."""
    out = parquet_dir("statsbomb") / "competitions.parquet"
    if out.exists() and not overwrite:
        return out
    rows = json.load(open(_SB / "competitions.json"))
    df = pl.from_dicts(rows, infer_schema_length=None)
    return write_parquet(df, out, overwrite=overwrite)


def extract_matches(overwrite: bool = False) -> Path:
    """Une matches/{comp}/{season}.json en 1 parquet (lista de matches)."""
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


# ---- Events por partido (con nested types) ----

def extract_events(match_ids: list[int] | None = None,
                    overwrite: bool = False) -> dict[int, Path]:
    """events/{match_id}.json -> parquet, 1 fichero por partido."""
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


# ---- Lineups por partido ----

def extract_lineups(match_ids: list[int] | None = None,
                     overwrite: bool = False) -> dict[int, Path]:
    """lineups/{match_id}.json -> parquet (2 equipos con su lineup nested)."""
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


# ---- Freeze frames por partido (360 frame por evento etiquetado) ----

def extract_freeze_frames(match_ids: list[int] | None = None,
                           overwrite: bool = False) -> dict[int, Path]:
    """three-sixty/{match_id}.json -> parquet.

    1 fila = 1 freeze-frame con event_uuid + visible_area (list[float]) +
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


# ---- All-in-one ----

def extract_all(overwrite: bool = False) -> dict:
    """Ejecuta los 5 extractores StatsBomb en orden seguro de memoria."""
    return {
        "competitions":  extract_competitions(overwrite=overwrite),
        "matches":       extract_matches(overwrite=overwrite),
        "events":        extract_events(overwrite=overwrite),
        "lineups":       extract_lineups(overwrite=overwrite),
        "freeze_frames": extract_freeze_frames(overwrite=overwrite),
    }
