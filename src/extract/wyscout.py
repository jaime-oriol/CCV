"""extract.wyscout - Open dataset Wyscout 2017/18 -> parquet.

5 ligas + WC 2018 + Euro 2016 (~1941 partidos, ~3.25M eventos).

Outputs (data/parquet/wyscout/):
    events_{competition}.parquet     eventos por competicion
    matches.parquet                  union de matches_*.json (anade competition)
    players.parquet / teams.parquet / coaches.parquet / referees.parquet
    playerank.parquet                rankings ML de Wyscout

LOSSLESS: cada parquet preserva todos los campos del JSON original.
Limpieza previa: convierte sentinelas Wyscout ("", "null") a None para
no perder tipos (subEventId, currentTeamId mezclados int/str).
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import polars as pl

from ._common import (
    DATA_PUB, clean_empty_strings, parquet_dir, write_parquet,
)

# ---- Rutas raw ----

_WYSCOUT = DATA_PUB / "wyscout"

# 7 competiciones del open dataset
_COMPETITIONS: list[str] = [
    "England", "France", "Germany", "Italy", "Spain",
    "European_Championship", "World_Cup",
]


# ---- Events por competicion ----

def extract_events(competitions: list[str] | None = None,
                    overwrite: bool = False) -> dict[str, Path]:
    """events_{competition}.json -> parquet por competicion.

    Polars infiere schema con nested types (positions como list[struct{x,y}],
    tags como list[struct{id}], etc.).
    """
    out_dir = parquet_dir("wyscout")
    comps = competitions or _COMPETITIONS
    written = {}

    for comp in comps:
        out = out_dir / f"events_{comp}.parquet"
        if out.exists() and not overwrite:
            written[comp] = out
            continue

        src = _WYSCOUT / f"events_{comp}.json"
        if not src.exists():
            continue

        rows = clean_empty_strings(json.load(open(src)))
        df = pl.from_dicts(rows, infer_schema_length=None)
        write_parquet(df, out, overwrite=overwrite)
        written[comp] = out

        del rows, df
        gc.collect()

    return written


# ---- Matches ----

def extract_matches(overwrite: bool = False) -> Path:
    """Une matches_*.json en 1 parquet, anadiendo columna competition."""
    out = parquet_dir("wyscout") / "matches.parquet"
    if out.exists() and not overwrite:
        return out
    rows = []
    for comp in _COMPETITIONS:
        src = _WYSCOUT / f"matches_{comp}.json"
        if not src.exists():
            continue
        for m in json.load(open(src)):
            m["competition"] = comp
            rows.append(m)
    rows = clean_empty_strings(rows)
    df = pl.from_dicts(rows, infer_schema_length=None)
    return write_parquet(df, out, overwrite=overwrite)


# ---- Catalogos (players, teams, coaches, referees, playerank) ----

def _extract_catalog(filename: str, overwrite: bool = False) -> Path | None:
    """Extrae un catalogo JSON. Tolera JSON corrupto (referees.json esta mal cerrado)."""
    out = parquet_dir("wyscout") / f"{Path(filename).stem}.parquet"
    if out.exists() and not overwrite:
        return out
    src = _WYSCOUT / filename
    try:
        rows = clean_empty_strings(json.load(open(src)))
    except json.JSONDecodeError as e:
        print(f"WARN: {filename} JSON corrupto, skip: {e}")
        return None
    df = pl.from_dicts(rows, infer_schema_length=None)
    return write_parquet(df, out, overwrite=overwrite)


def extract_players(overwrite: bool = False) -> Path:
    return _extract_catalog("players.json", overwrite)


def extract_teams(overwrite: bool = False) -> Path:
    return _extract_catalog("teams.json", overwrite)


def extract_coaches(overwrite: bool = False) -> Path:
    return _extract_catalog("coaches.json", overwrite)


def extract_referees(overwrite: bool = False) -> Path:
    return _extract_catalog("referees.json", overwrite)


def extract_playerank(overwrite: bool = False) -> Path:
    return _extract_catalog("playerank.json", overwrite)


# ---- All-in-one ----

def extract_all(overwrite: bool = False) -> dict:
    """Ejecuta todos los extractores Wyscout."""
    return {
        "events":    extract_events(overwrite=overwrite),
        "matches":   extract_matches(overwrite=overwrite),
        "players":   extract_players(overwrite=overwrite),
        "teams":     extract_teams(overwrite=overwrite),
        "coaches":   extract_coaches(overwrite=overwrite),
        "referees":  extract_referees(overwrite=overwrite),
        "playerank": extract_playerank(overwrite=overwrite),
    }
