"""extract.pff - Extrae el dataset PFF FC WC22 a parquet con tipos anidados.

4 extractores:
  extract_pff_events()    64 JSON Event Data -> 1 parquet por partido
                          (1 fila = 1 evento; sub-dicts como Struct,
                          listas como List(Struct))
  extract_pff_tracking()  64 jsonl.bz2 Tracking Data -> 1 parquet por
                          partido (streaming chunked, peak ~150-300 MB)
  extract_pff_metadata()  64 JSON Metadata -> 1 parquet unificado
  extract_pff_rosters()   64 JSON Rosters -> 1 parquet unificado

LOSSLESS: 1 fila parquet -> dict equivalente al JSON original. Verificable
con _common.roundtrip_check() (events) y streaming_roundtrip_tracking().
"""
from __future__ import annotations

import bz2
import gc
import json
from pathlib import Path
from typing import Iterator

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ._common import DATA_PFF, deep_equal, parquet_dir, write_parquet

# ---- Rutas raw ----

_EVENTS    = DATA_PFF / "Event Data"
_TRACKING  = DATA_PFF / "Tracking Data"
_METADATA  = DATA_PFF / "Metadata"
_ROSTERS   = DATA_PFF / "Rosters"


# ---- Listado de partidos ----

def list_event_match_ids() -> list[int]:
    """IDs de los 64 partidos con event data."""
    return sorted(int(f.stem) for f in _EVENTS.glob("*.json"))


def list_tracking_match_ids() -> list[int]:
    """IDs de los 64 partidos con tracking data."""
    return sorted(int(f.name.split(".")[0]) for f in _TRACKING.glob("*.jsonl.bz2"))


# ---- Events (carga completa por partido, ~17 MB JSON) ----

def extract_pff_events(match_ids: list[int] | None = None,
                        overwrite: bool = False) -> dict[int, Path]:
    """Event Data JSON -> parquet por partido con tipos anidados.

    Cada evento es un dict con sub-dicts (gameEvents, initialTouch,
    possessionEvents, fouls, grades, stadiumMetadata) y listas
    (homePlayers, awayPlayers, ball). Polars los preserva nativamente.
    """
    out_dir = parquet_dir("pff/events")
    ids = match_ids or list_event_match_ids()
    written = {}

    for gid in ids:
        out = out_dir / f"{gid}.parquet"
        if out.exists() and not overwrite:
            written[gid] = out
            continue

        rows = json.load(open(_EVENTS / f"{gid}.json"))
        df = pl.from_dicts(rows, infer_schema_length=None)
        write_parquet(df, out, overwrite=overwrite)
        written[gid] = out

        del rows, df
        gc.collect()

    return written


# ---- Tracking (streaming chunked anti-OOM) ----

# 5000 frames * ~5 KB/frame = ~25 MB JSON / chunk -> peak ~150 MB RAM
_TRACKING_CHUNK_FRAMES = 5000


def _iter_tracking_frames(path: Path) -> Iterator[dict]:
    """Yield 1 dict por linea del jsonl.bz2 (streaming)."""
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _peek_first_n(it: Iterator[dict], n: int) -> tuple[list[dict], Iterator[dict]]:
    """Saca las primeras N filas del iterador para inferir schema unificado.

    Devuelve (head_list, iterator_que_empieza_por_head_y_sigue_con_el_resto).
    """
    head = []
    for _ in range(n):
        try:
            head.append(next(it))
        except StopIteration:
            break

    def chained():
        yield from head
        yield from it

    return head, chained()


def extract_pff_tracking(match_ids: list[int] | None = None,
                          overwrite: bool = False,
                          chunk_frames: int = _TRACKING_CHUNK_FRAMES,
                          ) -> dict[int, Path]:
    """Tracking jsonl.bz2 -> parquet con streaming chunked (peak ~150-300 MB).

    Estrategia:
      1. Stream linea a linea (bz2 + json.loads)
      2. Peek 200 filas para inferir schema unificado (claves opcionales)
      3. Acumular chunk_frames, convertir a polars, escribir row group con
         pyarrow.ParquetWriter
      4. Liberar buffer entre chunks
    """
    out_dir = parquet_dir("pff/tracking")
    ids = match_ids or list_tracking_match_ids()
    written = {}

    for gid in ids:
        out = out_dir / f"{gid}.parquet"
        if out.exists() and not overwrite:
            written[gid] = out
            continue

        src = _TRACKING / f"{gid}.jsonl.bz2"
        # 200 frames es suficiente para cubrir game_event / possession_event
        # opcionales (aparecen sparse en el stream).
        head, stream = _peek_first_n(_iter_tracking_frames(src), 200)
        if not head:
            continue
        schema_df = pl.from_dicts(head, infer_schema_length=None)
        arrow_schema = schema_df.to_arrow().schema
        del schema_df, head

        writer = pq.ParquetWriter(str(out), arrow_schema, compression="snappy")
        try:
            buf: list[dict] = []
            for frame in stream:
                buf.append(frame)
                if len(buf) >= chunk_frames:
                    _flush_chunk(buf, writer, arrow_schema)
                    buf = []
                    gc.collect()
            if buf:
                _flush_chunk(buf, writer, arrow_schema)
        finally:
            writer.close()

        written[gid] = out
        gc.collect()

    return written


def _flush_chunk(rows: list[dict], writer: pq.ParquetWriter, schema: pa.Schema) -> None:
    """Convierte rows a Arrow Table conforme al schema y escribe 1 row group."""
    df = pl.from_dicts(rows, infer_schema_length=None)
    table = df.to_arrow()
    table = table.cast(schema, safe=False) if table.schema != schema else table
    writer.write_table(table)
    del df, table


# ---- Roundtrip lossless tracking (streaming, sin OOM) ----

def streaming_roundtrip_tracking(parquet_path: Path,
                                  jsonl_path: Path,
                                  sample_indices: list[int] | None = None,
                                  ) -> tuple[bool, list[str]]:
    """Verifica lossless en tracking sin cargar todo a memoria.

    Compara filas muestreadas (default: primeras 50 + ultimas 50).
    """
    if sample_indices is None:
        n = pl.scan_parquet(parquet_path).select(pl.len()).collect().item()
        sample_indices = list(range(50)) + list(range(n - 50, n))

    sample_set = set(sample_indices)
    parquet_rows: dict[int, dict] = {}
    df = pl.read_parquet(parquet_path).with_row_index("_idx")
    for r in df.filter(pl.col("_idx").is_in(sample_indices)).to_dicts():
        idx = r.pop("_idx")
        parquet_rows[idx] = r
    del df

    # Stream del JSONL, recoge solo las filas pedidas
    json_rows: dict[int, dict] = {}
    for i, frame in enumerate(_iter_tracking_frames(jsonl_path)):
        if i in sample_set:
            json_rows[i] = frame
        if len(json_rows) == len(sample_set):
            break

    errs = []
    for i in sample_indices:
        if i not in json_rows or i not in parquet_rows:
            errs.append(f"row {i} missing")
            continue
        if not deep_equal(json_rows[i], parquet_rows[i]):
            errs.append(f"row {i} differs")
    return len(errs) == 0, errs


# ---- Metadata + rosters (unificados) ----

def extract_pff_metadata(overwrite: bool = False) -> Path:
    """Junta los 64 JSON de Metadata en 1 parquet."""
    out = parquet_dir("pff") / "metadata.parquet"
    if out.exists() and not overwrite:
        return out
    rows = []
    for f in sorted(_METADATA.glob("*.json")):
        data = json.load(open(f))
        rows.extend(data if isinstance(data, list) else [data])
    df = pl.from_dicts(rows, infer_schema_length=None)
    return write_parquet(df, out, overwrite=overwrite)


def extract_pff_rosters(overwrite: bool = False) -> Path:
    """Junta los 64 JSON de Rosters en 1 parquet (anade match_id por fila)."""
    out = parquet_dir("pff") / "rosters.parquet"
    if out.exists() and not overwrite:
        return out
    rows = []
    for f in sorted(_ROSTERS.glob("*.json")):
        gid = int(f.stem)
        data = json.load(open(f))
        for r in data:
            r["match_id"] = gid
            rows.append(r)
    df = pl.from_dicts(rows, infer_schema_length=None)
    return write_parquet(df, out, overwrite=overwrite)


# ---- All-in-one (orden seguro de memoria: ligeros primero) ----

def extract_all(overwrite: bool = False) -> dict:
    """Ejecuta los 4 extractores PFF en orden seguro de memoria."""
    out = {}
    out["metadata"] = extract_pff_metadata(overwrite=overwrite)
    out["rosters"]  = extract_pff_rosters(overwrite=overwrite)
    out["events"]   = extract_pff_events(overwrite=overwrite)
    out["tracking"] = extract_pff_tracking(overwrite=overwrite)
    return out
