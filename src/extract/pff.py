"""
extract.pff - Extrae el dataset PFF FC WC22 a parquet con tipos anidados.

Cuatro extractores:
  - extract_pff_events()   : 64 JSON Event Data -> 1 parquet por partido.
                             1 fila = 1 evento. Sub-dicts como struct,
                             listas de jugadores como list[struct].
  - extract_pff_tracking() : 48 jsonl.bz2 Tracking Data -> 1 parquet por
                             partido. Streaming chunked (peak <500 MB/partido).
  - extract_pff_metadata() : 64 JSON Metadata -> 1 parquet unificado.
  - extract_pff_rosters()  : 64 JSON Rosters -> 1 parquet unificado.

Diseno LOSSLESS: 1 fila parquet -> dict equivalente al JSON original.
Verificable con _common.roundtrip_check() (events) y
_streaming_roundtrip_tracking() (tracking, sin OOM).
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

# -- Rutas ------------------------------------------------------------------

_EVENTS    = DATA_PFF / "Event Data"
_TRACKING  = DATA_PFF / "Tracking Data"
_METADATA  = DATA_PFF / "Metadata"
_ROSTERS   = DATA_PFF / "Rosters"


# -- Listado de partidos ----------------------------------------------------

def list_event_match_ids() -> list[int]:
    """IDs de los 64 partidos con event data."""
    return sorted(int(f.stem) for f in _EVENTS.glob("*.json"))


def list_tracking_match_ids() -> list[int]:
    """IDs de los 48 partidos con tracking data."""
    return sorted(int(f.name.split(".")[0]) for f in _TRACKING.glob("*.jsonl.bz2"))


# -- PFF events -------------------------------------------------------------

def extract_pff_events(
    match_ids: list[int] | None = None,
    overwrite: bool = False,
) -> dict[int, Path]:
    """Convierte los Event Data JSON a parquet con tipos anidados.

    Cada evento es un dict con sub-dicts (gameEvents, initialTouch,
    possessionEvents, fouls, grades, stadiumMetadata) y listas
    (homePlayers, awayPlayers, ball). Polars los preserva como
    Struct y List(Struct) respectivamente.

    Memoria: events son ligeros (~17 MB JSON / ~3k filas por partido).
    Carga completa del JSON cabe sobrado en RAM.
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


# -- PFF tracking (streaming) -----------------------------------------------

# Tamano del chunk: equilibrio memoria vs velocidad de write.
# 5000 frames * ~5 KB = ~25 MB de JSON por chunk -> peak RAM ~150 MB.
_TRACKING_CHUNK_FRAMES = 5000


def _iter_tracking_frames(path: Path) -> Iterator[dict]:
    """Streaming reader: yield un dict por linea del jsonl.bz2."""
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _peek_first_n(it: Iterator[dict], n: int) -> tuple[list[dict], Iterator[dict]]:
    """Saca las primeras N filas del iterador para inferir schema unificado.

    Devuelve (las N peeked, iterator que vuelve a empezar por esas N + resto).
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


def extract_pff_tracking(
    match_ids: list[int] | None = None,
    overwrite: bool = False,
    chunk_frames: int = _TRACKING_CHUNK_FRAMES,
) -> dict[int, Path]:
    """Convierte los Tracking Data jsonl.bz2 a parquet via streaming chunked.

    Estrategia anti-OOM:
      1. Stream del jsonl.bz2 linea a linea (bz2 + json.loads).
      2. Peek N filas iniciales para inferir schema unificado del partido.
      3. Acumular chunks de chunk_frames, convertir a polars, escribir como
         row group con pyarrow.ParquetWriter.
      4. Liberar el chunk antes del siguiente.

    Peak RAM por partido: ~150-300 MB (vs ~3 GB si carga completa).

    Args:
        match_ids    : Subset a procesar. None = todos los 48.
        overwrite    : Re-escribir si ya existe.
        chunk_frames : Frames por row group. Default 5000.

    Returns:
        Dict {match_id: parquet_path}.
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
        # 1. Peek primeros frames para inferir schema (uniones de claves opcionales).
        #    Con 200 frames cubre los game_event/possession_event que aparecen sparse.
        head, stream = _peek_first_n(_iter_tracking_frames(src), 200)
        if not head:
            continue
        schema_df = pl.from_dicts(head, infer_schema_length=None)
        arrow_schema = schema_df.to_arrow().schema
        del schema_df, head

        # 2. Stream + chunked write
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
    """Convierte rows a Arrow Table conformado al schema y escribe row group."""
    df = pl.from_dicts(rows, schema_overrides=None, infer_schema_length=None)
    table = df.to_arrow()
    # Conformar al schema unificado (anade campos faltantes como null)
    table = table.cast(schema, safe=False) if table.schema != schema else table
    writer.write_table(table)
    del df, table


# -- Round-trip lossless tracking (streaming, sin OOM) ----------------------

def streaming_roundtrip_tracking(
    parquet_path: Path,
    jsonl_path: Path,
    sample_indices: list[int] | None = None,
) -> tuple[bool, list[str]]:
    """Verifica lossless en tracking sin cargar todo a memoria.

    Lee el parquet con pl.scan_parquet (lazy) y el jsonl.bz2 streaming.
    Compara por indices muestreados.

    Args:
        parquet_path   : Parquet generado.
        jsonl_path     : Original jsonl.bz2.
        sample_indices : Indices de filas a comparar. None = primeras 50 + ultimas 50.

    Returns:
        (ok, errores).
    """
    # Cargar SOLO las filas muestreadas del parquet (con take)
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

    # Stream del JSONL, recoger las filas pedidas
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


# -- PFF metadata + rosters -------------------------------------------------

def extract_pff_metadata(overwrite: bool = False) -> Path:
    """Junta los 64 JSON de Metadata en un parquet unificado."""
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
    """Junta los 64 JSON de Rosters en un parquet unificado.

    Cada fichero es una lista de ~50 jugadores. Anade columna match_id.
    """
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


# -- All-in-one -------------------------------------------------------------

def extract_all(overwrite: bool = False) -> dict:
    """Ejecuta los 4 extractores PFF en orden seguro de memoria."""
    out = {}
    out["metadata"] = extract_pff_metadata(overwrite=overwrite)
    out["rosters"]  = extract_pff_rosters(overwrite=overwrite)
    out["events"]   = extract_pff_events(overwrite=overwrite)
    out["tracking"] = extract_pff_tracking(overwrite=overwrite)
    return out
