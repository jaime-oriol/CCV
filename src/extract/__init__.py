"""extract - Extractores raw -> parquet con tipos anidados.

3 fuentes (PFF WC22, StatsBomb open, Wyscout open) + auditor lossless.

Uso:
    from src.extract import pff, statsbomb, wyscout
    pff.extract_all()
    statsbomb.extract_all()
    wyscout.extract_all()

    # O todo de golpe:
    from src.extract import extract_everything
    extract_everything()
"""
from . import pff, statsbomb, wyscout
from ._common import (
    DATA, DATA_PFF, DATA_PUB, PARQUET,
    deep_equal, parquet_dir, roundtrip_check, scan_glob, write_parquet,
)


def extract_everything(overwrite: bool = False) -> dict:
    """Ejecuta los 3 extractores en orden y devuelve resumen."""
    return {
        "wyscout":   wyscout.extract_all(overwrite=overwrite),
        "statsbomb": statsbomb.extract_all(overwrite=overwrite),
        "pff":       pff.extract_all(overwrite=overwrite),
    }


__all__ = [
    "pff", "statsbomb", "wyscout",
    "extract_everything",
    "DATA", "DATA_PFF", "DATA_PUB", "PARQUET",
    "parquet_dir", "write_parquet", "roundtrip_check", "deep_equal",
]
