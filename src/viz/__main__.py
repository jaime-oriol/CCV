"""Runner unico de la capa viz. Renderiza las figuras core a outputs/viz/.

Uso:
    python -m src.viz                        # PPCF + scatter + event-study + report
    python -m src.viz radar "Messi"          # solo el radar geometrico
    python -m src.viz report "Messi"         # radar_report (radar + tabla)
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viz import radar_report, figures, ppcf, radar, scatter

_OUT = _SRC.parent / "outputs" / "viz"
_TABLE = _SRC.parent / "outputs" / "pcj_table.parquet"


def _render_radar(df: pl.DataFrame, query: str) -> Path:
    """Render del radar geometrico standalone (con header + logo)."""
    pid = radar._find(df, query)
    r = df.filter(pl.col("pff_player_id") == pid).row(0, named=True)
    out = _OUT / f"radar_{pid}.png"
    radar.player_radar(
        df, pid,
        title=f"{r['player_name']}  ·  Perfil Clutch del Jugador",
        subtitle=f"{r['team_name']}  ·  {r['position_group']}  ·  "
                  f"{int(r['minutes_played'])} min  ·  Mundial Qatar 2022",
        save_path=out)
    return out


def make_all() -> None:
    """Renderiza las 4 figuras core para la portada del repo / TFM."""
    print("[viz] PPCF — 2-2 de Mbappe (Final ARG-FRA, min 81, instante de la volea)...")
    # frame 164933 = instante EXACTO del remate (game_event OTB de Mbappe, sync via
    # start_frame del evento del disparo; P2 regulacion = tracking limpio, sin espejo).
    ppcf.plot_ppcf(10517, 164933, save_path=_OUT / "ppcf_mbappe_2_2_final.png")

    print("[viz] Scatters globales (Remontador x Cerrojo + ataque marcar vs presion, 511 jug)...")
    _tbl = pl.read_parquet(_TABLE)
    for _key in ("remontador_cerrojo", "ataque_marcar_presion"):
        scatter.diamond_scatter(_tbl, config=_key,
                                 save_path=_OUT / f"scatter_{_key}.png")

    print("[viz] Event-study causal (M12)...")
    figures.event_study(save_path=_OUT / "event_study.png")

    print("[viz] Radar report — Messi (portada)...")
    df = pl.read_parquet(_TABLE)
    radar_report.player_radar_report(df, radar_report._find(df, "Messi"))

    print(f"[viz] OK — figuras en {_OUT}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] in ("radar", "report"):
        df = pl.read_parquet(_TABLE)
        if sys.argv[1] == "radar":
            print(f"OK -> {_render_radar(df, sys.argv[2])}")
        else:
            pid = radar_report._find(df, sys.argv[2])
            print(f"OK -> {radar_report.player_radar_report(df, pid)}")
    else:
        make_all()
