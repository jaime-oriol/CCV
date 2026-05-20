"""Runner unico: renderiza las figuras core del PCJ a outputs/viz/.

Uso:
    python -m src.viz                        # PPCF + scatter + event-study + report
    python -m src.viz radar "Messi"          # solo el radar
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
    """Radar individual de un jugador (id o substring de nombre)."""
    pid = radar._find(df, query)
    r = df.filter(pl.col("pff_player_id") == pid).row(0, named=True)
    out = _OUT / f"radar_{pid}.png"
    radar.player_radar(
        df, pid,
        title=f"{r['player_name']}  ·  Perfil Clutch del Jugador",
        subtitle=f"{r['team_name']}  ·  {r['position_group']}  ·  "
                 f"{int(r['minutes_played'])} min  —  Mundial Qatar 2022",
        save_path=out)
    return out


def make_all() -> None:
    """Renderiza PPCF + scatter + event-study + report (jugador portada)."""
    print("[viz] PPCF — gol de Messi (ARG-MEX, periodo 2 min 63)...")
    fnum = ppcf.frame_for_clock(3835, period=2, clock_s=3812 - 3)
    ppcf.plot_ppcf(3835, fnum, save_path=_OUT / "ppcf_messi_arg_mex.png")

    print("[viz] Scatter Remontador x Cerrojo...")
    scatter.diamond_scatter(pl.read_parquet(_TABLE),
                             save_path=_OUT / "scatter_remontador_cerrojo.png")

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
