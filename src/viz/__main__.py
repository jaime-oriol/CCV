"""Runner unico de la capa viz. Renderiza toda la baraja de figuras a outputs/viz/.

Uso:
    python -m src.viz                        # baraja completa (default)
    python -m src.viz radar "Messi"          # solo el radar geometrico (8 ejes) standalone
    python -m src.viz report "Messi"         # solo el radar_report (radar + tabla) standalone

La baraja completa incluye:
  - PPCF Mbappe 2-2 (final ARG-FRA, frame 164933 = instante exacto del remate)
  - 2 scatter globales: Remontador x Cerrojo + Ataque tras marcar / tras encajar (511 jug)
  - 2 scatter de Francia: misma estetica que globales pero filtrado al equipo
  - 4 radar reports: Messi (1531), Hakimi (1681), Mbappe (3870), Brozovic (8129)

El event-study causal (figura de validacion / metodos) NO entra aqui — se genera
aparte via `python -m src.viz.figures`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viz import ppcf, radar, radar_report, scatter, scatter_team
from viz.common import TOURNAMENT_ES, team_es

_OUT = _SRC.parent / "outputs" / "viz"
_TABLE = _SRC.parent / "outputs" / "pcj_table.parquet"

# ----------------------------------------------------------------------------
# Players de la baraja completa (cara + escudo cacheados en outputs/assets/)
# ----------------------------------------------------------------------------
_REPORT_PLAYERS = [
    1531,   # Lionel Messi (Argentina · CF)
    1681,   # Achraf Hakimi (Morocco · RB)
    3870,   # Kylian Mbappe (France · LW)
    8129,   # Marcelo Brozovic (Croatia · CM)
]
_TEAM_SCATTER = "France"   # equipo pa los 2 scatter_team de la baraja

# Frame exacto del 2-2 de Mbappe (game_event OTB del remate, P2 = tracking sin espejo)
_PPCF_MATCH = 10517
_PPCF_FRAME = 164933


def _render_radar(df: pl.DataFrame, query: str) -> Path:
    """Render del radar geometrico standalone (con header + logo)."""
    pid = radar._find(df, query)
    r = df.filter(pl.col("pff_player_id") == pid).row(0, named=True)
    out = _OUT / f"radar_{pid}.png"
    radar.player_radar(
        df, pid,
        title=f"{r['player_name']}  ·  Perfil Clutch del Jugador",
        subtitle=(f"{team_es(r['team_name'])} · {r['position_group']} · "
                  f"{int(r['minutes_played'])} min  |  {TOURNAMENT_ES}"),
        save_path=out)
    return out


def make_all() -> None:
    """Renderiza la baraja COMPLETA de figuras de portada del TFM."""
    df = pl.read_parquet(_TABLE)

    # ---- 1) PPCF: 2-2 de Mbappe (Final ARG-FRA, instante exacto del remate) ----
    print(f"[viz] PPCF — 2-2 de Mbappe (Final ARG-FRA, frame {_PPCF_FRAME})...")
    ppcf.plot_ppcf(_PPCF_MATCH, _PPCF_FRAME,
                    save_path=_OUT / "ppcf_mbappe_2_2_final.png")

    # ---- 2) Scatter globales (Remontador x Cerrojo + Ataque tras marcar/encajar) ----
    print(f"[viz] Scatter globales x2 ({df.height} jugadores)...")
    for key in ("remontador_cerrojo", "ataque_marcar_encajar"):
        scatter.opta_scatter(df, config=key,
                              save_path=_OUT / f"scatter_{key}.png")

    # ---- 3) Scatter por equipo (Francia: mismos 2 conceptos, caras + nube torneo) ----
    print(f"[viz] Scatter equipo x2 ({_TEAM_SCATTER})...")
    scatter_team.scatter_team_all(df, _TEAM_SCATTER, _OUT)

    # ---- 4) Radar reports (radar 12 ejes + tabla percentiles) ----
    print(f"[viz] Radar reports x{len(_REPORT_PLAYERS)}...")
    for pid in _REPORT_PLAYERS:
        out = radar_report.player_radar_report(df, pid)
        print(f"       OK -> {out.name}")

    print(f"[viz] OK — todas las figuras en {_OUT}")


if __name__ == "__main__":
    # Subcomandos:
    #   python -m src.viz                    → baraja completa
    #   python -m src.viz radar "Messi"      → radar standalone
    #   python -m src.viz report "Messi"     → radar_report standalone
    if len(sys.argv) > 2 and sys.argv[1] in ("radar", "report"):
        df = pl.read_parquet(_TABLE)
        if sys.argv[1] == "radar":
            print(f"OK -> {_render_radar(df, sys.argv[2])}")
        else:
            pid = radar_report._find(df, sys.argv[2])
            print(f"OK -> {radar_report.player_radar_report(df, pid)}")
    else:
        make_all()
