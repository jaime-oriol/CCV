"""figures - Figuras analiticas del CCV (capa causal).

Estilo LIGHT OPTA paper sobre BG blanco: layout 2 (perspective) x 4 (canal)
con header PPCF-style (escudo WC22 + titulo + sub + JO), tipografia Chakra
Petch limpia, banda IC + linea beta, footer scout-style en 1 linea.

NO va en el make_all core showcase: es figura de METODO / validacion causal
del CCV. La interpretacion (pre-trends planos + efecto medio ≈ 0 → la
heterogeneidad se captura en el CATE bayesiano de M14) se cuenta en el TFM,
no en el chart.

Uso:
    python -m src.viz.figures
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viz.common import (ATT, BG, DEF, GRID, LEGEND, MASTER_FIGSIZE,
                         N_PLAYERS_WC22, TEXT, TOURNAMENT_ES, draw_header)

_DID    = _SRC.parent / "data" / "parquet" / "derived" / "did"
_LOGOS  = _SRC.parent / "outputs" / "assets" / "logos"
_WC22   = _LOGOS / "wc22.png"                                       # escudo del torneo (header)

# Nombres paper-quality + orden canonico (mismo lenguaje que scatter/radar).
_CHANNELS = ["ataque", "defensa", "offball", "fisico"]
_CH_LABEL = {
    "ataque":  "Producción ofensiva",
    "defensa": "Acciones defensivas",
    "offball": "Movimiento off-ball",
    "fisico":  "Carga física",
}

# Filas = perspective del shock. ATT (azul) tras encajar (chasing); DEF (rojo)
# tras marcar (protecting). Mismo codigo de color que scatter/radar.
_SHOCKS = [
    ("GOAL_AGAINST", "Tras encajar", ATT),
    ("GOAL_FOR",     "Tras marcar",  DEF),
]

_REF_GREY = "#999999"                                                 # linea y=0 + x=0 (gris medio)


def event_study(save_path=None):
    """Event-study Sun-Abraham (2021): 2 shocks x 4 canales, beta minuto a minuto.

    Layout 2x4 sobre MASTER_FIGSIZE 16x9. Header PPCF-style (escudo WC22 +
    titulo + JO), footer scout-style (1 linea LEGEND). La justificacion
    metodologica completa va en el TFM, no en el chart.
    """
    es = pl.read_parquet(_DID / "event_study.parquet")

    fig = plt.figure(figsize=MASTER_FIGSIZE, facecolor=BG)

    # ---- Header PPCF-style (escudo WC22 + titulo + sub + JO DCHA) ----
    draw_header(fig,
                title="Efecto del shock emocional sobre los 4 canales",
                subtitle=f"CCV  ·  {TOURNAMENT_ES}  ·  {N_PLAYERS_WC22} jugadores",
                escudo_path=_WC22)

    # ---- Grid 2x4: y=[0.10, 0.815]; x=[0.07, 0.985] (deja header arriba y footer abajo) ----
    gs = fig.add_gridspec(
        2, 4,
        left=0.07, right=0.985, top=0.815, bottom=0.10,
        wspace=0.22, hspace=0.32,
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2)]

    # ---- Paneles ----
    for ri, (sh, _row_lbl, color) in enumerate(_SHOCKS):
        for ci, ch in enumerate(_CHANNELS):
            ax = axes[ri][ci]
            ax.set_facecolor(BG)

            d = (es.filter((pl.col("channel") == ch) & (pl.col("shock_type") == sh))
                    .sort("relative_min"))
            x = d["relative_min"].to_numpy()
            b = d["beta"].to_numpy()
            lo, hi = d["ci_lo"].to_numpy(), d["ci_hi"].to_numpy()

            ax.axvspan(0, 10, color=color, alpha=0.05, zorder=0)        # zona post-shock tenue
            ax.axhline(0, color=_REF_GREY, lw=0.7, alpha=0.6, zorder=1) # ref y=0
            ax.axvline(0, color=_REF_GREY, lw=0.8, ls=(0, (3, 3)),
                       alpha=0.7, zorder=2)                              # instante del gol (x=0)
            ax.fill_between(x, lo, hi, color=color, alpha=0.18,
                             edgecolor="none", zorder=2)                 # IC 95%
            ax.plot(x, b, "-", color=color, lw=1.4, alpha=0.95, zorder=3)
            ax.plot(x, b, "o", color=color, ms=3.0, mec=BG, mew=0.6, zorder=4)

            if ri == 0:
                ax.set_title(_CH_LABEL[ch], color=TEXT,
                              fontsize=12, fontweight="bold", pad=8)
            if ri == 1:
                ax.set_xlabel("Minuto", color=TEXT,
                              fontsize=10, labelpad=3)
            if ci == 0:
                ax.set_ylabel("Efecto",
                              color=TEXT, fontsize=11, labelpad=4)

            ax.set_xlim(-10.5, 10.5)
            ax.set_xticks(range(-10, 11, 5))
            ax.tick_params(labelsize=9, colors=TEXT,
                            length=3, width=0.7, pad=2)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
                ax.spines[s].set_linewidth(0.7)

    # ---- Etiquetas de fila (perspective) rotadas a la izquierda ----
    for ri, (_, row_lbl, _) in enumerate(_SHOCKS):
        pos = axes[ri][0].get_position()
        y_center = pos.y0 + pos.height / 2.0
        fig.text(0.015, y_center, row_lbl, ha="left", va="center",
                  color=TEXT, fontsize=12, fontweight="bold", rotation=90)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, facecolor=BG, bbox_inches=None)
        plt.close(fig)
    return fig


if __name__ == "__main__":
    out = "outputs/viz/event_study.png"
    event_study(save_path=out)
    print(f"OK -> {out}")
