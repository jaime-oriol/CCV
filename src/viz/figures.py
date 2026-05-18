"""figures - Figuras analiticas del PCJ (capa causal).

Patron portado de jaime-oriol/Diagonality_3D (deliverable/make_figures.py):
ejes 'dark journal', grid-y suave, anotaciones n=, identidad Diagonality.

Figura principal: event-study Sun-Abraham (M12) — el efecto causal del shock
por minuto relativo, con banda de IC95%, en los 4 canales x 2 shocks.

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

from viz.common import ATT, BG, DEF, GRID, WHITE, add_logo, style_ax

_DID = _SRC.parent / "data" / "parquet" / "derived" / "did"

_CH_LABEL = {"ataque": "Empuje Ofensivo", "defensa": "Solidez Defensiva",
             "offball": "Inteligencia Off-ball", "fisico": "Pulso Fisico"}
_CH_ORDER = ["ataque", "defensa", "offball", "fisico"]
_SH = [("GOAL_AGAINST", "post-GA  (su equipo encaja)", ATT),
       ("GOAL_FOR",     "post-GF  (su equipo marca)",  DEF)]


def event_study(save_path=None):
    """Event-study Sun-Abraham: beta_tau +-10 min con banda IC95%, 4x2 paneles."""
    es = pl.read_parquet(_DID / "event_study.parquet")

    fig, axes = plt.subplots(4, 2, figsize=(11.5, 13.5), sharex=True)
    fig.set_facecolor(BG)

    for ri, ch in enumerate(_CH_ORDER):
        for ci, (sh, sh_lbl, color) in enumerate(_SH):
            ax = axes[ri, ci]
            style_ax(ax, ygrid=True)
            d = (es.filter((pl.col("channel") == ch) & (pl.col("shock_type") == sh))
                   .sort("relative_min"))
            x = d["relative_min"].to_numpy()
            b = d["beta"].to_numpy()
            lo = d["ci_lo"].to_numpy()
            hi = d["ci_hi"].to_numpy()

            # Referencias: 0 efecto + instante del shock (tau=0)
            ax.axhline(0, color="#7a7c7b", lw=1.0, ls=(0, (3, 3)), zorder=1)
            ax.axvline(0, color="#7a7c7b", lw=1.0, alpha=0.7, zorder=1)
            # Banda IC95% + linea de efecto
            ax.fill_between(x, lo, hi, color=color, alpha=0.22, zorder=2)
            ax.plot(x, b, "-o", color=color, ms=3.5, lw=1.6, zorder=3)

            ax.set_title(f"{_CH_LABEL[ch]}   ·   {sh_lbl}",
                         color=WHITE, fontsize=10.5, fontweight="bold", pad=8)
            if ri == 3:
                ax.set_xlabel("minuto relativo al shock", fontsize=9.5)
            if ci == 0:
                ax.set_ylabel("efecto causal (z)", fontsize=9.5)
            ax.set_xticks(range(-10, 11, 5))
            ax.tick_params(labelsize=8.5)

    fig.text(0.5, 0.975, "El efecto causal del shock emocional, minuto a minuto",
             ha="center", va="top", color=WHITE, fontsize=15, fontweight="bold")
    fig.text(0.5, 0.952,
             "Event-study Sun-Abraham (2021)  ·  DiD within-player  ·  "
             "bin de referencia tau = -1  ·  banda = IC 95%  (M12, WC22)",
             ha="center", va="top", color="#c8c8c8", fontsize=9.5)
    fig.tight_layout(rect=[0, 0.01, 1, 0.915])
    add_logo(fig, width_frac=0.10)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
    return fig


if __name__ == "__main__":
    out = "outputs/viz/event_study.png"
    event_study(save_path=out)
    print(f"OK -> {out}")
