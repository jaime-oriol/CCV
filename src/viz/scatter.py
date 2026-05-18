"""scatter - Scatter Remontador x Cerrojo de los jugadores del PCJ.

Estilo portado de jaime-oriol/footballdecoded (viz/scatter.py): puntos
coloreados por percentil combinado (node_cmap), top-10 etiquetados con bbox,
region sombreada P20-P80, lineas de referencia.

Adaptacion: los indices PCJ son CATEs con signo (centrados en 0), no stats
per-90 positivas — se usan ejes de cuadrante (0,0 al centro) en vez del
diamante rotado 45 grados, que asume metricas positivas.

Uso:
    python -m src.viz.scatter
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viz.common import BG, GRID, PCT_CMAP, PE, WHITE, add_logo, style_ax

_TABLE = _SRC.parent / "outputs" / "pcj_table.parquet"

try:
    import adjustText
    _HAS_ADJUST = True
except ImportError:
    _HAS_ADJUST = False


def _short(name: str) -> str:
    """'Lionel Messi' -> 'L. Messi'."""
    parts = name.split()
    return f"{parts[0][0]}. {parts[-1]}" if len(parts) > 1 else name


def diamond_scatter(df: pl.DataFrame,
                    x_metric: str = "chasing_clutch_idx",
                    y_metric: str = "protecting_clutch_idx",
                    x_label: str = "Indice Remontador  (empuje + off-ball | post-GA)",
                    y_label: str = "Indice Cerrojo  (defensa + fisico | post-GF)",
                    title: str = "Perfil Clutch del Jugador  —  Remontador x Cerrojo",
                    subtitle: str = "Mundial Qatar 2022  ·  CATE bayesiano por jugador "
                                    "(eta individual, neto de equipo y posicion)",
                    save_path=None):
    """Scatter de cuadrantes x_metric x y_metric con top-10 etiquetado."""
    pdf = df.to_pandas()
    x = pdf[x_metric].to_numpy(dtype=float)
    y = pdf[y_metric].to_numpy(dtype=float)

    # Color por percentil combinado (idem node_cmap de footballdecoded)
    rx = pdf[x_metric].rank(pct=True).to_numpy()
    ry = pdf[y_metric].rank(pct=True).to_numpy()
    cval = (rx + ry) / 2

    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    fig.set_facecolor(BG)
    style_ax(ax, ygrid=False)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)

    # Region sombreada P20-P80 en cualquiera de los dos ejes
    qx = np.quantile(x, [0.2, 0.8])
    qy = np.quantile(y, [0.2, 0.8])
    ax.axvspan(qx[0], qx[1], color="grey", alpha=0.10, zorder=0)
    ax.axhspan(qy[0], qy[1], color="grey", alpha=0.10, zorder=0)

    # Ejes de cuadrante en 0
    ax.axvline(0, color="#7a7c7b", lw=1.0, ls=(0, (3, 3)), zorder=1)
    ax.axhline(0, color="#7a7c7b", lw=1.0, ls=(0, (3, 3)), zorder=1)

    ax.scatter(x, y, c=cval, cmap=PCT_CMAP, edgecolor=WHITE, s=70, lw=0.5,
               alpha=0.85, zorder=3)

    # Top-10 por suma de percentiles -> etiquetado
    top_idx = np.argsort(-(rx + ry))[:10]
    texts = []
    for i in top_idx:
        name = _short(str(pdf.iloc[i].get("player_name", pdf.index[i])))
        t = ax.annotate(name, (x[i], y[i]), color="yellow", fontsize=8.5,
                        fontweight="bold", ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=BG,
                                  edgecolor="yellow", alpha=0.95, linewidth=1),
                        zorder=6)
        texts.append(t)
    if _HAS_ADJUST and texts:
        adjustText.adjust_text(texts, ax=ax, force_text=1.4,
                               expand_text=(1.8, 1.8),
                               arrowprops=dict(arrowstyle="-", color="yellow",
                                               alpha=0.8, lw=1.0))

    # Etiqueta del cuadrante "dual clutch"
    ax.text(0.985, 0.985, "DUAL CLUTCH", transform=ax.transAxes, ha="right",
            va="top", color="#c8c8c8", fontsize=10, fontweight="bold",
            style="italic", path_effects=PE)

    ax.set_xlabel(x_label, fontsize=11, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=11, fontweight="bold")
    fig.text(0.5, 0.975, title, ha="center", va="top", color=WHITE,
             fontsize=15, fontweight="bold")
    fig.text(0.5, 0.945, subtitle, ha="center", va="top", color="#c8c8c8",
             fontsize=9.5)
    fig.text(0.5, 0.022, "Region sombreada: percentil 20-80 en cada eje.",
             ha="center", color="#c8c8c8", fontsize=9, style="italic")
    add_logo(fig, width_frac=0.12)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
    return fig


if __name__ == "__main__":
    df = pl.read_parquet(_TABLE)
    out = "outputs/viz/scatter_remontador_cerrojo.png"
    diamond_scatter(df, save_path=out)
    print(f"OK -> {out}  ({df.height} jugadores)")
