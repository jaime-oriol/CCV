"""scatter - Diamond scatter global Remontador x Cerrojo (511 jugadores).

Diamante rotado 45 grados (floating_axes). Cada punto = 1 jugador del
torneo, coloreado por percentil combinado (PCT_CMAP). Top-10 etiquetado
con adjustText. Lineas mediana globales -> 4 cuadrantes.

Indices PCJ son CATEs con signo, asi que la normalizacion es min-max
(v-min)/(max-min) en vez de v/max.

Uso:
    python -m src.viz.scatter
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist.grid_finder import DictFormatter, FixedLocator
from PIL import Image

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viz.common import BG, PCT_CMAP, WHITE, _LOGO_PATH

_TABLE = _SRC.parent / "outputs" / "pcj_table.parquet"

try:
    import adjustText
    _HAS_ADJUST = True
except ImportError:
    _HAS_ADJUST = False

# 6 marcas por eje. Con 11 el diamante rotado se satura visualmente.
_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def diamond_scatter(df: pl.DataFrame,
                     x_metric: str = "chasing_clutch_idx",
                     y_metric: str = "protecting_clutch_idx",
                     x_pct: str = "pct_chasing_global",
                     y_pct: str = "pct_protecting_global",
                     save_path=None):
    """Diamond scatter rotado 45 grados, Remontador x Cerrojo."""
    pdf = df.to_pandas()
    left = pdf[x_metric].fillna(0.0)      # Remontador (eje izquierdo del diamante)
    right = pdf[y_metric].fillna(0.0)     # Cerrojo    (eje inferior del diamante)

    # Normalizacion min-max -> [0, 0.99]. Los CATEs son con signo, no v/max.
    lmin, lmax = float(left.min()), float(left.max())
    rmin, rmax = float(right.min()), float(right.max())
    left_n = 0.99 * (left - lmin) / (lmax - lmin)
    right_n = 0.99 * (right - rmin) / (rmax - rmin)

    l_med = float(left_n.median())        # mediana P50 Remontador (global)
    r_med = float(right_n.median())       # mediana P50 Cerrojo (global)

    # Top jugadores a etiquetar: P81+ en ambos ejes, fallback P75+
    px, py = pdf[x_pct].to_numpy() * 100.0, pdf[y_pct].to_numpy() * 100.0
    pdf = pdf.assign(_px=px, _py=py)
    top = pdf[(pdf["_px"] >= 81) & (pdf["_py"] >= 81)]
    if len(top) < 5:
        top = pdf[(pdf["_px"] >= 75) & (pdf["_py"] >= 75)]
    top = top.assign(_tot=top["_px"] + top["_py"]).nlargest(10, "_tot")

    fig = plt.figure(figsize=(10, 11.5), facecolor=BG)            # ↑ figsize -> figura MAS GRANDE

    # ---- Header style pass-plot: titulo + subtitulo + logo JO ----
    fig.text(0.50, 0.965, "Quien tira y quien sostiene",
              ha="center", va="center", color=WHITE, fontsize=20, fontweight="bold")
    fig.text(0.50, 0.935,
              "Mundial Qatar 2022  ·  511 jugadores  ·  cambio post-shock aislando "
              "lo que aporta el resto del equipo",
              ha="center", va="center", color=WHITE, fontsize=11)
    if _LOGO_PATH.exists():
        limg = Image.open(_LOGO_PATH)
        ab = AnnotationBbox(
            OffsetImage(np.asarray(limg.convert("RGBA")), zoom=0.13),  # ↑ zoom -> logo MAS GRANDE
            (0.87, 0.945), frameon=False,                              # ↑ X -> logo MAS a la DERECHA
            xycoords="figure fraction", box_alignment=(0.5, 0.5))      # ancla CENTRO del logo en (X,Y)
        ab.set_clip_on(False)
        fig.add_artist(ab)

    # ---- Marcas de eje con valor REAL (con signo) ----
    left_dict  = {i: f"{lmin + (i / 0.99) * (lmax - lmin):+.3f}" for i in _TICKS}
    right_dict = {i: f"{rmin + (i / 0.99) * (rmax - rmin):+.3f}" for i in _TICKS}

    # ---- floating_axes rotado 45 deg (diamante) ----
    transform = Affine2D().rotate_deg(45)
    helper = floating_axes.GridHelperCurveLinear(
        transform, (0, 1.001, 0, 1.001),
        grid_locator1=FixedLocator(_TICKS), grid_locator2=FixedLocator(_TICKS),
        tick_formatter1=DictFormatter(right_dict),
        tick_formatter2=DictFormatter(left_dict))
    ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=helper)
    # Posicion del diamante dentro de la figura.
    # [left, bottom, width, height] en fracciones de fig.
    ax.set_position([0.08, 0.07, 0.84, 0.78], which="both")        # ↑ width/height -> diamante MAS GRANDE
    aux = ax.get_aux_axes(transform)
    ax = fig.add_axes(ax)
    aux.patch = ax.patch

    # Estilo ejes: solo left + bottom del diamante (otros invisibles)
    ax.axis["left"].line.set_color(WHITE)
    ax.axis["bottom"].line.set_color(WHITE)
    ax.axis["right"].set_visible(False)
    ax.axis["top"].set_visible(False)
    ax.axis["left"].major_ticklabels.set(rotation=0, ha="center", fontsize=8.5)
    ax.axis["bottom"].major_ticklabels.set(fontsize=8.5)
    ax.axis["bottom"].major_ticklabels.set_pad(6)
    for side, lbl in (("left",   "REMONTADOR  ·  reaccion tras encajar un gol"),
                       ("bottom", "CERROJO  ·  aguante tras marcar un gol")):
        ax.axis[side].set_label(lbl)
        ax.axis[side].label.set(color=WHITE, fontweight="bold", fontsize=11)
        ax.axis[side].LABELPAD += 9                                  # separa label del eje
    ax.axis["left"].label.set_rotation(0)
    ax.grid(alpha=0.18, color=WHITE)

    # Puntos: color por suma de percentiles (PCT_CMAP), borde blanco
    aux.scatter(right_n, left_n, c=left_n + right_n, cmap=PCT_CMAP,
                 edgecolor=WHITE, s=58, lw=0.5, zorder=2, alpha=0.8)

    # Lineas mediana globales (parten el diamante en 4 cuadrantes)
    aux.plot([r_med, r_med], [0.0, 1.001], color=WHITE, lw=2.4, alpha=0.9,
              ls=(0, (7, 4)), zorder=6, solid_capstyle="round")
    aux.plot([0.0, 1.001], [l_med, l_med], color=WHITE, lw=2.4, alpha=0.9,
              ls=(0, (7, 4)), zorder=6, solid_capstyle="round")

    # ---- Top jugadores etiquetados (adjustText evita overlap) ----
    texts = []
    for i, p in top.iterrows():
        parts = str(p.get("player_name", i)).split()
        short = f"{parts[0][0]}. {parts[-1]}" if len(parts) > 1 else parts[0]
        texts.append(aux.annotate(
            short, xy=(right_n.loc[i], left_n.loc[i]), color="yellow",
            fontsize=8.5, fontweight="bold", ha="center", va="center", zorder=4,
            bbox=dict(boxstyle="round,pad=0.22", facecolor=BG, edgecolor="yellow",
                       alpha=0.95, linewidth=1)))
    if _HAS_ADJUST and texts:
        adjustText.adjust_text(texts, ax=aux, force_text=1.6,
                                expand_text=(2.1, 2.1),
                                arrowprops=dict(arrowstyle="-", color="yellow",
                                                 alpha=0.9, linewidth=1.2))

    # ---- Carteles laterales (lenguaje futbol) ----
    fig.text(0.20, 0.62, "Por este lado\nLOS QUE TIRAN DEL EQUIPO\ncuando toca remontar",
              ha="center", va="center", color=WHITE, fontsize=10, linespacing=1.5)
    # ↑ X (0.20) -> cartel a la DERECHA; ↑ Y -> cartel SUBE
    fig.text(0.80, 0.62, "Por este lado\nLOS QUE AGUANTAN EL RESULTADO\ncuando hay que cerrar",
              ha="center", va="center", color=WHITE, fontsize=10, linespacing=1.5)
    fig.text(0.50, 0.72, "ARRIBA: los que hacen LAS DOS COSAS",
              ha="center", va="center", color=WHITE, fontsize=10,
              fontweight="bold", style="italic")

    # Nota al pie (esquina inferior izquierda)
    fig.text(0.045, 0.04, "Lineas discontinuas: la mediana (percentil 50) "
              "de cada indice — parten el campo en 4 cuadrantes.",
              ha="left", va="bottom", color=WHITE, fontsize=8.5, style="italic")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, facecolor=BG, bbox_inches="tight")
        plt.close(fig)
    return fig


if __name__ == "__main__":
    df = pl.read_parquet(_TABLE)
    out = "outputs/viz/scatter_remontador_cerrojo.png"
    diamond_scatter(df, save_path=out)
    print(f"OK -> {out}  ({df.height} jugadores)")
