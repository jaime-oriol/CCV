"""common - Estilo + constantes + helpers compartidos por todas las vizs.

Convencion campo: 105x68 m, origen en el centro (0, 0), metros — encaja
directo con el tracking PFF. Aqui solo viven constantes, paleta, colormaps
y helpers genericos (draw_pitch, add_logo, save_fig). La logica de cada
figura va en su modulo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---- Estilo global (BG dark, texto blanco) ----

BG    = "#313332"
WHITE = "white"
GRID  = "#4a4c4b"
FONT  = "DejaVu Sans"

plt.style.use("default")
plt.rcParams.update({
    "font.family": FONT, "font.size": 10,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.facecolor": BG, "figure.facecolor": BG,
    "text.color": WHITE, "axes.labelcolor": WHITE,
    "xtick.color": WHITE, "ytick.color": WHITE,
})

# Contornos para texto sobre zonas con mucho color (numeros de dorsal,
# etiquetas sobre PPCF, etc.). PE usa contorno negro, PE_S usa BG.
PE   = [pe.withStroke(linewidth=1.5, foreground="black"), pe.Normal()]
PE_S = [pe.withStroke(linewidth=2.6, foreground=BG)]

# ---- Colores de equipo y elementos ----

ATT  = "deepskyblue"   # equipo en posesion / canal ataque
DEF  = "tomato"        # equipo defensor / canal defensa
GK   = "black"
BALL = WHITE

# ---- Colormaps ----

# PPCF: defensor rojo -> neutro gris -> atacante azul. Divergente
# perceptualmente uniforme para que el "tira-y-afloja" entre equipos se vea.
PPCF_CMAP = LinearSegmentedColormap.from_list(
    "ppcf", ["#8B0000", "#777777", "#004D98"])

# Percentil 0-100 (estilo stats_radar): frio (P0) -> calido (P100). Usado
# en la tabla del radar_report para colorear el numero de percentil.
PCT_CMAP = LinearSegmentedColormap.from_list(
    "pct", ["deepskyblue", "cyan", "lawngreen", "yellow",
            "gold", "lightpink", "tomato"])

# Divergente para CATEs con signo (negativo rojo -> 0 gris -> positivo azul).
CATE_CMAP = LinearSegmentedColormap.from_list(
    "cate", ["#8B0000", "#777777", "#004D98"])

# ---- Dimensiones campo PFF ----

PITCH_LENGTH = 105.0
PITCH_WIDTH  = 68.0

# Logo JO Jaime Oriol (firma de las figuras)
_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo.png"


# ---- draw_pitch: campo de futbol en matplotlib puro ----

def draw_pitch(ax: plt.Axes,
               pitch_length: float = PITCH_LENGTH,
               pitch_width: float = PITCH_WIDTH,
               color: str = WHITE, lw: float = 1.0) -> None:
    """Dibuja un campo de futbol centrado en (0,0), metros. Sin mplsoccer."""
    import numpy as np
    L, W = pitch_length / 2, pitch_width / 2

    # Rectangulo exterior + linea de medio campo + circulo central + punto
    ax.plot([-L, L, L, -L, -L], [-W, -W, W, W, -W], color=color, lw=lw, zorder=2)
    ax.plot([0, 0], [-W, W], color=color, lw=lw, zorder=2)
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, ec=color, lw=lw, zorder=2))
    ax.plot(0, 0, "o", ms=3, color=color, zorder=2)

    pa_w, pa_d = 20.16, 16.5   # area grande
    ga_w, ga_d = 9.16, 5.5     # area pequena
    for sign in (-1, 1):                # por cada porteria (izq -1, dcha +1)
        x0, x1 = sign * L, sign * (L - pa_d)
        ax.plot([x0, x1, x1, x0], [-pa_w, -pa_w, pa_w, pa_w],
                color=color, lw=lw, zorder=2)
        x2 = sign * (L - ga_d)
        ax.plot([x0, x2, x2, x0], [-ga_w, -ga_w, ga_w, ga_w],
                color=color, lw=lw, zorder=2)
        ax.plot(sign * (L - 11), 0, "o", ms=3, color=color, zorder=2)
        # Semicirculo del area (arco penalti). Recortado al borde del area.
        theta = np.linspace(-np.pi / 2, np.pi / 2, 50)
        arc_x = sign * (L - 11) + 9.15 * np.cos(theta) * sign
        arc_y = 9.15 * np.sin(theta)
        arc_x[np.abs(arc_x) > (L - pa_d)] = np.nan
        ax.plot(arc_x, arc_y, color=color, lw=lw, zorder=2)
        # Porteria (dos lineas horizontales + larguero detras del fondo)
        for yy in (-3.66, 3.66):
            ax.plot([sign * L, sign * (L + 1.5)], [yy, yy],
                    color=color, lw=lw * 1.5, zorder=2)
        ax.plot([sign * (L + 1.5), sign * (L + 1.5)], [-3.66, 3.66],
                color=color, lw=lw * 1.5, zorder=2)

    # +3 m de aire alrededor del rectangulo del campo
    ax.set_xlim(-L - 3, L + 3)
    ax.set_ylim(-W - 3, W + 3)
    ax.set_aspect("equal")
    ax.axis("off")


def make_pitch(figsize: Tuple[float, float] = (16, 10.4),
               pitch_length: float = PITCH_LENGTH,
               pitch_width: float = PITCH_WIDTH,
               ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Devuelve (fig, ax) con el campo ya dibujado. Si pasas ax, lo reusa."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_facecolor(BG)
    else:
        fig = ax.get_figure()
    draw_pitch(ax, pitch_length, pitch_width)
    return fig, ax


# ---- Helpers de figura ----

def add_logo(fig: plt.Figure, width_frac: float = 0.11,
             margin: float = 0.012, corner: str = "br") -> None:
    """Estampa el logo JO en una esquina como axes dedicado.

    width_frac : ancho del logo como fraccion del ancho de fig (sube -> logo MAS GRANDE)
    margin     : separacion del borde de la fig (sube -> MAS LEJOS del borde)
    corner     : 'br' abajo-dcha (default), 'bl' abajo-izq, 'tr', 'tl'.

    Uso axes dedicado en vez de AnnotationBbox para que sobreviva al
    savefig(bbox_inches='tight') sin que matplotlib lo recorte.
    """
    if not _LOGO_PATH.exists():
        return
    try:
        img = plt.imread(str(_LOGO_PATH))
        logo_aspect = img.shape[1] / img.shape[0]              # ancho / alto
        figW, figH = fig.get_size_inches()
        h_frac = width_frac * (figW / figH) / logo_aspect      # alto proporcional
        # Posicion del logo segun la esquina. 'l'->left, 'r'->right, 't'->top, 'b'->bottom.
        x = margin if corner in ("bl", "tl") else 1.0 - margin - width_frac
        y = margin if corner in ("bl", "br") else 1.0 - margin - h_frac
        ax_logo = fig.add_axes([x, y, width_frac, h_frac])
        ax_logo.imshow(img)
        ax_logo.axis("off")
    except Exception:
        pass


def style_ax(ax: plt.Axes, ygrid: bool = True) -> None:
    """Ejes estilo 'dark journal': sin spines top/right, grid-y suave opcional."""
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    if ygrid:
        ax.grid(True, axis="y", color=GRID, linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)


def save_fig(fig: plt.Figure, path, logo: bool = False) -> None:
    """Guarda la figura con identidad (BG dark, dpi 300).

    logo=True estampa el logo JO en br. Solo para renders de campo (PPCF,
    radar). NO usar en figuras analiticas (event-study) — choca con ejes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if logo:
        add_logo(fig)
    fig.savefig(path, dpi=300, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
