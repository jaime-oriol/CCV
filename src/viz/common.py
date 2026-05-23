"""common - Estilo + constantes + helpers compartidos por todas las vizs.

Identidad LIGHT OPTA-STYLE: fondo blanco, textos negros, leyenda gris medio,
accent saturado azul/rojo. Convencion campo: 105x68 m, origen en el centro
(0, 0), metros — encaja directo con tracking PFF. Helpers genericos
(draw_pitch, draw_header PPCF-style, add_logo, save_fig).

WHITE/PE retro-compat: WHITE apunta a TEXT (negro) — codigo viejo que pinta
con WHITE sigue funcionando (texto/lineas en negro sobre blanco, que es lo
correcto en este estilo). Para uso nuevo, preferir TEXT / LEGEND.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

# ---- Paleta light Opta ----

BG      = "#ffffff"          # fondo blanco puro
TEXT    = "#000000"          # textos principales (titulos, labels, ejes)
LEGEND  = "#666666"          # textos de leyenda / secundarios (gris Opta)
GRID    = "#dddddd"          # gridlines y spines secundarios
PITCH   = "#000000"          # lineas del campo

# Retro-compat: WHITE = TEXT (codigo viejo que pintaba con WHITE sigue OK,
# ahora pintando negro sobre blanco que es lo que toca en este estilo).
WHITE = TEXT

# Accent saturado (alto contraste sobre blanco)
ATT     = "#1d4ed8"          # azul Opta (atacante / canal ofensivo / chasing)
DEF     = "#dc2626"          # rojo Opta (defensor / canal defensivo / protecting)
GK      = "#000000"
BALL    = "#000000"
NEUTRAL = "#e5e7eb"          # gris muy claro para el centro de los cmaps divergentes

# Font stack profesional Opta-style: prioridad geometric sans modernos, con
# fallback a clasicos del sistema. matplotlib usa el primero disponible.
FONT_STACK = ["Inter", "Source Sans 3", "Source Sans Pro",
              "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
FONT = FONT_STACK[0]   # nombre primario (para legado)

plt.style.use("default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_STACK,
    "font.size": 10,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.facecolor": BG, "figure.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
})

# Path effects: contorno BG (blanco) para texto sobre zonas de color
PE   = [pe.withStroke(linewidth=1.5, foreground=BG), pe.Normal()]
PE_S = [pe.withStroke(linewidth=2.6, foreground=BG)]

# ---- Colormaps ----

# PPCF: defensor (rojo) -> neutro -> atacante (azul). Divergente perceptualmente
# uniforme; sobre BG blanco la NEUTRAL gris claro hace de "frontera invisible".
PPCF_CMAP = LinearSegmentedColormap.from_list("ppcf", [DEF, NEUTRAL, ATT])

# Percentil para scatter (combinado x+y): gradiente morado -> pink, vibrante
# y aesthetic sobre BG blanco. Todos los stops contrastan bien (sin yellow ni
# grey que se diluyan). 3 stops para gradiente suave.
PCT_CMAP = LinearSegmentedColormap.from_list("pct", [
    "#7e22ce",   # purple-700: percentil bajo (morado profundo)
    "#c026d3",   # fuchsia-600: percentil medio (vibrante)
    "#ec4899",   # pink-500: percentil alto (rosa Opta)
])

# CATE divergente (negativo rojo -> 0 neutro -> positivo azul).
CATE_CMAP = LinearSegmentedColormap.from_list("cate", [DEF, NEUTRAL, ATT])

# ---- Dimensiones canonicas ----

# TODAS las vizs (PPCF, scatters, radar, event_study) usan este figsize
# master landscape para coherencia visual + casar perfecto con ancho de
# documento. 14x8 = 1.75:1 paper-friendly.
MASTER_FIGSIZE = (14.0, 8.0)

PITCH_LENGTH = 105.0
PITCH_WIDTH  = 68.0

# Logo JO (version sobre fondo claro)
_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo_2.png"


# ---- draw_pitch: campo de futbol en matplotlib puro ----

def draw_pitch(ax: plt.Axes,
               pitch_length: float = PITCH_LENGTH,
               pitch_width: float = PITCH_WIDTH,
               color: str = PITCH, lw: float = 1.0) -> None:
    """Dibuja un campo de futbol centrado en (0,0), metros. Sin mplsoccer."""
    L, W = pitch_length / 2, pitch_width / 2

    # Rectangulo exterior + linea de medio campo + circulo central + punto
    ax.plot([-L, L, L, -L, -L], [-W, -W, W, W, -W], color=color, lw=lw, zorder=2)
    ax.plot([0, 0], [-W, W], color=color, lw=lw, zorder=2)
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, ec=color, lw=lw, zorder=2))
    ax.plot(0, 0, "o", ms=3, color=color, zorder=2)

    pa_w, pa_d = 20.16, 16.5
    ga_w, ga_d = 9.16, 5.5
    for sign in (-1, 1):
        x0, x1 = sign * L, sign * (L - pa_d)
        ax.plot([x0, x1, x1, x0], [-pa_w, -pa_w, pa_w, pa_w],
                color=color, lw=lw, zorder=2)
        x2 = sign * (L - ga_d)
        ax.plot([x0, x2, x2, x0], [-ga_w, -ga_w, ga_w, ga_w],
                color=color, lw=lw, zorder=2)
        ax.plot(sign * (L - 11), 0, "o", ms=3, color=color, zorder=2)
        theta = np.linspace(-np.pi / 2, np.pi / 2, 50)
        arc_x = sign * (L - 11) + 9.15 * np.cos(theta) * sign
        arc_y = 9.15 * np.sin(theta)
        arc_x[np.abs(arc_x) > (L - pa_d)] = np.nan
        ax.plot(arc_x, arc_y, color=color, lw=lw, zorder=2)
        for yy in (-3.66, 3.66):
            ax.plot([sign * L, sign * (L + 1.5)], [yy, yy],
                    color=color, lw=lw * 1.5, zorder=2)
        ax.plot([sign * (L + 1.5), sign * (L + 1.5)], [-3.66, 3.66],
                color=color, lw=lw * 1.5, zorder=2)

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


# ---- draw_header: header reusable PPCF-style ----

def _load_rgba(path) -> Optional[np.ndarray]:
    """Carga un PNG como np.array RGBA (maneja modo Palette de FotMob/etc)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return np.asarray(Image.open(p).convert("RGBA"))
    except Exception:
        return None


def draw_header(fig: plt.Figure, *, title: str,
                subtitle: Optional[str] = None,
                subtitle2: Optional[str] = None,
                escudo_path=None, logo_path=None,
                hdr_band: Tuple[float, float] = (0.84, 0.96),
                escudo_x: float = 0.04, text_x: float = 0.13, jo_x: float = 0.96,
                escudo_zoom: float = 1.20, jo_zoom: float = 0.15,
                title_size: float = 22, sub_size: float = 13) -> None:
    """Header PPCF-style: escudo grande IZQ | titulo TOP + sub(s) LEFT | JO DCHA.

    Replica exacta del layout de PPCF para coherencia entre todas las vizs.
    title arriba bold, hasta 2 subtitulos stacked debajo. Escudo y JO logo
    en la altura intermedia del bloque (entre title y sub1).

    hdr_band : (bot, top) en fig coords del bloque-header.
               Default (0.84, 0.96) = height 0.12.
    """
    bot, top = hdr_band
    h = top - bot
    # Posiciones internas (copia exacta de PPCF._draw_header)
    title_y  = bot + 0.82 * h
    sub_y    = bot + 0.46 * h
    sub2_y   = bot + 0.20 * h
    escudo_y = bot + 0.50 * h
    jo_y     = bot + 0.42 * h

    if escudo_path is not None:
        img = _load_rgba(escudo_path)
        if img is not None:
            ab = AnnotationBbox(
                OffsetImage(img, zoom=escudo_zoom),
                (escudo_x, escudo_y), frameon=False,
                xycoords="figure fraction", box_alignment=(0.0, 0.5),
            )
            ab.set_clip_on(False)
            fig.add_artist(ab)

    fig.text(text_x, title_y, title, ha="left", va="center",
             color=TEXT, fontsize=title_size, fontweight="bold")
    if subtitle:
        fig.text(text_x, sub_y, subtitle, ha="left", va="center",
                 color=TEXT, fontsize=sub_size)
    if subtitle2:
        fig.text(text_x, sub2_y, subtitle2, ha="left", va="center",
                 color=TEXT, fontsize=sub_size)

    img_jo = _load_rgba(logo_path if logo_path is not None else _LOGO_PATH)
    if img_jo is not None:
        ab = AnnotationBbox(
            OffsetImage(img_jo, zoom=jo_zoom),
            (jo_x, jo_y), frameon=False,
            xycoords="figure fraction", box_alignment=(1.0, 0.5),
        )
        ab.set_clip_on(False)
        fig.add_artist(ab)


# ---- Legacy helpers ----

def add_logo(fig: plt.Figure, width_frac: float = 0.11,
             margin: float = 0.012, corner: str = "br") -> None:
    """Estampa el logo JO en una esquina como axes dedicado. (Legacy — usar
    `draw_header` para nuevas vizs.)
    """
    if not _LOGO_PATH.exists():
        return
    try:
        img = plt.imread(str(_LOGO_PATH))
        logo_aspect = img.shape[1] / img.shape[0]
        figW, figH = fig.get_size_inches()
        h_frac = width_frac * (figW / figH) / logo_aspect
        x = margin if corner in ("bl", "tl") else 1.0 - margin - width_frac
        y = margin if corner in ("bl", "br") else 1.0 - margin - h_frac
        ax_logo = fig.add_axes([x, y, width_frac, h_frac])
        ax_logo.imshow(img)
        ax_logo.axis("off")
    except Exception:
        pass


def style_ax(ax: plt.Axes, ygrid: bool = True) -> None:
    """Ejes estilo paper light: sin spines top/right, spines suaves, grid suave."""
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(0.8)
    if ygrid:
        ax.grid(True, axis="y", color=GRID, linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)


def save_fig(fig: plt.Figure, path, logo: bool = False) -> None:
    """Guarda la figura con identidad (BG blanco, dpi 300)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if logo:
        add_logo(fig)
    fig.savefig(path, dpi=300, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
