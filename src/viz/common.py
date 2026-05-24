"""common - Estilo + constantes + helpers compartidos por todas las vizs.

Identidad LIGHT OPTA-STYLE: fondo BLANCO, textos NEGROS, leyenda gris medio.
Paleta azul / rojo (saturada light para PPCF) + cmap morado→fuchsia→rosa
(para percentiles en scatter y radar table). Tipografia Chakra Petch (cortes
angulares estilo Opta).

Exporta:
  - Paleta:       BG, TEXT, LEGEND, GRID, PITCH, WHITE(=TEXT), ATT, DEF, GK, NEUTRAL
  - Cmaps:        PPCF_CMAP (DEF→NEUTRAL→ATT), PCT_CMAP (violet→fuchsia→rose)
  - Path effects: PE (halo fino), PE_S (halo gordo, dorsales sobre PPCF)
  - Tipografia:   FONT_STACK (Chakra Petch + fallbacks) — aplicado a rcParams
  - Layout:       MASTER_FIGSIZE (16x9), HEADER_BAND (0.85..1.0), PITCH_LENGTH/WIDTH
  - Helpers:      draw_pitch (campo 105x68), draw_header (escudo+titulo+JO), add_logo (legacy)

WHITE/PE retro-compat: WHITE apunta a TEXT (negro) — codigo viejo que pinta
con WHITE sigue funcionando (negro sobre blanco). Para uso nuevo, preferir
TEXT (texto principal) / LEGEND (textos secundarios).

CHEAT-SHEET (todos los numeros aqui SON ajustables; cada constante documenta
su efecto con ↑/↓ → efecto):
  - MASTER_FIGSIZE     : tamano de scatter/scatter_team (16x9 landscape)
  - HEADER_BAND        : franja vertical reservada al header
  - draw_header(...)   : posiciones internas del escudo/titulo/sub/JO
  - ATT/DEF            : color principal atacante/defensor (PPCF y figures.py)
  - PCT_CMAP           : gradiente percentiles (scatter dots + radar legend)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.patheffects as pe                                # halo de fondo alrededor del texto
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm                               # pa registrar Chakra Petch
import numpy as np
from matplotlib.colors import LinearSegmentedColormap              # pa cmaps custom
from matplotlib.offsetbox import AnnotationBbox, OffsetImage       # pa pegar PNGs (escudos, caras)
from PIL import Image                                                # pa cargar/resize PNGs

# ============================================================================
# REGISTRO DE FUENTES — Chakra Petch al importar el modulo
# ============================================================================
# Carga Chakra Petch (todos los pesos) en el fontManager de matplotlib pa que
# rcParams pueda usarlo. Mira en outputs/assets/fonts y en ~/.local/share/fonts.
# Si no existe el font, matplotlib usa el siguiente del FONT_STACK (Inter, etc).
_CHAKRA_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "assets" / "fonts"
_SYS_FONT_DIR = Path.home() / ".local" / "share" / "fonts"
for _fdir in (_CHAKRA_DIR, _SYS_FONT_DIR):
    if _fdir.exists():
        for _f in _fdir.glob("ChakraPetch*.ttf"):
            try:
                _fm.fontManager.addfont(str(_f))                    # add font sin pasar por install
            except Exception:
                pass                                                 # font corrupto/duplicado → ignora

# ============================================================================
# PALETA LIGHT OPTA
# ============================================================================

BG      = "#ffffff"          # fondo de TODA la viz — cambia a off-white (#fafafa, #f5f5f5) si quieres mas tibio
TEXT    = "#000000"          # color de titulos, labels de ejes, ticks — negro puro siempre
LEGEND  = "#666666"          # color de notas secundarias, disclaimers, mediana inline
                              #   ↑ mas claro (#999) → leyenda casi invisible
                              #   ↓ mas oscuro (#444) → leyenda con mas presencia
GRID    = "#dddddd"          # color de gridlines + spines (gris muy suave)
                              #   ↑ mas oscuro (#bbb) → grid mas visible
                              #   ↓ mas claro  (#eee) → grid casi invisible
PITCH   = "#000000"          # color de las lineas del campo en draw_pitch — negro; pon "#555" pa gris

WHITE = TEXT                  # alias retro-compat — codigo viejo usa WHITE, sigue pintando negro sobre blanco

# colores accent — saturados pa contrastar bien sobre BG blanco
ATT     = "#3b82f6"          # azul vivo light: atacante, canal ofensivo, PPCF blue zones
                              #   ↑ #1d4ed8 → mas oscuro/navy (premium)
                              #   ↑ #2563eb → algo mas profundo (Tailwind blue-600)
DEF     = "#ef4444"          # rojo vivo light: defensor, canal defensivo, PPCF red zones
                              #   ↑ #dc2626 → mas oscuro
                              #   ↑ #b91c1c → carmesi profundo
GK      = "#000000"          # portero en el pitch (ppcf.py) — negro; pon "#555" pa distinguirlo menos
NEUTRAL = "#e5e7eb"          # gris muy claro pa el centro del PPCF (cuando control ≈ 0.5)
                              #   ↑ #d4d4d4 → zona 50/50 mas gris (mas visible)
                              #   ↓ #f3f4f6 → zona neutra casi desaparece

# ============================================================================
# TIPOGRAFIA
# ============================================================================

# matplotlib usa el primer font de la lista que encuentre en el sistema.
# Chakra Petch primero pa tener los cortes angulares Opta-style en mayusculas.
# DejaVu Sans al final = ultimo recurso, siempre esta disponible.
FONT_STACK = ["Chakra Petch", "Inter", "Source Sans 3", "Source Sans Pro",
              "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]

plt.style.use("default")                                            # reset completo antes de aplicar estilo
plt.rcParams.update({
    "font.family": "sans-serif",                                    # familia base — sans-serif Opta-style
    "font.sans-serif": FONT_STACK,                                  # prioridad: Chakra Petch primero
    "font.size": 10,                                                # tamano BASE de texto sin fontsize explicito
                                                                     #   ↑ 11 → todo un poco mas grande
                                                                     #   ↓ 9  → todo un poco mas pequeno
    "figure.dpi": 100,                                              # DPI de pantalla (no afecta al PNG guardado)
    "savefig.dpi": 300,                                             # DPI default al guardar — cada viz override con dpi=N
    "savefig.bbox": None,                                           # SIN tight-crop: el PNG sale EXACTO al figsize
                                                                     # (si fuera "tight" recortaria margenes y el tamano varia)
    "axes.facecolor": BG,                                           # fondo de TODOS los axes = BG (blanco)
    "figure.facecolor": BG,                                         # fondo de TODAS las figuras = BG (blanco)
    "text.color": TEXT,                                             # color default de TODO texto = negro
    "axes.labelcolor": TEXT,                                        # color de xlabel/ylabel = negro
    "xtick.color": TEXT,                                            # color de los ticks X = negro
    "ytick.color": TEXT,                                            # color de los ticks Y = negro
})

# Path effects: halo BG alrededor del texto pa que se lea sobre fondos coloreados.
PE   = [pe.withStroke(linewidth=1.5, foreground=BG), pe.Normal()]   # halo fino (texto normal sobre PPCF)
PE_S = [pe.withStroke(linewidth=2.6, foreground=BG)]                # halo gordo (dorsales sobre PPCF)
                                                                     #   ↑ linewidth → halo mas gordo

# ============================================================================
# COLORMAPS
# ============================================================================

# PPCF (pitch control): defensor (rojo) → neutro (gris claro) → atacante (azul).
# Se reconstruye automaticamente si cambias ATT/DEF/NEUTRAL arriba.
PPCF_CMAP = LinearSegmentedColormap.from_list("ppcf", [DEF, NEUTRAL, ATT])

# Percentil combinado pa scatter (color de cada punto) y radar legend (5 tramos).
# Morado → fuchsia → rosa: tonos vivos en BG blanco, con contraste a lo largo del rango.
PCT_CMAP = LinearSegmentedColormap.from_list("pct", [
    "#5b21b6",   # violet-800 — percentil bajo (mas profundo pa mas contraste)
    "#c026d3",   # fuchsia-600 — percentil medio
    "#f43f5e",   # rose-500 — percentil alto (mas vivido pa mas contraste)
])

# ============================================================================
# DIMENSIONES MASTER (LAYOUT GLOBAL)
# ============================================================================

# TODAS las vizs 16x9 (scatter, scatter_team) usan este figsize → PNG 2400x1350 @ 150dpi.
# PPCF NO usa este figsize (se ajusta al campo, ver ppcf.py).
MASTER_FIGSIZE = (16.0, 9.0)                                        # (ancho, alto) en pulgadas — figura landscape
                                                                     #   ↑ ancho 18 → mas wide pero rompe layouts
                                                                     #   si cambias alto, recalcula HEADER_BAND

# Franja vertical reservada al header (escudo IZQ + titulo+sub + JO DCHA).
HEADER_BAND = (0.85, 1.0)                                           # (y_inferior, y_superior) en fraccion de figura [0..1]
                                                                     #   ↑ y_inferior (0.88) → header mas delgado
                                                                     #   ↓ y_inferior (0.80) → header mas gordo
                                                                     # alto en pulgadas = (1.0 - 0.85) * 9 = 1.35 in

PITCH_LENGTH = 105.0                                                # metros — norma FIFA, no tocar
PITCH_WIDTH  = 68.0                                                 # metros — norma FIFA, no tocar

# Logo JO (Jaime Oriol) — esquina derecha del header en draw_header,
# o esquinas inferiores en add_logo (figures.py, radar.py).
_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo_2.png"


# ============================================================================
# draw_pitch — campo de futbol centrado en (0,0), metros
# ============================================================================

def draw_pitch(ax: plt.Axes,
               pitch_length: float = PITCH_LENGTH,
               pitch_width: float = PITCH_WIDTH,
               color: str = PITCH, lw: float = 1.0) -> None:
    """Dibuja un campo de futbol centrado en (0,0), en metros. Sin mplsoccer."""
    L, W = pitch_length / 2, pitch_width / 2                        # semieje long/ancho — origen en el centro

    # Borde exterior del campo
    ax.plot([-L, L, L, -L, -L], [-W, -W, W, W, -W], color=color, lw=lw, zorder=2)
    # Linea de medio campo vertical
    ax.plot([0, 0], [-W, W], color=color, lw=lw, zorder=2)
    # Circulo central — radio FIFA 9.15 m
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, ec=color, lw=lw, zorder=2))
    # Punto central del campo
    ax.plot(0, 0, "o", ms=3, color=color, zorder=2)                 # ↑ ms → punto mas grande

    pa_w, pa_d = 20.16, 16.5                                        # area grande: semianchura y profundidad (m)
    ga_w, ga_d = 9.16, 5.5                                          # area pequena: semianchura y profundidad (m)

    for sign in (-1, 1):                                            # sign=-1 = porteria IZQ; sign=+1 = porteria DCHA
        x0, x1 = sign * L, sign * (L - pa_d)                        # borde de gol y limite del area grande
        # Area grande (rectangulo)
        ax.plot([x0, x1, x1, x0], [-pa_w, -pa_w, pa_w, pa_w], color=color, lw=lw, zorder=2)
        x2 = sign * (L - ga_d)                                      # limite del area pequena
        # Area pequena (rectangulo interior)
        ax.plot([x0, x2, x2, x0], [-ga_w, -ga_w, ga_w, ga_w], color=color, lw=lw, zorder=2)
        # Punto de penalti (a 11 m de la linea de gol)
        ax.plot(sign * (L - 11), 0, "o", ms=3, color=color, zorder=2)
        # Arco del area (semicirculo r=9.15 desde el penalti, solo parte fuera del area)
        theta = np.linspace(-np.pi / 2, np.pi / 2, 50)              # 50 puntos = arco suave
        arc_x = sign * (L - 11) + 9.15 * np.cos(theta) * sign
        arc_y = 9.15 * np.sin(theta)
        arc_x[np.abs(arc_x) > (L - pa_d)] = np.nan                  # borra parte que cae dentro del area
        ax.plot(arc_x, arc_y, color=color, lw=lw, zorder=2)
        # Porteria: postes ±3.66 m del centro = porteria FIFA de 7.32 m
        for yy in (-3.66, 3.66):
            ax.plot([sign * L, sign * (L + 1.5)], [yy, yy], color=color, lw=lw * 1.5, zorder=2)
        ax.plot([sign * (L + 1.5), sign * (L + 1.5)], [-3.66, 3.66],
                color=color, lw=lw * 1.5, zorder=2)                 # palo trasero de la porteria

    ax.set_xlim(-L - 3, L + 3)                                      # 3 m de margen exterior (no cortar el campo)
    ax.set_ylim(-W - 3, W + 3)
    ax.set_aspect("equal")                                          # aspecto 1:1 (sin deformar el campo)
    ax.axis("off")                                                  # sin ejes/ticks/bordes alrededor del campo


# ============================================================================
# draw_header — PPCF-style header reusable (escudo IZQ | titulo+sub | JO DCHA)
# ============================================================================
# CHEAT-SHEET de tuneo del header:
#   - hdr_band    : (bot, top) en fraccion de figura — default HEADER_BAND=(0.85,1.0)
#                    ↓ bot → header mas gordo, mas sitio pa titulo largo
#   - escudo_x    : x del borde izquierdo del escudo [0..1 fig] — default 0.04
#                    ↑ → escudo mas a la derecha (aleja del borde)
#   - text_x      : x donde empieza el bloque titulo+sub [0..1 fig] — default 0.13
#                    ↑ → texto mas a la derecha (deja mas sitio al escudo)
#   - jo_x        : x del borde derecho del logo JO [0..1 fig] — default 0.96
#                    ↓ → logo JO mas a la izquierda (aleja del borde)
#   - escudo_zoom : escala del escudo despues de normalizar a 100px — default 1.20
#                    ↑ 1.5 → escudo mas grande; ↓ 0.8 → mas pequeno
#   - jo_zoom     : escala del logo JO — default 0.15
#                    ↑ 0.20 → JO mas grande; ↓ 0.10 → mas pequeno
#   - title_size  : fontsize del titulo (Chakra Petch 700) — default 22
#   - sub_size    : fontsize del subtitulo (Chakra Petch 500) — default 14
#
# Posiciones verticales dentro de hdr_band (fraccion del alto de la banda):
#   title_y  = bot + 0.85*h   → titulo arriba
#   sub_y    = bot + 0.55*h   → subtitulo en el centro
#   escudo_y = bot + 0.50*h   → escudo centrado vertical
#   jo_y     = bot + 0.42*h   → JO ligeramente bajo el centro

def _load_rgba(path) -> Optional[np.ndarray]:
    """Carga PNG como np.array RGBA. Maneja modo Palette de FotMob/sportlogos."""
    p = Path(path)
    if not p.exists():
        return None                                                 # archivo no existe → None silencioso
    try:
        return np.asarray(Image.open(p).convert("RGBA"))            # convierte a RGBA siempre pa uniformidad
    except Exception:
        return None                                                 # PNG corrupto/raro → None silencioso


_ESCUDO_TARGET_PX: int = 100                                        # alto al que se normalizan TODOS los escudos antes del zoom
                                                                     #   ↑ 150 → mas resolucion pero pesados
                                                                     #   ↓ 64  → mas pixelados si PNG es grande


def _normalize_img(img: np.ndarray, target_h: int = _ESCUDO_TARGET_PX) -> np.ndarray:
    """Redimensiona img a target_h px de alto (mantiene aspect ratio).

    Evita que logos con mas resolucion (ej. WC22 vs logos de equipo) salgan
    mas grandes que otros con el mismo zoom en draw_header.
    """
    h = img.shape[0]                                                # alto actual en pixeles
    if h == target_h:
        return img                                                  # ya tiene el tamano objetivo → no hace nada
    w = img.shape[1]                                                # ancho actual
    new_w = max(1, int(round(w * target_h / h)))                    # nuevo ancho proporcional (min 1px)
    return np.asarray(Image.fromarray(img).resize((new_w, target_h), Image.LANCZOS))
    # LANCZOS = mejor calidad al reducir; alternativa BILINEAR pa mas velocidad


def draw_header(fig: plt.Figure, *, title: str,
                subtitle: Optional[str] = None,
                escudo_path=None,
                hdr_band: Tuple[float, float] = HEADER_BAND,
                escudo_x: float = 0.04, text_x: float = 0.13, jo_x: float = 0.96,
                escudo_zoom: float = 1.20, jo_zoom: float = 0.15,
                title_size: float = 22, sub_size: float = 14) -> None:
    """Header PPCF-style: escudo grande IZQ | titulo TOP + subtitulo LEFT | JO DCHA.

    Args:
      fig:          la Figure de matplotlib donde pintar
      title:        texto del titulo (Chakra Petch 700, negro)
      subtitle:     subtitulo (Chakra Petch 500, negro). None = no se pinta
      escudo_path:  PNG del escudo a la izquierda (selecciones/torneo). None = no escudo
      hdr_band:     franja vertical reservada al header — default HEADER_BAND
      escudo_x/text_x/jo_x: posiciones x de cada elemento en fraccion [0..1] de figura
      escudo_zoom/jo_zoom:  escala de cada logo (sobre target px normalizado)
      title_size/sub_size:  fontsizes del texto
    """
    bot, top = hdr_band                                             # fraccion inferior/superior del header en figura
    h = top - bot                                                   # alto total del header en fraccion de figura

    # Posiciones verticales (fraccion de figura, NO del header)
    title_y  = bot + 0.85 * h                                       # titulo: arriba del todo
    sub_y    = bot + 0.55 * h                                       # subtitulo: centro
    escudo_y = bot + 0.50 * h                                       # escudo: centrado vertical
    jo_y     = bot + 0.42 * h                                       # JO: ligeramente bajo el centro

    # ---- Escudo IZQ (si se pasa path) ----
    if escudo_path is not None:
        img = _load_rgba(escudo_path)                               # carga PNG como RGBA
        if img is not None:
            img = _normalize_img(img)                               # normaliza a _ESCUDO_TARGET_PX px de alto
            ab = AnnotationBbox(
                OffsetImage(img, zoom=escudo_zoom),                 # escudo escalado — ↑ zoom → mas grande
                (escudo_x, escudo_y),                               # posicion (x,y) en fraccion de figura
                frameon=False,                                      # sin caja/borde
                xycoords="figure fraction",                         # coords relativas a la figura entera
                box_alignment=(0.0, 0.5),                           # ancla BORDE IZQ del escudo en escudo_x
            )
            ab.set_clip_on(False)                                   # permite que sobresalga del axes si es grande
            fig.add_artist(ab)                                      # pega directamente sobre la figura

    # ---- Titulo (Chakra Petch 700 = bold, negro) ----
    fig.text(text_x, title_y, title, ha="left", va="center",
             color=TEXT, fontsize=title_size, fontweight=700)

    # ---- Subtitulo (Chakra Petch 500 = medium, negro) ----
    if subtitle:
        fig.text(text_x, sub_y, subtitle, ha="left", va="center",
                 color=TEXT, fontsize=sub_size, fontweight=500)

    # ---- Logo JO en DCHA (default _LOGO_PATH del repo) ----
    img_jo = _load_rgba(_LOGO_PATH)
    if img_jo is not None:
        ab = AnnotationBbox(
            OffsetImage(img_jo, zoom=jo_zoom),                      # logo escalado — ↑ zoom → mas grande
            (jo_x, jo_y),                                           # pegado al borde derecho de la figura
            frameon=False,
            xycoords="figure fraction",
            box_alignment=(1.0, 0.5),                               # ancla BORDE DCHA del logo en jo_x
        )
        ab.set_clip_on(False)                                       # permite que sobresalga si hace falta
        fig.add_artist(ab)


# ============================================================================
# add_logo — LEGACY: estampa el logo JO en una esquina (radar.py, figures.py)
# ============================================================================
# draw_header ya maneja el JO en headers nuevos. add_logo se usa cuando la viz
# NO tiene header (ej. radar standalone) o quiere el logo ABAJO (figures.py).

def add_logo(fig: plt.Figure, width_frac: float = 0.11,
             margin: float = 0.012, corner: str = "br") -> None:
    """Estampa el logo JO en una esquina (axes dedicado).

    Args:
      width_frac:  ancho del logo en fraccion de figura — ↑ logo MAS GRANDE
      margin:      separacion del borde de la figura en fraccion — ↑ MAS lejos del borde
      corner:      "br"=bottom-right, "bl"=bottom-left, "tr"=top-right, "tl"=top-left
    """
    if not _LOGO_PATH.exists():
        return                                                      # logo no existe → no pinta nada (no peta)
    try:
        img = plt.imread(str(_LOGO_PATH))
        logo_aspect = img.shape[1] / img.shape[0]                   # ratio ancho/alto del PNG
        figW, figH = fig.get_size_inches()                          # tamano de la figura en pulgadas
        h_frac = width_frac * (figW / figH) / logo_aspect           # alto proporcional en fraccion de figura
        x = margin if corner in ("bl", "tl") else 1.0 - margin - width_frac
        y = margin if corner in ("bl", "br") else 1.0 - margin - h_frac
        ax_logo = fig.add_axes([x, y, width_frac, h_frac])          # axes dedicado pa el logo
        ax_logo.imshow(img)
        ax_logo.axis("off")                                         # sin ejes alrededor
    except Exception:
        pass                                                         # PNG raro/error → no pinta nada
