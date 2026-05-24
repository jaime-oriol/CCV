"""common - Estilo + constantes + helpers compartidos por todas las vizs.

Identidad LIGHT OPTA-STYLE: fondo blanco, textos negros, leyenda gris medio,
accent saturado azul/rojo (y cmap morado→fuchsia→rosa para scatters). Helpers
genericos: draw_pitch, draw_header (PPCF-style escudo+title+sub+JO), add_logo
(legacy), save_fig.

Convencion campo: 105x68 m, origen en el centro (0, 0), metros (PFF).

CHEAT-SHEET DE TUNEO MANUAL (todos los numeros aqui SON ajustables; los
efectos estan documentados en cada constante con ↑/↓ → efecto):

  - MASTER_FIGSIZE     : tamano de TODAS las vizs (16×9 landscape)
  - HEADER_BAND        : franja vertical reservada al header (escudo+titulo)
  - draw_header(...)   : posiciones internas del escudo/titulo/sub/logo
  - Paleta (ATT/DEF/..): colores principales (azul atacante / rojo defensor)
  - Cmaps              : gradientes (PCT_CMAP para scatters, etc.)

WHITE/PE retro-compat: WHITE apunta a TEXT (negro) → codigo viejo que pinta
con WHITE sigue funcionando (negro sobre blanco). Para uso nuevo, preferir
TEXT (texto principal) / LEGEND (textos secundarios).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.patheffects as pe   # pa hacer el stroke blanco alrededor del texto
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import numpy as np
from matplotlib.colors import LinearSegmentedColormap   # pa construir gradientes custom
from matplotlib.offsetbox import AnnotationBbox, OffsetImage   # pa pegar imagenes (escudos, caras)
from PIL import Image   # pa cargar y redimensionar PNGs

# registra Chakra Petch (todos los pesos) al importar el modulo
_CHAKRA_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "assets" / "fonts"
_SYS_FONT_DIR = Path.home() / ".local" / "share" / "fonts"
for _fdir in (_CHAKRA_DIR, _SYS_FONT_DIR):
    if _fdir.exists():
        for _f in _fdir.glob("ChakraPetch*.ttf"):
            try:
                _fm.fontManager.addfont(str(_f))
            except Exception:
                pass

# ============================================================================
# PALETA LIGHT OPTA
# ============================================================================

BG      = "#ffffff"          # fondo de TODA la viz — cambia aqui pa off-white (#fafafa, #f5f5f5)
TEXT    = "#000000"          # color de titulos, labels de ejes, ticks — negro puro siempre
LEGEND  = "#666666"          # color de notas secundarias, footnotes, mediana inline
                              #   ↑ mas claro (#999) → leyenda casi invisible
                              #   ↓ mas oscuro (#444) → leyenda con mas presencia
GRID    = "#dddddd"          # color de las gridlines y spines (gris suave)
                              #   ↑ mas oscuro (#bbb) → grid mas visible
                              #   ↓ mas claro  (#eee) → grid casi invisible
PITCH   = "#000000"          # color de las lineas del campo — negro; pon "#555" pa gris

WHITE = TEXT   # alias retro-compat: codigo viejo que usaba WHITE sigue pintando negro sobre blanco

# colores accent — deben ser SATURADOS pa contrastar bien sobre BG blanco
ATT     = "#3b82f6"          # azul vivo light: atacante, canal ofensivo, chasing — cambiar ej. a "#1d4ed8" pa mas oscuro/navy
DEF     = "#ef4444"          # rojo vivo light: defensor, canal defensivo, protecting — cambiar ej. a "#dc2626" pa mas oscuro
GK      = "#000000"          # portero en pitch viz — negro; pon un gris "#555" pa distinguirlo menos
BALL    = "#000000"          # balon en pitch viz — negro
NEUTRAL = "#e5e7eb"          # gris muy claro pa el centro de los cmaps divergentes (PPCF, CATE)
                              #   ↑ mas oscuro (#d4d4d4) → la zona "50/50" se ve mas gris
                              #   ↓ mas claro  (#f3f4f6) → la zona neutra casi desaparece

# ============================================================================
# TIPOGRAFIA
# ============================================================================

# matplotlib usa el primer font de la lista que encuentre en el sistema
FONT_STACK = ["Chakra Petch", "Inter", "Source Sans 3", "Source Sans Pro",
              "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]
                              # Chakra Petch primero: tiene los cortes angulares Opta-style
                              # "DejaVu Sans" es el ultimo recurso; siempre esta disponible
FONT = FONT_STACK[0]   # alias legado — no se usa directamente, es pa referencia

plt.style.use("default")   # reset completo antes de aplicar nuestro estilo
plt.rcParams.update({
    "font.family": "sans-serif",          # tipo de familia — sans-serif es Opta-style
    "font.sans-serif": FONT_STACK,        # orden de prioridad de fonts pa sans-serif
    "font.size": 10,                      # tamano BASE de todo texto que no especifique fontsize
                                           #   ↑ 11 → todo un poco mas grande globalmente
                                           #   ↓ 9  → todo un poco mas pequeno
    "figure.dpi": 100,                    # DPI pa pantalla (visualizacion interactiva) — no afecta al PNG
    "savefig.dpi": 300,                   # DPI por defecto al guardar — cada viz lo override con dpi=N
    "savefig.bbox": None,                 # sin tight-crop por defecto: el PNG sale EXACTO al figsize
                                           # (si fuera "tight" recortaria margenes y el tamano varia)
                                           # save_fig() y radar usan bbox_inches="tight" explicito
    "axes.facecolor": BG,                 # fondo de TODOS los axes = BG (blanco)
    "figure.facecolor": BG,               # fondo de TODAS las figuras = BG (blanco)
    "text.color": TEXT,                   # color por defecto de todo texto = negro
    "axes.labelcolor": TEXT,              # color de xlabel/ylabel = negro
    "xtick.color": TEXT,                  # color de los ticks del eje X = negro
    "ytick.color": TEXT,                  # color de los ticks del eje Y = negro
})

# path effects: halo de fondo alrededor del texto pa que se lea sobre el PPCF coloreado
PE   = [pe.withStroke(linewidth=1.5, foreground=BG), pe.Normal()]   # halo fino (pa texto normal)
PE_S = [pe.withStroke(linewidth=2.6, foreground=BG)]                # halo gordo (pa dorsales sobre PPCF)
                                                                      #   ↑ linewidth → halo mas gordo

# ============================================================================
# COLORMAPS
# ============================================================================

# PPCF (pitch control): defensor (rojo) → neutro → atacante (azul)
PPCF_CMAP = LinearSegmentedColormap.from_list("ppcf", [DEF, NEUTRAL, ATT])
            # cambiar DEF y ATT directamente arriba pa cambiar los colores del campo

# percentil combinado pa scatter y radar legend: morado → fuchsia → rosa (mas contraste)
PCT_CMAP = LinearSegmentedColormap.from_list("pct", [
    "#5b21b6",   # violet-800 — percentil bajo (mas profundo pa mas contraste)
    "#c026d3",   # fuchsia-600 — percentil medio
    "#f43f5e",   # rose-500 — percentil alto (mas vivido pa mas contraste)
])

# CATE divergente (efecto negativo rojo → cero neutro → positivo azul)
CATE_CMAP = LinearSegmentedColormap.from_list("cate", [DEF, NEUTRAL, ATT])
            # mismo patron que PPCF pa coherencia visual

# ============================================================================
# DIMENSIONES MASTER (LAYOUT GLOBAL)
# ============================================================================

# TODAS las vizs (scatter, ppcf, scatter_team) usan este figsize pa salir 2400x1350 px @ 150dpi
MASTER_FIGSIZE = (16.0, 9.0)   # (ancho, alto) en pulgadas — NO cambiar sin revisar todos los layouts
                                 #   ↑ ancho (18, 9) → mas ancho pero rompe proporciones del plot
                                 #   si cambias alto recalcula HEADER_BAND = (1 - 1.5/H, 1)

HEADER_BAND = (0.85, 1.0)   # (y_inferior, y_superior) del header en fraccion de figura [0..1]
                              #   ↑ y_inferior (0.88) → header mas delgado, mas sitio pa el plot
                              #   ↓ y_inferior (0.80) → header mas gordo, titulo mas grande
                              # la altura en pulgadas = (1.0 - 0.85) * 9 = 1.35 in

PITCH_LENGTH = 105.0   # metros — norma FIFA, no tocar
PITCH_WIDTH  = 68.0    # metros — norma FIFA, no tocar

_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo_2.png"
             # logo JO (Jaime Oriol) que va en la esquina derecha del header
             # pa cambiar el logo sustituye ese PNG (mantener fondo transparente)


# ============================================================================
# draw_pitch
# ============================================================================

def draw_pitch(ax: plt.Axes,
               pitch_length: float = PITCH_LENGTH,
               pitch_width: float = PITCH_WIDTH,
               color: str = PITCH, lw: float = 1.0) -> None:
    """Dibuja un campo de futbol centrado en (0,0), metros. Sin mplsoccer."""
    L, W = pitch_length / 2, pitch_width / 2   # semiancho y semilargo: el origen queda en el centro

    # borde exterior del campo
    ax.plot([-L, L, L, -L, -L], [-W, -W, W, W, -W], color=color, lw=lw, zorder=2)
    # linea de medio campo vertical
    ax.plot([0, 0], [-W, W], color=color, lw=lw, zorder=2)
    # circulo central (radio 9.15 m)
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, ec=color, lw=lw, zorder=2))
    # punto del centro del campo
    ax.plot(0, 0, "o", ms=3, color=color, zorder=2)   # ↑ ms → punto mas grande

    pa_w, pa_d = 20.16, 16.5   # area grande: semianchura y profundidad en metros
    ga_w, ga_d = 9.16, 5.5    # area pequena: semianchura y profundidad en metros

    for sign in (-1, 1):   # sign=-1 = porteria izquierda; sign=+1 = porteria derecha
        x0, x1 = sign * L, sign * (L - pa_d)   # borde de gol y limite del area grande
        # area grande (rectangulo)
        ax.plot([x0, x1, x1, x0], [-pa_w, -pa_w, pa_w, pa_w], color=color, lw=lw, zorder=2)
        x2 = sign * (L - ga_d)   # limite del area pequena
        # area pequena (rectangulo interior)
        ax.plot([x0, x2, x2, x0], [-ga_w, -ga_w, ga_w, ga_w], color=color, lw=lw, zorder=2)
        # punto de penalti (a 11 m de la linea de gol)
        ax.plot(sign * (L - 11), 0, "o", ms=3, color=color, zorder=2)
        # arco del area (semicirculo de radio 9.15 desde el penalti, solo la parte fuera del area)
        theta = np.linspace(-np.pi / 2, np.pi / 2, 50)   # 50 puntos = arco suave
        arc_x = sign * (L - 11) + 9.15 * np.cos(theta) * sign
        arc_y = 9.15 * np.sin(theta)
        arc_x[np.abs(arc_x) > (L - pa_d)] = np.nan   # borra la parte que cae dentro del area
        ax.plot(arc_x, arc_y, color=color, lw=lw, zorder=2)
        for yy in (-3.66, 3.66):   # postes: ±3.66 m del centro = porteria de 7.32 m
            ax.plot([sign * L, sign * (L + 1.5)], [yy, yy], color=color, lw=lw * 1.5, zorder=2)
        ax.plot([sign * (L + 1.5), sign * (L + 1.5)], [-3.66, 3.66],
                color=color, lw=lw * 1.5, zorder=2)   # palo trasero de la porteria

    ax.set_xlim(-L - 3, L + 3)   # 3 m de margen exterior pa que no se corte el campo
    ax.set_ylim(-W - 3, W + 3)
    ax.set_aspect("equal")   # aspecto 1:1 pa que el campo no quede deformado
    ax.axis("off")           # sin ejes, ticks ni bordes alrededor del campo


def make_pitch(figsize: Tuple[float, float] = MASTER_FIGSIZE,
               pitch_length: float = PITCH_LENGTH,
               pitch_width: float = PITCH_WIDTH,
               ax: Optional[plt.Axes] = None) -> Tuple[plt.Figure, plt.Axes]:
    """Devuelve (fig, ax) con el campo ya dibujado. Si pasas ax, lo reusa."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_facecolor(BG)
    else:
        fig = ax.get_figure()   # reutiliza la fig que ya tiene ese ax
    draw_pitch(ax, pitch_length, pitch_width)
    return fig, ax


# ============================================================================
# draw_header (PPCF-style, REUSABLE)
# ============================================================================
# CHEAT-SHEET de tuneo del header:
#   - hdr_band    : (bot, top) en fraccion de figura — default HEADER_BAND=(0.85,1.0)
#                    ↓ bot → header mas gordo y con mas sitio pa titulo largo
#   - escudo_x    : x del borde izquierdo del escudo [0..1 fig] — default 0.04
#                    ↑ → escudo mas a la derecha (aleja del borde)
#   - text_x      : x donde empieza el bloque titulo+subs [0..1 fig] — default 0.13
#                    ↑ → texto mas a la derecha (deja mas sitio al escudo)
#   - jo_x        : x del borde derecho del logo JO [0..1 fig] — default 0.96
#                    ↓ → logo JO mas a la izquierda (aleja del borde)
#   - escudo_zoom : escala del escudo despues de normalizar a 100px — default 1.20
#                    ↑ 1.5 → escudo mas grande; ↓ 0.8 → mas pequeno
#   - jo_zoom     : escala del logo JO — default 0.15
#                    ↑ 0.20 → JO mas grande; ↓ 0.10 → mas pequeno
#   - title_size  : fontsize del titulo bold — default 22
#   - sub_size    : fontsize de los subtitulos — default 14
#
# Posiciones verticales dentro de hdr_band (fraccion del alto de la banda):
#   title_y  = bot + 0.85*h   → titulo bien arriba
#   sub_y    = bot + 0.55*h   → sub1 en el centro
#   sub2_y   = bot + 0.25*h   → sub2 abajo
#   escudo_y = bot + 0.50*h   → escudo centrado vertical
#   jo_y     = bot + 0.42*h   → JO ligeramente bajo el centro

def _load_rgba(path) -> Optional[np.ndarray]:
    """Carga PNG como np.array RGBA (maneja modo Palette de FotMob/sportlogos)."""
    p = Path(path)
    if not p.exists():
        return None   # si no existe el archivo devuelve None en silencio
    try:
        return np.asarray(Image.open(p).convert("RGBA"))   # convierte a RGBA siempre pa uniformidad
    except Exception:
        return None   # si falla la lectura (PNG corrupto, etc.) devuelve None en silencio


_ESCUDO_TARGET_PX: int = 100   # todos los escudos se normalizan a esta altura antes del zoom
                                 # ↑ 150 → escudos con mas resolucion pero mas pesados
                                 # ↓ 64  → escudos mas pixelados si el PNG original es grande


def _normalize_img(img: np.ndarray, target_h: int = _ESCUDO_TARGET_PX) -> np.ndarray:
    """Redimensiona img a target_h px de alto (mantiene aspect ratio).

    Evita que logos con mas resolucion (ej. WC22 vs logos de equipo) salgan
    mas grandes que otros con el mismo zoom en draw_header.
    """
    h = img.shape[0]   # alto actual del PNG en pixeles
    if h == target_h:
        return img   # si ya tiene el tamano correcto no hace nada
    w = img.shape[1]   # ancho actual
    new_w = max(1, int(round(w * target_h / h)))   # nuevo ancho proporcional (min 1px)
    return np.asarray(Image.fromarray(img).resize((new_w, target_h), Image.LANCZOS))
    # LANCZOS = mejor calidad al reducir; alternativa BILINEAR pa mas velocidad


def draw_header(fig: plt.Figure, *, title: str,
                subtitle: Optional[str] = None,
                subtitle2: Optional[str] = None,
                escudo_path=None, logo_path=None,
                hdr_band: Tuple[float, float] = HEADER_BAND,
                escudo_x: float = 0.04, text_x: float = 0.13, jo_x: float = 0.96,
                escudo_zoom: float = 1.20, jo_zoom: float = 0.15,
                title_size: float = 22, sub_size: float = 14) -> None:
    """Header PPCF-style: escudo grande IZQ | titulo TOP + sub(s) LEFT | JO DCHA."""
    bot, top = hdr_band          # fraccion inferior y superior del header dentro de la figura
    h = top - bot                # alto total del header en fraccion de figura

    # posiciones verticales de cada elemento (en fraccion de figura, no del header)
    title_y  = bot + 0.85 * h   # titulo: arriba del todo del header
    sub_y    = bot + 0.55 * h   # subtitulo 1: centro del header
    sub2_y   = bot + 0.25 * h   # subtitulo 2: abajo del header
    escudo_y = bot + 0.50 * h   # escudo: centrado vertical del header
    jo_y     = bot + 0.42 * h   # logo JO: ligeramente debajo del centro

    if escudo_path is not None:
        img = _load_rgba(escudo_path)    # carga el PNG del escudo como array RGBA
        if img is not None:
            img = _normalize_img(img)    # normaliza a _ESCUDO_TARGET_PX px de alto antes del zoom
            ab = AnnotationBbox(
                OffsetImage(img, zoom=escudo_zoom),   # escudo escalado — ↑ zoom → mas grande
                (escudo_x, escudo_y),                 # posicion (x,y) en fraccion de figura
                frameon=False,                        # sin caja ni borde alrededor del escudo
                xycoords="figure fraction",           # coordenadas relativas a la figura entera
                box_alignment=(0.0, 0.5),             # ancla el BORDE IZQUIERDO del escudo en escudo_x
            )
            ab.set_clip_on(False)   # permite que el escudo sobresalga del axes si es grande
            fig.add_artist(ab)      # pega el escudo directamente sobre la figura

    # titulo principal: Chakra Petch 700 negro
    fig.text(text_x, title_y, title, ha="left", va="center",
             color=TEXT, fontsize=title_size, fontweight=700)

    if subtitle:
        # subtitulo: Chakra Petch 500 (medium) negro
        fig.text(text_x, sub_y, subtitle, ha="left", va="center",
                 color=TEXT, fontsize=sub_size, fontweight=500)

    if subtitle2:
        fig.text(text_x, sub2_y, subtitle2, ha="left", va="center",
                 color=TEXT, fontsize=sub_size, fontweight=500)

    # logo JO — si logo_path=None carga desde _LOGO_PATH (el default del repo)
    img_jo = _load_rgba(logo_path if logo_path is not None else _LOGO_PATH)
    if img_jo is not None:
        ab = AnnotationBbox(
            OffsetImage(img_jo, zoom=jo_zoom),   # logo escalado — ↑ jo_zoom → mas grande
            (jo_x, jo_y),                        # posicion: pegado al borde derecho de la figura
            frameon=False,
            xycoords="figure fraction",
            box_alignment=(1.0, 0.5),            # ancla el BORDE DERECHO del logo en jo_x
        )
        ab.set_clip_on(False)   # permite que el logo sobresalga si es necesario
        fig.add_artist(ab)


# ============================================================================
# LEGACY HELPERS (mantener para retro-compat)
# ============================================================================

def add_logo(fig: plt.Figure, width_frac: float = 0.11,
             margin: float = 0.012, corner: str = "br") -> None:
    """Estampa el logo JO en una esquina como axes dedicado.

    LEGACY — usar draw_header pa nuevas vizs. Solo radar_report sigue aqui.
    ↑ width_frac → logo mas grande. corner: "br" / "bl" / "tr" / "tl".
    """
    if not _LOGO_PATH.exists():
        return   # si no existe el logo no peta, simplemente no pinta nada
    try:
        img = plt.imread(str(_LOGO_PATH))
        logo_aspect = img.shape[1] / img.shape[0]   # ratio ancho/alto del PNG del logo
        figW, figH = fig.get_size_inches()           # tamano de la figura en pulgadas
        h_frac = width_frac * (figW / figH) / logo_aspect   # alto proporcional en fraccion de figura
        x = margin if corner in ("bl", "tl") else 1.0 - margin - width_frac   # x segun esquina
        y = margin if corner in ("bl", "br") else 1.0 - margin - h_frac       # y segun esquina
        ax_logo = fig.add_axes([x, y, width_frac, h_frac])   # axes dedicado pa el logo
        ax_logo.imshow(img)    # pinta el logo dentro del axes
        ax_logo.axis("off")    # sin ejes alrededor del logo
    except Exception:
        pass   # si algo falla (PNG raro, etc.) simplemente no pinta el logo


def style_ax(ax: plt.Axes, ygrid: bool = True) -> None:
    """Ejes estilo paper light: sin spines top/right, spines suaves, grid suave."""
    ax.set_facecolor(BG)   # fondo blanco del axes
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)   # elimina spines superior y derecho (Opta-style)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)        # spines restantes en gris suave
        ax.spines[s].set_linewidth(0.8)     # trazo fino — ↑ → spine mas grueso
    if ygrid:
        ax.grid(True, axis="y", color=GRID, linewidth=0.7, alpha=0.9)   # grid horizontal suave
        ax.set_axisbelow(True)   # el grid queda DETRAS de los datos (no los tapa)


def save_fig(fig: plt.Figure, path, logo: bool = False) -> None:
    """Guarda la figura con identidad (BG blanco, dpi 300, tight crop)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)   # crea el directorio si no existe
    if logo:
        add_logo(fig)   # estampa el logo JO en la esquina si se pide
    fig.savefig(path, dpi=300, facecolor=BG, bbox_inches="tight")
    # bbox_inches="tight" → recorta los margenes blancos sobrantes (OK pa radar)
    # pa scatter/ppcf NO se usa esta funcion: ellos salvan directamente con dpi=150
    plt.close(fig)   # libera memoria de la figura
