"""common - Estilo + constantes + helpers compartidos por todas las vizs.

Identidad LIGHT OPTA-STYLE: fondo BLANCO, textos NEGROS, leyenda gris medio.
Paleta azul / rojo (saturada light para PPCF) + cmap rosa→violeta→navy logo JO
(para percentiles en scatter y radar table). Tipografia Chakra Petch (cortes
angulares estilo Opta).

Exporta:
  - Paleta:       BG, TEXT, LEGEND, GRID, PITCH, WHITE(=TEXT), ATT, DEF, GK, NEUTRAL
  - Cmaps:        PPCF_CMAP (DEF→NEUTRAL→ATT), PCT_CMAP (rose→violet→navy)
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

# ----------------------------------------------------------------------------
# REGISTRO DE FUENTES — Chakra Petch al importar el modulo
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# PALETA LIGHT OPTA
# ----------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------
# TIPOGRAFIA
# ----------------------------------------------------------------------------

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

# Path effects: halo alrededor del texto pa que se lea sobre fondos coloreados.
PE   = [pe.withStroke(linewidth=1.5, foreground=BG), pe.Normal()]   # halo BLANCO fino (texto negro sobre PPCF claro)
PE_S = [pe.withStroke(linewidth=2.2, foreground="black")]           # halo NEGRO grueso (dorsales BLANCOS sobre nodos azul/rojo saturados)
                                                                     #   ↑ linewidth → halo mas gordo (mas contraste pero texto se ve menos)
                                                                     #   foreground="white" → invierte a halo blanco (pa fondos oscuros)

# ----------------------------------------------------------------------------
# COLORMAPS
# ----------------------------------------------------------------------------

# PPCF (pitch control): defensor (rojo) → neutro (gris claro) → atacante (azul).
# Se reconstruye automaticamente si cambias ATT/DEF/NEUTRAL arriba.
PPCF_CMAP = LinearSegmentedColormap.from_list("ppcf", [DEF, NEUTRAL, ATT])

# Percentil combinado pa scatter (color de cada punto) y radar legend (5 tramos).
# Rosa → violeta → navy del logo JO: warm→cool sunset, elite = brand navy.
PCT_CMAP = LinearSegmentedColormap.from_list("pct", [
    "#f43f5e",   # rose-500 — percentil bajo (rosa vivo, potente sobre blanco)
    "#9333ea",   # purple-600 — percentil medio (violeta puente warm→cool)
    "#1e3a8a",   # blue-900 — percentil alto (navy del logo JO, elite)
])

# ----------------------------------------------------------------------------
# DIMENSIONES MASTER (LAYOUT GLOBAL)
# ----------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------
# NAMING ESTANDAR (todo en ESPAÑOL, mismo wording en TODAS las vizs)
# ----------------------------------------------------------------------------

TOURNAMENT_ES   = "Mundial Qatar 2022"                              # nombre del torneo (subtitulos)
N_PLAYERS_WC22  = 511                                                # numero jugadores filtrados en ccv_table.parquet

# Mapeo nombres PFF (ingles) -> español. Pa traducir los team_name del parquet.
TEAM_NAME_ES: dict[str, str] = {
    "Argentina":     "Argentina",
    "Brazil":        "Brasil",
    "Ecuador":       "Ecuador",
    "Uruguay":       "Uruguay",
    "Belgium":       "Bélgica",
    "Croatia":       "Croacia",
    "Denmark":       "Dinamarca",
    "England":       "Inglaterra",
    "France":        "Francia",
    "Germany":       "Alemania",
    "Netherlands":   "Países Bajos",
    "Poland":        "Polonia",
    "Portugal":      "Portugal",
    "Serbia":        "Serbia",
    "Spain":         "España",
    "Switzerland":   "Suiza",
    "Wales":         "Gales",
    "Cameroon":      "Camerún",
    "Ghana":         "Ghana",
    "Morocco":       "Marruecos",
    "Senegal":       "Senegal",
    "Tunisia":       "Túnez",
    "Japan":         "Japón",
    "South Korea":   "Corea del Sur",
    "Iran":          "Irán",
    "Qatar":         "Qatar",
    "Saudi Arabia":  "Arabia Saudí",
    "Canada":        "Canadá",
    "Mexico":        "México",
    "United States": "Estados Unidos",
    "Costa Rica":    "Costa Rica",
    "Australia":     "Australia",
}


def team_es(name: str) -> str:
    """Traduce un team_name de PFF al español. Si no esta mapeado, devuelve igual."""
    return TEAM_NAME_ES.get(name, name)

# Logo JO (Jaime Oriol) — esquina derecha del header en draw_header,
# o esquinas inferiores en add_logo (figures.py, radar.py).
_LOGO_PATH = Path(__file__).resolve().parent.parent.parent / "outputs" / "assets" / "logo.png"
_JO_LOGO_SCALE = 1.15                                                # multiplier sobre el alto del escudo — ↑ logo JO MAS GRANDE en draw_header


# ----------------------------------------------------------------------------
# draw_pitch — campo de futbol centrado en (0,0), metros
# ----------------------------------------------------------------------------

def draw_pitch(ax: plt.Axes,
               pitch_length: float = PITCH_LENGTH,
               pitch_width: float = PITCH_WIDTH,
               color: str = PITCH, lw: float = 1.0,
               margin: float = 3.0,
               margin_x: Optional[float] = None) -> None:
    """Dibuja un campo de futbol centrado en (0,0), en metros. Sin mplsoccer.

    Margenes:
      - margin   = metros de margen Y (arriba/abajo del campo)
      - margin_x = metros de margen X (laterales). None → usa `margin` (margen uniforme)
                   IMPORTANTE: las porterias van de ±L a ±(L+1.5), asi que margin_x>=1.5
                   pa que NO se corten los postes traseros.
    Default ambos 3.0 (≈0.4" a 150dpi en vizs 16x9). Usalo junto con _DATA_RATIO
    de tu layout pa que set_aspect encaje sin huecos.
    """
    if margin_x is None:
        margin_x = margin
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

    ax.set_xlim(-L - margin_x, L + margin_x)                        # margen X (incluye porterias ±(L+1.5))
    ax.set_ylim(-W - margin, W + margin)                            # margen Y exterior
    ax.set_aspect("equal")                                          # aspecto 1:1 (sin deformar el campo)
    ax.axis("off")                                                  # sin ejes/ticks/bordes alrededor del campo


# ----------------------------------------------------------------------------
# draw_header — PPCF-style header reusable (escudo IZQ | titulo+sub | JO DCHA)
# ----------------------------------------------------------------------------
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


_TEXT_FILL_FACTOR = 1.2   # divisor pa convertir alto en pt → matplotlib fontsize (compensa line-spacing/padding)
                           #   ↑ 1.5 → texto MAS PEQUEÑO pa misma altura asignada (mas margen)
                           #   ↓ 1.0 → texto MAS GRANDE (toca los limites)


def draw_header(fig: plt.Figure, *, title: str,
                subtitle: Optional[str] = None,
                escudo_path=None,
                hdr_band: Tuple[float, float] = HEADER_BAND,
                escudo_x: float = 0.04, text_x: float = 0.13, jo_x: float = 0.96,
                logo_h_frac: float = 0.5,
                text_h_frac: float = 0.5,
                title_share: float = 2.0 / 3.0,
                title_size: Optional[float] = None,
                sub_size: Optional[float] = None) -> None:
    """Header PPCF-style: escudo grande IZQ | titulo + subtitulo CENTRO | JO DCHA.

    Proporciones (todo relativo al alto del header band):
      - Logos (escudo+JO):    centrados vertical, alto = logo_h_frac del band
                              (default 0.5 → van de 0.25 a 0.75)
      - Bloque titulo+sub:    centrado vertical, alto = text_h_frac del band
                              (default 0.5 → tambien 0.25-0.75, mismo que logos)
      - Dentro del bloque:    titulo = title_share del alto (default 2/3)
                              sub    = (1 - title_share) del alto (default 1/3)

    Args:
      fig:          la Figure de matplotlib donde pintar
      title:        texto del titulo (Chakra Petch 700, negro)
      subtitle:     subtitulo (Chakra Petch 500, negro). None = no se pinta
      escudo_path:  PNG del escudo IZQ (selecciones/torneo). None = sin escudo
      hdr_band:     franja vertical reservada al header — default HEADER_BAND
      escudo_x/text_x/jo_x: posiciones x de cada elemento en fraccion [0..1] de figura
      logo_h_frac:  alto de AMBOS logos en fraccion del band — ↑ logos MAS GRANDES
      text_h_frac:  alto total del bloque texto en fraccion del band — ↑ texto MAS GRANDE
      title_share:  cuanto del bloque ocupa el titulo (resto va al sub)
      title_size:   override fontsize del titulo. None = auto-derivado del espacio asignado
      sub_size:     override fontsize del subtitulo. None = auto-derivado
    """
    bot, top = hdr_band                                             # fraccion inferior/superior del header en figura
    h = top - bot                                                   # alto total del header en fraccion de figura

    fig_h_inches = fig.get_size_inches()[1]
    band_h_inches = h * fig_h_inches                                # alto del band en pulgadas

    # ---- Bloque de texto: titulo arriba (title_share) + sub abajo (1-title_share) ----
    # Bloque centrado vertical en el band, ocupa text_h_frac de su alto.
    sub_share = 1.0 - title_share
    block_bot_frac = 0.5 - text_h_frac / 2.0                        # default 0.25 (con text_h_frac=0.5)
    block_top_frac = 0.5 + text_h_frac / 2.0                        # default 0.75
    boundary_frac = block_bot_frac + sub_share * text_h_frac         # frontera titulo/sub
    # Centros verticales de cada texto (pa va="center")
    sub_center_frac   = (block_bot_frac + boundary_frac) / 2.0
    title_center_frac = (boundary_frac + block_top_frac) / 2.0

    title_y = bot + title_center_frac * h
    sub_y   = bot + sub_center_frac   * h
    logo_y  = bot + 0.5 * h                                         # logos centrados vertical en el band

    # ---- Auto-fontsize: el alto rendered del texto ≈ fontsize_pt * _TEXT_FILL_FACTOR
    # → fontsize_pt = alto_inches * 72 / _TEXT_FILL_FACTOR
    if title_size is None:
        title_size = (title_share * text_h_frac * band_h_inches) * 72.0 / _TEXT_FILL_FACTOR + 1
    if sub_size is None:
        sub_size = (sub_share * text_h_frac * band_h_inches) * 72.0 / _TEXT_FILL_FACTOR + 1

    # ---- Zoom de los logos pa que midan logo_h_frac del band ----
    # Formula OffsetImage: displayed_inches = image_px * zoom / fig.dpi
    # → zoom = (target_inches * fig.dpi) / image_px
    target_logo_px = logo_h_frac * band_h_inches * fig.dpi          # alto deseado en pixeles a fig.dpi

    # ---- Escudo IZQ (si se pasa path) ----
    if escudo_path is not None:
        img = _load_rgba(escudo_path)
        if img is not None:
            img = _normalize_img(img)                               # normaliza a _ESCUDO_TARGET_PX (uniformiza source)
            escudo_zoom = target_logo_px / img.shape[0]
            ab = AnnotationBbox(
                OffsetImage(img, zoom=escudo_zoom),
                (escudo_x, logo_y),
                frameon=False,
                xycoords="figure fraction",
                box_alignment=(0.0, 0.5),                           # ancla BORDE IZQ del escudo en escudo_x
            )
            ab.set_clip_on(False)
            fig.add_artist(ab)

    # ---- Titulo (Chakra Petch 700 = bold, negro) ----
    fig.text(text_x, title_y, title, ha="left", va="center",
             color=TEXT, fontsize=title_size, fontweight=700)

    # ---- Subtitulo (Chakra Petch 500 = medium, negro) ----
    if subtitle:
        fig.text(text_x, sub_y, subtitle, ha="left", va="center",
                 color=TEXT, fontsize=sub_size, fontweight=500)

    # ---- Logo JO en DCHA ----
    img_jo = _load_rgba(_LOGO_PATH)
    if img_jo is not None:
        img_jo = _normalize_img(img_jo)                             # normaliza a _ESCUDO_TARGET_PX → mismo source que escudo
        jo_zoom = (target_logo_px * _JO_LOGO_SCALE) / img_jo.shape[0]  # JO_SCALE×alto del escudo
        ab = AnnotationBbox(
            OffsetImage(img_jo, zoom=jo_zoom),
            (jo_x, logo_y),
            frameon=False,
            xycoords="figure fraction",
            box_alignment=(1.0, 0.5),                               # ancla BORDE DCHA del logo en jo_x
        )
        ab.set_clip_on(False)
        fig.add_artist(ab)


# ----------------------------------------------------------------------------
# add_logo — LEGACY: estampa el logo JO en una esquina (radar.py, figures.py)
# ----------------------------------------------------------------------------
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
