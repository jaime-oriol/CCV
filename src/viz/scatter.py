"""scatter - Diamond scatters globales del PCJ (511 jugadores del torneo).

Diamante rotado 45 grados (floating_axes). Cada punto = 1 jugador, coloreado
por percentil combinado (PCT_CMAP). Top combinado + destacados por eje, cada
uno con su CARA (FotMob). Lineas mediana globales -> 4 cuadrantes.

2 conceptos (config-driven via SCATTERS):
  1. remontador_cerrojo    : Remontador x Cerrojo (los 2 perfiles de la propuesta).
  2. ataque_marcar_presion : Ataque post-GF x Ataque pre-elim (los 2 unicos ejes
                             con spread real; donde el shock deja huella medible).

Misma estetica que scatter_team.py (header logo WC22 + leyenda dashed +
carteles en los triangulos vacios). Indices PCJ son CATEs con signo, asi que
la normalizacion es min-max (v-min)/(max-min) en vez de v/max. Los percentiles
para color + seleccion de top se computan al vuelo (rank), no de cols cacheadas.

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
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.transforms import Affine2D
from mpl_toolkits.axisartist.grid_finder import DictFormatter, FixedLocator
from PIL import Image

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viz.common import BG, PCT_CMAP, WHITE, _LOGO_PATH

_TABLE = _SRC.parent / "outputs" / "pcj_table.parquet"
_LOGOS = _SRC.parent / "outputs" / "assets" / "logos"
_FACES = _SRC.parent / "outputs" / "assets" / "faces"
# Logo del torneo (viz genericas, sin seleccion concreta): va arriba-izq
# igual que el escudo de la seleccion en las viz por equipo.
_WC22_LOGO = _LOGOS / "wc22.png"

# 6 marcas por eje. Con 11 el diamante rotado se satura visualmente.
_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Configuracion de los 2 scatters. x -> eje izquierdo del diamante (esquina
# izq), y -> eje inferior (esquina dcha). title/labels/carteles por concepto.
SCATTERS: dict[str, dict] = {
    # Los 2 perfiles agregados de la propuesta. Ejes apretados (la elite es
    # homogenea en los agregados), pero es el marco conceptual del TFM.
    "remontador_cerrojo": dict(
        x="chasing_clutch_idx", y="protecting_clutch_idx",
        title="Remontador  vs  Cerrojo",
        x_label="REMONTADOR: ataque y movimiento off-ball tras encajar",
        y_label="CERROJO: defensa e intensidad fisica tras marcar",
        top="ARRIBA: los que hacen LAS DOS COSAS",
        left="Por este lado\nLOS QUE TIRAN DEL EQUIPO\ncuando toca remontar",
        right="Por este lado\nLOS QUE AGUANTAN EL RESULTADO\ncuando hay que cerrar",
        foot=("cada punto = 1 jugador  ·  caras = top + destacados por eje  ·  ejes = "
              "cambio post-shock relativo al resto del equipo")),
    # Los 2 unicos ejes con estructura real: atk-GF (rango 0.87, 22 Sig) y
    # atk-Pressure (0.59). El mapa de produccion ofensiva tras el shock.
    "ataque_marcar_presion": dict(
        x="cate_ataque_GOAL_FOR_mean", y="cate_ataque_PRESSURE_mean",
        title="Ataque tras marcar  vs  bajo presion",
        x_label="Mas ataque tras poner a su equipo por delante",
        y_label="Mas ataque cuando crece el riesgo de eliminacion",
        top="ARRIBA: los que hacen LAS DOS COSAS",
        left="Por este lado\nNO ESPECULAN CON EL RESULTADO\nsiguen atacando tras marcar",
        right="Por este lado\nDAN UN PASO ADELANTE\ncuando el equipo roza la eliminacion",
        foot=("cada punto = 1 jugador  ·  caras = top + destacados por eje  ·  ejes = "
              "valor ofensivo post-shock")),
}


def diamond_scatter(df: pl.DataFrame, config: str | dict = "remontador_cerrojo",
                     save_path=None):
    """Diamond scatter rotado 45 grados. `config` = clave de SCATTERS o dict."""
    cfg = SCATTERS[config] if isinstance(config, str) else config
    x_metric, y_metric = cfg["x"], cfg["y"]

    pdf = df.to_pandas()
    left = pdf[x_metric].fillna(0.0)      # eje izquierdo del diamante
    right = pdf[y_metric].fillna(0.0)     # eje inferior del diamante

    # Normalizacion min-max -> [0, 0.99]. Los CATEs son con signo, no v/max.
    lmin, lmax = float(left.min()), float(left.max())
    rmin, rmax = float(right.min()), float(right.max())
    left_n = 0.99 * (left - lmin) / (lmax - lmin)
    right_n = 0.99 * (right - rmin) / (rmax - rmin)

    l_med = float(left_n.median())        # mediana P50 eje x (global)
    r_med = float(right_n.median())       # mediana P50 eje y (global)

    # Percentiles al vuelo (rank) -> color + seleccion de top, valido para
    # cualquier metrica (no depende de cols pct_* cacheadas).
    px = (pdf[x_metric].rank(pct=True) * 100.0).to_numpy()
    py = (pdf[y_metric].rank(pct=True) * 100.0).to_numpy()
    pdf = pdf.assign(_px=px, _py=py)
    # Caras = (a) top-10 combinado: extremos en AMBOS ejes a la vez ...
    combo = pdf[(pdf["_px"] >= 81) & (pdf["_py"] >= 81)]
    if len(combo) < 5:
        combo = pdf[(pdf["_px"] >= 75) & (pdf["_py"] >= 75)]
    combo = combo.assign(_tot=combo["_px"] + combo["_py"]).nlargest(10, "_tot")
    # ... + (b) los 2 mejores de CADA eje individual que NO esten ya en el
    # top-10 combinado (p.ej. un atacante top en ataque-bajo-presion pero
    # medio en ataque-tras-marcar igual merece su cara). Max 2 por eje.
    rest = pdf.loc[pdf.index.difference(combo.index)]
    spec_x = rest.nlargest(2, "_px").index
    spec_y = rest.nlargest(2, "_py").index
    top = pdf.loc[combo.index.union(spec_x).union(spec_y)]

    fig = plt.figure(figsize=(10, 11.5), facecolor=BG)            # ↑ figsize -> figura MAS GRANDE

    # ---- Header: logo torneo (izq) + titulo + subtitulos + logo JO (dcha) ----
    # Viz generica -> logo WC22 arriba-izq (en vez de escudo de seleccion).
    if _WC22_LOGO.exists():
        wimg = Image.open(_WC22_LOGO)
        ab = AnnotationBbox(
            OffsetImage(np.asarray(wimg.convert("RGBA")), zoom=0.17),  # ↑ zoom -> logo MAS GRANDE
            (0.15, 0.88), frameon=False,                              # ↑ X -> a la DERECHA; ↑ Y -> SUBE
            xycoords="figure fraction", box_alignment=(0.5, 0.5))
        ab.set_clip_on(False)
        fig.add_artist(ab)

    fig.text(0.50, 0.955, cfg["title"], ha="center", va="center",
              color=WHITE, fontsize=19, fontweight="bold")
    fig.text(0.50, 0.925, "FIFA Men's World Cup 2022",
              ha="center", va="center", color=WHITE, fontsize=14)
    fig.text(0.50, 0.905, "Los 511 jugadores del torneo",
              ha="center", va="center", color=WHITE, fontsize=12)

    if _LOGO_PATH.exists():
        limg = Image.open(_LOGO_PATH)
        ab = AnnotationBbox(
            OffsetImage(np.asarray(limg.convert("RGBA")), zoom=0.14),  # ↑ zoom -> logo MAS GRANDE
            (0.89, 0.90), frameon=False,                               # X -> borde DCHO del logo aqui
            xycoords="figure fraction", box_alignment=(1.0, 0.5))      # ancla RIGHT-edge (no se corta)
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
    # Posicion del diamante. ↑ width/height -> diamante MAS GRANDE.
    # ↑ bottom -> diamante SUBE. Top en ~0.84 (cerca del subtitulo).
    ax.set_position([0.07, 0.12, 0.86, 0.72], which="both")
    aux = ax.get_aux_axes(transform)
    ax = fig.add_axes(ax)
    aux.patch = ax.patch

    # Estilo ejes: solo left + bottom del diamante (otros invisibles)
    ax.axis["left"].line.set_color(WHITE)
    ax.axis["bottom"].line.set_color(WHITE)
    ax.axis["right"].set_visible(False)
    ax.axis["top"].set_visible(False)
    ax.axis["left"].major_ticklabels.set(rotation=0, ha="center", fontsize=10)
    ax.axis["bottom"].major_ticklabels.set(fontsize=10)
    ax.axis["bottom"].major_ticklabels.set_pad(6)
    for side, lbl in (("left", cfg["x_label"]), ("bottom", cfg["y_label"])):
        ax.axis[side].set_label(lbl)
        ax.axis[side].label.set(color=WHITE, fontweight="bold", fontsize=12)
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

    # ---- Top jugadores: CARA (FotMob) en vez de etiqueta de texto ----
    # RGBA explicito (los PNG FotMob vienen en modo Palette; sin convertir
    # OffsetImage aplica colormap default y se ven verdes). La cara tapa el
    # punto de debajo.
    for i, p in top.iterrows():
        pid = int(p["pff_player_id"])
        face_p = _FACES / f"{pid}.png"
        if not face_p.exists():
            continue
        img_arr = np.asarray(Image.open(face_p).convert("RGBA"))
        ab = AnnotationBbox(OffsetImage(img_arr, zoom=0.20),       # ↑ zoom -> cara MAS GRANDE
                             (right_n.loc[i], left_n.loc[i]), frameon=False,
                             pad=0.0, xycoords=aux.transData, zorder=5)
        aux.add_artist(ab)

    # ---- Carteles direccionales en los triangulos vacios del diamante ----
    fig.text(0.50, 0.855, cfg["top"],
              ha="center", va="center", color=WHITE, fontsize=12,
              fontweight="bold", style="italic")                          # vertice SUPERIOR
    fig.text(0.205, 0.725, cfg["left"],
              ha="center", va="center", color=WHITE, fontsize=12,
              linespacing=1.5)                                            # esquina IZQUIERDA
    fig.text(0.795, 0.725, cfg["right"],
              ha="center", va="center", color=WHITE, fontsize=12,
              linespacing=1.5)                                            # esquina DERECHA

    # ---- Leyenda real al pie: linea dashed = mediana + glosa ----
    leg_y = 0.058
    # Misma linea dashed EXACTA que las medianas del plot (lw, ls, caps, alpha):
    # matplotlib escala el dash por lw -> on=7*2.4, off=4*2.4 (~26pt/periodo).
    # Largo ~0.10 de figura para que salgan justo 3 dashes.
    fig.add_artist(Line2D([0.255, 0.355], [leg_y, leg_y], color=WHITE, lw=2.4,
                           alpha=0.9, ls=(0, (7, 4)), solid_capstyle="round",
                           transform=fig.transFigure))
    fig.text(0.365, leg_y, "mediana del torneo (P50, 511 jugadores)",
              ha="left", va="center", color=WHITE, fontsize=12)
    fig.text(0.50, leg_y - 0.030, cfg["foot"],
              ha="center", va="center", color=WHITE, fontsize=12, style="italic")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # bbox_inches=None (NO 'tight') para que logos no se corten
        fig.savefig(save_path, dpi=200, facecolor=BG, bbox_inches=None)
        plt.close(fig)
    return fig


if __name__ == "__main__":
    df = pl.read_parquet(_TABLE)
    for key in ("remontador_cerrojo", "ataque_marcar_presion"):
        out = f"outputs/viz/scatter_{key}.png"
        diamond_scatter(df, config=key, save_path=out)
        print(f"OK -> {out}  ({df.height} jugadores)")
