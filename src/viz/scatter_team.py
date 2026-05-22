"""scatter_team - 4 diamond scatters por seleccion con CARAS de jugadores.

Misma estetica que scatter.py (diamante rotado 45 grados, mediana global,
mensajes laterales) pero filtrado a 1 seleccion y ploteando la CARA del
jugador (FotMob, PNG transparente) como marker en vez de un punto.

2 paneles, 1 PNG cada uno:
  1. Remontador x Cerrojo        (chasing x protecting) — marco de la propuesta
  2. Ataque tras marcar x bajo presion  (atk-GF x atk-Pressure) — los 2 ejes
                                 con spread real (donde el shock deja huella)

Uso:
    python -m src.viz.scatter_team France
    python -m src.viz.scatter_team England
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

from viz.common import BG, WHITE, _LOGO_PATH

_TABLE = _SRC.parent / "outputs" / "pcj_table.parquet"
_FACES = _SRC.parent / "outputs" / "assets" / "faces"
_LOGOS = _SRC.parent / "outputs" / "assets" / "logos"

_TEAM_TO_SLUG = {
    "Argentina":"arg","Brazil":"bra","Ecuador":"ecu","Uruguay":"uru","Belgium":"bel",
    "Croatia":"cro","Denmark":"den","England":"eng","France":"fra","Germany":"ger",
    "Netherlands":"ned","Poland":"pol","Portugal":"por","Serbia":"srb","Spain":"esp",
    "Switzerland":"sui","Wales":"wal","Cameroon":"cmr","Ghana":"gha","Morocco":"mar",
    "Senegal":"sen","Tunisia":"tun","Japan":"jpn","South Korea":"kor","Iran":"irn",
    "Qatar":"qat","Saudi Arabia":"ksa","Canada":"can","Mexico":"mex",
    "United States":"usa","Costa Rica":"crc","Australia":"aus",
}

_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]      # 6 marcas por eje del diamante

# Configuracion de los 4 paneles. side_left/right/top = textos en cada esquina
# del diamante. La interpretacion fina de cada canal va en el TFM, no en la viz.
_PAIRS = [
    dict(
        x="chasing_clutch_idx",      y="protecting_clutch_idx",
        x_label="REMONTADOR: ataque y movimiento off-ball tras encajar",
        y_label="CERROJO: defensa e intensidad fisica tras marcar",
        side_left="SUBEN ATAQUE Y MOVIMIENTO\noff-ball post-encajar gol",
        side_right="MAS ACCIONES DEFENSIVAS\nE INTENSIDAD post-marcar gol",
        top="ARRIBA: hacen LAS DOS COSAS",
        title="Remontador  vs  Cerrojo",
        slug="remontador_cerrojo"),
    dict(
        x="cate_ataque_GOAL_FOR_mean", y="cate_ataque_PRESSURE_mean",
        x_label="Mas ataque tras poner a su equipo por delante",
        y_label="Mas ataque cuando crece el riesgo de eliminacion",
        side_left="NO ESPECULAN CON EL RESULTADO\nsiguen atacando tras marcar",
        side_right="DAN UN PASO ADELANTE\ncuando el equipo roza la eliminacion",
        top="ARRIBA: siguen atacando con ventaja y bajo presion",
        title="Ataque tras marcar  vs  ataque bajo presion",
        slug="killer_biggame"),
]


def diamond_team(df_full: pl.DataFrame, team: str, pair: dict,
                  save_path: Path) -> Path:
    """1 diamante con CARAS del equipo + mediana del global (511 jug WC22)."""
    pdf_full = df_full.to_pandas()
    pdf_team = pdf_full[pdf_full["team_name"] == team].reset_index(drop=True)

    x_metric, y_metric = pair["x"], pair["y"]
    xs_full = pdf_full[x_metric].fillna(0.0)
    ys_full = pdf_full[y_metric].fillna(0.0)
    xmin, xmax = float(xs_full.min()), float(xs_full.max())
    ymin, ymax = float(ys_full.min()), float(ys_full.max())
    # Normalizacion min-max sobre el GLOBAL: los jugadores del equipo se ven
    # en contexto del torneo, no escalados solo entre ellos.
    left_n = (0.99 * (pdf_team[x_metric].fillna(0.0) - xmin) / (xmax - xmin)).to_numpy()
    right_n = (0.99 * (pdf_team[y_metric].fillna(0.0) - ymin) / (ymax - ymin)).to_numpy()
    l_med = 0.99 * (float(xs_full.median()) - xmin) / (xmax - xmin)
    r_med = 0.99 * (float(ys_full.median()) - ymin) / (ymax - ymin)

    fig = plt.figure(figsize=(10, 11.5), facecolor=BG)            # ↑ figsize -> figura MAS GRANDE

    # ---- Header: escudo seleccion (izq) + titulo + subtitulos + logo JO (dcha) ----
    # Mismo patron que PPCF/radar. bbox_inches=None al guardar para que los
    # logos pegados al borde NO se corten.
    slug = _TEAM_TO_SLUG.get(team)
    team_logo_p = _LOGOS / f"{slug}.png" if slug else None
    if team_logo_p and team_logo_p.exists():
        img = Image.open(team_logo_p)
        ab = AnnotationBbox(
            OffsetImage(np.asarray(img.convert("RGBA")), zoom=1.15),  # ↑ zoom -> escudo MAS GRANDE
            (0.125, 0.92), frameon=False,                            # ↑ X -> escudo a la DERECHA; ↑ Y -> SUBE
            xycoords="figure fraction", box_alignment=(0.5, 0.5))     # ancla CENTRO en (X,Y)
        ab.set_clip_on(False)
        fig.add_artist(ab)

    # Titulo + subtitulos (centrados)
    fig.text(0.50, 0.955, pair["title"], ha="center", va="center", color=WHITE,
              fontsize=19, fontweight="bold")                          # ↑ Y -> titulo SUBE
    fig.text(0.50, 0.925, "FIFA Men's World Cup 2022",
              ha="center", va="center", color=WHITE, fontsize=14)
    fig.text(0.50, 0.905,
              f"Seleccion de {team} vs los 511 jugadores del torneo",
              ha="center", va="center", color=WHITE, fontsize=12)

    # Logo JO (esquina superior derecha)
    if _LOGO_PATH.exists():
        limg = Image.open(_LOGO_PATH)
        ab = AnnotationBbox(
            OffsetImage(np.asarray(limg.convert("RGBA")), zoom=0.14),  # ↑ zoom -> logo MAS GRANDE
            (0.89, 0.92), frameon=False,                              # X -> borde DCHO del logo aqui
            xycoords="figure fraction", box_alignment=(1.0, 0.5))      # ancla RIGHT-edge (no se corta)
        ab.set_clip_on(False)
        fig.add_artist(ab)

    # ---- Marcas de eje con valor REAL (con signo) ----
    left_dict  = {i: f"{xmin + (i / 0.99) * (xmax - xmin):+.3f}" for i in _TICKS}
    right_dict = {i: f"{ymin + (i / 0.99) * (ymax - ymin):+.3f}" for i in _TICKS}

    # ---- floating_axes rotado 45 deg (diamante) ----
    transform = Affine2D().rotate_deg(45)
    helper = floating_axes.GridHelperCurveLinear(
        transform, (0, 1.001, 0, 1.001),
        grid_locator1=FixedLocator(_TICKS), grid_locator2=FixedLocator(_TICKS),
        tick_formatter1=DictFormatter(right_dict),
        tick_formatter2=DictFormatter(left_dict))
    ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=helper)
    # Posicion del diamante. ↑ width/height -> diamante MAS GRANDE.
    # ↑ bottom -> diamante SUBE; ↓ -> baja. Top en ~0.84 (cerca del subtitulo).
    ax.set_position([0.07, 0.12, 0.86, 0.72], which="both")
    aux = ax.get_aux_axes(transform)
    ax = fig.add_axes(ax)
    aux.patch = ax.patch

    ax.axis["left"].line.set_color(WHITE)
    ax.axis["bottom"].line.set_color(WHITE)
    ax.axis["right"].set_visible(False)
    ax.axis["top"].set_visible(False)
    ax.axis["left"].major_ticklabels.set(rotation=0, ha="center", fontsize=10)
    ax.axis["bottom"].major_ticklabels.set(fontsize=10)
    ax.axis["bottom"].major_ticklabels.set_pad(6)
    for side, lbl in (("left", pair["x_label"]), ("bottom", pair["y_label"])):
        ax.axis[side].set_label(lbl)
        ax.axis[side].label.set(color=WHITE, fontweight="bold", fontsize=12)
        ax.axis[side].LABELPAD += 9
    ax.axis["left"].label.set_rotation(0)
    ax.grid(alpha=0.18, color=WHITE)

    # Lineas mediana del GLOBAL (parten el diamante en 4 cuadrantes)
    aux.plot([r_med, r_med], [0.0, 1.001], color=WHITE, lw=2.4, alpha=0.9,
              ls=(0, (7, 4)), zorder=6, solid_capstyle="round")
    aux.plot([0.0, 1.001], [l_med, l_med], color=WHITE, lw=2.4, alpha=0.9,
              ls=(0, (7, 4)), zorder=6, solid_capstyle="round")

    # ---- Caras del equipo como markers ----
    # RGBA explicito porque los PNG FotMob vienen en modo Palette (P);
    # sin convertir, OffsetImage aplica colormap default y se ven verdes.
    for i in range(len(pdf_team)):
        pid = int(pdf_team["pff_player_id"].iloc[i])
        x, y = float(right_n[i]), float(left_n[i])
        if np.isnan(x) or np.isnan(y):
            continue
        face_p = _FACES / f"{pid}.png"
        if not face_p.exists():
            continue
        img_arr = np.asarray(Image.open(face_p).convert("RGBA"))
        img = OffsetImage(img_arr, zoom=0.20)            # ↑ zoom -> cara MAS GRANDE
        ab = AnnotationBbox(img, (x, y), frameon=False, pad=0.0,
                             xycoords=aux.transData, zorder=5)
        aux.add_artist(ab)

    # ---- Carteles direccionales en los triangulos vacios del diamante ----
    # El diamante rotado deja huecos en vertice superior + esquinas izq/dcha;
    # ahi van los textos para NO pisar las caras.
    fig.text(0.50, 0.855, pair["top"], ha="center", va="center", color=WHITE,
              fontsize=12, fontweight="bold", style="italic")               # vertice SUPERIOR
    fig.text(0.205, 0.725, f"Por este lado\n{pair['side_left']}",
              ha="center", va="center", color=WHITE, fontsize=12,
              linespacing=1.5)                                              # esquina IZQUIERDA
    fig.text(0.795, 0.725, f"Por este lado\n{pair['side_right']}",
              ha="center", va="center", color=WHITE, fontsize=12,
              linespacing=1.5)                                              # esquina DERECHA

    # ---- Leyenda real al pie: linea dashed = mediana + glosa de que es cada cara ----
    leg_y = 0.058
    # Misma linea dashed EXACTA que las medianas del plot (lw, ls, caps, alpha):
    # matplotlib escala el dash por lw -> on=7*2.4, off=4*2.4 (~26pt/periodo).
    # Largo ~0.10 de figura para que salgan justo 3 dashes.
    fig.add_artist(Line2D([0.255, 0.355], [leg_y, leg_y], color=WHITE, lw=2.4,
                           alpha=0.9, ls=(0, (7, 4)), solid_capstyle="round",
                           transform=fig.transFigure))
    fig.text(0.365, leg_y, "mediana del torneo (P50, 511 jugadores)",
              ha="left", va="center", color=WHITE, fontsize=12)
    fig.text(0.50, leg_y - 0.030,
              "cada cara = 1 jugador de la seleccion  ·  ejes = cambio post-shock "
              "relativo al resto del equipo",
              ha="center", va="center", color=WHITE, fontsize=12, style="italic")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches=None (NO 'tight') para que escudo + logo JO no se corten
    fig.savefig(save_path, dpi=200, facecolor=BG, bbox_inches=None)
    plt.close(fig)
    return save_path


def scatter_team_all(df: pl.DataFrame, team: str, out_dir: Path) -> list[Path]:
    """Genera los 4 diamond scatters individuales del equipo."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    team_slug = team.lower().replace(" ", "_")
    paths = []
    for pair in _PAIRS:
        p = out_dir / f"scatter_{team_slug}_{pair['slug']}.png"
        diamond_team(df, team, pair, p)
        paths.append(p)
    return paths


if __name__ == "__main__":
    df = pl.read_parquet(_TABLE)
    team = sys.argv[1] if len(sys.argv) > 1 else "France"
    out_dir = _SRC.parent / "outputs" / "viz"
    paths = scatter_team_all(df, team, out_dir)
    for p in paths:
        print(f"OK -> {p}")
