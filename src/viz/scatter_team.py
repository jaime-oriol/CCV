"""scatter_team - 4 diamond scatters individuales por seleccion con CARAS.

Misma estetica que `scatter.py` (diamond rotado 45 grados, mensajes laterales,
mediana del dataset global, gradiente PCT_CMAP) pero filtrado a 1 seleccion
y plotando la CARA del jugador (FotMob) en vez de un punto. Genera 4 PNG
individuales, uno por par de dimensiones:

  1. Remontador × Cerrojo            (chasing_clutch_idx × protecting_clutch_idx)
  2. Remontador × Pressure           (chasing × pressure_response)
  3. Cerrojo × Pressure              (protecting × pressure)
  4. Huele sangre × Cierra atras     (atk_GF × def_GF — canales con senal real)

Uso:
    python -m src.viz.scatter_team France
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

_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# 4 paneles: x/y metric, ejes, mensajes laterales, matiz tecnico al pie.
# Etiquetas laterales = lectura intuitiva; matiz al pie = honestidad sobre
# que mide cada canal (evita interpretacion enganosa por posicion).
_PAIRS = [
    dict(
        x="chasing_clutch_idx",      y="protecting_clutch_idx",
        x_label="REMONTADOR  —  empuje ofensivo + off-ball post-encajar",
        y_label="CERROJO  —  acciones defensivas + intensidad fisica post-marcar",
        side_left="SUBEN EMPUJE Y MOVIMIENTO\noff-ball post-encajar gol",
        side_right="MAS ACCIONES DEFENSIVAS\nE INTENSIDAD post-marcar gol",
        top="ARRIBA: hacen LAS DOS COSAS",
        title="Reaccion post-encajar vs aguante post-marcar",
        matiz="CERROJO mide acciones defensivas (recuperaciones, presiones, intervenciones) + fisico (HSR, sprints). Interpretacion segun rol: delanteros = presion alta; defensas = repliegue al area.",
        slug="remontador_cerrojo"),
    dict(
        x="chasing_clutch_idx",      y="pressure_response_idx",
        x_label="REMONTADOR  —  empuje ofensivo + off-ball post-encajar",
        y_label="PRESSURE  —  pendiente individual respecto a elim_prox",
        side_left="REACCIONAN AL ENCAJAR\nsuben empuje + off-ball",
        side_right="RINDEN MAS BAJO PRESION\ncuando hay riesgo eliminacion",
        top="ARRIBA: reaccionan AL ENCAJAR y bajo PRESION",
        title="Reaccion al gol vs pulso bajo presion",
        matiz="PRESSURE = pendiente eta_pressure (Bayesian slope respecto a P(equipo NO clasifica)). Positivo = rinde MAS cuanto mayor el riesgo. Mide cambio en CATE multivariate, no es score absoluto.",
        slug="remontador_pressure"),
    dict(
        x="protecting_clutch_idx",   y="pressure_response_idx",
        x_label="CERROJO  —  acciones defensivas + intensidad fisica post-marcar",
        y_label="PRESSURE  —  pendiente individual respecto a elim_prox",
        side_left="SUBEN DEFENSA Y FISICO\npost-marcar gol",
        side_right="RINDEN MAS BAJO PRESION\ncuando hay riesgo eliminacion",
        top="ARRIBA: cierran atras Y aparecen bajo presion",
        title="Aguante tras marcar vs pulso bajo presion",
        matiz="CERROJO = vdep + xpress + maejima + fisico. Por rol: delanteros = presion alta; defensas = repliegue. PRESSURE = pendiente Bayesian respecto a elim_prox.",
        slug="cerrojo_pressure"),
    dict(
        x="cate_ataque_GOAL_FOR_mean", y="cate_defensa_GOAL_FOR_mean",
        x_label="EMPUJE OFENSIVO post-GF (cambio en atomic-VAEP)",
        y_label="ACCIONES DEFENSIVAS post-GF (vdep + xpress + maejima)",
        side_left="SIGUEN EMPUJANDO\nmas valor ofensivo post-marcar",
        side_right="MAS ACCIONES DEFENSIVAS\npresion alta o repliegue, segun rol",
        top="ARRIBA: SIGUEN ATACANDO Y DEFENDIENDO (raro)",
        title="Tras marcar gol — atacar o defender",
        matiz="DEF post-GF NO mide posicion fisica. Mide ACCIONES defensivas (recuperaciones, presiones, intervenciones). Delanteros con DEF positivo = presion alta post-marcar (Mbappe). Centrales = repliegue al area.",
        slug="atkGF_defGF"),
]


def _short_name(name: str) -> str:
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {parts[-1]}"
    return name


def diamond_team(df_full: pl.DataFrame, team: str, pair: dict,
                  save_path: Path) -> Path:
    """1 diamond scatter rotado 45 grados con CARAS del equipo + lineas
    mediana del dataset global (contexto de los 511 jug WC22)."""
    pdf_full = df_full.to_pandas()
    pdf_team = pdf_full[pdf_full["team_name"] == team].reset_index(drop=True)

    x_metric, y_metric = pair["x"], pair["y"]
    xs_full = pdf_full[x_metric].fillna(0.0)
    ys_full = pdf_full[y_metric].fillna(0.0)
    xmin, xmax = float(xs_full.min()), float(xs_full.max())
    ymin, ymax = float(ys_full.min()), float(ys_full.max())
    # Normalizacion min-max sobre el GLOBAL (los puntos del equipo se ven en
    # contexto del torneo entero, no escalados solo a si mismos).
    left_n = (0.99 * (pdf_team[x_metric].fillna(0.0) - xmin) / (xmax - xmin)).to_numpy()
    right_n = (0.99 * (pdf_team[y_metric].fillna(0.0) - ymin) / (ymax - ymin)).to_numpy()
    l_med = 0.99 * (float(xs_full.median()) - xmin) / (xmax - xmin)
    r_med = 0.99 * (float(ys_full.median()) - ymin) / (ymax - ymin)

    fig = plt.figure(figsize=(10, 11.5), facecolor=BG)

    # ---- Header style pass-plot: escudo seleccion + titulo + logo JO ----
    slug = _TEAM_TO_SLUG.get(team)
    team_logo_p = _LOGOS / f"{slug}.png" if slug else None
    fig_w = 10; fig_h = 11.5
    if team_logo_p and team_logo_p.exists():
        img = Image.open(team_logo_p)
        ab = AnnotationBbox(OffsetImage(np.asarray(img.convert("RGBA")), zoom=0.70),
                             (0.13, 0.945), frameon=False,
                             xycoords="figure fraction", box_alignment=(0.5, 0.5))
        ab.set_clip_on(False)
        fig.add_artist(ab)
    # Titulo
    fig.text(0.50, 0.965, pair["title"], ha="center", va="center", color=WHITE,
              fontsize=20, fontweight="bold")
    fig.text(0.50, 0.935,
              f"Seleccion {team}  ·  Mundial Qatar 2022  ·  "
              f"{len(pdf_team)} jugadores  ·  vs los 511 del torneo",
              ha="center", va="center", color=WHITE, fontsize=11)
    # Logo JO
    if _LOGO_PATH.exists():
        limg = Image.open(_LOGO_PATH)
        ab = AnnotationBbox(OffsetImage(np.asarray(limg.convert("RGBA")), zoom=0.13),
                             (0.87, 0.945), frameon=False,
                             xycoords="figure fraction", box_alignment=(0.5, 0.5))
        ab.set_clip_on(False)
        fig.add_artist(ab)

    # Marcas de eje con valor real (signed)
    left_dict = {i: f"{xmin + (i / 0.99) * (xmax - xmin):+.3f}" for i in _TICKS}
    right_dict = {i: f"{ymin + (i / 0.99) * (ymax - ymin):+.3f}" for i in _TICKS}
    transform = Affine2D().rotate_deg(45)
    helper = floating_axes.GridHelperCurveLinear(
        transform, (0, 1.001, 0, 1.001),
        grid_locator1=FixedLocator(_TICKS), grid_locator2=FixedLocator(_TICKS),
        tick_formatter1=DictFormatter(right_dict),
        tick_formatter2=DictFormatter(left_dict))
    ax = floating_axes.FloatingSubplot(fig, 111, grid_helper=helper)
    # Diamond mas pegado al header (antes 0.10 -> 0.18 top space; ahora menos gap)
    ax.set_position([0.08, 0.07, 0.84, 0.78], which="both")
    aux = ax.get_aux_axes(transform)
    ax = fig.add_axes(ax)
    aux.patch = ax.patch

    ax.axis["left"].line.set_color(WHITE)
    ax.axis["bottom"].line.set_color(WHITE)
    ax.axis["right"].set_visible(False)
    ax.axis["top"].set_visible(False)
    ax.axis["left"].major_ticklabels.set(rotation=0, ha="center", fontsize=8.5)
    ax.axis["bottom"].major_ticklabels.set(fontsize=8.5)
    ax.axis["bottom"].major_ticklabels.set_pad(6)
    for side, lbl in (("left", pair["x_label"]), ("bottom", pair["y_label"])):
        ax.axis[side].set_label(lbl)
        ax.axis[side].label.set(color=WHITE, fontweight="bold", fontsize=11)
        ax.axis[side].LABELPAD += 9
    ax.axis["left"].label.set_rotation(0)
    ax.grid(alpha=0.18, color=WHITE)

    # Lineas mediana del GLOBAL (4 cuadrantes)
    aux.plot([r_med, r_med], [0.0, 1.001], color=WHITE, lw=2.4, alpha=0.9,
             ls=(0, (7, 4)), zorder=6, solid_capstyle="round")
    aux.plot([0.0, 1.001], [l_med, l_med], color=WHITE, lw=2.4, alpha=0.9,
             ls=(0, (7, 4)), zorder=6, solid_capstyle="round")

    # Caras del equipo, sin etiqueta de nombre (limpia visual). RGBA
    # explicito porque los PNG FotMob vienen en modo Palette (P) y sin
    # convertir se renderizan con tinte verde via colormap por defecto.
    for i in range(len(pdf_team)):
        pid = int(pdf_team["pff_player_id"].iloc[i])
        x, y = float(right_n[i]), float(left_n[i])
        if np.isnan(x) or np.isnan(y):
            continue
        face_p = _FACES / f"{pid}.png"
        if not face_p.exists():
            continue
        img_arr = np.asarray(Image.open(face_p).convert("RGBA"))
        img = OffsetImage(img_arr, zoom=0.17)       # 0.13 -> 0.17 (un pelin mas)
        ab = AnnotationBbox(img, (x, y), frameon=False, pad=0.0,
                             xycoords=aux.transData, zorder=5)
        aux.add_artist(ab)

    # Carteles laterales y top (todo blanco, posiciones ajustadas al nuevo layout)
    fig.text(0.20, 0.62, f"Por este lado\n{pair['side_left']}",
              ha="center", va="center", color=WHITE, fontsize=10, linespacing=1.5)
    fig.text(0.80, 0.62, f"Por este lado\n{pair['side_right']}",
              ha="center", va="center", color=WHITE, fontsize=10, linespacing=1.5)
    fig.text(0.50, 0.72, pair["top"], ha="center", va="center", color=WHITE,
              fontsize=10, fontweight="bold", style="italic")

    fig.text(0.045, 0.04, "Lineas discontinuas: mediana del torneo "
              "(percentil 50 de los 511 jug) — parten el campo en 4 cuadrantes.",
              ha="left", va="bottom", color=WHITE, fontsize=8.5, style="italic")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, facecolor=BG, bbox_inches="tight")
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
