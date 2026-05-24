"""scatter - Scatters globales del xCV estilo Opta (landscape, paper-grade).

Scatter X-Y nativo (NO rotado). Layout landscape 16x9 que casa con ancho de
doc. Header PPCF-style (escudo WC22 IZQ | titulo+sub LEFT | JO DCHA). Todos
los puntos coloreados por PCT_CMAP (gradiente percentil combinado x+y).
Caras (FotMob) sobre top combinado + 2 destacados por eje. Resto del torneo
en el gradiente. Mediana del torneo dashed con label INLINE.

2 conceptos config-driven (via SCATTERS):
  1. remontador_cerrojo    : Remontador x Cerrojo (los 2 perfiles agregados)
  2. ataque_marcar_encajar : Ataque tras GF x Ataque tras GA (2 ejes con
                             spread real; donde el shock deja huella medible)

Identidad LIGHT OPTA: BG blanco, textos negros, leyenda gris medio, paleta
PCT_CMAP morado→fuchsia→rosa. Sin carteles direccionales (los ejes hablan).

Uso:
    python -m src.viz.scatter
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage           # pa pegar caras FotMob
from PIL import Image

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viz.common import (BG, GRID, LEGEND, MASTER_FIGSIZE, N_PLAYERS_WC22, PCT_CMAP,
                         TEXT, TOURNAMENT_ES, draw_header)

_TABLE = _SRC.parent / "outputs" / "xcv_table.parquet"                # tabla scout final (M15)
_LOGOS = _SRC.parent / "outputs" / "assets" / "logos"                 # escudos selecciones
_FACES = _SRC.parent / "outputs" / "assets" / "faces"                 # caras jugadores (FotMob)
_WC22_LOGO = _LOGOS / "wc22.png"                                       # escudo del torneo (header del scatter global)

# ----------------------------------------------------------------------------
# Configuracion de los 2 scatters (pattern Opta)
# ----------------------------------------------------------------------------
#   title       = noun phrase corto que dice QUIEN se mapea
#   subtitle    = "Comparando X y Y | contexto" — los ejes no necesitan label largo
#   x_label /
#   y_label     = concepto puro, breve (los ejes hablan por si mismos)
#   foot        = nota al pie con descripcion metodologica (gris)

SCATTERS: dict[str, dict] = {
    # Los 2 perfiles agregados de la propuesta. Ejes apretados (la elite es
    # homogenea en los agregados), pero es el marco conceptual del TFM.
    "remontador_cerrojo": dict(
        x="chasing_clutch_idx", y="protecting_clutch_idx",
        title="Remontador y Cerrojo",
        subtitle=f"Comparando los dos perfiles del shock emocional  |  {TOURNAMENT_ES} · {N_PLAYERS_WC22} jugadores",
        x_label="Índice Remontador",
        y_label="Índice Cerrojo",
        foot="*Cambio en el rendimiento individual tras el shock emocional, ajustado por el resto del equipo"),
    "ataque_marcar_encajar": dict(
        x="cate_ataque_GOAL_FOR_mean", y="cate_ataque_PRESSURE_mean",
        title="Atacantes clutch",
        subtitle=f"Ataque tras marcar vs tras encajar  |  {TOURNAMENT_ES} · {N_PLAYERS_WC22} jugadores",
        x_label="Producción ofensiva tras marcar",
        y_label="Producción ofensiva tras encajar",
        foot="*Cambio en la producción ofensiva tras marcar o encajar, ajustado por el resto del equipo"),
}

# ----------------------------------------------------------------------------
# Tuneo visual (puntos, caras, mediana)
# ----------------------------------------------------------------------------
_DOT_SIZE        = 70          # ↑ → puntos MAS GRANDES (todos coloreados por PCT_CMAP)
_FACE_ZOOM       = 0.2        # ↑ → caras FotMob MAS GRANDES sobre el punto
_MED_LINE_COLOR  = "#888888"   # gris medio pa la mediana del torneo (dashed inline)
_MED_LINE_LW     = 1.0         # grosor de la mediana — ↑ MAS GRUESO


def opta_scatter(df: pl.DataFrame, config: str | dict = "remontador_cerrojo",
                 save_path=None):
    """Scatter Opta-style landscape. `config` = clave de SCATTERS o dict propio."""
    cfg = SCATTERS[config] if isinstance(config, str) else config
    x_metric, y_metric = cfg["x"], cfg["y"]

    # ---- Datos: valores raw + percentiles al vuelo ----
    pdf = df.to_pandas().reset_index(drop=True)
    xs = pdf[x_metric].fillna(0.0).to_numpy()
    ys = pdf[y_metric].fillna(0.0).to_numpy()
    # Percentiles → color de TODOS los puntos (gradiente PCT_CMAP)
    px = pdf[x_metric].rank(pct=True).to_numpy() * 100.0
    py = pdf[y_metric].rank(pct=True).to_numpy() * 100.0
    pdf = pdf.assign(_px=px, _py=py)
    color_norm = (px + py) / 200.0                                    # 0..1 pa PCT_CMAP

    # ---- Top pa pegar caras = combinado + destacados por eje ----
    # (a) top combinado: extremos en AMBOS ejes a la vez (P>81). Si pocos, baja a P>75.
    combo = pdf[(pdf["_px"] >= 81) & (pdf["_py"] >= 81)]
    if len(combo) < 5:
        combo = pdf[(pdf["_px"] >= 75) & (pdf["_py"] >= 75)]
    combo = combo.assign(_tot=combo["_px"] + combo["_py"]).nlargest(10, "_tot")
    # (b) + los 2 mejores de CADA eje individual fuera del top combinado
    rest = pdf.loc[pdf.index.difference(combo.index)]
    spec_x = rest.nlargest(2, "_px").index
    spec_y = rest.nlargest(2, "_py").index
    top_idx = combo.index.union(spec_x).union(spec_y)

    # ---- Mediana del torneo (sobre valores raw, no percentiles) ----
    x_med, y_med = float(np.median(xs)), float(np.median(ys))

    # ---- Figura landscape MASTER (16x9 @ 150dpi = 2400x1350) ----
    fig = plt.figure(figsize=MASTER_FIGSIZE, facecolor=BG)

    # Header PPCF-style: escudo WC22 IZQ + titulo+sub LEFT + JO DCHA
    draw_header(fig, title=cfg["title"], subtitle=cfg["subtitle"],
                escudo_path=_WC22_LOGO)

    # Plot area: header arriba y=[0.85,1.0]; plot y=[0.12, 0.815]; footer abajo
    # Si subes HEADER_BAND, baja el top del axes y (=0.815) en proporcion.
    ax = fig.add_axes([0.07, 0.12, 0.88, 0.695])
    ax.set_facecolor(BG)

    # ---- Lineas mediana del torneo (dashed grey) ----
    ax.axvline(x_med, color=_MED_LINE_COLOR, lw=_MED_LINE_LW, ls="--",
                alpha=0.75, zorder=2)
    ax.axhline(y_med, color=_MED_LINE_COLOR, lw=_MED_LINE_LW, ls="--",
                alpha=0.75, zorder=2)

    # ---- TODOS los puntos coloreados por gradiente PCT_CMAP ----
    # vmin=0/vmax=1 → mapea color_norm directo al cmap morado→rosa
    # edgecolor=TEXT (negro) lw=0.4 → contorno fino que define el punto
    ax.scatter(xs, ys, s=_DOT_SIZE, c=color_norm, cmap=PCT_CMAP,
                vmin=0.0, vmax=1.0, edgecolor=TEXT, lw=0.4, alpha=0.92,
                zorder=4)

    # ---- Caras (FotMob) en el top (sobre el punto, tapandolo) ----
    top = pdf.loc[top_idx]
    for _, r in top.iterrows():
        x, y = float(r[x_metric]), float(r[y_metric])
        pid = int(r["pff_player_id"])
        face_p = _FACES / f"{pid}.png"
        if not face_p.exists():
            continue                                                  # si no hay PNG, salta sin error
        img_arr = np.asarray(Image.open(face_p).convert("RGBA"))
        ab = AnnotationBbox(OffsetImage(img_arr, zoom=_FACE_ZOOM),
                             (x, y), frameon=False, pad=0, zorder=6)
        ax.add_artist(ab)

    # ---- Labels mediana INLINE (Opta-style, encima de las lineas) ----
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    # Label horizontal arriba del axvline (mediana X)
    ax.text(x_med, ylim[1], "  Mediana", color=LEGEND, fontsize=10,
            ha="left", va="top", style="italic", zorder=5,
            bbox=dict(facecolor=BG, edgecolor="none", pad=1, alpha=0.85))
    # Label horizontal a la dcha del axhline (mediana Y)
    ax.text(xlim[1], y_med, "Mediana  ", color=LEGEND, fontsize=10,
            ha="right", va="bottom", style="italic", zorder=5,
            bbox=dict(facecolor=BG, edgecolor="none", pad=1, alpha=0.85))

    # ---- Ejes (Opta-style: label limpio bold, ticks pequenos) ----
    ax.set_xlabel(cfg["x_label"], fontsize=14, fontweight="bold",
                   color=TEXT, labelpad=8)
    ax.set_ylabel(cfg["y_label"], fontsize=14, fontweight="bold",
                   color=TEXT, labelpad=8)
    ax.tick_params(colors=TEXT, labelsize=10, length=3, width=0.7)
    # Quita spines top + right (estilo Opta paper)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # Spines left + bottom en gris claro (suaves)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(0.9)
    # Gridlines suaves en ambos ejes
    ax.grid(True, alpha=0.55, color=GRID, lw=0.5, axis="both")
    ax.set_axisbelow(True)                                            # grid detras de los puntos

    # ---- Footer Opta-style: nota metodologica gris abajo-DCHA ----
    fig.text(0.925, 0.03, cfg["foot"], color=LEGEND, fontsize=12,
             ha="right", style="italic")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # bbox_inches=None pa que el PNG salga EXACTO 16x9 (= 2400x1350 @ 150dpi)
        fig.savefig(save_path, dpi=150, facecolor=BG, bbox_inches=None)
        plt.close(fig)
    return fig


if __name__ == "__main__":
    df = pl.read_parquet(_TABLE)
    for key in ("remontador_cerrojo", "ataque_marcar_encajar"):
        out = f"outputs/viz/scatter_{key}.png"
        opta_scatter(df, config=key, save_path=out)
        print(f"OK -> {out}  ({df.height} jugadores)")
