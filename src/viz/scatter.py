"""scatter - Scatters globales del PCJ estilo Opta (landscape, paper-grade).

Scatter X-Y nativo (NO rotado). Layout landscape 14x8 que casa perfecto con
ancho de doc. Header PPCF-style (escudo izq | titulo+sub LEFT | JO dcha).
Spotlight Opta: jugadores top combinado + destacados por eje en color bright
con FACE + NOMBRE; resto del torneo en grey low-alpha (la "nube" del torneo).
Mediana del torneo dashed con label INLINE en la propia linea.

2 conceptos (config-driven via SCATTERS):
  1. remontador_cerrojo    : Remontador x Cerrojo (los 2 perfiles de la propuesta).
  2. ataque_marcar_presion : Ataque post-GF x Ataque pre-elim (los 2 unicos ejes
                             con spread real; donde el shock deja huella medible).

Identidad LIGHT OPTA: BG blanco, textos negros, leyenda gris medio, paleta
bright vibrante (PCT_CMAP indigo->violeta->rosa->naranja). Caras + nombres
junto al punto. Sin carteles direccionales (los ejes hablan).

Uso:
    python -m src.viz.scatter
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viz.common import BG, GRID, LEGEND, PCT_CMAP, TEXT, draw_header

_TABLE = _SRC.parent / "outputs" / "pcj_table.parquet"
_LOGOS = _SRC.parent / "outputs" / "assets" / "logos"
_FACES = _SRC.parent / "outputs" / "assets" / "faces"
_WC22_LOGO = _LOGOS / "wc22.png"

# Configuracion de los 2 scatters. Pattern Opta:
#   title = noun phrase corto que dice QUIEN se esta mapeando
#   subtitle = "Comparando X y Y | contexto" -> los ejes no necesitan label largo
#   x_label / y_label = concepto puro, breve
SCATTERS: dict[str, dict] = {
    # Los 2 perfiles agregados de la propuesta. Ejes apretados (la elite es
    # homogenea en los agregados), pero es el marco conceptual del TFM.
    "remontador_cerrojo": dict(
        x="chasing_clutch_idx", y="protecting_clutch_idx",
        title="Remontador y Cerrojo",
        subtitle="Comparando los dos perfiles del shock emocional",
        subtitle2="Mundial Qatar 2022  |  511 jugadores",
        x_label="Índice Remontador",
        y_label="Índice Cerrojo",
        foot="*Cambio en el rendimiento individual tras el shock emocional, ajustado por el resto del equipo"),
    # Los 2 unicos ejes con estructura real: atk-GF (rango 0.87, 22 Sig) y
    # atk-Pressure (0.59). El mapa de produccion ofensiva tras el shock.
    "ataque_marcar_presion": dict(
        x="cate_ataque_GOAL_FOR_mean", y="cate_ataque_PRESSURE_mean",
        title="Atacantes clutch",
        subtitle="Comparando ataque tras marcar y ataque bajo presión",
        subtitle2="Mundial Qatar 2022  |  511 jugadores",
        x_label="Producción ofensiva tras marcar",
        y_label="Producción ofensiva bajo presión",
        foot="*Cambio en la producción ofensiva tras el shock emocional"),
}

# ---- Tamaños / estetica de puntos ----
_DOT_SIZE        = 70          # todos los puntos (gradiente cmap)
_FACE_ZOOM       = 0.16        # cara sobre el punto (top selection)
_MED_LINE_COLOR  = "#888888"
_MED_LINE_LW     = 1.0


def opta_scatter(df: pl.DataFrame, config: str | dict = "remontador_cerrojo",
                 save_path=None):
    """Scatter Opta-style landscape. `config` = clave de SCATTERS o dict."""
    cfg = SCATTERS[config] if isinstance(config, str) else config
    x_metric, y_metric = cfg["x"], cfg["y"]

    pdf = df.to_pandas().reset_index(drop=True)
    xs = pdf[x_metric].fillna(0.0).to_numpy()
    ys = pdf[y_metric].fillna(0.0).to_numpy()

    # Percentiles al vuelo -> color de TODOS los puntos (gradiente cmap)
    px = pdf[x_metric].rank(pct=True).to_numpy() * 100.0
    py = pdf[y_metric].rank(pct=True).to_numpy() * 100.0
    pdf = pdf.assign(_px=px, _py=py)
    color_norm = (px + py) / 200.0      # 0..1 para PCT_CMAP

    # Top para caras = (a) top-10 combinado: extremos en AMBOS ejes a la vez ...
    combo = pdf[(pdf["_px"] >= 81) & (pdf["_py"] >= 81)]
    if len(combo) < 5:
        combo = pdf[(pdf["_px"] >= 75) & (pdf["_py"] >= 75)]
    combo = combo.assign(_tot=combo["_px"] + combo["_py"]).nlargest(10, "_tot")
    # ... + (b) los 2 mejores de CADA eje individual fuera del top-10.
    rest = pdf.loc[pdf.index.difference(combo.index)]
    spec_x = rest.nlargest(2, "_px").index
    spec_y = rest.nlargest(2, "_py").index
    top_idx = combo.index.union(spec_x).union(spec_y)

    # Mediana del torneo (sobre valores raw, no normalizados)
    x_med, y_med = float(np.median(xs)), float(np.median(ys))

    # ---- Figura landscape MASTER (Opta-style) ----
    from viz.common import MASTER_FIGSIZE
    fig = plt.figure(figsize=MASTER_FIGSIZE, facecolor=BG)

    # Header PPCF-style: escudo torneo (izq) + titulo+sub+sub2 LEFT + JO dcha
    draw_header(fig, title=cfg["title"], subtitle=cfg["subtitle"],
                subtitle2=cfg.get("subtitle2"),
                escudo_path=_WC22_LOGO)

    # Plot area: header arriba y=[0.833,1.0]; plot y=[0.12, 0.815]; footer abajo
    ax = fig.add_axes([0.07, 0.12, 0.88, 0.695])
    ax.set_facecolor(BG)

    # ---- Lineas mediana del torneo (dashed grey) ----
    ax.axvline(x_med, color=_MED_LINE_COLOR, lw=_MED_LINE_LW, ls="--",
                alpha=0.75, zorder=2)
    ax.axhline(y_med, color=_MED_LINE_COLOR, lw=_MED_LINE_LW, ls="--",
                alpha=0.75, zorder=2)

    # ---- TODOS los puntos con gradiente PCT_CMAP (morado->fuchsia->pink) ----
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
            continue
        img_arr = np.asarray(Image.open(face_p).convert("RGBA"))
        ab = AnnotationBbox(OffsetImage(img_arr, zoom=_FACE_ZOOM),
                             (x, y), frameon=False, pad=0, zorder=6)
        ax.add_artist(ab)

    # ---- Labels mediana INLINE (Opta-style) ----
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.text(x_med, ylim[1], "  mediana del torneo", color=LEGEND, fontsize=9,
            ha="left", va="top", style="italic", zorder=5,
            bbox=dict(facecolor=BG, edgecolor="none", pad=1, alpha=0.85))
    ax.text(xlim[1], y_med, "mediana del torneo  ", color=LEGEND, fontsize=9,
            ha="right", va="bottom", style="italic", zorder=5,
            bbox=dict(facecolor=BG, edgecolor="none", pad=1, alpha=0.85))

    # ---- Ejes (Opta-style: label limpio, ticks pequeños) ----
    ax.set_xlabel(cfg["x_label"], fontsize=12, fontweight="bold",
                   color=TEXT, labelpad=8)
    ax.set_ylabel(cfg["y_label"], fontsize=12, fontweight="bold",
                   color=TEXT, labelpad=8)
    ax.tick_params(colors=TEXT, labelsize=10, length=3, width=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(0.9)
    ax.grid(True, alpha=0.55, color=GRID, lw=0.5, axis="both")
    ax.set_axisbelow(True)

    # ---- Footer Opta-style: gris pequeño abajo-DCHA al nivel del eje X ----
    fig.text(0.95, 0.06, cfg["foot"], color=LEGEND, fontsize=9,
             ha="right", style="italic")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, facecolor=BG, bbox_inches=None)
        plt.close(fig)
    return fig


# Alias para retro-compat con __main__
diamond_scatter = opta_scatter


if __name__ == "__main__":
    df = pl.read_parquet(_TABLE)
    for key in ("remontador_cerrojo", "ataque_marcar_presion"):
        out = f"outputs/viz/scatter_{key}.png"
        opta_scatter(df, config=key, save_path=out)
        print(f"OK -> {out}  ({df.height} jugadores)")
