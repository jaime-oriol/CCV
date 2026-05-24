"""scatter_team - 2 scatters Opta-style por seleccion (landscape, paper-grade).

Misma estetica light Opta que scatter.py (X-Y nativo, landscape 14x8, header
PPCF-style, mediana inline, footer gris). Filtrado al equipo: jugadores del
torneo en grey low-alpha (la "nube" del torneo) + jugadores del equipo en
TEAM_ACCENT con su CARA. Aesthetic + identitario.

2 paneles, 1 PNG cada uno:
  1. Remontador y Cerrojo               (chasing x protecting) — marco de la propuesta
  2. Atacantes clutch                   (atk-GF x atk-Pressure) — donde el shock deja huella

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
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viz.common import BG, GRID, LEGEND, TEXT, draw_header

_TABLE = _SRC.parent / "outputs" / "pcj_table.parquet"
_FACES = _SRC.parent / "outputs" / "assets" / "faces"
_LOGOS = _SRC.parent / "outputs" / "assets" / "logos"

# pff team_name -> iso3 slug
_TEAM_TO_SLUG = {
    "Argentina":"arg","Brazil":"bra","Ecuador":"ecu","Uruguay":"uru","Belgium":"bel",
    "Croatia":"cro","Denmark":"den","England":"eng","France":"fra","Germany":"ger",
    "Netherlands":"ned","Poland":"pol","Portugal":"por","Serbia":"srb","Spain":"esp",
    "Switzerland":"sui","Wales":"wal","Cameroon":"cmr","Ghana":"gha","Morocco":"mar",
    "Senegal":"sen","Tunisia":"tun","Japan":"jpn","South Korea":"kor","Iran":"irn",
    "Qatar":"qat","Saudi Arabia":"ksa","Canada":"can","Mexico":"mex",
    "United States":"usa","Costa Rica":"crc","Australia":"aus",
}

# Color accent primario por seleccion (bright + paper-friendly, alto contraste).
TEAM_ACCENT: dict[str, str] = {
    "Argentina":   "#5eb3e4",   # celeste
    "France":      "#1d4ed8",   # azul rey
    "Croatia":     "#ef4444",   # rojo
    "Morocco":     "#16a34a",   # verde
    "Brazil":      "#fbbf24",   # amarillo
    "England":     "#dc2626",   # rojo
    "Spain":       "#dc2626",   # rojo
    "Portugal":    "#16a34a",   # verde
    "Netherlands": "#f97316",   # oranje
    "Germany":     "#000000",   # negro
    "Belgium":     "#dc2626",   # rojo
    "Japan":       "#1e40af",   # azul samurai
    "Mexico":      "#16a34a",   # verde
    "Uruguay":     "#5eb3e4",   # celeste
    "Senegal":     "#16a34a",   # verde
    "United States":"#1d4ed8",
    "Switzerland": "#dc2626",
    "Poland":      "#dc2626",
    "Denmark":     "#dc2626",
    "Australia":   "#16a34a",
    "Ecuador":     "#fbbf24",
    "Qatar":       "#7c2d12",   # granate
    "Saudi Arabia":"#16a34a",
    "Iran":        "#16a34a",
    "Wales":       "#dc2626",
    "Canada":      "#dc2626",
    "Costa Rica":  "#dc2626",
    "Serbia":      "#dc2626",
    "Cameroon":    "#16a34a",
    "Ghana":       "#dc2626",
    "South Korea": "#dc2626",
    "Tunisia":     "#dc2626",
}

# Configuracion de los 2 paneles. Mismos textos que scatter global por
# concepto -> coherencia global<->team.
_PAIRS = [
    dict(
        x="chasing_clutch_idx", y="protecting_clutch_idx",
        title_concept="Remontador y Cerrojo",
        subtitle_concept="Comparando los dos perfiles del shock emocional",
        x_label="Índice Remontador",
        y_label="Índice Cerrojo",
        foot="*Cambio en el rendimiento individual tras el shock emocional, ajustado por el resto del equipo",
        slug="remontador_cerrojo"),
    dict(
        x="cate_ataque_GOAL_FOR_mean", y="cate_ataque_PRESSURE_mean",
        title_concept="Atacantes clutch",
        subtitle_concept="Comparando ataque tras marcar y ataque bajo presión",
        x_label="Producción ofensiva tras marcar",
        y_label="Producción ofensiva bajo presión",
        foot="*Cambio en la producción ofensiva tras el shock emocional",
        slug="ataque_marcar_presion"),
]

# ---- Estetica de puntos ----
_DOT_BG_SIZE     = 22                  # nube del torneo (rest)
_DOT_BG_COLOR    = "#d4d4d4"           # grey low-alpha
_DOT_BG_ALPHA    = 0.45
_DOT_TEAM_SIZE   = 110                 # puntos del equipo (team accent)
_FACE_ZOOM       = 0.14                # caras del equipo
_MED_LINE_COLOR  = "#888888"
_MED_LINE_LW     = 1.0


def opta_scatter_team(df_full: pl.DataFrame, team: str, pair: dict,
                       save_path: Path) -> Path:
    """1 scatter Opta-style con caras del equipo + fondo de torneo en grey.

    El equipo se ve EN CONTEXTO del torneo: nube grey de los 511, equipo
    coloreado en TEAM_ACCENT + cara. Mediana del torneo dashed inline.
    """
    accent = TEAM_ACCENT.get(team, "#ec4899")    # fallback pink bright
    x_metric, y_metric = pair["x"], pair["y"]

    pdf_full = df_full.to_pandas().reset_index(drop=True)
    xs_full = pdf_full[x_metric].fillna(0.0).to_numpy()
    ys_full = pdf_full[y_metric].fillna(0.0).to_numpy()
    team_mask = (pdf_full["team_name"] == team).to_numpy()

    x_med, y_med = float(np.median(xs_full)), float(np.median(ys_full))

    # ---- Figura landscape MASTER ----
    from viz.common import MASTER_FIGSIZE
    fig = plt.figure(figsize=MASTER_FIGSIZE, facecolor=BG)

    # Header PPCF-style: escudo del EQUIPO (izq) + title + sub + sub2 + JO dcha
    slug = _TEAM_TO_SLUG.get(team)
    team_logo = (_LOGOS / f"{slug}.png") if slug else None
    title = f"{team}  ·  {pair['title_concept']}"
    subtitle = pair["subtitle_concept"]
    subtitle2 = f"Selección de {team}  |  contexto: 511 jugadores del Mundial Qatar 2022"
    draw_header(fig, title=title, subtitle=subtitle, subtitle2=subtitle2,
                escudo_path=team_logo)

    # Plot area: header arriba y=[0.833,1.0]; plot y=[0.12, 0.815]; footer abajo
    ax = fig.add_axes([0.07, 0.12, 0.88, 0.695])
    ax.set_facecolor(BG)

    # ---- Lineas mediana del torneo (dashed grey) ----
    ax.axvline(x_med, color=_MED_LINE_COLOR, lw=_MED_LINE_LW, ls="--",
                alpha=0.75, zorder=2)
    ax.axhline(y_med, color=_MED_LINE_COLOR, lw=_MED_LINE_LW, ls="--",
                alpha=0.75, zorder=2)

    # ---- Nube del torneo (jugadores que NO son del equipo) ----
    ax.scatter(xs_full[~team_mask], ys_full[~team_mask], s=_DOT_BG_SIZE,
                c=_DOT_BG_COLOR, alpha=_DOT_BG_ALPHA, edgecolor="none",
                zorder=3)

    # ---- Jugadores del equipo: dot accent + face ----
    team_pdf = pdf_full[team_mask]
    ax.scatter(team_pdf[x_metric].to_numpy(), team_pdf[y_metric].to_numpy(),
                s=_DOT_TEAM_SIZE, c=accent, edgecolor=TEXT, lw=0.6,
                alpha=0.95, zorder=5)
    for _, r in team_pdf.iterrows():
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
            ha="left", va="top", style="italic", zorder=4,
            bbox=dict(facecolor=BG, edgecolor="none", pad=1, alpha=0.85))
    ax.text(xlim[1], y_med, "mediana del torneo  ", color=LEGEND, fontsize=9,
            ha="right", va="bottom", style="italic", zorder=4,
            bbox=dict(facecolor=BG, edgecolor="none", pad=1, alpha=0.85))

    # ---- Ejes (Opta-style) ----
    ax.set_xlabel(pair["x_label"], fontsize=12, fontweight="bold",
                   color=TEXT, labelpad=8)
    ax.set_ylabel(pair["y_label"], fontsize=12, fontweight="bold",
                   color=TEXT, labelpad=8)
    ax.tick_params(colors=TEXT, labelsize=10, length=3, width=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(0.9)
    ax.grid(True, alpha=0.55, color=GRID, lw=0.5, axis="both")
    ax.set_axisbelow(True)

    # ---- Footer Opta-style: gris pequeño abajo-DCHA al nivel del eje X ----
    fig.text(0.95, 0.06, pair["foot"], color=LEGEND, fontsize=9,
             ha="right", style="italic")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, facecolor=BG, bbox_inches=None)
    plt.close(fig)
    return save_path


# Alias retro-compat
diamond_team = opta_scatter_team


def scatter_team_all(df: pl.DataFrame, team: str, out_dir: Path) -> list[Path]:
    """Genera los 2 scatters Opta-style individuales del equipo."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    team_slug = team.lower().replace(" ", "_")
    paths = []
    for pair in _PAIRS:
        p = out_dir / f"scatter_{team_slug}_{pair['slug']}.png"
        opta_scatter_team(df, team, pair, p)
        paths.append(p)
    return paths


if __name__ == "__main__":
    df = pl.read_parquet(_TABLE)
    team = sys.argv[1] if len(sys.argv) > 1 else "France"
    out_dir = _SRC.parent / "outputs" / "viz"
    paths = scatter_team_all(df, team, out_dir)
    for p in paths:
        print(f"OK -> {p}")
