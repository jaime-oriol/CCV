"""radar_report - Ficha scout: radar geometrico + tabla de percentiles.

Layout combinado: radar (IZQ) + tabla (DCHA), pegados a igual altura sobre BG.
La tabla muestra valor + percentil coloreado (PCT_CMAP morado→fuchsia→rosa)
por las 12 dimensiones del CATE (4 canales x 3 contextos: tras GA, tras GF,
bajo presión), con zebra alterno y leyenda de 5 tramos.

Cabecera de la tabla: escudo de la seleccion (IZQ) + cara FotMob del jugador
(centro-IZQ) + nombre + contexto (equipo + posicion) + logo JO (top DCHA).

Uso:
    python -m src.viz.radar_report "Messi"
    python -m src.viz.radar_report 1531
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from PIL import Image

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viz.common import ATT, BG, DEF, LEGEND, PCT_CMAP, WHITE, _LOGO_PATH
from viz.radar import PCJ_METRICS_12, PCJ_TITLES_12, TEAM_COLORS, _find, player_radar

_TABLE = _SRC.parent / "outputs" / "pcj_table.parquet"                # tabla scout final (M15)
_FACES = _SRC.parent / "outputs" / "assets" / "faces"                 # caras jugadores (FotMob)
_LOGOS = _SRC.parent / "outputs" / "assets" / "logos"                 # escudos selecciones

# team_name -> iso3 (sportlogos). Hardcoded pa no depender de un parquet auxiliar
# — son 32 selecciones WC22, no cambia.
_TEAM_TO_SLUG = {
    "Argentina":"arg","Brazil":"bra","Ecuador":"ecu","Uruguay":"uru","Belgium":"bel",
    "Croatia":"cro","Denmark":"den","England":"eng","France":"fra","Germany":"ger",
    "Netherlands":"ned","Poland":"pol","Portugal":"por","Serbia":"srb","Spain":"esp",
    "Switzerland":"sui","Wales":"wal","Cameroon":"cmr","Ghana":"gha","Morocco":"mar",
    "Senegal":"sen","Tunisia":"tun","Japan":"jpn","South Korea":"kor","Iran":"irn",
    "Qatar":"qat","Saudi Arabia":"ksa","Canada":"can","Mexico":"mex",
    "United States":"usa","Costa Rica":"crc","Australia":"aus",
}

# ============================================================================
# 12 dimensiones de la tabla (4 canales x 3 contextos)
# ============================================================================
# Agrupadas POR CANAL (los 3 contextos de cada canal consecutivos) → mismo orden
# que el radar 12 ejes con reorder=False.
TABLE_METRICS = [
    "cate_ataque_GOAL_AGAINST_mean",  "cate_ataque_GOAL_FOR_mean",  "cate_ataque_PRESSURE_mean",
    "cate_defensa_GOAL_AGAINST_mean", "cate_defensa_GOAL_FOR_mean", "cate_defensa_PRESSURE_mean",
    "cate_offball_GOAL_AGAINST_mean", "cate_offball_GOAL_FOR_mean", "cate_offball_PRESSURE_mean",
    "cate_fisico_GOAL_AGAINST_mean",  "cate_fisico_GOAL_FOR_mean",  "cate_fisico_PRESSURE_mean",
]
TABLE_TITLES = [
    "Ataque · tras GA", "Ataque · tras GF", "Ataque · bajo presión",
    "Defensa · tras GA", "Defensa · tras GF", "Defensa · bajo presión",
    "Off-ball · tras GA", "Off-ball · tras GF", "Off-ball · bajo presión",
    "Físico · tras GA", "Físico · tras GF", "Físico · bajo presión",
]

_NAME_COLOR = WHITE                                                  # color del nombre del jugador (=TEXT=negro en light)


def _short(name: str, max_len: int = 16) -> str:
    """Acorta nombre largo: 'Lionel Messi' → 'L. Messi' si excede max_len."""
    if len(name) <= max_len:
        return name
    parts = name.split()
    return f"{parts[0][0]}. {parts[-1]}" if len(parts) >= 2 else name


def _fmt(v) -> str:
    """Formato de celda de valor. CATEs (<1) con signo + 3 decimales; resto adaptativo."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "0.000"
    if abs(v) < 1:
        return f"{v:+.3f}"                                            # +/-0.123 (CATEs)
    if abs(v) < 10:
        return f"{v:.1f}"
    return f"{int(v)}"


def create_stats_table(df: pl.DataFrame, player_id: int,
                        metrics: list[str] = TABLE_METRICS,
                        metric_titles: list[str] = TABLE_TITLES,
                        footer_text: str = "percentil dentro de su posición",
                        save_path=None):
    """Tabla de stats con valor + percentil coloreado.

    El percentil se calcula DENTRO DEL position_group (mismo puesto). El color
    viene de PCT_CMAP (morado=bajo → rosa=alto).
    """
    pdf = df.to_pandas()
    # Percentil dentro de la posicion pa cada metrica
    for m in metrics:
        pdf[f"{m}_pct"] = pdf.groupby("position_group")[m].rank(pct=True) * 100.0
    p1 = pdf[pdf["pff_player_id"] == player_id].iloc[0]

    node_cmap = PCT_CMAP                                              # cmap del color del percentil
    percentile_norm = Normalize(vmin=0, vmax=100)                     # mapea 0-100 → 0-1 pa el cmap

    # ---- Figura: portrait 7.5 x 8.5 — combinada despues con el radar (cuadrado) ----
    fig = plt.figure(figsize=(7.5, 8.5), facecolor=BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 8.5)                                               # ancho del lienzo data coords
    ax.set_ylim(0, 15)                                                # alto del lienzo data coords
    ax.axis("off")

    # ---- TUNEO de la cabecera (DATA coords: 0..8.5 X, 0..15 Y) ----
    y_start = 13.4                                                    # y del nombre — ↑ sube TODO el header
    text1_x = 3                                                       # x del bloque de texto (nombre+equipo)
    p1_value_x = 4.9                                                  # x de la columna VALOR — ↑ mas a la DCHA
    p1_pct_x = 5.3                                                    # x de la columna PERCENTIL — ↑ mas a la DCHA

    # ---- LOGO JO en FIG COORDS (0..1 X, 0..1 Y) — esquina sup-dcha ----
    figW, figH = fig.get_size_inches()
    logo_w = 0.2                                                      # ancho logo en fraccion fig — ↑ MAS GRANDE
    right_edge = 0.755                                                # borde dcho del logo (fraccion fig) — ↑ mas a la DCHA
    top_edge = 0.265                                                  # bottom-edge tras restar el alto — ↑ logo MAS ABAJO
    if _LOGO_PATH.exists():
        try:
            limg = plt.imread(str(_LOGO_PATH))
            aspect = limg.shape[1] / limg.shape[0]                    # w/h del PNG
            h = logo_w * (figW / figH) / aspect                       # alto proporcional al ancho
            ax_logo = fig.add_axes([right_edge - logo_w, top_edge - h, logo_w, h])
            ax_logo.imshow(limg); ax_logo.axis("off")
        except Exception:
            pass                                                       # logo raro → no peta

    # ---- CARA + ESCUDO en FIG COORDS, a la altura del nombre ----
    # Layout: escudo (IZQ) → cara (centro-IZQ) → texto nombre (centro/DCHA)
    header_fig_y_bottom = 0.760                                       # y inferior de cara y escudo — ↑ SUBEN
    face_w = 0.08                                                     # tamano cara — ↑ MAS GRANDE
    crest_w = 0.07                                                    # tamano escudo (menor que cara) — ↑ MAS GRANDE
    escudo_left_x = 0.18                                              # x del escudo (fraccion fig)
    face_left_x = 0.26                                                # x de la cara (tras el escudo)

    team_name = str(p1.get("team_name", ""))
    slug = _TEAM_TO_SLUG.get(team_name)

    # 1) Escudo a la IZQ. +0.005 centra vertical (face_w > crest_w → ajuste)
    if slug:
        logo_path = _LOGOS / f"{slug}.png"
        if logo_path.exists():
            crest_ax = fig.add_axes([escudo_left_x, header_fig_y_bottom + 0.005,
                                       crest_w, crest_w])
            crest_ax.imshow(Image.open(logo_path)); crest_ax.axis("off")

    # 2) Cara del jugador (FotMob) a la dcha del escudo
    face_path = _FACES / f"{player_id}.png"
    if face_path.exists():
        face_ax = fig.add_axes([face_left_x, header_fig_y_bottom, face_w, face_w])
        face_ax.imshow(Image.open(face_path)); face_ax.axis("off")

    # 3) Nombre + contexto (equipo · posicion) a la dcha de la cara
    name1 = _short(p1.get("player_name", str(player_id)))
    ax.text(text1_x, y_start, name1, fontweight="bold", fontsize=14,
             color=_NAME_COLOR, ha="left", va="center")
    ax.text(text1_x, y_start - 0.5,
             f"{p1.get('team_name', '')}  ·  {p1.get('position_group', '')}",
             fontsize=12, color=WHITE, alpha=0.9, ha="left")

    # ---- Separador horizontal bajo el header ----
    y_line = y_start - 0.7                                            # ↑ separador SUBE (mas cerca del header)
    ax.plot([0.5, 8.5], [y_line, y_line], color="grey", linewidth=0.5, alpha=0.6)

    # ---- Bloque contexto (Minutos / Partidos jugados) ----
    y_context = y_start - 1.2
    ax.text(0.7, y_context, "Minutos jugados", fontsize=10, color=WHITE,
             fontweight="bold")
    ax.text(p1_value_x, y_context, f"{int(p1.get('minutes_played', 0))}",
             fontsize=11, color=WHITE, ha="right")
    y_context -= 0.4                                                  # ↓ fila Partidos pegada a Minutos
    ax.text(0.7, y_context, "Partidos jugados", fontsize=10, color=WHITE,
             fontweight="bold")
    ax.text(p1_value_x, y_context, f"{int(p1.get('n_matches_played', 0))}",
             fontsize=11, color=WHITE, ha="right")

    # Separador bajo el bloque contexto
    y_line = y_context - 0.3
    ax.plot([0.5, 8.5], [y_line, y_line], color="grey", linewidth=0.5, alpha=0.6)

    # ---- Filas de metricas (12 dimensiones) ----
    y_metrics = y_context - 0.7                                       # y de la PRIMERA fila — ↑ SUBE
    # 8 dims → filas anchas (row_height=1.0); 12 dims → mas juntas (0.62)
    grp = 2 if len(metrics) <= 8 else 3                               # zebra agrupa: 2 (pareja GA/GF) o 3 (canal)
    row_height = 1.0 if len(metrics) <= 8 else 0.62
    half = row_height * 0.45                                          # alto de la banda zebra (un poco menor que row_height)
    for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        y_pos = y_metrics - idx * row_height
        # Zebra: bloques de `grp` filas alternados con stripe gris claro
        if (idx // grp) % 2 == 0:
            ax.add_patch(Rectangle((0.5, y_pos - half), 8.0, 2 * half,
                                    facecolor="#f5f5f5", alpha=1.0))   # gris claro visible sobre BG blanco
        ax.text(0.7, y_pos, title, fontsize=10, color=WHITE, fontweight="bold",
                 va="center")
        pct = p1.get(f"{metric}_pct", 0)
        pct = 0 if pct is None or np.isnan(pct) else float(pct)
        # Columna VALOR (formato adaptativo via _fmt)
        ax.text(p1_value_x, y_pos, _fmt(p1.get(metric)), fontsize=11, color=WHITE,
                 ha="right", va="center")
        # Columna PERCENTIL coloreada segun PCT_CMAP (morado=bajo, rosa=alto)
        ax.text(p1_pct_x, y_pos, f"{int(pct)}", fontsize=10,
                 color=node_cmap(percentile_norm(pct)), ha="left", va="center")

    # ---- Footer (disclaimer gris, italic, sin bold) ----
    footer_y = y_metrics - len(metrics) * row_height
    ax.text(0.7, footer_y, f"*{footer_text}", fontsize=9, color=LEGEND,
             ha="left", style="italic", va="center")

    # ---- Leyenda: 5 tramos de percentil (5 lineas + etiquetas debajo) ----
    legend_y = footer_y - 0.8                                         # ↑ leyenda SUBE (cerca del footer)
    intervals = [(0, 20), (21, 40), (41, 60), (61, 80), (81, 100)]
    spacing = 0.8                                                     # separacion horizontal entre tramos — ↑ MAS SEPARADOS
    for i, (lo, hi) in enumerate(intervals):
        x_pos = 1.0 + i * spacing                                     # ↑ leyenda completa MAS a la DCHA
        ax.plot([x_pos - 0.25, x_pos + 0.25], [legend_y, legend_y],
                 color=node_cmap(percentile_norm(i * 25)), linewidth=3,
                 solid_capstyle="round")
        ax.text(x_pos, legend_y - 0.4, f"{lo}-{hi}", fontsize=9, color=WHITE,
                 ha="center")

    # ---- Flecha BAJO → ALTO debajo de la leyenda ----
    arrow_y = legend_y - 0.8                                          # ↑ flecha SUBE (cerca de la leyenda)
    ax.annotate("", xy=(4.0, arrow_y), xytext=(1.2, arrow_y),
                 arrowprops=dict(arrowstyle="->", color=WHITE, lw=1))
    ax.text(1.1, arrow_y, "BAJO", fontsize=9, color=WHITE, ha="right",
             va="center")
    ax.text(4.1, arrow_y, "ALTO", fontsize=9, color=WHITE, ha="left",
             va="center")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
    return fig


def _combine(radar_path: Path, table_path: Path, out_path: Path) -> None:
    """Pega radar (IZQ) + tabla (DCHA) a igual altura sobre lienzo BG."""
    radar = Image.open(radar_path).convert("RGB")
    table = Image.open(table_path).convert("RGB")
    H = max(radar.height, table.height)                               # altura comun = max de los 2
    rad = radar.resize((int(radar.width * H / radar.height), H), Image.LANCZOS)
    tab = table.resize((int(table.width * H / table.height), H), Image.LANCZOS)
    canvas = Image.new("RGB", (rad.width + tab.width, H), color=BG)
    canvas.paste(rad, (0, 0))
    canvas.paste(tab, (rad.width, 0))
    canvas.save(out_path, dpi=(300, 300))


def player_radar_report(df: pl.DataFrame, player_id: int, save_path=None) -> Path:
    """Reporte completo: radar geometrico (IZQ) + tabla percentiles (DCHA)."""
    tmp = Path(tempfile.gettempdir())
    radar_p, table_p = tmp / f"_rad_{player_id}.png", tmp / f"_tab_{player_id}.png"

    # Radar 12 ejes (reorder=False → wedges por canal, mismo orden que la tabla).
    # Colores de la seleccion del jugador (fallback ATT/DEF). Sin titulo ni logo:
    # la identidad (nombre + escudo + JO) vive en la tabla.
    team = str(df.filter(pl.col("pff_player_id") == player_id)["team_name"][0])
    team_colors = TEAM_COLORS.get(team, (ATT, DEF))
    player_radar(df, player_id, PCJ_METRICS_12, PCJ_TITLES_12, colors=team_colors,
                  title="", subtitle="", logo=False, reorder=False, save_path=radar_p)
    create_stats_table(df, player_id, save_path=table_p)

    if save_path is None:
        save_path = _SRC.parent / "outputs" / "viz" / f"radar_report_{player_id}.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    _combine(radar_p, table_p, save_path)
    radar_p.unlink(missing_ok=True)
    table_p.unlink(missing_ok=True)
    return save_path


if __name__ == "__main__":
    df = pl.read_parquet(_TABLE)
    pid = _find(df, sys.argv[1] if len(sys.argv) > 1 else "Messi")
    out = player_radar_report(df, pid)
    print(f"OK -> {out}")
