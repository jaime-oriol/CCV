"""radar - Radar geometrico del Perfil Clutch del Jugador.

8 ejes (4 canales x 2 contextos post-GA / post-GF) sobre circulos
concentricos, rangos por percentil 1-99 del dataset, poligono con anillos
de color alternos clipeados a su forma.

Uso:
    python -m src.viz.radar 1234        # por pff_player_id
    python -m src.viz.radar "Messi"     # por substring del nombre
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viz.common import ATT, BG, DEF, WHITE, add_logo

_TABLE = _SRC.parent / "outputs" / "pcj_table.parquet"

# Orden de los 8 ejes: bloque post-GA (4 canales) seguido del bloque post-GF
# (mismos 4 canales). Con la rotacion interna (primer eje fijo + resto
# invertido) el eje i queda enfrentado al i+4 -> cada canal sale con su GA y
# su GF a 180 grados (Ataque-GA arriba <-> Ataque-GF abajo, etc).
PCJ_METRICS = [
    "cate_ataque_GOAL_AGAINST_mean",  "cate_defensa_GOAL_AGAINST_mean",
    "cate_offball_GOAL_AGAINST_mean", "cate_fisico_GOAL_AGAINST_mean",
    "cate_ataque_GOAL_FOR_mean",      "cate_defensa_GOAL_FOR_mean",
    "cate_offball_GOAL_FOR_mean",     "cate_fisico_GOAL_FOR_mean",
]
PCJ_TITLES = [
    "Ataque\npost-GA", "Defensa\npost-GA", "Off-ball\npost-GA", "Fisico\npost-GA",
    "Ataque\npost-GF", "Defensa\npost-GF", "Off-ball\npost-GF", "Fisico\npost-GF",
]

# Variante 12 ejes: 4 canales x 3 contextos (post-GA, post-GF, pre-elim).
# Agrupados POR CANAL (los 3 contextos de cada canal juntos), MISMO orden que la
# tabla del report -> se usa con reorder=False (horario desde arriba): wedge
# Ataque, wedge Defensa, wedge Off-ball, wedge Fisico.
PCJ_METRICS_12 = [
    "cate_ataque_GOAL_AGAINST_mean",  "cate_ataque_GOAL_FOR_mean",  "cate_ataque_PRESSURE_mean",
    "cate_defensa_GOAL_AGAINST_mean", "cate_defensa_GOAL_FOR_mean", "cate_defensa_PRESSURE_mean",
    "cate_offball_GOAL_AGAINST_mean", "cate_offball_GOAL_FOR_mean", "cate_offball_PRESSURE_mean",
    "cate_fisico_GOAL_AGAINST_mean",  "cate_fisico_GOAL_FOR_mean",  "cate_fisico_PRESSURE_mean",
]
PCJ_TITLES_12 = [
    "Ataque\npost-GA", "Ataque\npost-GF", "Ataque\npre-elim",
    "Defensa\npost-GA", "Defensa\npost-GF", "Defensa\npre-elim",
    "Off-ball\npost-GA", "Off-ball\npost-GF", "Off-ball\npre-elim",
    "Fisico\npost-GA", "Fisico\npost-GF", "Fisico\npre-elim",
]

# Colores de seleccion (primario, secundario) para el radar: pares vivos que
# contrastan entre si y brillan sobre el fondo oscuro. Fallback a (ATT, DEF).
TEAM_COLORS: dict[str, tuple[str, str]] = {
    "Argentina":   ("#6CACE4", "#FFFFFF"),   # celeste + blanco
    "France":      ("#3B7DD8", "#E1314A"),   # azul + rojo
    "Croatia":     ("#F1414F", "#FFFFFF"),   # rojo + blanco (damero)
    "Morocco":     ("#E63946", "#19A35A"),   # rojo + verde
    "Brazil":      ("#FFDF00", "#1CA64C"),   # amarillo + verde
    "England":     ("#FFFFFF", "#E8344E"),   # blanco + rojo
    "Spain":       ("#E0314B", "#FFC83D"),   # rojo + amarillo
    "Portugal":    ("#E8344E", "#1CA64C"),   # rojo + verde
    "Netherlands": ("#FF7A1A", "#FFFFFF"),   # naranja + blanco
    "Germany":     ("#F2F2F2", "#FFD23B"),   # blanco + dorado
    "Belgium":     ("#E83B4E", "#FFD23B"),   # rojo + dorado
    "Japan":       ("#3A6FE0", "#FFFFFF"),   # azul + blanco
    "Mexico":      ("#1CA64C", "#E8344E"),   # verde + rojo
    "Uruguay":     ("#5CBFEB", "#FFFFFF"),   # celeste + blanco
    "Senegal":     ("#1CA64C", "#E8344E"),   # verde + rojo
    "USA":         ("#3A6FE0", "#E8344E"),   # azul + rojo
    "Switzerland": ("#F1414F", "#FFFFFF"),   # rojo + blanco
    "Poland":      ("#FFFFFF", "#E8344E"),   # blanco + rojo
    "Denmark":     ("#F1414F", "#FFFFFF"),   # rojo + blanco
    "Australia":   ("#FFD23B", "#1CA64C"),   # oro + verde
    "Ecuador":     ("#FFD23B", "#3A6FE0"),   # amarillo + azul
    "Qatar":       ("#9E2B4E", "#FFFFFF"),   # granate + blanco
    "Saudi Arabia":("#1CA64C", "#FFFFFF"),   # verde + blanco
    "Iran":        ("#1CA64C", "#E8344E"),   # verde + rojo
    "Wales":       ("#E8344E", "#1CA64C"),   # rojo + verde
    "Canada":      ("#F1414F", "#FFFFFF"),   # rojo + blanco
    "Costa Rica":  ("#E8344E", "#3A6FE0"),   # rojo + azul
    "Serbia":      ("#E0314B", "#3A6FE0"),   # rojo + azul
    "Cameroon":    ("#1CA64C", "#E8344E"),   # verde + rojo
    "Ghana":       ("#E8344E", "#FFD23B"),   # rojo + dorado
    "South Korea": ("#E8344E", "#3A6FE0"),   # rojo + azul
    "Tunisia":     ("#E8344E", "#FFFFFF"),   # rojo + blanco
}


def player_radar(df: pl.DataFrame, player_id: int,
                  metrics: list[str] = PCJ_METRICS,
                  metric_titles: list[str] = PCJ_TITLES,
                  colors: tuple[str, str] = (ATT, DEF),
                  title: str = "", subtitle: str = "",
                  logo: bool = True, reorder: bool = True, save_path=None):
    """Radar geometrico de 1 jugador.

    Rangos por percentil P1-P99 del `df` completo (no del propio jugador).
    Anillos de color alternos (ATT/DEF) clipeados al poligono.

    reorder=True (8 ejes): primer eje fijo arriba + resto invertido (rotacion
    footballdecoded; cada canal GA queda enfrentado a su GF). reorder=False
    (12 ejes): se respeta el orden dado, horario desde arriba (arcos por contexto).
    """
    pdf = df.to_pandas()
    row = pdf[pdf["pff_player_id"] == player_id].iloc[0]

    if reorder:
        reordered = [metrics[0]] + list(reversed(metrics[1:]))
        reordered_titles = [metric_titles[0]] + list(reversed(metric_titles[1:]))
    else:
        reordered, reordered_titles = list(metrics), list(metric_titles)

    # Rangos del dataset por eje (P1-P99 evita outliers extremos)
    ranges = []
    for m in reordered:
        d = pdf[m].dropna()
        ranges.append((np.percentile(d, 1), np.percentile(d, 99)))

    # ---- Figura + ejes geometricos ----
    fig, ax = plt.subplots(figsize=(9, 10), facecolor=BG)      # ↑ figsize -> radar MAS GRANDE
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.set(xlim=(-22, 22), ylim=(-23, 25))                      # margen extra arriba para titulo

    values = [row[m] for m in reordered]

    # Circulos concentricos del radar. radius_circles define los anillos:
    # [3, ..., 20.5]. El primero (3) es el centro, el ultimo el borde.
    radius_circles = [3, 5.5, 8, 10.5, 13, 15.5, 18, 20.5]
    for i, rad in enumerate(radius_circles):
        if i == 0:
            continue                                            # el centro no se dibuja
        if i == len(radius_circles) - 1:
            color, lw, alpha = "white", 1.2, 1.0               # borde exterior nitido
        else:
            color, lw, alpha = "grey", 1, 0.4
        ax.add_patch(plt.Circle((0, 0), rad, fc="none", ec=color, lw=lw, alpha=alpha))

    n = len(reordered)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)        # 8 ejes equiespaciados

    # ---- Etiquetas de eje (Ataque post-GA, Off-ball post-GA, ...) ----
    lbl_fs = 10 if n <= 8 else 8                                  # 12 ejes -> fuente menor (no se pisan)
    for angle, t in zip(angles, reordered_titles):
        x, y = 21.5 * np.sin(angle), 21.5 * np.cos(angle)        # 21.5 = radio etiqueta (>20.5 borde)
        rot = -np.rad2deg(angle)
        if y < 0:
            rot += 180                                           # mantiene texto legible abajo
        ax.text(x, y, t, rotation=rot, ha="center", va="center",
                fontsize=lbl_fs, fontweight="bold", color=WHITE, family="DejaVu Sans")

    # Lineas radiales (separadores entre ejes)
    for angle in angles:
        ax.plot([0, 20.5 * np.sin(angle)], [0, 20.5 * np.cos(angle)],
                color="grey", linewidth=0.5, alpha=0.4)

    # ---- Etiquetas de valor en cada anillo (formato dependiente de magnitud) ----
    for rad in [4.25, 6.75, 9.25, 11.75, 14.25, 16.75, 19.25]:    # entre cada par de anillos
        for angle, (mn, mx) in zip(angles, ranges):
            rad_norm = (rad - 3) / (20.5 - 3)                    # 0..1 a lo largo del radio
            val = mn if mx == mn else mn + rad_norm * (mx - mn)
            # Formato adaptativo: 3 decimales para magnitudes <0.01 (CATEs),
            # 2 para <1, 1 para <10, entero el resto.
            if abs(val) < 0.01:
                label = f"{val:.3f}"
            elif abs(val) < 1:
                label = f"{val:.2f}"
            elif abs(val) < 10:
                label = f"{val:.1f}"
            else:
                label = f"{int(val)}"
            ax.text(rad * np.sin(angle), rad * np.cos(angle), label,
                    ha="center", va="center", size=7, color=WHITE,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=BG,
                              edgecolor="none", alpha=0.9), family="DejaVu Sans")

    # ---- Poligono del jugador (coords polares -> cartesianas) ----
    vertices = []
    for value, (mn, mx) in zip(values, ranges):
        if mx == mn:
            nv = 11.75                                           # rango cero -> radio medio
        else:
            nv = 3 + (value - mn) / (mx - mn) * 17.5             # mapea [mn,mx] a [3, 20.5]
        nv = max(3, min(20.5, nv))                               # clip a la corona dibujable
        idx = len(vertices)
        vertices.append([nv * np.sin(angles[idx]), nv * np.cos(angles[idx])])

    # Anillos de color alternos (ATT/DEF) clipeados al poligono del jugador.
    # Da el efecto visual "telarana coloreada" del radar de footballdecoded.
    poly = Polygon(vertices, fc="none", alpha=1.0, zorder=1)
    ax.add_patch(poly)
    central = plt.Circle((0, 0), radius_circles[0], fc=colors[0], ec="none",
                          alpha=0.45, zorder=2)
    central.set_clip_path(poly)
    ax.add_patch(central)
    theta = np.linspace(0, 2 * np.pi, 100)
    for i in range(len(radius_circles) - 1):
        ri, ro = radius_circles[i], radius_circles[i + 1]
        cidx = (i + 1) % 2                                       # alterna 0 y 1 -> ATT/DEF
        ring = list(zip(ro * np.cos(theta), ro * np.sin(theta))) + \
                list(zip(ri * np.cos(theta[::-1]), ri * np.sin(theta[::-1])))
        rp = Polygon(ring, fc=colors[cidx], alpha=0.45, zorder=2)
        rp.set_clip_path(poly)
        ax.add_patch(rp)
    closed = vertices + [vertices[0]]
    ax.plot([v[0] for v in closed], [v[1] for v in closed],
            color=colors[0], linewidth=3, zorder=10)              # contorno grueso del poligono

    ax.axis("off")

    # ---- Header opcional (title + subtitle + logo JO) ----
    # Cuando radar_report combina radar + tabla, llama con title="" para que la
    # identidad viva en la tabla. En uso standalone si que se ven.
    if title:
        fig.text(0.5, 0.965, title, ha="center", va="top", color=WHITE,
                  fontsize=17, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="top", color=WHITE,
                  fontsize=11)
    if logo:
        add_logo(fig, width_frac=0.13)                            # ↑ logo MAS GRANDE

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
    return fig


def _find(df: pl.DataFrame, query: str):
    """Resuelve un jugador por id (entero) o substring del nombre."""
    try:
        return int(query)
    except ValueError:
        sub = df.filter(pl.col("player_name").str.contains(query, literal=False))
        if sub.height == 0:
            raise SystemExit(f"jugador no encontrado: {query!r}")
        return int(sub["pff_player_id"][0])


if __name__ == "__main__":
    df = pl.read_parquet(_TABLE)
    query = sys.argv[1] if len(sys.argv) > 1 else "Messi"
    pid = _find(df, query)
    r = df.filter(pl.col("pff_player_id") == pid).row(0, named=True)
    out = f"outputs/viz/radar_{pid}.png"
    player_radar(
        df, pid,
        title=f"{r['player_name']}  ·  Perfil Clutch del Jugador",
        subtitle=f"{r['team_name']}  ·  {r['position_group']}  ·  "
                  f"{int(r['minutes_played'])} min  ·  Mundial Qatar 2022",
        save_path=out,
    )
    print(f"OK -> {out}")
