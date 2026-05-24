"""radar - Radar geometrico del xCV (Expected Clutch Value).

Variantes:
  - 8 ejes  (XCV_METRICS / XCV_TITLES):    4 canales x 2 contextos (tras GA, tras GF)
  - 12 ejes (XCV_METRICS_12 / _TITLES_12): 4 canales x 3 contextos (tras GA, tras GF, bajo presión)

Convencion polar: x=sin(angle), y=cos(angle) — angle=0 apunta al NORTE (arriba)
y crece en sentido horario. Cada eje se dibuja desde el centro hasta el anillo
exterior (radio 20.5). Las etiquetas viven a radio 21.5 (just fuera del borde).

Rangos por percentil P1-P99 del df completo (no del propio jugador) → evita
outliers extremos comprimiendo la escala visual. Anillos de color alternos
(colors[0]/colors[1]) clipeados al poligono del jugador → efecto telarana
coloreada estilo footballdecoded.

reorder=True (8 ejes): primer eje fijo arriba + resto invertido → cada canal GA
queda enfrentado a su GF (180 grados). reorder=False (12 ejes): orden literal,
horario desde arriba → wedges por canal (Ataque / Defensa / Off-ball / Físico).

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

from viz.common import ATT, BG, DEF, TOURNAMENT_ES, WHITE, add_logo, team_es

_TABLE = _SRC.parent / "outputs" / "xcv_table.parquet"

# ----------------------------------------------------------------------------
# 8 ejes: 4 canales x 2 contextos (tras GA + tras GF), bloque GA y bloque GF
# ----------------------------------------------------------------------------
# Orden: GA primero (4) + GF (4). Con reorder=True el eje i queda enfrentado al
# i+4 → cada canal sale con su GA arriba y su GF abajo (Ataque-GA <-> Ataque-GF).
XCV_METRICS = [
    "cate_ataque_GOAL_AGAINST_mean",  "cate_defensa_GOAL_AGAINST_mean",
    "cate_offball_GOAL_AGAINST_mean", "cate_fisico_GOAL_AGAINST_mean",
    "cate_ataque_GOAL_FOR_mean",      "cate_defensa_GOAL_FOR_mean",
    "cate_offball_GOAL_FOR_mean",     "cate_fisico_GOAL_FOR_mean",
]
XCV_TITLES = [
    "Ataque\ntras GA", "Defensa\ntras GA", "Off-ball\ntras GA", "Físico\ntras GA",
    "Ataque\ntras GF", "Defensa\ntras GF", "Off-ball\ntras GF", "Físico\ntras GF",
]

# ----------------------------------------------------------------------------
# 12 ejes: 4 canales x 3 contextos (tras GA + tras GF + bajo presion)
# ----------------------------------------------------------------------------
# Agrupados POR CANAL (los 3 contextos de cada canal consecutivos) — mismo orden
# que la tabla de radar_report. Se usa con reorder=False (horario desde arriba):
# wedge Ataque, wedge Defensa, wedge Off-ball, wedge Físico.
XCV_METRICS_12 = [
    "cate_ataque_GOAL_AGAINST_mean",  "cate_ataque_GOAL_FOR_mean",  "cate_ataque_PRESSURE_mean",
    "cate_defensa_GOAL_AGAINST_mean", "cate_defensa_GOAL_FOR_mean", "cate_defensa_PRESSURE_mean",
    "cate_offball_GOAL_AGAINST_mean", "cate_offball_GOAL_FOR_mean", "cate_offball_PRESSURE_mean",
    "cate_fisico_GOAL_AGAINST_mean",  "cate_fisico_GOAL_FOR_mean",  "cate_fisico_PRESSURE_mean",
]
XCV_TITLES_12 = [
    "Ataque\ntras GA", "Ataque\ntras GF", "Ataque\nbajo presión",
    "Defensa\ntras GA", "Defensa\ntras GF", "Defensa\nbajo presión",
    "Off-ball\ntras GA", "Off-ball\ntras GF", "Off-ball\nbajo presión",
    "Físico\ntras GA", "Físico\ntras GF", "Físico\nbajo presión",
]

# ----------------------------------------------------------------------------
# Colores de seleccion (primario, secundario)
# ----------------------------------------------------------------------------
# Para el fill alterno de los anillos del radar. Cada par = 2 colores vivos que
# contrastan entre si Y se ven bien sobre BG BLANCO a alpha=0.45.
# Regla: evitar blanco (#FFF) y amarillos muy claros — quedan invisibles a
# alpha=0.45. Fallback a (ATT, DEF) si la seleccion no esta aqui.
TEAM_COLORS: dict[str, tuple[str, str]] = {
    "Argentina":    ("#5BAEE0", "#003087"),   # celeste + azul navy
    "France":       ("#3B7DD8", "#E1314A"),   # bleu + rouge
    "Croatia":      ("#EE1D23", "#0033A0"),   # rojo damero + azul croata
    "Morocco":      ("#E63946", "#19A35A"),   # rojo + verde
    "Brazil":       ("#D4B000", "#1CA64C"),   # dorado oscuro + verde
    "England":      ("#CC2222", "#00205B"),   # rojo St George + azul navy
    "Spain":        ("#E0314B", "#D4A017"),   # rojo + dorado oscuro
    "Portugal":     ("#E8344E", "#1CA64C"),   # rojo + verde
    "Netherlands":  ("#FF6B00", "#21468B"),   # oranje + azul real
    "Germany":      ("#1a1a1a", "#D4AF00"),   # negro + dorado
    "Belgium":      ("#E83B4E", "#D4A017"),   # rojo + dorado oscuro
    "Japan":        ("#2B5CBF", "#BC002D"),   # azul + rojo hinomaru
    "Mexico":       ("#1CA64C", "#E8344E"),   # verde + rojo
    "Uruguay":      ("#5BAEE0", "#003087"),   # celeste + azul navy
    "Senegal":      ("#1CA64C", "#E8344E"),   # verde + rojo
    "United States":("#1A3A8A", "#B22234"),   # azul + rojo
    "Switzerland":  ("#D52B1E", "#5a5a5a"),   # rojo suizo + gris oscuro
    "Poland":       ("#DC143C", "#5a5a5a"),   # rojo polaco + gris oscuro
    "Denmark":      ("#C60C30", "#1a1a1a"),   # rojo danes + negro
    "Australia":    ("#D4A017", "#1CA64C"),   # dorado + verde
    "Ecuador":      ("#D4A017", "#3A6FE0"),   # dorado + azul
    "Qatar":        ("#8B1839", "#3a3a3a"),   # granate + gris oscuro
    "Saudi Arabia": ("#007A3D", "#003082"),   # verde + azul navy
    "Iran":         ("#1CA64C", "#E8344E"),   # verde + rojo
    "Wales":        ("#E8344E", "#1CA64C"),   # rojo + verde
    "Canada":       ("#D52B1E", "#5a5a5a"),   # rojo + gris oscuro
    "Costa Rica":   ("#E8344E", "#3A6FE0"),   # rojo + azul
    "Serbia":       ("#E0314B", "#3A6FE0"),   # rojo + azul
    "Cameroon":     ("#1CA64C", "#E8344E"),   # verde + rojo
    "Ghana":        ("#E8344E", "#D4A017"),   # rojo + dorado oscuro
    "South Korea":  ("#E8344E", "#3A6FE0"),   # rojo + azul
    "Tunisia":     ("#CC2222", "#5a5a5a"),   # rojo + gris oscuro
}


def player_radar(df: pl.DataFrame, player_id: int,
                  metrics: list[str] = XCV_METRICS,
                  metric_titles: list[str] = XCV_TITLES,
                  colors: tuple[str, str] = (ATT, DEF),
                  title: str = "", subtitle: str = "",
                  logo: bool = True, reorder: bool = True, save_path=None):
    """Radar geometrico de 1 jugador.

      df:            tabla completa (pa calcular percentiles P1-P99 por eje)
      player_id:     pff_player_id del jugador a pintar
      metrics:       lista de columnas (8 o 12)
      metric_titles: lista de etiquetas (multilinea con \\n permitido)
      colors:        (primario, secundario) — fill alterno de los anillos + contorno
      title/subtitle: cabecera centrada (vacios cuando se combina con tabla en radar_report)
      logo:          estampa logo JO esquina abajo-DCHA (legacy add_logo)
      reorder=True:  primer eje fijo + resto invertido (8 ejes) → cada canal GA vs GF a 180°
      reorder=False: orden literal horario desde arriba (12 ejes) → wedges por canal
    """
    pdf = df.to_pandas()
    row = pdf[pdf["pff_player_id"] == player_id].iloc[0]

    # ---- Orden de ejes ----
    if reorder:
        reordered = [metrics[0]] + list(reversed(metrics[1:]))        # [m0, m7, m6, m5, m4, m3, m2, m1]
        reordered_titles = [metric_titles[0]] + list(reversed(metric_titles[1:]))
    else:
        reordered, reordered_titles = list(metrics), list(metric_titles)

    # ---- Rangos del dataset por eje (P1-P99 evita outliers extremos) ----
    ranges = []
    for m in reordered:
        d = pdf[m].dropna()
        ranges.append((np.percentile(d, 1), np.percentile(d, 99)))

    # ---- Figura + axes geometrico ----
    fig, ax = plt.subplots(figsize=(9, 10), facecolor=BG)             # ↑ figsize → radar MAS GRANDE
    ax.set_facecolor(BG)
    ax.set_aspect("equal")                                            # 1:1 pa que los circulos no se deformen
    ax.set(xlim=(-22, 22), ylim=(-23, 25))                            # margen extra arriba pal titulo (si lo hay)

    values = [row[m] for m in reordered]

    # ---- Circulos concentricos del radar ----
    # radius_circles: [3 (centro mas pequeno, NO se dibuja), 5.5, 8, ..., 20.5 (borde)]
    # El primero (3) es solo el limite del area central coloreada; el ultimo (20.5) es el borde exterior.
    radius_circles = [3, 5.5, 8, 10.5, 13, 15.5, 18, 20.5]
    for i, rad in enumerate(radius_circles):
        if i == 0:
            continue                                                  # el centro no se dibuja (es referencia)
        if i == len(radius_circles) - 1:
            color, lw, alpha = "black", 1.5, 1.0                      # borde exterior NEGRO bold sobre BG blanco
        else:
            color, lw, alpha = "grey", 1, 0.4                         # anillos intermedios gris suave
        ax.add_patch(plt.Circle((0, 0), rad, fc="none", ec=color, lw=lw, alpha=alpha))

    n = len(reordered)                                                # numero de ejes (8 o 12)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)             # angulos equiespaciados (0 = norte)

    # ---- Etiquetas de eje (Ataque tras GA, Off-ball tras GA, ...) ----
    # Rotacion: el texto se gira pa quedar tangente al circulo. Si esta abajo
    # (y<0), se suma 180° pa que NO salga al reves (legibilidad).
    lbl_fs = 10 if n <= 8 else 8                                      # 12 ejes → fuente menor (no se pisan)
    for angle, t in zip(angles, reordered_titles):
        x, y = 21.5 * np.sin(angle), 21.5 * np.cos(angle)             # 21.5 = radio etiqueta (>20.5 borde)
        rot = -np.rad2deg(angle)
        if y < 0:
            rot += 180                                                # mantiene texto legible abajo
        ax.text(x, y, t, rotation=rot, ha="center", va="center",
                fontsize=lbl_fs, fontweight="bold", color=WHITE)

    # ---- Lineas radiales (separadores entre ejes) ----
    for angle in angles:
        ax.plot([0, 20.5 * np.sin(angle)], [0, 20.5 * np.cos(angle)],
                color="grey", linewidth=0.5, alpha=0.4)

    # ---- Etiquetas de valor en cada anillo (formato adaptativo) ----
    # Se pintan en los puntos MEDIOS entre cada par de anillos (4.25, 6.75, ...).
    # El valor mostrado interpola linealmente entre el P1 y P99 del eje.
    for rad in [4.25, 6.75, 9.25, 11.75, 14.25, 16.75, 19.25]:
        for angle, (mn, mx) in zip(angles, ranges):
            rad_norm = (rad - 3) / (20.5 - 3)                         # 0..1 a lo largo del radio
            val = mn if mx == mn else mn + rad_norm * (mx - mn)
            # Formato dependiente de magnitud: 3 decimales pa CATEs (<0.01),
            # 2 pa <1, 1 pa <10, entero el resto.
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
                              edgecolor="none", alpha=0.9))            # halo BG pa que el numero se lea sobre el fill

    # ---- Poligono del jugador (coords polares → cartesianas) ----
    # Cada valor del jugador se mapea linealmente del rango [P1, P99] del eje
    # al radio fisico [3, 20.5]. Si el valor cae fuera del rango, se clipa.
    vertices = []
    for value, (mn, mx) in zip(values, ranges):
        if mx == mn:
            nv = 11.75                                                # rango cero → radio medio
        else:
            nv = 3 + (value - mn) / (mx - mn) * 17.5                  # mapea [mn,mx] → [3, 20.5]
        nv = max(3, min(20.5, nv))                                    # clip a la corona dibujable
        idx = len(vertices)
        vertices.append([nv * np.sin(angles[idx]), nv * np.cos(angles[idx])])

    # ---- Anillos de color alternos (colors[0]/colors[1]) clipeados al poligono ----
    # Efecto "telarana coloreada": cada anillo se pinta solo donde toca al poligono.
    poly = Polygon(vertices, fc="none", alpha=1.0, zorder=1)
    ax.add_patch(poly)
    # Centro: circulo del radio mas pequeno (3) con colors[0] = primario, clipped al poly
    central = plt.Circle((0, 0), radius_circles[0], fc=colors[0], ec="none",
                          alpha=0.45, zorder=2)
    central.set_clip_path(poly)
    ax.add_patch(central)
    # Anillos intermedios: construimos un polygon en forma de "corona" (anillo)
    # alternando colors[0] y colors[1].
    theta = np.linspace(0, 2 * np.pi, 100)
    for i in range(len(radius_circles) - 1):
        ri, ro = radius_circles[i], radius_circles[i + 1]
        cidx = (i + 1) % 2                                            # alterna 0 y 1 → colors[0]/colors[1]
        # Polygon = circulo exterior (ccw) + circulo interior (cw) = corona/anillo
        ring = list(zip(ro * np.cos(theta), ro * np.sin(theta))) + \
                list(zip(ri * np.cos(theta[::-1]), ri * np.sin(theta[::-1])))
        rp = Polygon(ring, fc=colors[cidx], alpha=0.45, zorder=2)
        rp.set_clip_path(poly)
        ax.add_patch(rp)
    # Contorno grueso del poligono encima de los rings (cierra el dibujo)
    closed = vertices + [vertices[0]]
    ax.plot([v[0] for v in closed], [v[1] for v in closed],
            color=colors[0], linewidth=3, zorder=10)

    ax.axis("off")                                                    # sin ejes/ticks alrededor del radar

    # ---- Header opcional (titulo + subtitulo + logo JO) ----
    # Cuando radar_report combina radar + tabla, llama con title="" pa que la
    # identidad viva en la tabla. En uso standalone si que se ven.
    if title:
        fig.text(0.5, 0.965, title, ha="center", va="top", color=WHITE,
                  fontsize=17, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.93, subtitle, ha="center", va="top", color=WHITE,
                  fontsize=11)
    if logo:
        add_logo(fig, width_frac=0.15)                                # ↑ logo MAS GRANDE

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # bbox_inches="tight" pal radar standalone (recorta margen blanco extra)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
    return fig


def _find(df: pl.DataFrame, query: str):
    """Resuelve un jugador por id (entero) o substring del nombre.

    Util pa la CLI: `python -m src.viz.radar 1234` o `python -m src.viz.radar Messi`.
    """
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
        subtitle=(f"{team_es(r['team_name'])} · {r['position_group']} · "
                  f"{int(r['minutes_played'])} min  |  {TOURNAMENT_ES}"),
        save_path=out,
    )
    print(f"OK -> {out}")
