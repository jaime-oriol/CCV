"""figures - Figuras analiticas del PCJ (capa causal).

Estilo dark journal paper-style: layout factorial 2 (perspective) × 4 (canal)
con cabeceras de fila + columna, tipografia limpia, banda IC + linea β,
caption metodologico al pie. NO va en el `make_all` core showcase — es
figura de METODO/validacion causal (pre-trends planos + efecto medio
≈0 -> heterogeneidad individual en CATE bayesiano de M14).

Uso:
    python -m src.viz.figures
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import matplotlib.pyplot as plt

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viz.common import ATT, BG, DEF, WHITE, add_logo

_DID = _SRC.parent / "data" / "parquet" / "derived" / "did"

# Nombres paper-quality (no jerga colloquial) y orden canonico.
_CHANNELS = ["ataque", "defensa", "offball", "fisico"]
_CH_LABEL = {
    "ataque":  "Produccion ofensiva",
    "defensa": "Acciones defensivas",
    "offball": "Movimiento off-ball",
    "fisico":  "Carga fisica",
}
# Filas = perspective del shock. Color heredado de la identidad del repo
# (azul ATT para post-GA, rojo DEF para post-GF; consistente con scatter/radar).
_SHOCKS = [
    ("GOAL_AGAINST", "Tras encajar", ATT),
    ("GOAL_FOR",     "Tras marcar",  DEF),
]

_GRID_C   = "#5a5c5b"          # gris medio: spines + lineas de referencia
_REF_GREY = "#7a7c7b"          # gris algo mas claro: y=0


def event_study(save_path=None):
    """Event-study Sun-Abraham 2021: 2 shocks × 4 canales, β minuto a minuto.

    Layout 2×4 (rows = perspective, cols = canal). Row labels rotados a la
    izquierda; col labels solo en la fila superior. X compartido; Y libre por
    panel (las escalas difieren mucho entre canales). Paleta sobria, lineas
    finas, IC sombreado al 18% alpha. Caption metodologico al pie.
    """
    es = pl.read_parquet(_DID / "event_study.parquet")

    fig, axes = plt.subplots(
        2, 4, figsize=(14, 7), sharex=True, sharey=False, facecolor=BG,
    )
    fig.subplots_adjust(
        left=0.075, right=0.985, top=0.83, bottom=0.16,
        wspace=0.22, hspace=0.32,
    )

    # ---- Cabecera: titulo declarativo + subtitulo metodologico ----
    fig.text(0.5, 0.965,
             "Efecto causal del shock emocional sobre los 4 canales del juego",
             ha="center", va="top", color=WHITE,
             fontsize=15, fontweight="bold")
    fig.text(0.5, 0.927,
             "Event-study DiD (Sun & Abraham, 2021)  ·  ventana ±10 min "
             "respecto al gol  ·  Mundial Qatar 2022",
             ha="center", va="top", color=WHITE, fontsize=10, alpha=0.78)

    # ---- Grid de paneles ----
    for ri, (sh, _row_lbl, color) in enumerate(_SHOCKS):
        for ci, ch in enumerate(_CHANNELS):
            ax = axes[ri, ci]
            ax.set_facecolor(BG)

            d = (es.filter((pl.col("channel") == ch) & (pl.col("shock_type") == sh))
                    .sort("relative_min"))
            x = d["relative_min"].to_numpy()
            b = d["beta"].to_numpy()
            lo, hi = d["ci_lo"].to_numpy(), d["ci_hi"].to_numpy()

            # Sombreado tenue de la zona post-shock (0, +10 min)
            ax.axvspan(0, 10, color=color, alpha=0.05, zorder=0)
            # Linea de referencia y = 0
            ax.axhline(0, color=_REF_GREY, lw=0.7, alpha=0.6, zorder=1)
            # Instante del gol (x = 0)
            ax.axvline(0, color=WHITE, lw=0.8, ls=(0, (3, 3)),
                       alpha=0.6, zorder=2)
            # Banda IC 95% + linea de coeficientes + puntos discretos
            ax.fill_between(x, lo, hi, color=color, alpha=0.18,
                             edgecolor="none", zorder=2)
            ax.plot(x, b, "-", color=color, lw=1.4, alpha=0.95, zorder=3)
            ax.plot(x, b, "o", color=color, ms=3.0, mec=BG, mew=0.6, zorder=4)

            # Cabecera de columna (canal) solo en fila superior
            if ri == 0:
                ax.set_title(_CH_LABEL[ch], color=WHITE,
                              fontsize=11, fontweight="bold", pad=8)
            # Label X solo en fila inferior
            if ri == 1:
                ax.set_xlabel(r"$\tau$ (min)", color=WHITE,
                              fontsize=9.5, labelpad=3)
            # Label Y solo en primera columna
            if ci == 0:
                ax.set_ylabel(r"$\hat{\beta}_{\tau}$",
                              color=WHITE, fontsize=10.5, labelpad=4)

            ax.set_xlim(-10.5, 10.5)
            ax.set_xticks(range(-10, 11, 5))
            ax.tick_params(labelsize=8, colors=WHITE,
                            length=3, width=0.7, pad=2)
            # Spines: solo left + bottom, finos, gris medio
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(_GRID_C)
                ax.spines[s].set_linewidth(0.7)

    # ---- Etiquetas de fila (perspective) rotadas a la izquierda ----
    for ri, (_, row_lbl, _) in enumerate(_SHOCKS):
        pos = axes[ri, 0].get_position()
        y_center = pos.y0 + pos.height / 2.0
        fig.text(0.012, y_center, row_lbl, ha="left", va="center",
                  color=WHITE, fontsize=11.5, fontweight="bold", rotation=90)

    # ---- Caption metodologico al pie (paper-style) ----
    fig.text(
        0.5, 0.085,
        r"Coeficientes $\hat{\beta}_{\tau}$ del event-study Sun-Abraham (J Econometrics 225, 2021); "
        r"bin de referencia $\tau = -1$. Banda = IC 95% con errores estandar agrupados por jugador (CRV1).",
        ha="center", va="top", color=WHITE, fontsize=8.7, alpha=0.85,
    )
    fig.text(
        0.5, 0.052,
        "Pre-trends planos sostienen la identificacion causal; efecto medio "
        "≈ 0 implica que la respuesta al shock es individual,",
        ha="center", va="top", color=WHITE, fontsize=8.7, alpha=0.85,
    )
    fig.text(
        0.5, 0.028,
        "no poblacional — la heterogeneidad se captura en el CATE bayesiano de M14.",
        ha="center", va="top", color=WHITE, fontsize=8.7, alpha=0.85,
    )

    # Logo JO pequeno esquina inferior derecha
    add_logo(fig, width_frac=0.055, margin=0.010, corner="br")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
    return fig


if __name__ == "__main__":
    out = "outputs/viz/event_study.png"
    event_study(save_path=out)
    print(f"OK -> {out}")
