"""figures_cap6 - Figuras Cap 6 TFM (validacion + sustantivos).

Estilo Diagonality-paper: figura cientifica simple (BG blanco, y-grid claro,
sin spines top/right, titulo bold pad, sin header dashboard, sin logos).

Funciones:
    psxg_calibration()        Reliability + Brier decomp del PSxG vs holdout
                              sagrado WC22.
    forest_pressure_clutch()  Top jugadores por probabilidad posterior
                              pressure-clutch.
    ate_bar_channels()        ATE DiD within-player por canal x contexto
                              (GOAL_AGAINST, GOAL_FOR). IC95% clusterizado.
    honestdid_bounds()        ANEXO. Sensibilidad HonestDiD M in {0.5, 1, 2}
                              sobre los 4 canales x 2 contextos.

Uso:
    python -m src.viz.figures_cap6
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import numpy as np
import matplotlib.pyplot as plt

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from viz.common import ATT, BG, DEF, GRID, LEGEND, TEXT

_BASE   = _SRC.parent
_DERIV  = _BASE / "data" / "parquet" / "derived"
_PSXG   = _DERIV / "psxg" / "calibration"
_DID    = _DERIV / "did"
_CCV    = _BASE / "outputs" / "ccv_table.parquet"
_OUTDIR = _BASE / "outputs" / "viz"

_CHANNELS = ["ataque", "defensa", "offball", "fisico"]
_CH_LABEL = {
    "ataque":  "Ofensivo",
    "defensa": "Defensivo",
    "offball": "Off-ball",
    "fisico":  "Físico",
}


# ----------------------------------------------------------------------------
# helper estilo: clean axes (Diagonality-paper)
# ----------------------------------------------------------------------------

def _style(ax, ygrid=True, xgrid=False):
    """Estilo paper limpio: spines top/right fuera, grid sutil."""
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(0.8)
    if ygrid:
        ax.grid(True, axis="y", color=GRID, linewidth=0.7, alpha=0.7, zorder=0)
    if xgrid:
        ax.grid(True, axis="x", color=GRID, linewidth=0.7, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10, colors=TEXT, length=3, width=0.7, pad=2)


def _savefig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# FIG 1: PSxG calibration (reliability curve + Brier decomposition)
# ============================================================================

def psxg_calibration(save_path=None):
    """Reliability curve del PSxG calibrado: OOF (cross-dataset) vs holdout WC22.

    Layout 1x2: izq reliability (predicho vs observado), dcha Brier decomp.
    """
    curve = pl.read_parquet(_PSXG / "calibration_curve.parquet")
    metrics = pl.read_parquet(_PSXG / "calibration_metrics.parquet")
    brier = pl.read_parquet(_PSXG / "brier_decomposition.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=BG)

    # ---- (a) Reliability curve ----
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", color=LEGEND, lw=1.0, alpha=0.7,
            zorder=1, label="Calibración perfecta")

    for model, color, label in [
        ("oof_psxg_calibrated", ATT, "OOF cross-dataset"),
        ("wc22_psxg_calibrated", DEF, "Holdout Mundial 2022"),
    ]:
        d = curve.filter(pl.col("model") == model).sort("pred_mean")
        if d.height == 0:
            continue
        x = d["pred_mean"].to_numpy()
        y = d["frac_positive"].to_numpy()
        n = d["n"].to_numpy()
        ax.plot(x, y, "-", color=color, lw=1.5, alpha=0.85, zorder=2)
        ax.scatter(x, y, s=40 + 100 * (n / max(n.max(), 1)),
                   color=color, edgecolor=BG, linewidth=0.8,
                   zorder=3, label=label)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("xG predicho", fontsize=11)
    ax.set_ylabel("Frecuencia observada de gol", fontsize=11)
    ax.set_title("(a) Reliability curve", fontsize=12, fontweight="bold", pad=10, loc="left")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95,
              edgecolor=GRID, fancybox=False)
    _style(ax)

    # ---- (b) Brier decomposition ----
    ax = axes[1]
    models = [("oof_psxg_calibrated", "OOF\ncross-dataset"),
              ("wc22_psxg_calibrated", "Holdout\nMundial 2022"),
              ("oof_sb_xg", "StatsBomb xG\n(baseline)")]
    rel, res = [], []
    for m, _ in models:
        d = brier.filter(pl.col("model") == m)
        if d.height == 0:
            rel.append(0); res.append(0)
            continue
        rel.append(float(d["reliability"][0]))
        res.append(float(d["resolution"][0]))

    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w/2, rel, w, color=DEF, alpha=0.85,
           edgecolor=TEXT, linewidth=0.5, label="Reliability  (menor = mejor)")
    ax.bar(x + w/2, res, w, color=ATT, alpha=0.85,
           edgecolor=TEXT, linewidth=0.5, label="Resolution  (mayor = mejor)")
    for xi, (r, s) in enumerate(zip(rel, res)):
        ax.text(xi - w/2, r, f"{r:.4f}", ha="center", va="bottom",
                fontsize=8.5, color=TEXT)
        ax.text(xi + w/2, s, f"{s:.4f}", ha="center", va="bottom",
                fontsize=8.5, color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in models], fontsize=9.5)
    ax.set_ylabel("Componente del Brier score", fontsize=11)
    ax.set_title("(b) Descomposición de Brier", fontsize=12,
                 fontweight="bold", pad=10, loc="left")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95,
              edgecolor=GRID, fancybox=False)
    _style(ax)

    # Footer numerico (ECE + AUC)
    ece_oof = float(metrics.filter(pl.col("model") == "oof_psxg_calibrated")["ece"][0])
    ece_wc = float(metrics.filter(pl.col("model") == "wc22_psxg_calibrated")["ece"][0])
    auc_oof = float(metrics.filter(pl.col("model") == "oof_psxg_calibrated")["auc"][0])
    auc_wc = float(metrics.filter(pl.col("model") == "wc22_psxg_calibrated")["auc"][0])
    fig.text(0.5, -0.04,
             f"ECE — OOF {ece_oof:.4f}  ·  WC22 {ece_wc:.4f}      "
             f"AUC — OOF {auc_oof:.3f}  ·  WC22 {auc_wc:.3f}",
             ha="center", va="top", color=LEGEND, fontsize=9)

    fig.tight_layout()
    if save_path:
        _savefig(fig, save_path)
    return fig


# ============================================================================
# FIG 2: Forest plot pressure-clutch top
# ============================================================================

def forest_pressure_clutch(save_path=None, top_n=12):
    """Top jugadores por probabilidad posterior P(beta > 0 | datos)
    sobre el indice pressure-clutch. IC80% como barra horizontal.
    """
    df = pl.read_parquet(_CCV)
    sub = (df.filter(
                (pl.col("low_sample") == False)
                & pl.col("pressure_response_idx").is_not_null()
                & pl.col("p_pressure_clutch_positive").is_not_null()
           )
           .sort("p_pressure_clutch_positive", descending=True)
           .head(top_n))

    n = sub.height
    fig, ax = plt.subplots(figsize=(10, 0.55 * n + 1.8), facecolor=BG)

    y = np.arange(n)[::-1]
    means = sub["pressure_response_idx"].to_numpy()
    lo = sub["pressure_response_lo80"].to_numpy()
    hi = sub["pressure_response_hi80"].to_numpy()
    probs = sub["p_pressure_clutch_positive"].to_numpy()
    names = sub["player_name"].to_list()
    teams = sub["team_name"].to_list()

    # Color segun cruza el umbral P >= 0.85
    colors = [ATT if p >= 0.85 else LEGEND for p in probs]

    ax.axvline(0, color=LEGEND, lw=0.9, ls=(0, (3, 3)), alpha=0.7, zorder=1)

    for yi, (m, l, h, c) in enumerate(zip(means, lo, hi, colors)):
        ax.plot([l, h], [y[yi], y[yi]], color=c, lw=2.2, alpha=0.75,
                solid_capstyle="round", zorder=2)
        ax.plot([l, l], [y[yi] - 0.18, y[yi] + 0.18], color=c, lw=1.1, zorder=2)
        ax.plot([h, h], [y[yi] - 0.18, y[yi] + 0.18], color=c, lw=1.1, zorder=2)
        ax.scatter(m, y[yi], s=70, color=c, edgecolor=BG, linewidth=1.0, zorder=3)

    # Y labels: nombre (bold) + equipo (gris)
    yticklabels = [f"{nm}\n{tm}" for nm, tm in zip(names, teams)]
    ax.set_yticks(y)
    ax.set_yticklabels(yticklabels, fontsize=9.5)

    # Probabilidad posterior anotada al final de cada IC
    for yi, (h, p) in enumerate(zip(hi, probs)):
        ax.text(h + 0.005, y[yi], f"P = {p:.2f}",
                ha="left", va="center",
                color=ATT if p >= 0.85 else LEGEND,
                fontsize=9, fontweight=600 if p >= 0.85 else 400)

    ax.set_xlabel("Efecto pressure-clutch  ·  media posterior (IC 80%)",
                  fontsize=11, labelpad=6)
    ax.set_title(f"Top {top_n} jugadores por probabilidad posterior pressure-clutch",
                 fontsize=12, fontweight="bold", pad=10, loc="left")
    _style(ax, ygrid=False, xgrid=True)

    # Leyenda manual
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=ATT,
                   markersize=9, label="P ≥ 0.85  (significativo)"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=LEGEND,
                   markersize=9, label="P < 0.85  (no significativo)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9,
              framealpha=0.95, edgecolor=GRID, fancybox=False)

    fig.tight_layout()
    if save_path:
        _savefig(fig, save_path)
    return fig


# ============================================================================
# FIG 3: ATE bar chart canal x contexto
# ============================================================================

def ate_bar_channels(save_path=None):
    """ATE DiD within-player por canal x contexto. Bars agrupadas, IC 95%."""
    ate = pl.read_parquet(_DID / "ate_population.parquet")

    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor=BG)

    x = np.arange(len(_CHANNELS))
    w = 0.36

    ate_dict = {(r["channel"], r["shock_type"]): (r["ate"], r["ci_lo"], r["ci_hi"])
                for r in ate.iter_rows(named=True)}

    for offset, sh, color, label in [
        (-w/2, "GOAL_AGAINST", ATT, "Tras encajar"),
        (+w/2, "GOAL_FOR",     DEF, "Tras marcar"),
    ]:
        vals = [ate_dict.get((ch, sh), (0, 0, 0))[0] for ch in _CHANNELS]
        los  = [ate_dict.get((ch, sh), (0, 0, 0))[1] for ch in _CHANNELS]
        his  = [ate_dict.get((ch, sh), (0, 0, 0))[2] for ch in _CHANNELS]
        err_lo = [max(0, vals[i] - los[i]) for i in range(len(vals))]
        err_hi = [max(0, his[i] - vals[i]) for i in range(len(vals))]

        ax.bar(x + offset, vals, w, color=color, alpha=0.85,
               edgecolor=TEXT, linewidth=0.5, label=label, zorder=2)
        ax.errorbar(x + offset, vals, yerr=[err_lo, err_hi],
                    fmt="none", ecolor=TEXT, elinewidth=0.9,
                    capsize=4, capthick=0.9, zorder=3)
        # Valor encima/debajo de la barra
        for xi, v in enumerate(vals):
            yt = max(his[xi], 0) + 0.001 if v >= 0 else min(los[xi], 0) - 0.001
            va = "bottom" if v >= 0 else "top"
            ax.text(x[xi] + offset, yt, f"{v:+.4f}",
                    ha="center", va=va, fontsize=8.5, color=TEXT)

    ax.axhline(0, color=LEGEND, lw=0.9, alpha=0.7, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([_CH_LABEL[ch] for ch in _CHANNELS],
                       fontsize=11, fontweight=600)
    ax.set_ylabel("ATE  ·  desviaciones estándar del canal", fontsize=11)
    ax.set_title("Efecto poblacional medio del shock por canal y contexto",
                 fontsize=12, fontweight="bold", pad=10, loc="left")
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95,
              edgecolor=GRID, fancybox=False)
    _style(ax)

    # Footer
    n_shocks_cell = int(ate["n_shocks"].mode()[0])
    fig.text(0.5, -0.02,
             f"DiD TWFE within-player  ·  IC 95% clusterizado por jugador  ·  n ≈ {n_shocks_cell} shocks por celda",
             ha="center", va="top", color=LEGEND, fontsize=9)

    fig.tight_layout()
    if save_path:
        _savefig(fig, save_path)
    return fig


# ============================================================================
# ANEXO FIG: HonestDiD sensitivity bounds
# ============================================================================

def honestdid_bounds(save_path=None):
    """Cotas HonestDiD para M in {0.5, 1, 2} sobre los 4 canales x 2 contextos.

    Layout 2x4. En cada panel, banda del ATE original (color tenue) + 3 barras
    horizontales (M=0.5, 1, 2) con su IC robustecido.
    """
    hd = pl.read_parquet(_DID / "honest_did.parquet")
    ate = pl.read_parquet(_DID / "ate_population.parquet")

    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.5), facecolor=BG,
                              sharey=True)

    shocks = [("GOAL_AGAINST", "Tras encajar", ATT),
              ("GOAL_FOR",     "Tras marcar",  DEF)]
    M_levels = [0.5, 1.0, 2.0]
    M_labels = ["M = 0.5", "M = 1", "M = 2"]

    for ri, (sh, row_lbl, color) in enumerate(shocks):
        for ci, ch in enumerate(_CHANNELS):
            ax = axes[ri][ci]

            ate_row = ate.filter((pl.col("channel") == ch) &
                                  (pl.col("shock_type") == sh))
            if ate_row.height > 0:
                ate_val = float(ate_row["ate"][0])
                ate_lo = float(ate_row["ci_lo"][0])
                ate_hi = float(ate_row["ci_hi"][0])
                ax.axvspan(ate_lo, ate_hi, color=color, alpha=0.12, zorder=1)
                ax.axvline(ate_val, color=color, lw=1.4, alpha=0.9,
                           zorder=4)

            for yp, M, lbl in zip([2, 1, 0], M_levels, M_labels):
                row = hd.filter((pl.col("channel") == ch) &
                                 (pl.col("shock_type") == sh) &
                                 (pl.col("M") == M))
                if row.height == 0:
                    continue
                lo = float(row["ci_lo_robust"][0])
                hi = float(row["ci_hi_robust"][0])
                pt = float(row["ate_robust"][0])
                ax.plot([lo, hi], [yp, yp], color=TEXT, lw=1.8,
                        solid_capstyle="round", alpha=0.7, zorder=3)
                ax.plot([lo, lo], [yp - 0.18, yp + 0.18], color=TEXT, lw=0.9, zorder=3)
                ax.plot([hi, hi], [yp - 0.18, yp + 0.18], color=TEXT, lw=0.9, zorder=3)
                ax.scatter(pt, yp, s=22, color=TEXT, edgecolor=BG,
                           linewidth=0.5, zorder=4)

            ax.axvline(0, color=LEGEND, lw=0.8, ls=(0, (3, 3)),
                       alpha=0.6, zorder=2)
            ax.set_ylim(-0.6, 2.6)
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(M_labels if ci == 0 else ["", "", ""], fontsize=9)
            if ri == 0:
                ax.set_title(_CH_LABEL[ch], fontsize=11, fontweight="bold", pad=8, loc="center")
            if ri == 1:
                ax.set_xlabel("ATE robustecido", fontsize=10, labelpad=4)
            if ci == 0:
                ax.set_ylabel(row_lbl, fontsize=11, fontweight=600, labelpad=8)
            _style(ax, ygrid=False, xgrid=True)

    fig.suptitle("Sensibilidad HonestDiD — cotas frente a tendencias paralelas",
                 fontsize=13, fontweight="bold", x=0.02, ha="left", y=0.98)
    fig.text(0.5, -0.01,
             "Banda tenue = IC 95% original  ·  Barra negra = cota robustecida para cada nivel M de relajación",
             ha="center", va="top", color=LEGEND, fontsize=9)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    if save_path:
        _savefig(fig, save_path)
    return fig


# ============================================================================
# CLI runner
# ============================================================================

def main():
    out = _OUTDIR
    out.mkdir(parents=True, exist_ok=True)
    print(f"[figures_cap6] outputs to {out}")
    psxg_calibration(save_path=out / "fig_cap6_psxg_calibration.png")
    print("  OK  fig_cap6_psxg_calibration.png")
    forest_pressure_clutch(save_path=out / "fig_cap6_forest_pressure_clutch.png")
    print("  OK  fig_cap6_forest_pressure_clutch.png")
    ate_bar_channels(save_path=out / "fig_cap6_ate_canal_contexto.png")
    print("  OK  fig_cap6_ate_canal_contexto.png")
    honestdid_bounds(save_path=out / "fig_anexo_honestdid.png")
    print("  OK  fig_anexo_honestdid.png")


if __name__ == "__main__":
    main()
