"""figures_cap6 - Figuras Cap 6 TFM (validacion + sustantivos).

Estilo Diagonality-paper: figura cientifica simple (BG blanco, y-grid claro,
sin spines top/right, titulo bold pad, sin header dashboard, sin logos).

Funciones:
    psxg_calibration()        Reliability + Brier decomp del PSxG vs holdout
                              sagrado WC22.
    cate_heterogeneity()      Distribucion de CATE individual por canal y
                              contexto. La heterogeneidad que el ATE
                              poblacional NO captura.
    window_sensitivity()      Sensibilidad del ATE a la ventana pre/post
                              (+-3/5/7/10/15 min). Valida el horizonte +-10.
    honestdid_bounds()        ANEXO. Sensibilidad HonestDiD M in {0.5, 1, 2}.

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
_VAL    = _DERIV / "did_validation"
_CATE   = _DERIV / "cate"
_OUTDIR = _BASE / "outputs" / "viz"

_CHANNELS = ["ataque", "defensa", "offball", "fisico"]
_CH_LABEL = {
    "ataque":  "Ofensivo",
    "defensa": "Defensivo",
    "offball": "Off-ball",
    "fisico":  "Físico",
}

# Color saturado para la curva WC22 (no el #EF4444 light de PPCF, que se diluye)
_RED_DEEP = "#DC2626"

# Color para PRESSURE en CATE heterogeneity (violet del PCT_CMAP)
_PURPLE = "#9333EA"


# ----------------------------------------------------------------------------
# helper estilo
# ----------------------------------------------------------------------------

def _style(ax, ygrid=True, xgrid=False):
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
# FIG 1: PSxG calibration (reliability + Brier decomp)
# ============================================================================

def psxg_calibration(save_path=None):
    """Reliability curve + Brier decomp. Fix: rojo profundo + legend abajo."""
    curve = pl.read_parquet(_PSXG / "calibration_curve.parquet")
    metrics = pl.read_parquet(_PSXG / "calibration_metrics.parquet")
    brier = pl.read_parquet(_PSXG / "brier_decomposition.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), facecolor=BG)

    # ---- (a) Reliability curve ----
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", color=LEGEND, lw=1.0, alpha=0.6, zorder=1)

    for model, color, label, lw in [
        ("oof_psxg_calibrated", ATT, "OOF cross-dataset", 2.0),
        ("wc22_psxg_calibrated", _RED_DEEP, "Holdout Mundial 2022", 2.0),
    ]:
        d = curve.filter(pl.col("model") == model).sort("pred_mean")
        if d.height == 0:
            continue
        x = d["pred_mean"].to_numpy()
        y = d["frac_positive"].to_numpy()
        n = d["n"].to_numpy()
        ax.plot(x, y, "-", color=color, lw=lw, alpha=0.9, zorder=2,
                label=label)
        ax.scatter(x, y, s=60 + 140 * (n / max(n.max(), 1)),
                   color=color, edgecolor=BG, linewidth=1.0, zorder=3)

    # Diagonal label
    ax.text(0.05, 0.93, "Calibración perfecta", color=LEGEND,
            fontsize=9, style="italic", rotation=45,
            transform=ax.transAxes, ha="left", va="top")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("xG predicho", fontsize=11)
    ax.set_ylabel("Frecuencia observada de gol", fontsize=11)
    ax.set_title("(a) Reliability curve", fontsize=12, fontweight="bold",
                 pad=10, loc="left")
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95,
              edgecolor=GRID, fancybox=False)
    _style(ax)

    # ---- (b) Brier decomposition + AUC ----
    ax = axes[1]
    models = [("oof_psxg_calibrated", "OOF\ncross-dataset", ATT),
              ("wc22_psxg_calibrated", "Holdout\nMundial 2022", _RED_DEEP),
              ("oof_sb_xg", "StatsBomb xG\n(baseline)", LEGEND)]
    rel, res, auc, brier_v = [], [], [], []
    for m, _, _ in models:
        d = brier.filter(pl.col("model") == m)
        m2 = metrics.filter(pl.col("model") == m)
        rel.append(float(d["reliability"][0]) if d.height else 0)
        res.append(float(d["resolution"][0]) if d.height else 0)
        auc.append(float(m2["auc"][0]) if m2.height else 0)
        brier_v.append(float(d["brier"][0]) if d.height else 0)

    x = np.arange(len(models))
    w = 0.35
    bars_rel = ax.bar(x - w/2, rel, w, color="#E5E7EB",
                     edgecolor=TEXT, linewidth=0.5, label="Reliability  (↓ mejor)")
    bars_res = ax.bar(x + w/2, res, w,
                     color=[c for _, _, c in models], alpha=0.85,
                     edgecolor=TEXT, linewidth=0.5, label="Resolution  (↑ mejor)")
    for xi, (r, s) in enumerate(zip(rel, res)):
        ax.text(xi - w/2, r + 0.001, f"{r:.4f}", ha="center", va="bottom",
                fontsize=8.5, color=TEXT)
        ax.text(xi + w/2, s + 0.001, f"{s:.4f}", ha="center", va="bottom",
                fontsize=8.5, color=TEXT, fontweight=600)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl, _ in models], fontsize=9.5)
    ax.set_ylabel("Componente del Brier score", fontsize=11)
    ax.set_title("(b) Descomposición de Brier", fontsize=12,
                 fontweight="bold", pad=10, loc="left")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95,
              edgecolor=GRID, fancybox=False)
    _style(ax)

    # Footer numerico (AUC)
    aucs_str = "    ".join([f"{lbl.replace(chr(10), ' ')}: AUC = {a:.3f}"
                             for (_, lbl, _), a in zip(models, auc)])
    fig.text(0.5, -0.03, aucs_str,
             ha="center", va="top", color=TEXT, fontsize=9.5, fontweight=500)

    fig.tight_layout()
    if save_path:
        _savefig(fig, save_path)
    return fig


# ============================================================================
# FIG 2: CATE heterogeneity por canal y contexto
# ============================================================================

def cate_heterogeneity(save_path=None):
    """Distribucion del CATE individual sobre los 598 jugadores, por canal x
    contexto. La VARIANCIA es la senal que el promedio cero esconde.

    Layout 1x4 (un panel por canal). Densidades para GOAL_AGAINST,
    GOAL_FOR y PRESSURE. Anotado: sd, range.
    """
    df = pl.read_parquet(_CATE / "posterior_player.parquet")

    shocks = [
        ("GOAL_AGAINST", "Tras encajar", ATT),
        ("GOAL_FOR",     "Tras marcar",  _RED_DEEP),
        ("PRESSURE",     "Presión alta", _PURPLE),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), facecolor=BG, sharey=True)

    for ci, ch in enumerate(_CHANNELS):
        ax = axes[ci]
        for sh, lbl, color in shocks:
            sub = df.filter((pl.col("channel") == ch) & (pl.col("shock_type") == sh))
            if sub.height == 0:
                continue
            vals = sub["cate_mean"].to_numpy()
            sd = vals.std()
            # KDE simple via histograma normalizado + suavizado
            bins = np.linspace(vals.min(), vals.max(), 40)
            hist, edges = np.histogram(vals, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            # KDE bandwidth via Silverman
            bw = 1.06 * sd * (len(vals) ** (-1/5))
            xs = np.linspace(vals.min() - 2*bw, vals.max() + 2*bw, 300)
            density = np.zeros_like(xs)
            for v in vals:
                density += np.exp(-0.5 * ((xs - v) / bw) ** 2)
            density /= (len(vals) * bw * np.sqrt(2 * np.pi))
            ax.fill_between(xs, density, color=color, alpha=0.25, zorder=2)
            ax.plot(xs, density, color=color, lw=1.8, label=lbl, zorder=3)

        ax.axvline(0, color=LEGEND, lw=0.9, ls=(0, (3, 3)), alpha=0.6, zorder=1)
        ax.set_title(_CH_LABEL[ch], fontsize=12, fontweight="bold", pad=8, loc="left")
        ax.set_xlabel("CATE individual", fontsize=10.5)
        if ci == 0:
            ax.set_ylabel("Densidad", fontsize=11)
            ax.legend(loc="upper right", fontsize=9, framealpha=0.95,
                      edgecolor=GRID, fancybox=False)
        # Anotar σ de GA + GF
        sd_text = []
        for sh, _, color in shocks[:2]:  # GA + GF
            sub = df.filter((pl.col("channel") == ch) & (pl.col("shock_type") == sh))
            if sub.height > 0:
                sd_text.append(f"σ={float(sub['cate_mean'].std()):.3f}")
        if sd_text:
            ax.text(0.97, 0.97, "\n".join(sd_text), transform=ax.transAxes,
                    ha="right", va="top", fontsize=9, color=TEXT,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                              edgecolor=GRID, linewidth=0.6))
        _style(ax)

    fig.suptitle("Heterogeneidad del CATE por jugador — el promedio cero esconde la señal individual",
                 fontsize=12.5, fontweight="bold", x=0.02, ha="left", y=1.02)
    fig.text(0.5, -0.03,
             "Distribución sobre 598 jugadores  ·  media posterior por jugador  ·  CATE jerárquico bayesiano multivariate (NUTS HMC)",
             ha="center", va="top", color=LEGEND, fontsize=9)

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    if save_path:
        _savefig(fig, save_path)
    return fig


# ============================================================================
# FIG 3: Window sensitivity
# ============================================================================

def window_sensitivity(save_path=None):
    """ATE estimado a 5 horizontes pre/post (+-3, 5, 7, 10, 15 min) por canal
    y contexto. Valida la eleccion del horizonte +-10 y muestra donde la
    senal cambia de signo / significancia con la ventana.
    """
    ws = pl.read_parquet(_VAL / "window_sensitivity.parquet")

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3), facecolor=BG)

    for ci, ch in enumerate(_CHANNELS):
        ax = axes[ci]
        for sh, color, marker, label in [
            ("GOAL_AGAINST", ATT, "o", "Tras encajar"),
            ("GOAL_FOR",     _RED_DEEP, "s", "Tras marcar"),
        ]:
            d = (ws.filter((pl.col("channel") == ch) & (pl.col("shock_type") == sh))
                 .sort("window_min"))
            if d.height == 0:
                continue
            x = d["window_min"].to_numpy()
            y = d["ate"].to_numpy()
            lo = d["ci_lo"].to_numpy()
            hi = d["ci_hi"].to_numpy()
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, zorder=2)
            ax.plot(x, y, "-", color=color, lw=1.6, marker=marker, ms=6,
                    mec=BG, mew=0.7, alpha=0.9, zorder=3, label=label)

        # Eje x personalizado: solo los 5 horizontes
        ax.axvline(10, color=LEGEND, lw=0.7, ls=(0, (2, 2)), alpha=0.6, zorder=1)
        ax.axhline(0, color=LEGEND, lw=0.8, alpha=0.6, zorder=1)
        ax.set_xticks([3, 5, 7, 10, 15])
        ax.set_xlabel("Ventana ± min", fontsize=10.5)
        if ci == 0:
            ax.set_ylabel("ATE  ·  desv. estándar canal", fontsize=10.5)
            ax.legend(loc="best", fontsize=9, framealpha=0.95,
                      edgecolor=GRID, fancybox=False)
        ax.set_title(_CH_LABEL[ch], fontsize=12, fontweight="bold", pad=8, loc="left")
        _style(ax)

    fig.suptitle("Sensibilidad del ATE al horizonte de ventana pre/post",
                 fontsize=12.5, fontweight="bold", x=0.02, ha="left", y=1.02)
    fig.text(0.5, -0.04,
             "Banda = IC 95% clusterizado por jugador  ·  Línea punteada vertical = ventana de referencia ± 10 min",
             ha="center", va="top", color=LEGEND, fontsize=9)

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    if save_path:
        _savefig(fig, save_path)
    return fig


# ============================================================================
# ANEXO FIG: HonestDiD sensitivity bounds
# ============================================================================

def honestdid_bounds(save_path=None):
    """Cotas HonestDiD para M in {0.5, 1, 2} sobre los 4 canales x 2 contextos."""
    hd = pl.read_parquet(_DID / "honest_did.parquet")
    ate = pl.read_parquet(_DID / "ate_population.parquet")

    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.5), facecolor=BG, sharey=True)

    shocks = [("GOAL_AGAINST", "Tras encajar", ATT),
              ("GOAL_FOR",     "Tras marcar",  _RED_DEEP)]
    M_levels = [0.5, 1.0, 2.0]
    M_labels = ["M = 0.5", "M = 1", "M = 2"]

    for ri, (sh, row_lbl, color) in enumerate(shocks):
        for ci, ch in enumerate(_CHANNELS):
            ax = axes[ri][ci]

            ate_row = ate.filter((pl.col("channel") == ch) & (pl.col("shock_type") == sh))
            if ate_row.height > 0:
                ate_val = float(ate_row["ate"][0])
                ate_lo = float(ate_row["ci_lo"][0])
                ate_hi = float(ate_row["ci_hi"][0])
                ax.axvspan(ate_lo, ate_hi, color=color, alpha=0.12, zorder=1)
                ax.axvline(ate_val, color=color, lw=1.4, alpha=0.9, zorder=4)

            for yp, M, lbl in zip([2, 1, 0], M_levels, M_labels):
                row = hd.filter((pl.col("channel") == ch) & (pl.col("shock_type") == sh) & (pl.col("M") == M))
                if row.height == 0:
                    continue
                lo = float(row["ci_lo_robust"][0]); hi = float(row["ci_hi_robust"][0])
                pt = float(row["ate_robust"][0])
                ax.plot([lo, hi], [yp, yp], color=TEXT, lw=1.6, solid_capstyle="round", alpha=0.75, zorder=3)
                ax.plot([lo, lo], [yp - 0.18, yp + 0.18], color=TEXT, lw=0.9, zorder=3)
                ax.plot([hi, hi], [yp - 0.18, yp + 0.18], color=TEXT, lw=0.9, zorder=3)
                ax.scatter(pt, yp, s=22, color=TEXT, edgecolor=BG, linewidth=0.5, zorder=4)

            ax.axvline(0, color=LEGEND, lw=0.8, ls=(0, (3, 3)), alpha=0.6, zorder=2)
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
    cate_heterogeneity(save_path=out / "fig_cap6_cate_heterogeneity.png")
    print("  OK  fig_cap6_cate_heterogeneity.png")
    window_sensitivity(save_path=out / "fig_cap6_window_sensitivity.png")
    print("  OK  fig_cap6_window_sensitivity.png")
    honestdid_bounds(save_path=out / "fig_anexo_honestdid.png")
    print("  OK  fig_anexo_honestdid.png")


if __name__ == "__main__":
    main()
