"""figures_cap6 - Figuras Cap 6 TFM (validacion + sustantivos).

Estilo paper homogeneo: BG blanco, logo JO top-right, titulos centrados,
y-grid claro, sin spines top/right. Mismos tamaños base (14x5 para 1x2/1x4
panels, 14x6.5 para 2x4).

Funciones:
    psxg_calibration()    Reliability + Brier decomp del PSxG vs holdout WC22.
    cate_heterogeneity()  Distribucion de CATE individual por canal y shock.
    window_sensitivity()  ATE estimado a 5 horizontes pre/post.
    honestdid_bounds()    ANEXO. Sensibilidad HonestDiD M in {0.5, 1, 2}.

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

from viz.common import ATT, BG, DEF, GRID, LEGEND, TEXT, add_logo

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

# Tercer color saturado para PRESSURE (violet vivo)
_PURPLE = "#9333EA"

# -- Estandar visual --
_TITLE_SIZE   = 12.5
_SUPTITLE_SZ  = 13.5
_AXLABEL_SZ   = 11
_TICK_SZ      = 10
_LEGEND_SZ    = 9.5
_FOOTER_SZ    = 9
# --- Logo JO (top-right) — parámetros de posicionamiento ---
# Geometría (todo en fracción [0..1] de figura, esquina top-right):
#     x_left = 1.0 - _LOGO_MARGIN_X - _LOGO_FRAC     (borde izq del logo)
#     y_bot  = 1.0 - _LOGO_MARGIN_Y - h_frac          (borde inf del logo)
#     h_frac = _LOGO_FRAC * (figW/figH) / aspect_logo (alto en fracción figura)
#
# Cómo mover el logo (separados X / Y, no comparten margen):
#   * MÁS GRANDE       → ↑ _LOGO_FRAC (e.g. 0.14). Crece ancho y alto a la vez.
#   * MÁS PEQUEÑO      → ↓ _LOGO_FRAC (e.g. 0.07).
#   * MÁS A LA DERECHA → ↓ _LOGO_MARGIN_X (0.0 = pegado al borde derecho).
#   * MÁS A LA IZQUIERDA → ↑ _LOGO_MARGIN_X (lo aleja del borde derecho).
#   * MÁS ARRIBA       → ↓ _LOGO_MARGIN_Y (0.0 = pegado al borde superior).
#   * MÁS ABAJO        → ↑ _LOGO_MARGIN_Y (lo aleja del borde superior).
_LOGO_FRAC      = 0.125    # ancho del logo en fracción de figura
_LOGO_MARGIN_X  = 0.03     # separación al borde derecho — ↑ = más a la izq
_LOGO_MARGIN_Y  = -0.015    # separación al borde superior — negativo = logo top sube hasta nivel del título (suptitle en y=1.02)


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
    ax.tick_params(labelsize=_TICK_SZ, colors=TEXT, length=3, width=0.7, pad=2)


def _logo_tr(fig):
    """Logo JO en top-right con márgenes X / Y independientes.

    Implementación local que no usa add_logo() de common.py porque aquella
    comparte un único `margin` para X e Y; aquí queremos separarlos para
    ajustar finamente la posición sobre el título sin tocar el lateral.
    """
    from viz.common import _LOGO_PATH                                   # path al PNG del logo JO
    if not _LOGO_PATH.exists():
        return                                                          # silencioso si falta el asset
    try:
        img = plt.imread(str(_LOGO_PATH))
        logo_aspect = img.shape[1] / img.shape[0]                       # ratio ancho/alto del PNG
        figW, figH = fig.get_size_inches()
        h_frac = _LOGO_FRAC * (figW / figH) / logo_aspect               # alto en fracción de figura
        x = 1.0 - _LOGO_MARGIN_X - _LOGO_FRAC                           # borde izq del logo
        y = 1.0 - _LOGO_MARGIN_Y - h_frac                               # borde inf del logo
        ax_logo = fig.add_axes([x, y, _LOGO_FRAC, h_frac])              # axes dedicado al logo
        ax_logo.imshow(img)
        ax_logo.axis("off")
    except Exception:
        pass                                                            # PNG corrupto/raro → no pinta


def _savefig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# FIG 1: PSxG calibration
# ============================================================================

def psxg_calibration(save_path=None):
    """Reliability curve + Brier decomp del PSxG."""
    curve = pl.read_parquet(_PSXG / "calibration_curve.parquet")
    metrics = pl.read_parquet(_PSXG / "calibration_metrics.parquet")
    brier = pl.read_parquet(_PSXG / "brier_decomposition.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)

    # ---- (a) Reliability curve ----
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", color=LEGEND, lw=1.0, alpha=0.5,
            zorder=1, label="Calibración perfecta")

    for model, color, label in [
        ("oof_psxg_calibrated",  ATT, "OOF cross-dataset"),
        ("wc22_psxg_calibrated", DEF, "Holdout Mundial 2022"),
    ]:
        d = curve.filter(pl.col("model") == model).sort("pred_mean")
        if d.height == 0:
            continue
        x = d["pred_mean"].to_numpy()
        y = d["frac_positive"].to_numpy()
        n = d["n"].to_numpy()
        ax.plot(x, y, "-", color=color, lw=2.0, alpha=0.9, zorder=2,
                label=label)
        ax.scatter(x, y, s=60 + 140 * (n / max(n.max(), 1)),
                   color=color, edgecolor=BG, linewidth=1.0, zorder=3)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("xG predicho", fontsize=_AXLABEL_SZ)
    ax.set_ylabel("Frecuencia observada de gol", fontsize=_AXLABEL_SZ)
    ax.set_title("Curva de fiabilidad", fontsize=_TITLE_SIZE,
                 fontweight="bold", pad=12, loc="center")
    ax.legend(loc="lower right", fontsize=_LEGEND_SZ, framealpha=0.95,
              edgecolor=GRID, fancybox=False)
    _style(ax)

    # ---- (b) Brier decomposition ----
    ax = axes[1]
    models = [("oof_psxg_calibrated",  "OOF\ncross-dataset",       ATT),
              ("wc22_psxg_calibrated", "Holdout\nMundial 2022",    DEF),
              ("oof_sb_xg",            "StatsBomb xG\n(baseline)", LEGEND)]
    rel, res, auc = [], [], []
    for m, _, _ in models:
        d = brier.filter(pl.col("model") == m)
        m2 = metrics.filter(pl.col("model") == m)
        rel.append(float(d["reliability"][0]) if d.height else 0)
        res.append(float(d["resolution"][0]) if d.height else 0)
        auc.append(float(m2["auc"][0]) if m2.height else 0)

    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w/2, rel, w, color="#E5E7EB",
           edgecolor=TEXT, linewidth=0.5, label="Reliability")
    ax.bar(x + w/2, res, w,
           color=[c for _, _, c in models], alpha=0.9,
           edgecolor=TEXT, linewidth=0.5, label="Resolution")
    for xi, (r, s) in enumerate(zip(rel, res)):
        ax.text(xi - w/2, r + 0.001, f"{r:.4f}", ha="center", va="bottom",
                fontsize=8.5, color=TEXT)
        ax.text(xi + w/2, s + 0.001, f"{s:.4f}", ha="center", va="bottom",
                fontsize=8.5, color=TEXT, fontweight=600)

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl, _ in models], fontsize=9.5)
    ax.set_ylabel("Componente del Brier score", fontsize=_AXLABEL_SZ)
    ax.set_title("Descomposición de Brier", fontsize=_TITLE_SIZE,
                 fontweight="bold", pad=12, loc="center")
    ax.legend(loc="upper left", fontsize=_LEGEND_SZ, framealpha=0.95,
              edgecolor=GRID, fancybox=False)
    _style(ax)

    # Footer AUC
    aucs_str = "    ".join([
        f"{lbl.replace(chr(10), ' ')}: AUC = {a:.3f}"
        for (_, lbl, _), a in zip(models, auc)
    ])
    fig.text(0.5, -0.02, aucs_str, ha="center", va="top",
             color=TEXT, fontsize=_FOOTER_SZ, fontweight=500)

    fig.tight_layout()
    _logo_tr(fig)
    if save_path:
        _savefig(fig, save_path)
    return fig


# ============================================================================
# FIG 2: CATE heterogeneity
# ============================================================================

def cate_heterogeneity(save_path=None):
    """Densidad del CATE individual sobre 598 jugadores por canal y shock."""
    df = pl.read_parquet(_CATE / "posterior_player.parquet")

    shocks = [
        ("GOAL_AGAINST", "Tras encajar", ATT),
        ("GOAL_FOR",     "Tras marcar",  DEF),
        ("PRESSURE",     "Presión alta", _PURPLE),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), facecolor=BG, sharey=False)

    for ci, ch in enumerate(_CHANNELS):
        ax = axes[ci]
        for sh, lbl, color in shocks:
            sub = df.filter((pl.col("channel") == ch) & (pl.col("shock_type") == sh))
            if sub.height == 0:
                continue
            vals = sub["cate_mean"].to_numpy()
            sd = vals.std()
            bw = 1.06 * sd * (len(vals) ** (-1/5))
            xs = np.linspace(vals.min() - 2*bw, vals.max() + 2*bw, 300)
            density = np.zeros_like(xs)
            for v in vals:
                density += np.exp(-0.5 * ((xs - v) / bw) ** 2)
            density /= (len(vals) * bw * np.sqrt(2 * np.pi))
            ax.fill_between(xs, density, color=color, alpha=0.22, zorder=2)
            ax.plot(xs, density, color=color, lw=1.9, label=lbl, zorder=3)

        ax.axvline(0, color=LEGEND, lw=0.9, ls=(0, (3, 3)), alpha=0.6, zorder=1)
        ax.set_title(_CH_LABEL[ch], fontsize=_TITLE_SIZE, fontweight="bold",
                     pad=10, loc="center")
        ax.set_xlabel("CATE individual", fontsize=_AXLABEL_SZ - 0.5)
        if ci == 0:
            ax.set_ylabel("Densidad", fontsize=_AXLABEL_SZ)
            ax.legend(loc="upper right", fontsize=_LEGEND_SZ, framealpha=0.95,
                      edgecolor=GRID, fancybox=False)
        # Anotar σ de GA + GF en esquina
        sd_lines = []
        for sh, _, _ in shocks[:2]:
            sub = df.filter((pl.col("channel") == ch) & (pl.col("shock_type") == sh))
            if sub.height > 0:
                short = "GA" if sh == "GOAL_AGAINST" else "GF"
                sd_lines.append(f"σ {short} = {float(sub['cate_mean'].std()):.3f}")
        if sd_lines:
            ax.text(0.03, 0.97, "\n".join(sd_lines), transform=ax.transAxes,
                    ha="left", va="top", fontsize=9, color=TEXT,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=BG,
                              edgecolor=GRID, linewidth=0.6))
        _style(ax)

    fig.suptitle(
        "Distribución del CATE individual por canal y contexto del shock",
        fontsize=_SUPTITLE_SZ, fontweight="bold", x=0.5, ha="center", y=1.02)
    fig.text(0.5, -0.03,
             "598 jugadores · media posterior por jugador · CATE jerárquico bayesiano multivariate (NUTS HMC)",
             ha="center", va="top", color=LEGEND, fontsize=_FOOTER_SZ)

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    _logo_tr(fig)
    if save_path:
        _savefig(fig, save_path)
    return fig


# ============================================================================
# FIG 3: Window sensitivity
# ============================================================================

def window_sensitivity(save_path=None):
    """ATE a 5 horizontes pre/post (±3, 5, 7, 10, 15 min) por canal y contexto."""
    ws = pl.read_parquet(_VAL / "window_sensitivity.parquet")

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), facecolor=BG)

    for ci, ch in enumerate(_CHANNELS):
        ax = axes[ci]
        for sh, color, marker, label in [
            ("GOAL_AGAINST", ATT, "o", "Tras encajar"),
            ("GOAL_FOR",     DEF, "s", "Tras marcar"),
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
            ax.plot(x, y, "-", color=color, lw=1.8, marker=marker, ms=7,
                    mec=BG, mew=0.7, alpha=0.95, zorder=3, label=label)

        ax.axvline(10, color=LEGEND, lw=0.7, ls=(0, (2, 2)), alpha=0.6, zorder=1)
        ax.axhline(0, color=LEGEND, lw=0.8, alpha=0.6, zorder=1)
        ax.set_xticks([3, 5, 7, 10, 15])
        ax.set_xlabel("Ventana ± min", fontsize=_AXLABEL_SZ - 0.5)
        if ci == 0:
            ax.set_ylabel("ATE  ·  desv. estándar canal", fontsize=_AXLABEL_SZ - 0.5)
            ax.legend(loc="best", fontsize=_LEGEND_SZ, framealpha=0.95,
                      edgecolor=GRID, fancybox=False)
        ax.set_title(_CH_LABEL[ch], fontsize=_TITLE_SIZE, fontweight="bold",
                     pad=10, loc="center")
        _style(ax)

    fig.suptitle("Sensibilidad del ATE al horizonte de ventana pre/post",
                 fontsize=_SUPTITLE_SZ, fontweight="bold", x=0.5, ha="center", y=1.02)
    fig.text(0.5, -0.04,
             "Banda = IC 95% clusterizado por jugador  ·  Línea punteada vertical = ventana de referencia ± 10 min",
             ha="center", va="top", color=LEGEND, fontsize=_FOOTER_SZ)

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    _logo_tr(fig)
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

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), facecolor=BG, sharey=True)

    shocks = [("GOAL_AGAINST", "Tras encajar", ATT),
              ("GOAL_FOR",     "Tras marcar",  DEF)]
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
                ax.set_title(_CH_LABEL[ch], fontsize=_TITLE_SIZE - 0.5,
                              fontweight="bold", pad=10, loc="center")
            if ri == 1:
                ax.set_xlabel("ATE robustecido", fontsize=_AXLABEL_SZ - 0.5, labelpad=4)
            if ci == 0:
                ax.set_ylabel(row_lbl, fontsize=_AXLABEL_SZ, fontweight=600, labelpad=8)
            _style(ax, ygrid=False, xgrid=True)

    fig.suptitle("Sensibilidad HonestDiD — cotas frente a tendencias paralelas",
                 fontsize=_SUPTITLE_SZ, fontweight="bold", x=0.5, ha="center", y=0.98)
    fig.text(0.5, -0.01,
             "Banda tenue = IC 95% original  ·  Barra negra = cota robustecida para cada nivel M de relajación",
             ha="center", va="top", color=LEGEND, fontsize=_FOOTER_SZ)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    _logo_tr(fig)
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
