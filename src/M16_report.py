"""M16_report - Informes individuales scout-facing + limitations autogenerado.

Consume `outputs/pcj_table.parquet` + `data/parquet/derived/cate/posterior_player.parquet`
+ samples NUTS de M14 y produce:

  outputs/reports/{player_id}.pdf       una pagina por jugador con minutos minimos
  outputs/figures/                      figuras de la memoria
  outputs/limitations.md                lo que la metrica NO captura (propuesta §Honestidad)

Diseno PDF (1 pagina, leible por DT no estadistico en <2 min):

  +--------------------------------------------------------------+
  | TARJETA  | NOMBRE  POSICION  EQUIPO  EDAD  MINUTOS  N_SHOCKS  |
  +--------------------------------------------------------------+
  | INDICES  | Remontador  IC80  IC95  P(>0)  Tier_certain        |
  |          | Cerrojo     IC80  IC95  P(>0)  Tier_certain        |
  |          | Pressure    IC80  IC95  P(>0)  Tier_certain        |
  +--------------------------------------------------------------+
  | TABLA 4 canales x 3 contextos = 12 coeficientes con IC80      |
  |   Empuje Ofensivo / Solidez Defensiva / Off-ball / Pulso Fis. |
  |   x  Post-GA  /  Post-GF  /  Pressure                          |
  +--------------------------------------------------------------+
  | RANKING within rol + global                                   |
  | NOTAS HONESTIDAD: shocks vividos, contexto (avg elim_prox,    |
  |   % min KO), limitaciones aplicables al jugador                |
  +--------------------------------------------------------------+

Decisiones:
  - Solo jugadores >= MIN_MINUTES (=270 = 3 partidos) — alineado con M15.
  - Tiers: usa `tier_*_global_certain` (Elite_certain / Top_certain / etc.).
  - IC80 en headlines, IC95 disponible en tabla detallada.
  - Sig flag: Sig_remontador / Sig_cerrojo / Sig_pressure_clutch / Inconclusive.
  - matplotlib + reportlab? Mantengo matplotlib stack (ya en deps); savefig PDF.
  - Layout: matplotlib gridspec 1 pagina A4 portrait.

Uso:
    python M16_report.py                    # genera PDFs para top 30 + selectos
    python M16_report.py --all              # genera para los ~234 con minutes>=270
    python M16_report.py --player 1234      # 1 jugador concreto
    python M16_report.py --limitations-only # solo regenera limitations.md
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl

_REPO = Path(__file__).resolve().parents[1]
_PCJ = _REPO / "outputs" / "pcj_table.parquet"
_POSTERIOR = _REPO / "data" / "parquet" / "derived" / "cate" / "posterior_player.parquet"
_MODEL = _REPO / "data" / "parquet" / "derived" / "cate" / "model" / "cate_nuts.pkl"
_REPORTS_DIR = _REPO / "outputs" / "reports"
_FIGURES_DIR = _REPO / "outputs" / "figures"
_LIMITATIONS = _REPO / "outputs" / "limitations.md"

MIN_MINUTES = 270
SHOCK_TYPES_ORDERED = ("GOAL_AGAINST", "GOAL_FOR", "PRESSURE")
CHANNELS_ORDERED = ("ataque", "defensa", "offball", "fisico")
CHANNEL_LABELS = {
    "ataque":  "Empuje Ofensivo",
    "defensa": "Solidez Defensiva",
    "offball": "Off-ball",
    "fisico":  "Pulso Fisico",
}
SHOCK_LABELS = {
    "GOAL_AGAINST": "Post-GA (Remontador)",
    "GOAL_FOR":     "Post-GF (Cerrojo)",
    "PRESSURE":     "Eliminacion (Pressure)",
}


# ---------------------------------------------------------------------------
# PDF builder via matplotlib (1 pagina A4 portrait)
# ---------------------------------------------------------------------------

def _pcj_row_to_player_dict(pcj: pl.DataFrame, player_id: int) -> dict:
    sub = pcj.filter(pl.col("pff_player_id") == player_id)
    if sub.height == 0:
        raise ValueError(f"player {player_id} no esta en pcj_table")
    return sub.to_dicts()[0]


def _posterior_for_player(post: pl.DataFrame, player_id: int) -> pl.DataFrame:
    return post.filter(pl.col("pff_player_id") == player_id).sort([
        "shock_type", "channel"
    ])


def _format_ci(mean: float, lo: float, hi: float, fmt: str = "+.3f") -> str:
    return f"{mean:{fmt}}  [{lo:{fmt}}, {hi:{fmt}}]"


def _tier_color(tier: str) -> str:
    if not tier or tier == "Inconclusive":
        return "#9aa0a6"
    if "Elite" in tier:
        return "#1a73e8" if "_certain" in tier else "#a8c7fa"
    if "Top" in tier:
        return "#34a853" if "_certain" in tier else "#a8dab5"
    if "Above_avg" in tier:
        return "#fbbc04"
    if "Below_avg" in tier or "Bottom" in tier:
        return "#ea4335"
    return "#5f6368"


def _player_limitations(p: dict) -> list[str]:
    """Notas de honestidad aplicables al jugador concreto."""
    notes = []
    n_total = (p.get("n_shocks_for") or 0) + (p.get("n_shocks_against") or 0)
    if n_total < 5:
        notes.append(f"Pocos shocks vividos ({n_total}); IC anchos por exposicion baja.")
    if (p.get("n_shocks_ko") or 0) == 0:
        notes.append("No jugo fase KO; pressure_response inferido por priors jerarquicos.")
    if p.get("avg_elim_prox_at_shock") and p["avg_elim_prox_at_shock"] < 0.3:
        notes.append("Exposicion baja a contextos de eliminacion (avg elim_prox<0.3).")
    if (p.get("sig_chasing") == "Inconclusive"
        and p.get("sig_protecting") == "Inconclusive"
        and p.get("sig_pressure") == "Inconclusive"):
        notes.append("Tres dimensiones inconclusivas: muestra insuficiente para "
                     "diferenciar del bloque (fenomeno sin captura clutch detectable).")
    return notes


def render_player_pdf(player_id: int, pcj: pl.DataFrame, post: pl.DataFrame,
                       output_dir: Path = _REPORTS_DIR) -> Path:
    """Genera 1 PDF por jugador. Layout A4 portrait."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle

    p = _pcj_row_to_player_dict(pcj, player_id)
    pp = _posterior_for_player(post, player_id)

    fig = plt.figure(figsize=(8.27, 11.69))   # A4 portrait inches
    gs = GridSpec(5, 1, figure=fig,
                   height_ratios=[0.6, 1.5, 3.5, 1.0, 0.7],
                   hspace=0.35, left=0.07, right=0.95, top=0.95, bottom=0.04)

    # 1) Tarjeta
    ax_card = fig.add_subplot(gs[0])
    ax_card.axis("off")
    title = f"{p['player_name']}  —  {p['team_name']}  ({p['position_group']})"
    subtitle = (f"Edad {p.get('age_years', 0):.0f}  |  "
                f"Minutos {p.get('minutes_played', 0)}  |  "
                f"Partidos {p.get('n_matches_played', 0)}  |  "
                f"Shocks: GF={p.get('n_shocks_for', 0)} GA={p.get('n_shocks_against', 0)}  "
                f"|  KO={p.get('n_shocks_ko', 0)}  |  HighElim={p.get('n_elimination_shocks', 0)}")
    ax_card.text(0, 0.7, title, fontsize=16, fontweight="bold")
    ax_card.text(0, 0.25, subtitle, fontsize=9, color="#3c4043")

    # 2) Indices con IC80 y P(idx>0)
    ax_idx = fig.add_subplot(gs[1])
    ax_idx.axis("off")
    ax_idx.set_title("Indices clutch (3 dimensiones, relativo al bloque)", loc="left",
                     fontsize=11, fontweight="bold", pad=8)
    rows = [
        ("Remontador (post-GA)", p["chasing_clutch_idx"],
         p["chasing_clutch_lo80"], p["chasing_clutch_hi80"],
         p["p_chasing_positive"], p.get("tier_chasing_global_certain"),
         p.get("sig_chasing")),
        ("Cerrojo (post-GF)", p["protecting_clutch_idx"],
         p["protecting_clutch_lo80"], p["protecting_clutch_hi80"],
         p["p_protecting_positive"], p.get("tier_protecting_global_certain"),
         p.get("sig_protecting")),
        ("Pressure Response (high elim_prox)",
         p.get("pressure_response_idx") or 0.0,
         p.get("pressure_response_lo80") or 0.0,
         p.get("pressure_response_hi80") or 0.0,
         p.get("p_pressure_clutch_positive") or 0.5,
         p.get("tier_pressure_global"), p.get("sig_pressure")),
    ]
    headers = ["Dimension", "Indice (IC80)", "P(>0)", "Tier", "Significancia"]
    col_x = [0.0, 0.30, 0.55, 0.65, 0.80]
    for i, h in enumerate(headers):
        ax_idx.text(col_x[i], 0.93, h, fontsize=9, fontweight="bold")
    for i, (name, m, lo, hi, p_pos, tier, sig) in enumerate(rows):
        y = 0.7 - i * 0.25
        ax_idx.text(col_x[0], y, name, fontsize=10)
        ax_idx.text(col_x[1], y, _format_ci(m, lo, hi), fontsize=9,
                     family="monospace")
        ax_idx.text(col_x[2], y, f"{p_pos:.2f}", fontsize=9, family="monospace")
        ax_idx.text(col_x[3], y, tier or "-", fontsize=9,
                     color=_tier_color(tier or ""))
        ax_idx.text(col_x[4], y, sig or "-", fontsize=8, color="#5f6368")

    # 3) Tabla 4 canales x 3 contextos
    ax_tab = fig.add_subplot(gs[2])
    ax_tab.axis("off")
    ax_tab.set_title("Coeficientes individuales (relativos al bloque, IC80)",
                      loc="left", fontsize=11, fontweight="bold", pad=8)
    # header
    col_w = 0.22
    x_off = 0.18
    for i, st in enumerate(SHOCK_TYPES_ORDERED):
        ax_tab.text(x_off + i * col_w + col_w/2, 0.97, SHOCK_LABELS[st],
                     fontsize=9, fontweight="bold", ha="center")
    for r, ch in enumerate(CHANNELS_ORDERED):
        y = 0.86 - r * 0.21
        ax_tab.text(0.0, y - 0.02, CHANNEL_LABELS[ch], fontsize=10,
                     fontweight="bold")
        for c, st in enumerate(SHOCK_TYPES_ORDERED):
            cell = pp.filter((pl.col("channel") == ch)
                              & (pl.col("shock_type") == st))
            if cell.height == 0:
                txt = "n/a"
            else:
                cv = cell.row(0, named=True)
                txt = (f"{cv['cate_mean']:+.3f}\n"
                       f"[{cv['ci_lo80']:+.3f}, {cv['ci_hi80']:+.3f}]")
            xpos = x_off + c * col_w + col_w / 2
            ax_tab.text(xpos, y - 0.02, txt, fontsize=8, family="monospace",
                         ha="center", va="top")

    # 4) Ranking + tiers
    ax_rk = fig.add_subplot(gs[3])
    ax_rk.axis("off")
    ax_rk.set_title("Rankings", loc="left", fontsize=11, fontweight="bold", pad=4)
    rkings = [
        ("Remontador  global", p.get("rank_chasing_global")),
        ("Remontador  in-rol", p.get("rank_chasing_in_position")),
        ("Cerrojo  global", p.get("rank_protecting_global")),
        ("Cerrojo  in-rol", p.get("rank_protecting_in_position")),
        ("Pressure  global", p.get("rank_pressure_global")),
        ("Pressure  in-rol", p.get("rank_pressure_in_position")),
    ]
    for i, (lab, r) in enumerate(rkings):
        col, row = i % 3, i // 3
        ax_rk.text(0.05 + col * 0.32, 0.7 - row * 0.4, f"{lab}: ",
                    fontsize=9)
        ax_rk.text(0.05 + col * 0.32 + 0.16, 0.7 - row * 0.4,
                    str(r) if r is not None else "-",
                    fontsize=9, fontweight="bold", color="#1a73e8")

    # 5) Notas honestidad
    ax_notes = fig.add_subplot(gs[4])
    ax_notes.axis("off")
    ax_notes.set_title("Notas de honestidad (lo que el modelo NO captura para este jugador)",
                        loc="left", fontsize=10, fontweight="bold", pad=4,
                        color="#5f6368")
    notes = _player_limitations(p)
    if not notes:
        notes = ["Sin alertas especificas; muestra suficiente, contexto cubierto."]
    for i, n in enumerate(notes):
        ax_notes.text(0.0, 0.6 - i * 0.18, "- " + n, fontsize=8, color="#3c4043",
                       wrap=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{player_id}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# limitations.md autogenerado
# ---------------------------------------------------------------------------

def write_limitations(pcj: pl.DataFrame) -> Path:
    """Genera `outputs/limitations.md` con limitaciones agregadas torneo + flagged jugadores."""
    n_total = pcj.height
    n_kos = int(pcj.filter(pl.col("n_shocks_ko") > 0).height)
    n_low_exposure = int(pcj.filter(
        (pl.col("n_shocks_for") + pl.col("n_shocks_against")) < 5
    ).height)
    n_inconclusive = int(pcj.filter(
        (pl.col("sig_chasing") == "Inconclusive")
        & (pl.col("sig_protecting") == "Inconclusive")
    ).height)
    n_h5_changed = int(pcj.filter(
        pl.col("h5_chasing_tier_changed") | pl.col("h5_protecting_tier_changed")
    ).height) if "h5_chasing_tier_changed" in pcj.columns else 0

    md = f"""# Limitaciones metodologicas — PCJ Mundial Qatar 2022

> Este documento enumera explicitamente lo que la metrica PCJ **NO captura**.
> Generado automaticamente desde `outputs/pcj_table.parquet`. Forma parte
> del compromiso de honestidad de la propuesta (§Honestidad metodologica).

## Cobertura

- Jugadores con minutos minimos (>= {MIN_MINUTES} min): **{n_total}**.
- Jugadores con al menos 1 shock en KO: **{n_kos}** ({100 * n_kos / n_total:.0f}%).
- Jugadores con exposicion baja (< 5 shocks totales): **{n_low_exposure}**
  ({100 * n_low_exposure / n_total:.0f}%) — sus IC son anchos.
- Jugadores con las dos dimensiones (Remontador + Cerrojo) inconclusivas:
  **{n_inconclusive}** ({100 * n_inconclusive / n_total:.0f}%).
- Jugadores que cambian de tier entre ranking ABSOLUTO y RELATIVO al bloque
  (test H5): **{n_h5_changed}** ({100 * n_h5_changed / n_total:.0f}%) —
  evidencia directa de que la metrica relativa **no es redundante** con la
  absoluta.

## Lo que el PCJ NO captura

1. **Outliers explosivos concentrados (1-2 minutos).** El CATE multivariate
   modela respuesta sostenida en ventana ±10 min. Un jugador que aparece en
   una jugada decisiva pero su delta promedio en la ventana es modesto
   queda subestimado. Caso paradigma: Mbappe min 80-81 final WC22 (doblete
   en 2 min) — el modelo lo etiqueta Inconclusive porque la consistencia es
   menor que la concentracion.

2. **Tanda de penaltis.** Excluida del scope causal (deporte distinto). No
   entra en n_shocks ni en CATE. Los IC y rankings son sobre regulacion + ET.

3. **Ventanas fuera de ±10 min.** Reacciones que se materializan minute >10
   tras shock no se capturan. Mitigado por `eta_pressure[i,k] · elim_prox_z`
   que captura clutch sostenido bajo high elim_prox sin ventana fija.

4. **Tarjetas / sustituciones / lesiones como shock.** Solo goles + near-miss
   (M06) cuentan como tratamiento. Tarjetas y subs entran como flags de
   exclusion (`sub_in_window`, `truncated_pre/post`) pero no como shock
   propio.

5. **Comportamiento colectivo del rival.** El bloque LOO es del PROPIO equipo
   del focal. La reaccion del rival queda absorbida en el delta del bloque
   propio, no se atribuye.

6. **Identificacion causal triangulada solo en M12 + M13.** El CATE
   multivariate (M14) hereda los supuestos identificadores del DiD
   within-player (parallel trends en residuo del bloque). HonestDiD M=2
   reportado en `data/parquet/derived/did/honest_did.parquet`.

7. **Granularidad temporal: 1 minuto.** Los moduladores continuos
   (minute_norm, score_diff_post_z, week_idx_norm, leverage_z, elim_prox_z)
   estan a granularidad shock, no segundo. Cambios sub-minuto en contexto
   no entran.

## Limitaciones del canal defensivo (M09)

- **VDEP "like" no fiel a Toda 2022.** M09 reusa la cabeza P(concedes) de
  atomic-VAEP en lugar de entrenar P(recovery) − C·P(attacked) dedicada.
  Equivalente bajo mismo horizonte y corpus, **no es VDEP stricto**. Mejora
  pendiente: cabeza dedicada exPress (Lee et al. 2025) sobre PFF tracking.
- **Asignacion al jugador EJECUTOR de la accion**, no al defensor mas
  cercano frame-level (Maejima 2024). Mejora pendiente: nearest-defender
  attribution via tracking 25Hz.

## Limitaciones de near-miss (M06)

- **Offside milimetrico** detectado via SB 360 freeze-frame (umbral 1.5m);
  fallback proxy `att_x > 110` cuando 360 ausente.
- **GLT-denied via PFF tracking ball.z**: detecta balones que cruzan linea
  de gol fisicamente. Casos extremadamente raros en WC22 (~0-2).

## Reportar al lector

Cuando se cite un coeficiente PCJ, hacerlo **siempre con su IC80 + posterior
probability**, NUNCA solo con el punto-estimado. La incertidumbre forma
parte del output, no es decoracion.
"""
    _LIMITATIONS.parent.mkdir(parents=True, exist_ok=True)
    _LIMITATIONS.write_text(md)
    return _LIMITATIONS


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Genera PDFs para todos")
    parser.add_argument("--player", type=int, default=None, help="player_id concreto")
    parser.add_argument("--top", type=int, default=30,
                         help="Genera PDFs para los TOP-N (default 30)")
    parser.add_argument("--limitations-only", action="store_true",
                         help="Solo regenera outputs/limitations.md")
    args = parser.parse_args()

    if not _PCJ.exists():
        print(f"FALTA: {_PCJ}. Ejecuta `python src/M15_pcj.py` primero.")
        sys.exit(1)
    pcj = pl.read_parquet(_PCJ)
    if not _POSTERIOR.exists():
        print(f"FALTA: {_POSTERIOR}. Ejecuta M14 compute_all primero.")
        sys.exit(1)
    post = pl.read_parquet(_POSTERIOR)

    print(f"[M16] PCJ table: {pcj.height} jugadores; posterior: {post.height} filas")

    print("[M16] Generando outputs/limitations.md...")
    p = write_limitations(pcj)
    print(f"  -> {p}")

    if args.limitations_only:
        return

    if args.player is not None:
        out = render_player_pdf(args.player, pcj, post)
        print(f"  PDF -> {out}")
        return

    if args.all:
        targets = pcj["pff_player_id"].to_list()
    else:
        # TOP por dual-clutch + algunos selectos significativos
        top_chasing = pcj.sort("chasing_clutch_idx", descending=True).head(args.top)
        top_protecting = pcj.sort("protecting_clutch_idx", descending=True).head(args.top)
        top_pressure = pcj.filter(pl.col("pressure_response_idx").is_not_null()) \
                          .sort("pressure_response_idx", descending=True).head(args.top)
        targets = list(set(
            top_chasing["pff_player_id"].to_list()
            + top_protecting["pff_player_id"].to_list()
            + top_pressure["pff_player_id"].to_list()
        ))

    print(f"[M16] Generando {len(targets)} PDFs en {_REPORTS_DIR}...")
    for i, pid in enumerate(targets):
        try:
            render_player_pdf(int(pid), pcj, post)
        except Exception as e:
            print(f"  skip {pid}: {e}")
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(targets)} PDFs OK")
    print(f"[M16] DONE")


if __name__ == "__main__":
    main()
