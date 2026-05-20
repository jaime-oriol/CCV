"""M15_pcj — Perfil Clutch del Jugador: ensamblaje scout-facing final.

Combina outputs del modelo unificado M14 (CATE bayesiano jerarquico con eta_ga,
eta_gf, eta_pressure) + M04/M06/M08-M13 + metadata para producir la tabla
maestra `outputs/pcj_table.parquet`.

3 dimensiones de clutch identificadas:
  1. **Remontador** (chasing-clutch) — eta_ga[atk] + eta_ga[off] post-GA
     captura "el que sube cuando hay que remontar"
  2. **Cerrojo** (protecting-clutch) — eta_gf[def] + eta_gf[phys] post-GF
     captura "el que aguanta cuando hay que aguantar"
  3. **Pressure Response** — eta_pressure[i,:] (pendiente individual respecto
     a elim_prox_z) — captura "el que aparece cuando estamos a punto de ser
     eliminados". 3a eta del modelo M14 unificado junto a eta_ga + eta_gf.

Decisiones de diseno:
  - Dual threshold no-arbitrario:
      * Clutch CATE rankings: n_shocks_total >= 3 (identificabilidad random
        effect jerarquico Bayes; <3 obs el shrinkage colapsa al prior posicion)
      * Per-90 fisicas: minutes_played >= 90 (estabilidad ratios per-90;
        <90 min el HSR_per90 / sprints_per90 es ruido)
      * Flag low_sample = (n_shocks<8 OR minutes<270) para CI anchos
    Razonamiento: el modelo Bayesiano YA aplica shrinkage automatico segun N.
    Minutes no es el sample size estadistico, los shocks lo son.
  - 8 CATEs preservados (4 canales × 2 shocks) — chasing vs protecting
    son fenomenos distintos, agregar cancelaria signos
  - Vector PCJ summary 4-canal directional:
      pcj_atk = cate_atk_GA, pcj_def = cate_def_GF,
      pcj_off = cate_off_GA, pcj_phys = max(|GA|,|GF|) signed
  - Tier labels percentile-based (global + within-position)
  - Significance flag bayesiano: P(idx>0|data) > 0.95 → Sig_clutch
    (lo que diferencia esto de Wyscout/InStat: ellos no tienen IC posterior)
  - Tier_certain = Elite/Top SOLO si IC80 excluye 0 (robust al ruido)

Outputs:
  outputs/pcj_table.parquet         (1 fila por jugador, ~277 cols)
  outputs/pcj_aux/top10_{chasing,protecting,pressure}_per_position.parquet
  outputs/pcj_aux/top10_{chasing,protecting,pressure}_per_bucket.parquet
  outputs/pcj_aux/dual_clutch_top.parquet
  outputs/pcj_aux/by_team.parquet

Uso:
    python M15_pcj.py [overwrite]
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import polars as pl

_REPO = Path(__file__).resolve().parents[1]
_CATE_DIR    = _REPO / "data" / "parquet" / "derived" / "cate"
_DID_DIR     = _REPO / "data" / "parquet" / "derived" / "did"
_AIPW_DIR    = _REPO / "data" / "parquet" / "derived" / "aipw"
_DIDV_DIR    = _REPO / "data" / "parquet" / "derived" / "did_validation"
_WP          = _REPO / "data" / "parquet" / "derived" / "wp" / "per_minute.parquet"
_NM          = _REPO / "data" / "parquet" / "derived" / "nearmiss" / "nearmiss_table.parquet"
_SHOCKS      = _REPO / "data" / "parquet" / "derived" / "shocks" / "shocks_table.parquet"
_PFF_GRADES  = _REPO / "data" / "parquet" / "derived" / "preprocess" / "pff_grades.parquet"
_OBSO        = _REPO / "data" / "parquet" / "derived" / "offball" / "per_minute.parquet"
_FISICO_PM   = _REPO / "data" / "parquet" / "derived" / "fisico" / "per_minute.parquet"
_ATAQUE_PM   = _REPO / "data" / "parquet" / "derived" / "ataque" / "per_minute.parquet"
_DEFENSA_PM  = _REPO / "data" / "parquet" / "derived" / "defensa" / "per_minute.parquet"
_PLAYERS_CSV = _REPO / "data_mundial" / "players.csv"
_OUT_DIR     = _REPO / "outputs"
_AUX_DIR     = _OUT_DIR / "pcj_aux"

MIN_SHOCKS = 3                # identificabilidad random effect jerarquico
MIN_MINUTES_FOR_PER90 = 90    # estabilidad ratios per-90 fisicas (1 partido)
LOW_SAMPLE_SHOCKS = 8         # bajo este N, flag low_sample (CI ancho)
LOW_SAMPLE_MINUTES = 270      # bajo este min, flag low_sample (sample limitado)
# Probability of direction (Makowski 2019). 0.85 = umbral principal Sig
# scout-friendly (90%+ ya muy conservador con N=172 shocks WC22). 0.95 =
# bonus tier "strong" para los pocos jugadores que SI alcanzan ese nivel
# de certeza posterior — dual flag para reportar maximo nivel de confianza.
SIG_THRESHOLD = 0.85
SIG_STRONG_THRESHOLD = 0.95
# Acute window +-5min: M12B window_sensitivity mostro decay 7x de w3 a w10
# en fisico/offball; +-10 dilute el efecto, +-5 captura el numero scout.
ACUTE_WINDOW = 5
HIGH_LEVERAGE_THRESHOLD = 0.05   # M04 leverage > 0.05 = "high stress" shock

# Mapping 16 labels PFF -> 4 buckets scout-friendly (GK / DEF / MED / ATA).
# Buckets dan sample size suficiente (~50-100 jug/bucket) para rankings
# significativos vs los 16 granulares (10-30 por label, ranking ruidoso).
# AM (attacking mid / "10") va a ATA por perfil ofensivo (Griezmann, Neymar),
# LWB/RWB van a DEF (carrileros tacticos defensivos aunque suban).
POS_BUCKET_MAP: dict[str, str] = {
    "GK":  "GK",
    "LCB": "DEF", "MCB": "DEF", "RCB": "DEF",
    "LB":  "DEF", "RB":  "DEF", "LWB": "DEF", "RWB": "DEF",
    "DM":  "MED", "CM":  "MED", "LM":  "MED", "RM":  "MED",
    "AM":  "ATA", "LW":  "ATA", "RW":  "ATA", "CF":  "ATA",
}


# ----------------------------------------------------------------------------
# Schema contract estable de la tabla PCJ
# ----------------------------------------------------------------------------
# Campos requeridos en pcj_table.parquet. Si falta alguno, M15 falla con
# mensaje claro antes de persistir — protege a downstream consumers
# (notebooks scout, exports, etc.) de schemas rotos.
PCJ_REQUIRED_COLS: dict[str, type] = {
    # Identidad (6)
    "pff_player_id": pl.Int64, "player_name": pl.String,
    "team_name": pl.String, "position_group": pl.String,
    "position_bucket": pl.String,
    "age_years": pl.Float64,
    # Exposicion (10)
    "minutes_played": pl.Int64, "n_matches_played": pl.UInt32,
    "n_shocks_for": pl.UInt32, "n_shocks_against": pl.UInt32,
    "n_shocks_groups": pl.UInt32, "n_shocks_ko": pl.UInt32,
    "n_high_leverage_shocks": pl.UInt32,
    "avg_leverage_at_shock": pl.Float64,
    "n_elimination_shocks": pl.UInt32,
    "avg_elim_prox_at_shock": pl.Float64,
    "n_nearmiss_exposure": pl.Float64,
    # Indices (8)
    "chasing_clutch_idx": pl.Float64, "chasing_clutch_lo80": pl.Float64,
    "chasing_clutch_hi80": pl.Float64,
    "protecting_clutch_idx": pl.Float64, "protecting_clutch_lo80": pl.Float64,
    "protecting_clutch_hi80": pl.Float64,
    # Posterior probs (3)
    "p_chasing_positive": pl.Float64, "p_protecting_positive": pl.Float64,
    "p_dual_positive": pl.Float64,
    # PCJ 4-vec summary (4)
    "pcj_atk": pl.Float64, "pcj_def": pl.Float64,
    "pcj_off": pl.Float64, "pcj_phys": pl.Float64,
    # Rankings (6) global + within-position (16) + within-bucket (4)
    "rank_chasing_global": pl.UInt32, "rank_protecting_global": pl.UInt32,
    "rank_chasing_in_position": pl.UInt32,
    "rank_protecting_in_position": pl.UInt32,
    "rank_chasing_in_bucket": pl.UInt32,
    "rank_protecting_in_bucket": pl.UInt32,
    # Tier + Sig (4)
    "tier_chasing_global": pl.String, "tier_protecting_global": pl.String,
    "sig_chasing": pl.String, "sig_protecting": pl.String,
    # Pressure response 3a dimension (eta_pressure de M14 unificado)
    "pressure_response_idx": pl.Float64,
    "pressure_response_lo80": pl.Float64,
    "pressure_response_hi80": pl.Float64,
    "p_pressure_clutch_positive": pl.Float64,
    "sig_pressure": pl.String,
    "tier_pressure_global": pl.String,
    # H5: ranking ABSOLUTO paralelo (test "varios cambian de tier abs vs rel")
    "chasing_clutch_idx_absolute":   pl.Float64,
    "protecting_clutch_idx_absolute": pl.Float64,
    "h5_chasing_tier_changed":   pl.Boolean,
    "h5_protecting_tier_changed": pl.Boolean,
    # Escenarios contextualizados: 4 canales x 2 shocks x 2 directions = 16
    # cells, cada uno con mean + ppos (los stats {sd,lo80,hi80} tambien estan
    # pero no se validan aqui — se validan los principales).
    **{f"clutch_{ch}_{sh}_{sc}_{stat}": pl.Float64
       for ch in ("atk", "def", "off", "phys")
       for sh in ("GOAL_AGAINST", "GOAL_FOR")
       for sc in ("team_attacks", "team_defends")
       for stat in ("mean", "ppos")},
}


def validate_pcj_schema(df: pl.DataFrame) -> None:
    """Falla con mensaje claro si faltan cols requeridas o tipos no coinciden.

    Garantia de estabilidad del contrato pcj_table.parquet. Llamada
    obligatoria antes de write_parquet en main().
    """
    missing = [c for c in PCJ_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"PCJ schema FALTAN cols: {missing}")
    bad_types = []
    for c, expected in PCJ_REQUIRED_COLS.items():
        actual = df[c].dtype
        # Permitimos UInt32/Int64 intercambiables y Float32/Float64 intercambiables
        if expected in (pl.UInt32, pl.Int64) and actual in (pl.UInt32, pl.Int64,
                                                              pl.UInt64, pl.Int32):
            continue
        if expected in (pl.Float32, pl.Float64) and actual in (pl.Float32, pl.Float64):
            continue
        if expected == pl.String and actual == pl.String:
            continue
        if actual != expected:
            bad_types.append(f"{c}: expected {expected} got {actual}")
    if bad_types:
        raise TypeError(f"PCJ schema TYPES wrong: {bad_types}")

CHANNELS = ["ataque", "defensa", "offball", "fisico"]
SHOCK_TYPES = ["GOAL_AGAINST", "GOAL_FOR"]   # GA=0, GF=1 (M14 sort order)
PER_MIN_PATHS = {"ataque": _ATAQUE_PM, "defensa": _DEFENSA_PM,
                 "offball": _OBSO,    "fisico": _FISICO_PM}
PER_MIN_OUTCOL = {"ataque": "score_atk_v2_minute", "defensa": "score_def_v4_minute",
                  "offball": "c_obso_mean",     "fisico": "score_phys"}


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def _load_m14() -> dict:
    """Lee outputs M14 (parquets). YA NO necesita cate_nuts.pkl: M14 ahora
    vuelca a parquet los 4 derivados del posterior que antes se computaban
    aqui desde samples (ver M14 _dump_all_sample_derivatives).
    """
    posterior        = pl.read_parquet(_CATE_DIR / "posterior_player.parquet")
    indices          = pl.read_parquet(_CATE_DIR / "indices.parquet")
    rankings         = pl.read_parquet(_CATE_DIR / "rankings.parquet")
    posterior_probs  = pl.read_parquet(_CATE_DIR / "posterior_probs.parquet")
    pressure         = pl.read_parquet(_CATE_DIR / "pressure_player.parquet")
    intra_corr       = pl.read_parquet(_CATE_DIR / "intra_corr_player.parquet")
    scenarios_wide   = pl.read_parquet(_CATE_DIR / "scenarios_player.parquet")
    return dict(posterior=posterior, indices=indices, rankings=rankings,
                 posterior_probs=posterior_probs, pressure=pressure,
                 intra_corr=intra_corr, scenarios_wide=scenarios_wide)


def _load_player_meta() -> pl.DataFrame:
    """Identidad: pff_player_id, player_name, position_group, team_name.

    Combina players.csv (nicknames) + pff_grades.parquet (position_group + team
    derivados de rosters).
    """
    players = pl.read_csv(_PLAYERS_CSV).select(
        pl.col("id").alias("pff_player_id"),
        pl.col("nickname").alias("player_name"),
    ).unique("pff_player_id")
    grades = pl.read_parquet(_PFF_GRADES).select(
        ["pff_player_id", "team_name", "position_group"])
    out = players.join(grades, on="pff_player_id", how="inner")
    # 16 labels PFF -> 4 buckets (GK/DEF/MED/ATA). Granular para inferencia
    # M14, agrupado para rankings scout-facing en M15.
    out = out.with_columns(
        pl.col("position_group").replace_strict(POS_BUCKET_MAP, default="UNK")
            .alias("position_bucket")
    )
    return out


def _load_player_age_height() -> pl.DataFrame:
    """Edad (a fecha 2022-11-20 inicio WC) + altura desde players.csv."""
    p = pl.read_csv(_PLAYERS_CSV).select(
        pl.col("id").alias("pff_player_id"),
        pl.col("dob").alias("dob"),
        pl.col("height").alias("height_cm"),
    ).unique("pff_player_id")
    p = p.with_columns([
        pl.col("dob").str.to_date(strict=False).alias("dob_d"),
    ])
    wc_start = pl.lit("2022-11-20").str.to_date()
    p = p.with_columns(
        ((wc_start - pl.col("dob_d")).dt.total_days() / 365.25)
            .alias("age_years")
    ).select(["pff_player_id", "age_years", "height_cm"])
    return p


def _load_channel_credibility() -> pl.DataFrame:
    """Por (channel, shock_type): pre-trend OK + AIPW same-sign + sensitivity robusto.

    Combina M12 diagnostics (pre-trend F-test) + M13 comparison_m12 (cross-val) +
    M13 sensitivity (Cinelli-Hazlett robustness value).
    """
    pre = (pl.read_parquet(_DID_DIR / "diagnostics.parquet")
             .select(["channel", "shock_type", "p_pretrend"]))
    cmp = (pl.read_parquet(_AIPW_DIR / "comparison_m12.parquet")
             .select(["channel", "shock_type", "same_sign", "diff_normalized"]))
    sens = (pl.read_parquet(_AIPW_DIR / "sensitivity.parquet")
             .rename({"perspective": "shock_type"})
             .select(["channel", "shock_type", "robustness_value", "interpretation"]))
    cred = pre.join(cmp, on=["channel", "shock_type"], how="left") \
              .join(sens, on=["channel", "shock_type"], how="left")
    cred = cred.with_columns([
        (pl.col("p_pretrend") > 0.05).alias("pretrend_clean"),
        (pl.col("p_pretrend") > 0.05).cast(pl.Int8).alias("c1"),
        (pl.col("interpretation") == "robusto").cast(pl.Int8).alias("c2"),
        pl.col("same_sign").fill_null(False).cast(pl.Int8).alias("c3"),
    ]).with_columns(
        (pl.col("c1") + pl.col("c2") + pl.col("c3")).alias("credibility_score")
    ).with_columns(
        pl.when(pl.col("credibility_score") >= 3).then(pl.lit("HIGH"))
          .when(pl.col("credibility_score") == 2).then(pl.lit("MED"))
          .otherwise(pl.lit("LOW"))
          .alias("channel_credibility")
    ).select(["channel", "shock_type", "p_pretrend", "same_sign",
              "robustness_value", "interpretation",
              "credibility_score", "channel_credibility"])
    return cred


def _load_power_flags() -> pl.DataFrame:
    """Per (channel, shock_type) flag de poder estadistico desde M12B power_analysis."""
    pwr = pl.read_parquet(_DIDV_DIR / "power_analysis.parquet").select(
        ["channel", "shock_type", "power_observed", "n_effective"])
    return pwr.with_columns(
        pl.when(pl.col("power_observed") >= 0.80).then(pl.lit("WELL_POWERED"))
          .when(pl.col("power_observed") >= 0.50).then(pl.lit("MARGINAL"))
          .otherwise(pl.lit("UNDERPOWERED"))
          .alias("power_flag")
    )


def _load_player_minutes() -> pl.DataFrame:
    """Suma minutos por jugador a lo largo de los 64 partidos WC22."""
    sys.path.insert(0, str(_REPO / "src"))
    from M01_loader_pff import list_event_match_ids
    from M03_preprocess import player_minutes
    rows = []
    for mid in list_event_match_ids():
        pm = player_minutes(mid)
        rows.append(pm.select(["player_id", "minutes_played"]))
    pm_all = pl.concat(rows)
    return (pm_all.group_by("player_id").agg([
                pl.col("minutes_played").sum().alias("minutes_played"),
                pl.col("minutes_played").gt(0).sum().alias("n_matches_played"),
            ]).rename({"player_id": "pff_player_id"}))


def _load_leverage_exposure() -> pl.DataFrame:
    """Per jugador: leverage + elim_prox M04 en los minutos cuando vivio shocks.

    leverage = sensibilidad WP a un gol mas (pivotal moment).
    elim_prox = P(equipo NO clasifica) — "cerca de irnos a casa", propuesta_final.md:27.
    Shocks bajo high-leverage / high-elim_prox son los REALMENTE clutch.
    """
    sh = pl.read_parquet(_SHOCKS).rename({"player_id": "pff_player_id",
                                          "match_id": "pff_match_id"})
    return (sh.group_by("pff_player_id").agg([
        pl.col("leverage_at_shock").mean().alias("avg_leverage_at_shock"),
        pl.col("leverage_at_shock").max().alias("max_leverage_at_shock"),
        (pl.col("leverage_at_shock") > HIGH_LEVERAGE_THRESHOLD).sum()
            .alias("n_high_leverage_shocks"),
        # Elim prox del equipo del jugador (perspectiva real)
        pl.col("elim_prox_player_at_shock").mean()
            .alias("avg_elim_prox_at_shock"),
        pl.col("elim_prox_player_at_shock").max()
            .alias("max_elim_prox_at_shock"),
        (pl.col("elim_prox_player_at_shock") > 0.7).sum()
            .alias("n_elimination_shocks"),
    ]))


def _load_nearmiss_exposure() -> pl.DataFrame:
    """Per jugador: total near-miss eventos vividos en campo (cualquier equipo).

    Nota: M06 nearmiss_table usa SB team_id, M03 player_minutes usa PFF team_id
    (universos distintos). En vez de mapear SB→PFF a nivel team, contamos
    exposición total: cuantos near-miss eventos ocurrieron mientras el jugador
    estaba en campo en su partido. Mide "stress por casi-gol".
    """
    nm = pl.read_parquet(_NM).select(
        pl.col("sb_match_id"), pl.col("minute"))
    sys.path.insert(0, str(_REPO / "src"))
    from M03_preprocess import player_minutes, pff_to_sb_match_id
    from M01_loader_pff import list_event_match_ids
    pff2sb = pff_to_sb_match_id()
    rows = []
    for pff_mid in list_event_match_ids():
        sb_mid = pff2sb.get(int(pff_mid))
        if sb_mid is None:
            continue
        nm_match = nm.filter(pl.col("sb_match_id") == sb_mid)
        if nm_match.height == 0:
            continue
        pm = player_minutes(pff_mid)
        for nm_row in nm_match.iter_rows(named=True):
            on_field = pm.filter(
                pl.col("minute_in").is_not_null() &
                (pl.col("minute_in") <= nm_row["minute"]) &
                (pl.col("minute_out") >= nm_row["minute"]))
            for r in on_field.iter_rows(named=True):
                rows.append(int(r["player_id"]))
    if not rows:
        return pl.DataFrame(schema={"pff_player_id": pl.Int64,
                                     "n_nearmiss_exposure": pl.UInt32})
    df = pl.DataFrame({"pff_player_id": rows})
    return df.group_by("pff_player_id").agg(pl.len().alias("n_nearmiss_exposure"))


def _load_physical_per90() -> pl.DataFrame:
    """Metricas fisicas Bradley 2024 per-90 desde tracking PFF 25Hz.

    Agrega M11 raw_per_minute por jugador y devuelve totals + peak speed.
    La normalizacion per-90 se hace en build_pcj_table tras join con minutes.
    """
    raw = pl.read_parquet(_FISICO_PM.parent / "raw_per_minute.parquet")
    return (raw.group_by("pff_player_id").agg([
        pl.col("distance_m").sum().alias("_phys_dist_total_m"),
        pl.col("hsr_s").sum().alias("_phys_hsr_s_total"),
        pl.col("sprint_count").sum().alias("_phys_sprints_total"),
        pl.col("n_high_accel").sum().alias("_phys_accels_total"),
        pl.col("hmld_m").sum().alias("_phys_hmld_total_m"),
        pl.col("psv95").max().alias("physical_peak_speed_mps"),
    ]))


def _load_baseline_channels() -> pl.DataFrame:
    """Baseline absoluto per jugador outside shock windows (M10 obso, M11 score_phys,
    M08 score_atk, M09 score_def). Permite distinguir 'PCJ_off bajo porque baseline
    elite ya' vs 'PCJ_off bajo porque mediocre'.
    """
    rows = []
    for ch, path in PER_MIN_PATHS.items():
        col = PER_MIN_OUTCOL[ch]
        df = pl.read_parquet(path).select(["pff_player_id", col])
        agg = df.group_by("pff_player_id").agg(
            pl.col(col).mean().alias(f"baseline_{ch}"))
        rows.append(agg)
    out = rows[0]
    for r in rows[1:]:
        out = out.join(r, on="pff_player_id", how="full", coalesce=True)
    return out


def _load_shock_exposure() -> pl.DataFrame:
    """Por jugador: n_shocks_for, n_shocks_against, n_groups, n_ko."""
    sh = pl.read_parquet(_SHOCKS).rename({"player_id": "pff_player_id"})
    return (sh.group_by("pff_player_id").agg([
        pl.col("shock_type").eq("GOAL_FOR").sum().alias("n_shocks_for"),
        pl.col("shock_type").eq("GOAL_AGAINST").sum().alias("n_shocks_against"),
        pl.col("stage").eq("groups").sum().alias("n_shocks_groups"),
        pl.col("stage").eq("ko").sum().alias("n_shocks_ko"),
    ]))


# ----------------------------------------------------------------------------
# Posterior probabilities desde samples
# ----------------------------------------------------------------------------
def _compute_acute_window_per_player(window: int = ACUTE_WINDOW) -> pl.DataFrame:
    """Per (player, channel, shock_type): within-player diff (post-pre) en
    ventana ACUTA +-window min, computed desde per_minute parquets.

    M12B window_sensitivity mostro que los efectos son acutos (decay 7x de
    w3 a w10 en fisico/offball). El M14 a +-10 DILUYE el efecto. Esta col
    da el numero scout-relevante.
    """
    sh = (pl.read_parquet(_SHOCKS).rename({"player_id": "pff_player_id",
                                            "match_id": "pff_match_id"})
          .filter(~pl.col("truncated_pre") & ~pl.col("truncated_post") &
                   ~pl.col("overlap_flag") & ~pl.col("sub_in_window")))
    rows = []
    for ch, path in PER_MIN_PATHS.items():
        col = PER_MIN_OUTCOL[ch]
        pm = pl.read_parquet(path).select(
            ["pff_match_id", "pff_player_id", "period",
             "minute_in_period", col])
        pm = pm.with_columns(
            ((pl.col("period") - 1) * 45 + pl.col("minute_in_period"))
                .alias("minute_global"))
        for sh_type in SHOCK_TYPES:
            sh_sub = sh.filter(pl.col("shock_type") == sh_type)
            # Cross-join per shock con +-window minutes
            rels = pl.DataFrame({"relative_min":
                [r for r in range(-window, window + 1) if r != 0]})
            sh_exp = sh_sub.select(["pff_match_id", "pff_player_id",
                                     "shock_id", "minute"]).join(
                rels, how="cross").with_columns(
                (pl.col("minute") + pl.col("relative_min")).alias("minute_global"))
            joined = sh_exp.join(
                pm.select(["pff_match_id", "pff_player_id",
                           "minute_global", col]),
                on=["pff_match_id", "pff_player_id", "minute_global"],
                how="left").drop_nulls(col)
            if joined.height == 0:
                continue
            # Per (player, shock): mean pre, mean post, diff
            joined = joined.with_columns(
                (pl.col("relative_min") > 0).cast(pl.Int8).alias("post"))
            per_ps = (joined.group_by(["pff_player_id", "shock_id", "post"])
                      .agg(pl.col(col).mean().alias("m"))
                      .sort(["pff_player_id", "shock_id", "post"]))
            wide = per_ps.pivot("post", index=["pff_player_id", "shock_id"],
                                values="m").drop_nulls()
            cols = wide.columns
            if "0" in cols and "1" in cols:
                wide = wide.with_columns((pl.col("1") - pl.col("0")).alias("d"))
            elif 0 in cols and 1 in cols:
                wide = wide.with_columns((pl.col(1) - pl.col(0)).alias("d"))
            else:
                continue
            per_player = wide.group_by("pff_player_id").agg(
                pl.col("d").mean().alias(f"acute_{ch}_{sh_type}"))
            rows.append(per_player)
    if not rows:
        return pl.DataFrame({"pff_player_id": []})
    out = rows[0]
    for r in rows[1:]:
        out = out.join(r, on="pff_player_id", how="full", coalesce=True)
    return out


def _compute_absolute_indices_for_h5(panel_abs: pl.DataFrame) -> pl.DataFrame:
    """Indices ABSOLUTOS paralelos (H5: comparacion absoluto vs relativo).

    panel_abs viene de build_delta_panel(relative=False) — delta = post-pre raw.
    Por jugador, mean del delta absoluto en cada (channel, shock_type) — no
    es bayesiano, es punto-estimado para fines de H5 ranking comparison.

    Returns 1 fila por jugador con cols *_idx_absolute.
    """
    if panel_abs is None or panel_abs.height == 0:
        return pl.DataFrame(schema={"pff_player_id": pl.Int64})
    pivoted = (panel_abs.group_by(["pff_player_id", "channel", "shock_type"])
               .agg(pl.col("delta").mean().alias("d"))
               .pivot(values="d", index="pff_player_id",
                      on=["channel", "shock_type"]))
    cols_map = {}
    for c in pivoted.columns:
        if c == "pff_player_id":
            continue
        # polars pivot crea col names tipo '{"ataque","GOAL_AGAINST"}' en list. simplifica.
        cols_map[c] = c.replace('{', '').replace('}', '').replace('"', '').replace(',', '_')
    pivoted = pivoted.rename(cols_map)
    out = pivoted.with_columns([
        ((pl.col("ataque_GOAL_AGAINST") + pl.col("offball_GOAL_AGAINST")) / 2)
            .alias("chasing_clutch_idx_absolute"),
        ((pl.col("defensa_GOAL_FOR") + pl.col("fisico_GOAL_FOR")) / 2)
            .alias("protecting_clutch_idx_absolute"),
    ])
    return out.select([
        "pff_player_id",
        "chasing_clutch_idx_absolute",
        "protecting_clutch_idx_absolute",
    ])


# ----------------------------------------------------------------------------
# CATEs 8-valores preservados (4 canales × 2 shocks)
# ----------------------------------------------------------------------------
def _build_cate_wide(posterior: pl.DataFrame) -> pl.DataFrame:
    """Pivot posterior_player (long) → wide con 32 cols (8 channels x 4 stats)."""
    rows = []
    for r in posterior.iter_rows(named=True):
        pid = r["pff_player_id"]
        prefix = f"cate_{r['channel']}_{r['shock_type']}"
        rows.append((pid, f"{prefix}_mean", r["cate_mean"]))
        rows.append((pid, f"{prefix}_sd",   r["cate_sd"]))
        rows.append((pid, f"{prefix}_lo80", r["ci_lo80"]))
        rows.append((pid, f"{prefix}_hi80", r["ci_hi80"]))
    long = pl.DataFrame(rows, schema=["pff_player_id", "key", "val"], orient="row")
    return long.pivot("key", index="pff_player_id", values="val")


# ----------------------------------------------------------------------------
# Vector PCJ summary 4-canal directional
# ----------------------------------------------------------------------------
def _build_pcj_summary_vector(cate_wide: pl.DataFrame) -> pl.DataFrame:
    """4-vector directional: cada canal usa shock_type de máxima leverage.

    pcj_atk  = cate_ataque_GOAL_AGAINST_mean  (chasing)
    pcj_def  = cate_defensa_GOAL_FOR_mean    (protecting)
    pcj_off  = cate_offball_GOAL_AGAINST_mean (chasing)
    pcj_phys = cate_fisico con max-magnitude signed (GA o GF, el más reactivo)
    """
    return cate_wide.with_columns([
        pl.col("cate_ataque_GOAL_AGAINST_mean").alias("pcj_atk"),
        pl.col("cate_defensa_GOAL_FOR_mean").alias("pcj_def"),
        pl.col("cate_offball_GOAL_AGAINST_mean").alias("pcj_off"),
        pl.when(pl.col("cate_fisico_GOAL_AGAINST_mean").abs() >=
                pl.col("cate_fisico_GOAL_FOR_mean").abs())
          .then(pl.col("cate_fisico_GOAL_AGAINST_mean"))
          .otherwise(pl.col("cate_fisico_GOAL_FOR_mean"))
          .alias("pcj_phys"),
    ])


# ----------------------------------------------------------------------------
# Tier labels (percentile-based)
# ----------------------------------------------------------------------------
def _tier_from_percentile(pct: float) -> str:
    if pct >= 0.95:  return "Elite"
    if pct >= 0.85:  return "Top"
    if pct >= 0.60:  return "Above_avg"
    if pct >= 0.40:  return "Average"
    if pct >= 0.15:  return "Below_avg"
    return "Bottom"


def _add_tiers(df: pl.DataFrame) -> pl.DataFrame:
    """Tier labels: global + within-position para 3 dimensiones (chasing,
    protecting, pressure_response). Percentile-based.
    """
    n = df.height
    # 4 percentiles core (chasing/protecting × global/in_position)
    df = df.with_columns([
        (pl.col("chasing_clutch_idx").rank(method="ordinal") / n)
            .alias("pct_chasing_global"),
        (pl.col("protecting_clutch_idx").rank(method="ordinal") / n)
            .alias("pct_protecting_global"),
        (pl.col("chasing_clutch_idx").rank(method="ordinal").over("position_group") /
            pl.col("position_group").count().over("position_group"))
            .alias("pct_chasing_in_position"),
        (pl.col("protecting_clutch_idx").rank(method="ordinal").over("position_group") /
            pl.col("position_group").count().over("position_group"))
            .alias("pct_protecting_in_position"),
        (pl.col("chasing_clutch_idx").rank(method="ordinal").over("position_bucket") /
            pl.col("position_bucket").count().over("position_bucket"))
            .alias("pct_chasing_in_bucket"),
        (pl.col("protecting_clutch_idx").rank(method="ordinal").over("position_bucket") /
            pl.col("position_bucket").count().over("position_bucket"))
            .alias("pct_protecting_in_bucket"),
    ])
    # Pressure_response tiers (eta_pressure de M14 unificado)
    if "pressure_response_idx" in df.columns:
        df = df.with_columns([
            (pl.col("pressure_response_idx").rank(method="ordinal") / n)
                .alias("pct_pressure_global"),
            (pl.col("pressure_response_idx").rank(method="ordinal").over("position_group") /
                pl.col("position_group").count().over("position_group"))
                .alias("pct_pressure_in_position"),
            (pl.col("pressure_response_idx").rank(method="ordinal").over("position_bucket") /
                pl.col("position_bucket").count().over("position_bucket"))
                .alias("pct_pressure_in_bucket"),
        ])
    pct_cols = [c for c in df.columns if c.startswith("pct_")]
    for col in pct_cols:
        tier_col = "tier_" + col.replace("pct_", "")
        df = df.with_columns(
            pl.col(col).map_elements(_tier_from_percentile, return_dtype=pl.String)
                       .alias(tier_col)
        )
    return df


def _add_tier_with_uncertainty(df: pl.DataFrame) -> pl.DataFrame:
    """Tier_certain: Elite/Top SOLO si IC80 excluye 0 (signo certero).
    Sino degrade a Elite_uncertain / Top_uncertain. Resto se mantiene.
    """
    def cert_tier(tier, lo, hi, sign):
        if tier in ("Elite", "Top"):
            ic_excludes_0 = (lo > 0 and hi > 0) if sign == "+" else (lo < 0 and hi < 0)
            return tier + ("_certain" if ic_excludes_0 else "_uncertain")
        return tier

    # Map manual via Python rows
    df_pd = df.to_pandas()
    df_pd["tier_chasing_global_certain"] = [
        cert_tier(t, lo, hi, "+")
        for t, lo, hi in zip(df_pd["tier_chasing_global"],
                              df_pd["chasing_clutch_lo80"],
                              df_pd["chasing_clutch_hi80"])
    ]
    df_pd["tier_protecting_global_certain"] = [
        cert_tier(t, lo, hi, "+")
        for t, lo, hi in zip(df_pd["tier_protecting_global"],
                              df_pd["protecting_clutch_lo80"],
                              df_pd["protecting_clutch_hi80"])
    ]
    return pl.from_pandas(df_pd)


def _add_significance(df: pl.DataFrame) -> pl.DataFrame:
    """Dual sig flags por dimension:
       sig_*       : Sig si P(>0)>=SIG_THRESHOLD (0.85), anti si <=0.15.
       sig_*_strong: STRONG si P(>0)>=SIG_STRONG_THRESHOLD (0.95).
    3 dimensiones: Remontador / Cerrojo / Pressure.
    """
    def _sig_expr(p_col: str, name: str, thr: float, anti_label: str | None = None
                   ) -> pl.Expr:
        if anti_label:
            return (pl.when(pl.col(p_col) >= thr).then(pl.lit(f"Sig_{name}"))
                      .when(pl.col(p_col) <= 1 - thr).then(pl.lit(f"Sig_anti_{name}"))
                      .otherwise(pl.lit("Inconclusive")))
        return (pl.when(pl.col(p_col) >= thr).then(pl.lit(f"Sig_{name}_strong"))
                  .when(pl.col(p_col) <= 1 - thr).then(pl.lit(f"Sig_anti_{name}_strong"))
                  .otherwise(pl.lit("Inconclusive")))

    out = df.with_columns([
        _sig_expr("p_chasing_positive",   "remontador", SIG_THRESHOLD, "anti")
            .alias("sig_chasing"),
        _sig_expr("p_chasing_positive",   "remontador", SIG_STRONG_THRESHOLD)
            .alias("sig_chasing_strong"),
        _sig_expr("p_protecting_positive","cerrojo",    SIG_THRESHOLD, "anti")
            .alias("sig_protecting"),
        _sig_expr("p_protecting_positive","cerrojo",    SIG_STRONG_THRESHOLD)
            .alias("sig_protecting_strong"),
    ])
    if "p_pressure_clutch_positive" in df.columns:
        out = out.with_columns([
            _sig_expr("p_pressure_clutch_positive", "pressure_clutch",
                       SIG_THRESHOLD, "anti").alias("sig_pressure"),
            _sig_expr("p_pressure_clutch_positive", "pressure_clutch",
                       SIG_STRONG_THRESHOLD).alias("sig_pressure_strong"),
        ])
    return out


# ----------------------------------------------------------------------------
# Rankings (global + in-position)
# ----------------------------------------------------------------------------
def _add_rankings(df: pl.DataFrame) -> pl.DataFrame:
    """Rankings global + within-position (16 labels) + within-bucket (4 grupos)
    para las 3 dimensiones.
    """
    out = df.with_columns([
        pl.col("chasing_clutch_idx").rank(method="ordinal", descending=True)
            .alias("rank_chasing_global"),
        pl.col("protecting_clutch_idx").rank(method="ordinal", descending=True)
            .alias("rank_protecting_global"),
        pl.col("chasing_clutch_idx").rank(method="ordinal", descending=True)
            .over("position_group").alias("rank_chasing_in_position"),
        pl.col("protecting_clutch_idx").rank(method="ordinal", descending=True)
            .over("position_group").alias("rank_protecting_in_position"),
        pl.col("chasing_clutch_idx").rank(method="ordinal", descending=True)
            .over("position_bucket").alias("rank_chasing_in_bucket"),
        pl.col("protecting_clutch_idx").rank(method="ordinal", descending=True)
            .over("position_bucket").alias("rank_protecting_in_bucket"),
    ])
    if "pressure_response_idx" in df.columns:
        out = out.with_columns([
            pl.col("pressure_response_idx").rank(method="ordinal", descending=True)
                .alias("rank_pressure_global"),
            pl.col("pressure_response_idx").rank(method="ordinal", descending=True)
                .over("position_group").alias("rank_pressure_in_position"),
            pl.col("pressure_response_idx").rank(method="ordinal", descending=True)
                .over("position_bucket").alias("rank_pressure_in_bucket"),
        ])
    return out


# ----------------------------------------------------------------------------
# Build maestro
# ----------------------------------------------------------------------------
def build_pcj_table() -> pl.DataFrame:
    print("[M15] Cargando M14 outputs (parquets, sin pkl)...")
    m14 = _load_m14()
    posterior = m14["posterior"]
    posterior_probs = m14["posterior_probs"]
    scenarios_wide  = m14["scenarios_wide"]
    intra           = m14["intra_corr"]
    pressure        = m14["pressure"]
    print(f"  posterior_probs: {posterior_probs.height} jugadores")
    print(f"  scenarios_wide:  {scenarios_wide.height} jugadores, "
          f"{scenarios_wide.width-1} cols clutch_*")
    print(f"  pressure:        {pressure.height} jugadores con pressure_response")

    print("[M15] Cargando metadata jugadores...")
    meta = _load_player_meta()
    print(f"  meta: {meta.height} jugadores con identidad completa")

    print("[M15] Calculando minutos jugados (sumando 64 partidos)...")
    minutes = _load_player_minutes()
    print(f"  minutes: {minutes.height} jugadores")

    print("[M15] Cargando exposicion shocks...")
    shocks = _load_shock_exposure()

    print("[M15] Construyendo CATE wide (5 perspectivas x 4 canales x 4 stats)...")
    cate_wide = _build_cate_wide(posterior)
    print(f"  cate_wide: {cate_wide.height} jugadores, {cate_wide.width} cols")

    print("[M15] Vector PCJ summary 4-canal directional...")
    cate_wide = _build_pcj_summary_vector(cate_wide)

    print("[M15] Acute window CATE +-5 min per player (M12B window_sensitivity)...")
    acute = _compute_acute_window_per_player(window=ACUTE_WINDOW)
    print(f"  acute: {acute.height} jugadores con acute deltas")

    print("[M15] Channel credibility (M12 pre-trend + M13 AIPW + sensitivity)...")
    cred = _load_channel_credibility()

    print("[M15] Power flags per channel desde M12B power_analysis...")
    pwr = _load_power_flags()

    print("[M15] Player metadata (age, height) desde players.csv...")
    age_h = _load_player_age_height()

    print("[M15] Leverage exposure desde M04 WP...")
    lev = _load_leverage_exposure()

    print("[M15] Near-miss exposure desde M06...")
    nm = _load_nearmiss_exposure()

    print("[M15] Baseline absoluto canales (M08-M11 outside windows)...")
    baselines = _load_baseline_channels()

    print("[M15] Metricas fisicas per-90 Bradley 2024 (tracking PFF 25Hz)...")
    physical = _load_physical_per90()

    print("[M15] Indices ABSOLUTOS paralelos (H5: ranking absoluto vs relativo)...")
    sys.path.insert(0, str(_REPO / "src"))
    from M14_cate import build_delta_panel as _bdp
    panel_abs = _bdp(cache=True, relative=False)
    abs_idx = _compute_absolute_indices_for_h5(panel_abs)
    print(f"  abs_idx: {abs_idx.height} jugadores con absolutos para H5")

    print("[M15] Joining + filtrando minutos minimos...")
    df = (cate_wide
            .join(posterior_probs,  on="pff_player_id", how="inner")
            .join(scenarios_wide,   on="pff_player_id", how="left")
            .join(meta,             on="pff_player_id", how="left")
            .join(minutes,          on="pff_player_id", how="left")
            .join(shocks,           on="pff_player_id", how="left")
            .join(acute,            on="pff_player_id", how="left")
            .join(intra,            on="pff_player_id", how="left")
            .join(age_h,            on="pff_player_id", how="left")
            .join(lev,              on="pff_player_id", how="left")
            .join(nm,               on="pff_player_id", how="left")
            .join(baselines,        on="pff_player_id", how="left")
            .join(physical,         on="pff_player_id", how="left")
            .join(pressure,         on="pff_player_id", how="left")
            .join(abs_idx,          on="pff_player_id", how="left"))
    n_total = df.height
    # n_shocks_total = exposicion al canal causal (sample size del modelo Bayes)
    df = df.with_columns(
        (pl.col("n_shocks_for").fill_null(0) + pl.col("n_shocks_against").fill_null(0))
            .cast(pl.UInt32).alias("n_shocks_total")
    )
    df = df.filter(pl.col("n_shocks_total") >= MIN_SHOCKS)
    print(f"  {df.height}/{n_total} jugadores con n_shocks_total>={MIN_SHOCKS}")
    # Flag low_sample: CI anchos / per-90 fisicas no fiables. Scout-facing
    # advertencia, no filtra.
    df = df.with_columns(
        ((pl.col("n_shocks_total") < LOW_SAMPLE_SHOCKS) |
         (pl.col("minutes_played") < LOW_SAMPLE_MINUTES))
            .alias("low_sample")
    )

    # Compute physical per-90 desde totals + minutes_played. NULL si <90 min
    # (per-90 ratios inestables con sample chico).
    _ok = pl.col("minutes_played") >= MIN_MINUTES_FOR_PER90
    df = df.with_columns([
        pl.when(_ok)
          .then(pl.col("_phys_dist_total_m") / 1000 / pl.col("minutes_played") * 90)
          .otherwise(None).alias("physical_distance_km_per90"),
        pl.when(_ok)
          .then(pl.col("_phys_hsr_s_total") * 5.5 / pl.col("minutes_played") * 90)
          .otherwise(None).alias("physical_hsr_m_per90"),    # 5.5 m/s = HSR threshold
        pl.when(_ok)
          .then(pl.col("_phys_sprints_total") / pl.col("minutes_played") * 90)
          .otherwise(None).alias("physical_sprints_per90"),
        pl.when(_ok)
          .then(pl.col("_phys_accels_total") / pl.col("minutes_played") * 90)
          .otherwise(None).alias("physical_high_accels_per90"),
        pl.when(_ok)
          .then(pl.col("_phys_hmld_total_m") / pl.col("minutes_played") * 90)
          .otherwise(None).alias("physical_hmld_m_per90"),
    ]).drop([c for c in df.columns if c.startswith("_phys_")])

    print("[M15] Rankings + tiers + sig flags + uncertainty + channel credibility wide...")
    df = _add_rankings(df)
    df = _add_tiers(df)
    df = _add_tier_with_uncertainty(df)
    df = _add_significance(df)

    # H5: tier_changed entre absoluto y relativo (test propuesta_final §H5)
    n = df.height
    df = df.with_columns([
        (pl.col("chasing_clutch_idx_absolute").rank(method="ordinal") / n)
            .map_elements(_tier_from_percentile, return_dtype=pl.String)
            .alias("tier_chasing_global_absolute"),
        (pl.col("protecting_clutch_idx_absolute").rank(method="ordinal") / n)
            .map_elements(_tier_from_percentile, return_dtype=pl.String)
            .alias("tier_protecting_global_absolute"),
    ]).with_columns([
        (pl.col("tier_chasing_global") != pl.col("tier_chasing_global_absolute"))
            .alias("h5_chasing_tier_changed"),
        (pl.col("tier_protecting_global") != pl.col("tier_protecting_global_absolute"))
            .alias("h5_protecting_tier_changed"),
    ])

    # Channel-level metadata (cred + power) repetidos como cols planas para
    # que scout queries no requieran join externo. Se mantienen estaticos a
    # lo largo del torneo (heuristica HIGH/MED/LOW agregada population-level).
    cred_dict = {f"cred_{r['channel']}_{r['shock_type']}":
                 r["channel_credibility"] for r in cred.iter_rows(named=True)}
    pwr_dict = {f"power_{r['channel']}_{r['shock_type']}": r["power_flag"]
                for r in pwr.iter_rows(named=True)}
    df = df.with_columns([pl.lit(v).alias(k) for k, v in cred_dict.items()] +
                          [pl.lit(v).alias(k) for k, v in pwr_dict.items()])

    # Reordenar columnas: identidad → exposicion → indices → posterior probs →
    # 4-vec → CATEs 8 → acute → baselines → meta → rankings → tiers → sig → cred
    front = ["pff_player_id", "player_name", "team_name", "position_group",
             "age_years", "height_cm",
             "minutes_played", "n_matches_played", "low_sample",
             "n_shocks_total", "n_shocks_for", "n_shocks_against",
             "n_shocks_groups", "n_shocks_ko",
             "n_high_leverage_shocks", "avg_leverage_at_shock",
             "max_leverage_at_shock",
             "n_elimination_shocks", "avg_elim_prox_at_shock",
             "max_elim_prox_at_shock",
             "n_nearmiss_exposure",
             "chasing_clutch_idx", "chasing_clutch_sd",
             "chasing_clutch_lo80", "chasing_clutch_hi80",
             "protecting_clutch_idx", "protecting_clutch_sd",
             "protecting_clutch_lo80", "protecting_clutch_hi80",
             "p_chasing_positive", "p_protecting_positive", "p_dual_positive",
             "intra_corr_chasing_atk_off", "intra_corr_protecting_def_phys",
             "pcj_atk", "pcj_def", "pcj_off", "pcj_phys",
             "rank_chasing_global", "rank_protecting_global",
             "rank_chasing_in_position", "rank_protecting_in_position",
             "tier_chasing_global", "tier_protecting_global",
             "tier_chasing_global_certain", "tier_protecting_global_certain",
             "tier_chasing_in_position", "tier_protecting_in_position",
             "sig_chasing", "sig_protecting",
             # Pressure response 3a dimension (eta_pressure de M14 unificado)
             "pressure_response_idx", "pressure_response_sd",
             "pressure_response_lo80", "pressure_response_hi80",
             "p_pressure_clutch_positive",
             "rank_pressure_global", "rank_pressure_in_position",
             "tier_pressure_global", "tier_pressure_in_position",
             "sig_pressure",
             # Physical Bradley 2024 per-90
             "physical_distance_km_per90", "physical_hsr_m_per90",
             "physical_sprints_per90", "physical_high_accels_per90",
             "physical_hmld_m_per90", "physical_peak_speed_mps",
             # H5: ranking absoluto vs relativo
             "chasing_clutch_idx_absolute", "protecting_clutch_idx_absolute",
             "tier_chasing_global_absolute", "tier_protecting_global_absolute",
             "h5_chasing_tier_changed", "h5_protecting_tier_changed"]
    cate_cols     = sorted([c for c in df.columns if c.startswith("cate_")])
    clutch_cols   = sorted([c for c in df.columns if c.startswith("clutch_")])
    acute_cols    = sorted([c for c in df.columns if c.startswith("acute_")])
    base_cols     = sorted([c for c in df.columns if c.startswith("baseline_")])
    cred_cols     = sorted([c for c in df.columns if c.startswith("cred_")])
    power_cols    = sorted([c for c in df.columns if c.startswith("power_")])
    pct_cols      = [c for c in df.columns if c.startswith("pct_")]
    cols = ([c for c in front if c in df.columns] + cate_cols + clutch_cols +
            acute_cols + base_cols + cred_cols + power_cols + pct_cols)
    cols += [c for c in df.columns if c not in cols]
    df = df.select(cols)
    return df


def build_aux_tables(pcj: pl.DataFrame) -> dict:
    """Tablas auxiliares scout-friendly."""
    aux = {}
    # Top10 chasing per position
    aux["top10_chasing_per_position"] = (pcj.sort("chasing_clutch_idx", descending=True)
        .group_by("position_group", maintain_order=True).head(10)
        .select(["position_group", "rank_chasing_in_position",
                 "player_name", "team_name", "chasing_clutch_idx",
                 "chasing_clutch_lo80", "chasing_clutch_hi80",
                 "p_chasing_positive", "tier_chasing_in_position",
                 "sig_chasing", "minutes_played"]))
    aux["top10_protecting_per_position"] = (pcj.sort("protecting_clutch_idx", descending=True)
        .group_by("position_group", maintain_order=True).head(10)
        .select(["position_group", "rank_protecting_in_position",
                 "player_name", "team_name", "protecting_clutch_idx",
                 "protecting_clutch_lo80", "protecting_clutch_hi80",
                 "p_protecting_positive", "tier_protecting_in_position",
                 "sig_protecting", "minutes_played"]))
    # Top10 por bucket (4 grupos: GK/DEF/MED/ATA) — ranking scout-friendly
    # con sample size suficiente. Excluye GK del scout principal de outfield.
    outfield = pcj.filter(pl.col("position_bucket") != "GK")
    aux["top10_chasing_per_bucket"] = (outfield.sort("chasing_clutch_idx", descending=True)
        .group_by("position_bucket", maintain_order=True).head(10)
        .select(["position_bucket", "position_group", "rank_chasing_in_bucket",
                 "player_name", "team_name", "chasing_clutch_idx",
                 "chasing_clutch_lo80", "chasing_clutch_hi80",
                 "p_chasing_positive", "tier_chasing_in_bucket",
                 "sig_chasing", "minutes_played"]))
    aux["top10_protecting_per_bucket"] = (outfield.sort("protecting_clutch_idx", descending=True)
        .group_by("position_bucket", maintain_order=True).head(10)
        .select(["position_bucket", "position_group", "rank_protecting_in_bucket",
                 "player_name", "team_name", "protecting_clutch_idx",
                 "protecting_clutch_lo80", "protecting_clutch_hi80",
                 "p_protecting_positive", "tier_protecting_in_bucket",
                 "sig_protecting", "minutes_played"]))
    # Dual clutch top: (chasing + protecting), filtered to both significant
    dual = (pcj.with_columns(
                (pl.col("chasing_clutch_idx") + pl.col("protecting_clutch_idx"))
                .alias("dual_score"))
              .sort("dual_score", descending=True)
              .head(30)
              .select(["player_name", "team_name", "position_group",
                       "chasing_clutch_idx", "protecting_clutch_idx",
                       "dual_score", "p_chasing_positive", "p_protecting_positive",
                       "p_dual_positive", "minutes_played"]))
    aux["dual_clutch_top"] = dual
    # Top10 pressure response per position (eta_pressure de M14 unificado)
    if "pressure_response_idx" in pcj.columns:
        aux["top10_pressure_per_position"] = (pcj
            .filter(pl.col("pressure_response_idx").is_not_null())
            .sort("pressure_response_idx", descending=True)
            .group_by("position_group", maintain_order=True).head(10)
            .select(["position_group", "rank_pressure_in_position",
                     "player_name", "team_name", "pressure_response_idx",
                     "pressure_response_lo80", "pressure_response_hi80",
                     "p_pressure_clutch_positive", "tier_pressure_in_position",
                     "sig_pressure", "minutes_played"]))
        aux["top10_pressure_per_bucket"] = (outfield
            .filter(pl.col("pressure_response_idx").is_not_null())
            .sort("pressure_response_idx", descending=True)
            .group_by("position_bucket", maintain_order=True).head(10)
            .select(["position_bucket", "position_group", "rank_pressure_in_bucket",
                     "player_name", "team_name", "pressure_response_idx",
                     "pressure_response_lo80", "pressure_response_hi80",
                     "p_pressure_clutch_positive", "tier_pressure_in_bucket",
                     "sig_pressure", "minutes_played"]))
    # Por equipo: agg de minutos + indices
    by_team = (pcj.group_by("team_name").agg([
        pl.len().alias("n_players"),
        pl.col("chasing_clutch_idx").mean().alias("team_chasing_mean"),
        pl.col("protecting_clutch_idx").mean().alias("team_protecting_mean"),
        pl.col("p_chasing_positive").mean().alias("team_p_chasing"),
        pl.col("p_protecting_positive").mean().alias("team_p_protecting"),
    ]).sort("team_chasing_mean", descending=True))
    aux["by_team"] = by_team
    return aux


def main():
    pcj = build_pcj_table()
    validate_pcj_schema(pcj)             # falla si schema contract roto
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _AUX_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "pcj_table.parquet"
    pcj.write_parquet(out_path)
    print(f"\n[M15] Saved {out_path} ({pcj.height} jugadores, {pcj.width} cols)")

    aux = build_aux_tables(pcj)
    for name, df in aux.items():
        path = _AUX_DIR / f"{name}.parquet"
        df.write_parquet(path)
        print(f"  + aux: {name}.parquet ({df.height} rows)")

    # Resumen sanity
    print(f"\n=== PCJ Table summary ===")
    n_low = int(pcj["low_sample"].sum())
    print(f"Jugadores n_shocks>={MIN_SHOCKS}: {pcj.height} "
           f"({n_low} con low_sample flag)")
    print(f"Posiciones cubiertas: {pcj['position_group'].n_unique()}")
    print(f"Equipos cubiertos: {pcj['team_name'].n_unique()}")
    print(f"\nDistribucion sig_chasing:")
    print(pcj.group_by("sig_chasing").len().sort("len", descending=True))
    print(f"\nDistribucion sig_protecting:")
    print(pcj.group_by("sig_protecting").len().sort("len", descending=True))
    print(f"\nTop 10 Remontador globales:")
    print(pcj.sort("rank_chasing_global").head(10).select(
        ["rank_chasing_global", "player_name", "team_name", "position_group",
         "chasing_clutch_idx", "p_chasing_positive", "sig_chasing"]))
    print(f"\nTop 10 Cerrojo globales:")
    print(pcj.sort("rank_protecting_global").head(10).select(
        ["rank_protecting_global", "player_name", "team_name", "position_group",
         "protecting_clutch_idx", "p_protecting_positive", "sig_protecting"]))


if __name__ == "__main__":
    main()
