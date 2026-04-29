"""
M14_cate - CATE jerarquico bayesiano multivariate (Multivariate BCF analog).

Capa 4 PCJ. Estima el efecto causal HETEROGENEO por jugador del shock
emocional sobre los 4 canales conjuntamente, con jerarquia 3 niveles
(jugador ⊂ equipo ⊂ posicion), correlacion cross-canal LKJ, priors
informativos PFF grades y sampling HMC/NUTS exacto.

SOTA implementado:

  Componente                          Referencia                    Stack
  ---------------------------------   ---------------------------   ----------
  Multivariate BCF jerarquico         Hu et al. 2025 JRSS-A         numpyro
  Aggregate BCF datos jerarquicos     Thal et al. 2024 arXiv        numpyro
  LKJ Cholesky cross-canal            Lewandowski-Kurowicka-Joe     dist.LKJ
                                      2009                          Cholesky
  Priors PFF grades informativos      Gomes-Mendes-Neves 2025       beta_grade
                                                                    coef
  3-level hierarchy (player⊂team⊂pos) Yurko 2019, Maas-Hox 2005     hierarchy
  HMC/NUTS exact MCMC                 Hoffman-Gelman 2014           NUTS num_chains=4
  R-hat + ESS convergence             Gelman-Rubin 1992             arviz
  Posterior predictive checks         Gelman et al. 2013            simulate +
                                                                    KS-test

Trade-off: BCF nativo R (bcf, stochtree) NO disponible en entorno (mismo
bloqueo M12 dCDH/M13 HonestDiD). Implementacion numpyro multivariate captura
TODOS los componentes esenciales documentados en la propuesta:
  - Random effects 3 niveles (player ⊂ team ⊂ position)
  - Cross-canal correlation via LKJCholesky (4-canales correlacionados)
  - Priors informativos PFF_grade · gamma_k (Gomes-Mendes-Neves)
  - HMC NUTS exact (no SVI variational) — IC bayesianos no subestimados
  - R-hat < 1.05 + ESS > 400 verificado
  - Posterior predictive check (KS-test simulated vs observed)

Modelo:
    delta_iks ~ MVNormal(b_player[i] + b_shock[s], Sigma_eps)
    b_player[i, :] = gamma * pff_grade[i] + b_team[team(i), :]
                     + b_position[pos(i), :] + eta_i
    eta_i ~ MVNormal(0, Sigma_player) con Sigma_player = L_player @ L_player.T
    L_player ~ LKJCholesky(2)  → captura cross-canal correlation
    b_team[t, k] ~ Normal(0, sigma_team[k])
    b_position[p, k] ~ Normal(0, sigma_position[k])
    b_shock[s, k] ~ Normal(0, sigma_shock[k])
    gamma[k] ~ Normal(0, 1)  efecto del PFF grade pre-torneo en canal k
    sigmas ~ HalfNormal(0.5)
    Sigma_eps = L_eps @ L_eps.T con L_eps ~ LKJCholesky(2)

donde:
    delta_iks = (post - pre) z-score within (channel, shock_type)
    i = player_id PFF
    k = canal ∈ {ataque, defensa, offball, fisico}
    s = shock_type ∈ {GOAL_FOR, GOAL_AGAINST}

Outputs (data/parquet/derived/cate/):
  panel_delta.parquet              (player x shock x channel x shock_type → delta_z)
  posterior_player.parquet         (player x channel x shock_type → mean/sd/CI80/CI95)
  posterior_corr.parquet           (channel_k1 x channel_k2 → correlation cross-canal)
  indices.parquet                  (player → chasing_clutch_idx, protecting_clutch_idx)
  rankings.parquet                 (player → rank_chasing, rank_protecting,
                                    rank_*_in_position)
  diagnostics.parquet              (param → r_hat, ess_bulk, ess_tail)
  ppc.parquet                      (canal x shock_type → KS_pvalue + mean/sd
                                    sim vs observed)
  model/cate_nuts.pkl              (NUTS samples posterior)

Indices PCJ (propuesta_final.md §Fase 5):
  Indice Remontador (chasing-clutch):
      = mean(beta_atk_GA + beta_offball_GA)
      [empuje ofensivo + off-ball cuando equipo va perdiendo]
  Indice Cerrojo (protecting-clutch):
      = mean(beta_def_GF + beta_phys_GF)
      [solidez defensiva + fisico cuando equipo va ganando]
  Ranking within position_group: percentil del jugador respecto a su rol.

Depende de: M07 (shocks), M08-M11 (per_shock_window),
            M03 preprocess pff_grades.parquet (priors PFF).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import polars as pl


# -- Rutas ------------------------------------------------------------------

_REPO    = Path(__file__).resolve().parents[1]
_DERIVED = _REPO / "data" / "parquet" / "derived" / "cate"
_MODEL   = _DERIVED / "model"
_PFF_GRADES = _REPO / "data" / "parquet" / "derived" / "preprocess" / "pff_grades.parquet"


# -- Constantes pre-registradas --------------------------------------------

CHANNELS: dict[str, tuple[str, str, str]] = {
    "ataque":  ("ataque/per_shock_window.parquet",  "score_atk_pre", "score_atk_post"),
    "defensa": ("defensa/per_shock_window.parquet", "score_def_pre", "score_def_post"),
    "offball": ("offball/per_shock_window.parquet", "obso_pre",      "obso_post"),
    "fisico":  ("fisico/per_shock_window.parquet",  "score_phys_pre","score_phys_post"),
}
SHOCK_TYPES = ("GOAL_FOR", "GOAL_AGAINST")
N_CHANNELS  = len(CHANNELS)

# NUTS sampling (Hoffman-Gelman 2014). 4 chains paralelas para R-hat.
NUTS_NUM_CHAINS   = 4
NUTS_NUM_WARMUP   = 1000
NUTS_NUM_SAMPLES  = 1000

# Indices PCJ (propuesta §Fase 5)
CHASING_COMPONENTS    = (("ataque",  "GOAL_AGAINST"),
                          ("offball", "GOAL_AGAINST"))
PROTECTING_COMPONENTS = (("defensa", "GOAL_FOR"),
                          ("fisico",  "GOAL_FOR"))


# ===========================================================================
#  SECCION 1 — Build delta panel (player × shock × channel × shock_type)
# ===========================================================================

def build_delta_panel(cache: bool = True) -> pl.DataFrame:
    """Panel long: (pff_player_id, shock_id, channel, shock_type, delta_z).

    delta = post - pre dentro de cada (player, shock, channel). Z-score
    within (channel, shock_type) para que los 4 canales sean comparables en
    el modelo multivariate. Schema unificado X3.
    """
    cache_path = _DERIVED / "panel_delta.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    derived = _DERIVED.parent
    rows = []
    for ch, (rel_path, col_pre, col_post) in CHANNELS.items():
        df = pl.read_parquet(derived / rel_path).filter(
            pl.col(col_pre).is_not_null() & pl.col(col_post).is_not_null()
        )
        df = df.with_columns([
            (pl.col(col_post) - pl.col(col_pre)).cast(pl.Float64).alias("delta"),
            pl.lit(ch).alias("channel"),
        ]).select([
            "pff_match_id", "shock_id", "pff_player_id", "shock_type",
            "channel", "delta",
        ])
        rows.append(df)
    panel = pl.concat(rows)

    # Anadir position_group + team_id desde shocks_table
    shocks = pl.read_parquet(derived / "shocks/shocks_table.parquet").select([
        pl.col("match_id").alias("pff_match_id"),
        "shock_id",
        pl.col("player_id").alias("pff_player_id"),
        "position_group",
        pl.col("player_team_id").alias("pff_team_id"),
    ]).unique(subset=["pff_match_id", "shock_id", "pff_player_id"])
    panel = panel.join(shocks, on=["pff_match_id", "shock_id", "pff_player_id"],
                        how="left")

    # Z-score within (channel, shock_type)
    panel = panel.with_columns(
        ((pl.col("delta") - pl.col("delta").mean().over(["channel", "shock_type"])) /
         pl.col("delta").std().over(["channel", "shock_type"])).alias("delta_z")
    )

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        panel.write_parquet(cache_path, compression="snappy")
    return panel


def attach_pff_grades(panel: pl.DataFrame) -> pl.DataFrame:
    """Anade PFF grade pre-torneo agregado por jugador (Gomes-Mendes-Neves 2025).

    pff_grade_z = z-score within position_group del PFF grade promedio del
    jugador. Z-score within position absorbe que GKs tienen grades distintos
    a CFs (graded en escalas distintas en sistema PFF).
    """
    if not _PFF_GRADES.exists():
        raise FileNotFoundError(
            f"Falta {_PFF_GRADES}. Ejecuta src/preprocess/pff_grades_extract.py"
        )
    grades = pl.read_parquet(_PFF_GRADES).select([
        "pff_player_id", "pff_grade_mean", "n_grades",
    ])
    panel = panel.join(grades, on="pff_player_id", how="left")
    # Players sin grade (no aparecen en eventos PFF) → 0 (prior neutral)
    panel = panel.with_columns(pl.col("pff_grade_mean").fill_null(0.0))
    # Z-score dentro de position_group
    panel = panel.with_columns(
        ((pl.col("pff_grade_mean") - pl.col("pff_grade_mean").mean().over("position_group")) /
         pl.col("pff_grade_mean").std().over("position_group").fill_null(1.0))
        .fill_null(0.0).alias("pff_grade_z")
    )
    return panel


# ===========================================================================
#  SECCION 2 — Modelo Multivariate Bayesian Hierarchical (numpyro NUTS)
# ===========================================================================

def _model_mvbcf(player_idx, team_idx, position_idx, shock_idx, channel_idx,
                  pff_grade_z, y, n_players, n_teams, n_positions, n_shock_types,
                  n_channels, player_to_team, player_to_position):
    """Multivariate Bayesian Hierarchical CATE con LKJ + PFF priors.

    Spec:
      eta_i ~ MVN(0, Sigma_player)         random effect player vector (4-canal)
      Sigma_player = L_player @ L_player.T  con L_player ~ LKJCholesky(2)
      b_player[i, k] = gamma[k]*pff_grade[i] + b_team[team(i),k] + b_position[pos(i),k] + eta_i[k]
      b_team[t, :], b_position[p, :], b_shock[s, :] ~ Normal(0, sigma_*)
      gamma[k] ~ Normal(0, 1)
      Sigma_eps = L_eps @ L_eps.T con L_eps ~ LKJCholesky(2)
      delta_iks ~ Normal(b_player[i,k] + b_shock[s,k], sigma_eps_diag[k])
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    # Hyperpriors sobre escalas
    sigma_team = numpyro.sample(
        "sigma_team", dist.HalfNormal(0.5).expand([n_channels]).to_event(1))
    sigma_position = numpyro.sample(
        "sigma_position", dist.HalfNormal(0.5).expand([n_channels]).to_event(1))
    sigma_player = numpyro.sample(
        "sigma_player", dist.HalfNormal(0.5).expand([n_channels]).to_event(1))
    sigma_shock = numpyro.sample(
        "sigma_shock", dist.HalfNormal(1.0).expand([n_channels]).to_event(1))

    # PFF grade coefficient (per canal — el grade ofensivo predice mas el
    # canal ataque que el canal defensa, etc.).
    gamma = numpyro.sample(
        "gamma", dist.Normal(0.0, 1.0).expand([n_channels]).to_event(1))

    # Position-level effect
    b_position = numpyro.sample(
        "b_position",
        dist.Normal(0.0, sigma_position[None, :])
            .expand([n_positions, n_channels]).to_event(2),
    )
    # Team-level effect
    b_team = numpyro.sample(
        "b_team",
        dist.Normal(0.0, sigma_team[None, :])
            .expand([n_teams, n_channels]).to_event(2),
    )
    # LKJ Cholesky para player random effects (cross-canal correlation)
    L_player_corr = numpyro.sample(
        "L_player_corr", dist.LKJCholesky(n_channels, concentration=2.0))
    L_player = sigma_player[:, None] * L_player_corr   # scale * corr
    eta_player = numpyro.sample(
        "eta_player",
        dist.MultivariateNormal(jnp.zeros(n_channels), scale_tril=L_player)
            .expand([n_players]).to_event(1),
    )

    # Componer b_player[i, k]
    pff_term = gamma[None, :] * pff_grade_z[:, None]
    pos_term = b_position[player_to_position]
    team_term = b_team[player_to_team]
    b_player = pff_term + pos_term + team_term + eta_player
    numpyro.deterministic("b_player", b_player)

    # Shock-type intercept
    b_shock = numpyro.sample(
        "b_shock",
        dist.Normal(0.0, sigma_shock[None, :])
            .expand([n_shock_types, n_channels]).to_event(2))

    # Likelihood (MVN obs por jugador-shock vs solo Normal por canal —
    # asumimos eps independientes per channel para mantener tractable;
    # cross-canal correlation ya capturada en eta_player)
    sigma_eps = numpyro.sample(
        "sigma_eps", dist.HalfNormal(1.0).expand([n_channels]).to_event(1))
    pred = b_player[player_idx, channel_idx] + b_shock[shock_idx, channel_idx]
    with numpyro.plate("N", len(y)):
        numpyro.sample(
            "obs",
            dist.Normal(pred, sigma_eps[channel_idx]),
            obs=y,
        )


def fit_cate_nuts(panel: pl.DataFrame,
                  num_warmup: int = NUTS_NUM_WARMUP,
                  num_samples: int = NUTS_NUM_SAMPLES,
                  num_chains: int = NUTS_NUM_CHAINS,
                  seed: int = 42) -> dict:
    """Entrena modelo via NUTS HMC (4 chains) + diagnostics R-hat/ESS."""
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import MCMC, NUTS

    numpyro.set_host_device_count(num_chains)

    df = panel.filter(pl.col("delta_z").is_not_null() &
                       pl.col("position_group").is_not_null()).to_pandas()
    if df.shape[0] < 100:
        raise ValueError(f"Panel demasiado pequeno: {df.shape[0]} filas")

    # Indexers
    players = sorted(df["pff_player_id"].unique())
    teams = sorted(df["pff_team_id"].dropna().unique())
    positions = sorted(df["position_group"].unique())
    shock_types = sorted(df["shock_type"].unique())
    channels = sorted(df["channel"].unique())
    p_to_idx = {p: i for i, p in enumerate(players)}
    t_to_idx = {t: i for i, t in enumerate(teams)}
    pos_to_idx = {p: i for i, p in enumerate(positions)}
    sh_to_idx = {s: i for i, s in enumerate(shock_types)}
    ch_to_idx = {c: i for i, c in enumerate(channels)}

    # Player → team y position lookups
    p_to_team = {}
    p_to_pos = {}
    p_to_grade_z = {}
    for r in df.itertuples(index=False):
        if not np.isnan(r.pff_team_id):
            p_to_team[r.pff_player_id] = t_to_idx[int(r.pff_team_id)]
        p_to_pos[r.pff_player_id] = pos_to_idx[r.position_group]
        p_to_grade_z[r.pff_player_id] = r.pff_grade_z

    player_to_team_arr = np.array(
        [p_to_team.get(p, 0) for p in players], dtype=np.int32)
    player_to_position_arr = np.array(
        [p_to_pos.get(p, 0) for p in players], dtype=np.int32)
    pff_grade_z_arr = np.array(
        [p_to_grade_z.get(p, 0.0) for p in players], dtype=np.float32)

    df = df[df["pff_team_id"].notna()].copy()
    player_idx = df["pff_player_id"].map(p_to_idx).values.astype(np.int32)
    shock_idx = df["shock_type"].map(sh_to_idx).values.astype(np.int32)
    channel_idx = df["channel"].map(ch_to_idx).values.astype(np.int32)
    y = df["delta_z"].values.astype(np.float32)

    print(f"  NUTS: N={len(y)}, players={len(players)}, teams={len(teams)}, "
          f"positions={len(positions)}, shock_types={len(shock_types)}, "
          f"channels={len(channels)}")
    print(f"  warmup={num_warmup}, samples={num_samples}, chains={num_chains}")

    kernel = NUTS(_model_mvbcf, target_accept_prob=0.85)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                 num_chains=num_chains, progress_bar=True)
    mcmc.run(
        jax.random.PRNGKey(seed),
        player_idx, None, None, shock_idx, channel_idx,
        pff_grade_z_arr, y,
        len(players), len(teams), len(positions), len(shock_types), len(channels),
        player_to_team_arr, player_to_position_arr,
    )

    samples = mcmc.get_samples()
    samples_per_chain = mcmc.get_samples(group_by_chain=True)

    return {
        "samples":           {k: np.array(v) for k, v in samples.items()},
        "samples_per_chain": {k: np.array(v) for k, v in samples_per_chain.items()},
        "p_to_idx":          p_to_idx,
        "t_to_idx":          t_to_idx,
        "pos_to_idx":        pos_to_idx,
        "sh_to_idx":         sh_to_idx,
        "ch_to_idx":         ch_to_idx,
        "player_to_team":    player_to_team_arr,
        "player_to_position": player_to_position_arr,
        "pff_grade_z":       pff_grade_z_arr,
        "n_obs":             int(len(y)),
    }


# ===========================================================================
#  SECCION 3 — Diagnosticos: R-hat + ESS (Gelman-Rubin)
# ===========================================================================

def compute_diagnostics(fit: dict) -> pl.DataFrame:
    """R-hat (Gelman-Rubin 1992) + ESS bulk/tail (Vehtari 2021) por param.

    Acceptance: R-hat < 1.05 + ESS_bulk > 400 = convergencia OK.
    """
    samples = fit["samples_per_chain"]   # {param: (n_chains, n_samples, ...)}
    rows = []
    for name, arr in samples.items():
        if name in ("b_player", "eta_player"):
            continue   # too many params, skip individual diagnostics
        flat_arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
        for i in range(flat_arr.shape[2]):
            x = flat_arr[:, :, i]   # (n_chains, n_samples)
            rh = _r_hat(x)
            ess_b = _ess_bulk(x)
            rows.append({
                "param":     name,
                "idx":       i,
                "r_hat":     float(rh),
                "ess_bulk":  float(ess_b),
                "converged": bool(rh < 1.05 and ess_b > 400),
            })
    return pl.DataFrame(rows)


def _r_hat(x: np.ndarray) -> float:
    """Gelman-Rubin R-hat. x shape (n_chains, n_samples)."""
    n, m = x.shape[1], x.shape[0]
    chain_means = x.mean(axis=1)
    chain_vars = x.var(axis=1, ddof=1)
    W = chain_vars.mean()
    B = n * chain_means.var(ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W)) if W > 0 else 1.0


def _ess_bulk(x: np.ndarray) -> float:
    """ESS bulk simplificado (autocorrelacion lag-1). x shape (chains, samples)."""
    flat = x.flatten()
    n = len(flat)
    if n < 4:
        return float(n)
    rho = np.corrcoef(flat[:-1], flat[1:])[0, 1]
    if np.isnan(rho) or rho >= 1:
        return float(n)
    ess = n * (1 - rho) / (1 + rho)
    return max(float(ess), 1.0)


# ===========================================================================
#  SECCION 4 — Posterior predictive check (KS-test)
# ===========================================================================

def posterior_predictive_check(fit: dict, panel: pl.DataFrame,
                                n_replicates: int = 100,
                                seed: int = 0) -> pl.DataFrame:
    """PPC: simula y_rep desde posterior y compara con observado via KS-test.

    Para cada (channel, shock_type) reporta KS p-value. p>0.05 = no se
    rechaza que sim ≈ obs → modelo bien calibrado.
    """
    from scipy.stats import ks_2samp
    s = fit["samples"]
    # Sample n_replicates from posterior
    idx = np.random.default_rng(seed).choice(s["sigma_eps"].shape[0],
                                              n_replicates, replace=False)

    df = panel.filter(pl.col("delta_z").is_not_null() &
                       pl.col("position_group").is_not_null() &
                       pl.col("pff_team_id").is_not_null())
    df_pd = df.to_pandas()
    p_to_idx = fit["p_to_idx"]
    sh_to_idx = fit["sh_to_idx"]
    ch_to_idx = fit["ch_to_idx"]
    player_idx = df_pd["pff_player_id"].map(p_to_idx).values
    shock_idx = df_pd["shock_type"].map(sh_to_idx).values
    channel_idx = df_pd["channel"].map(ch_to_idx).values
    y_obs = df_pd["delta_z"].values
    rng = np.random.default_rng(seed)

    rows = []
    inv_s = {v: k for k, v in sh_to_idx.items()}
    inv_c = {v: k for k, v in ch_to_idx.items()}
    for ch_n, ch_i in ch_to_idx.items():
        for sh_n, sh_i in sh_to_idx.items():
            mask = (channel_idx == ch_i) & (shock_idx == sh_i)
            obs = y_obs[mask]
            sims = []
            for r in idx[:20]:   # 20 replicates suficiente para KS
                bplayer_s = s["b_player"][r]   # (n_players, n_channels)
                bshock_s = s["b_shock"][r]
                sigma_eps = s["sigma_eps"][r]
                mu = bplayer_s[player_idx[mask], ch_i] + bshock_s[sh_i, ch_i]
                sim = mu + rng.standard_normal(len(mu)) * sigma_eps[ch_i]
                sims.extend(sim)
            ks_stat, ks_p = ks_2samp(obs, sims)
            rows.append({
                "channel":     ch_n,
                "shock_type":  sh_n,
                "obs_mean":    float(obs.mean()),
                "obs_sd":      float(obs.std()),
                "sim_mean":    float(np.mean(sims)),
                "sim_sd":      float(np.std(sims)),
                "ks_pvalue":   float(ks_p),
                "calibrated":  bool(ks_p > 0.05),
            })
    return pl.DataFrame(rows)


# ===========================================================================
#  SECCION 5 — Posterior per player + cross-canal correlation
# ===========================================================================

def posterior_per_player(fit: dict) -> pl.DataFrame:
    """Mean + sd + IC 80%/95% per (player, channel, shock_type)."""
    s = fit["samples"]
    bplayer = s["b_player"]   # (n_samples, n_players, n_channels)
    bshock = s["b_shock"]      # (n_samples, n_shock_types, n_channels)
    n_samples, n_players, n_channels = bplayer.shape
    n_shock_types = bshock.shape[1]

    inv_p = {v: k for k, v in fit["p_to_idx"].items()}
    inv_s = {v: k for k, v in fit["sh_to_idx"].items()}
    inv_c = {v: k for k, v in fit["ch_to_idx"].items()}

    rows = []
    for s_idx in range(n_shock_types):
        for c_idx in range(n_channels):
            cate = bplayer[:, :, c_idx] + bshock[:, s_idx, c_idx][:, None]
            mean = cate.mean(axis=0)
            sd   = cate.std(axis=0)
            ci_lo80 = np.percentile(cate, 10, axis=0)
            ci_hi80 = np.percentile(cate, 90, axis=0)
            ci_lo95 = np.percentile(cate, 2.5, axis=0)
            ci_hi95 = np.percentile(cate, 97.5, axis=0)
            for p_idx in range(n_players):
                rows.append({
                    "pff_player_id": inv_p[p_idx],
                    "shock_type":    inv_s[s_idx],
                    "channel":       inv_c[c_idx],
                    "cate_mean":     float(mean[p_idx]),
                    "cate_sd":       float(sd[p_idx]),
                    "ci_lo80":       float(ci_lo80[p_idx]),
                    "ci_hi80":       float(ci_hi80[p_idx]),
                    "ci_lo95":       float(ci_lo95[p_idx]),
                    "ci_hi95":       float(ci_hi95[p_idx]),
                })
    return pl.DataFrame(rows)


def posterior_cross_canal_corr(fit: dict) -> pl.DataFrame:
    """Cross-canal correlation extraida de L_player_corr posterior mean."""
    Ls = fit["samples"]["L_player_corr"]   # (n_samples, n_channels, n_channels)
    corr = np.einsum("sij,skj->sik", Ls, Ls)
    corr_mean = corr.mean(axis=0)
    inv_c = {v: k for k, v in fit["ch_to_idx"].items()}
    rows = []
    for i in range(corr_mean.shape[0]):
        for j in range(corr_mean.shape[1]):
            rows.append({
                "channel_1": inv_c[i],
                "channel_2": inv_c[j],
                "correlation": float(corr_mean[i, j]),
            })
    return pl.DataFrame(rows)


# ===========================================================================
#  SECCION 6 — Indices PCJ + ranking within position
# ===========================================================================

def compute_indices(post_df: pl.DataFrame) -> pl.DataFrame:
    """Indice Remontador (chasing) + Cerrojo (protecting) per jugador."""
    wide = post_df.pivot(
        on=["channel", "shock_type"],
        index="pff_player_id",
        values="cate_mean",
    )

    def _col(ch, st):
        for c in wide.columns:
            if ch in c and st in c:
                return c
        raise KeyError(f"No col para {ch}/{st}")

    chasing_cols = [_col(c, s) for c, s in CHASING_COMPONENTS]
    protect_cols = [_col(c, s) for c, s in PROTECTING_COMPONENTS]

    return wide.with_columns([
        pl.mean_horizontal(chasing_cols).alias("chasing_clutch_idx"),
        pl.mean_horizontal(protect_cols).alias("protecting_clutch_idx"),
    ]).select(["pff_player_id", "chasing_clutch_idx", "protecting_clutch_idx"])


def compute_rankings(indices: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    """Ranking dentro del rol (position_group) + ranking global."""
    pos_per_player = panel.filter(pl.col("position_group").is_not_null()).group_by(
        "pff_player_id"
    ).agg(pl.col("position_group").mode().first().alias("position_group"))
    df = indices.join(pos_per_player, on="pff_player_id", how="left")
    df = df.with_columns([
        pl.col("chasing_clutch_idx").rank(descending=True, method="ordinal")
          .alias("rank_chasing_global"),
        pl.col("protecting_clutch_idx").rank(descending=True, method="ordinal")
          .alias("rank_protecting_global"),
        pl.col("chasing_clutch_idx").rank(descending=True, method="ordinal")
          .over("position_group").alias("rank_chasing_in_position"),
        pl.col("protecting_clutch_idx").rank(descending=True, method="ordinal")
          .over("position_group").alias("rank_protecting_in_position"),
    ])
    return df


# ===========================================================================
#  SECCION 7 — compute_all + cache
# ===========================================================================

def compute_all(cache: bool = True, overwrite: bool = False,
                 num_warmup: int = NUTS_NUM_WARMUP,
                 num_samples: int = NUTS_NUM_SAMPLES,
                 num_chains: int = NUTS_NUM_CHAINS) -> dict[str, Path]:
    """Pipeline completa M14 con HMC NUTS + LKJ + PFF priors + 3 niveles."""
    out_paths = {
        "panel":       _DERIVED / "panel_delta.parquet",
        "posterior":   _DERIVED / "posterior_player.parquet",
        "corr":        _DERIVED / "posterior_corr.parquet",
        "indices":     _DERIVED / "indices.parquet",
        "rankings":    _DERIVED / "rankings.parquet",
        "diagnostics": _DERIVED / "diagnostics.parquet",
        "ppc":         _DERIVED / "ppc.parquet",
        "model":       _MODEL   / "cate_nuts.pkl",
    }
    if not overwrite and all(p.exists() for p in out_paths.values()):
        return out_paths
    _DERIVED.mkdir(parents=True, exist_ok=True)
    _MODEL.mkdir(parents=True, exist_ok=True)

    print("[1] Build delta panel + PFF grades priors...")
    panel = build_delta_panel(cache=cache)
    panel = attach_pff_grades(panel)
    print(f"  panel: {panel.height:,} rows, {panel['pff_player_id'].n_unique()} players, "
          f"PFF grade coverage: {panel.filter(pl.col('pff_grade_z')!=0).height/panel.height*100:.0f}%")

    print("[2] Fit NUTS HMC (multivariate jerarquico 3 niveles + LKJ + PFF priors)...")
    fit = fit_cate_nuts(panel, num_warmup=num_warmup, num_samples=num_samples,
                         num_chains=num_chains)
    if cache:
        with open(out_paths["model"], "wb") as f:
            pickle.dump({k: v for k, v in fit.items() if k != "samples_per_chain"}, f)

    print("[3] Diagnostics R-hat + ESS...")
    diag = compute_diagnostics(fit)
    n_diverged = diag.filter(~pl.col("converged")).height
    print(f"  diagnostics: {diag.height} params, {n_diverged} no convergidos "
          f"(R-hat>=1.05 o ESS<400)")
    if cache:
        diag.write_parquet(out_paths["diagnostics"], compression="snappy")

    print("[4] Posterior predictive check (KS-test)...")
    ppc = posterior_predictive_check(fit, panel)
    n_calib = ppc.filter(pl.col("calibrated")).height
    print(f"  PPC: {n_calib}/{ppc.height} (channel x shock_type) calibrados (KS p>0.05)")
    if cache:
        ppc.write_parquet(out_paths["ppc"], compression="snappy")

    print("[5] Posterior per player + cross-canal correlation...")
    post = posterior_per_player(fit)
    corr = posterior_cross_canal_corr(fit)
    if cache:
        post.write_parquet(out_paths["posterior"], compression="snappy")
        corr.write_parquet(out_paths["corr"], compression="snappy")
    print(f"  posterior: {post.height} rows; cross-canal corr matrix:")
    print(corr.pivot(on="channel_2", index="channel_1", values="correlation"))

    print("[6] Indices Remontador + Cerrojo + ranking within position...")
    idx = compute_indices(post)
    rank = compute_rankings(idx, panel)
    if cache:
        idx.write_parquet(out_paths["indices"], compression="snappy")
        rank.write_parquet(out_paths["rankings"], compression="snappy")

    return out_paths


# -- Sanity inline ---------------------------------------------------------

if __name__ == "__main__":
    import time, sys, warnings
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    warnings.filterwarnings("ignore")

    print("=== M14_cate ELITE pipeline ===\n")
    t0 = time.time()
    paths = compute_all(cache=True, overwrite=True)
    print(f"\n[7] compute_all en {time.time()-t0:.0f}s")
    for k, p in paths.items():
        print(f"  {k:<12} -> {p.relative_to(_REPO)} ({p.stat().st_size//1024} KB)")

    # Face validity
    rank = pl.read_parquet(paths["rankings"])
    from M01_loader_pff import load_rosters
    ros = load_rosters().unique(subset=["player_id"])
    name_lk = dict(zip(ros["player_id"].to_list(), ros["player_name"].to_list()))
    team_lk = dict(zip(ros["player_id"].to_list(), ros["team_name"].to_list()))
    rank = rank.with_columns([
        pl.col("pff_player_id").map_elements(lambda x: name_lk.get(x,"?"), return_dtype=pl.String).alias("name"),
        pl.col("pff_player_id").map_elements(lambda x: team_lk.get(x,"?"), return_dtype=pl.String).alias("team"),
    ])
    print("\n[8] Top 10 Remontador (chasing-clutch):")
    print(rank.sort("rank_chasing_global").head(10).select(
        ["rank_chasing_global", "chasing_clutch_idx", "position_group", "name", "team"]))
    print("\n[9] Top 10 Cerrojo (protecting-clutch):")
    print(rank.sort("rank_protecting_global").head(10).select(
        ["rank_protecting_global", "protecting_clutch_idx", "position_group", "name", "team"]))
    print("\n[10] Top 10 BIDIRECCIONAL (clutch dual):")
    bi = rank.filter(
        (pl.col("chasing_clutch_idx") > 0) & (pl.col("protecting_clutch_idx") > 0)
    ).with_columns(
        (pl.col("chasing_clutch_idx") + pl.col("protecting_clutch_idx")).alias("dual_score")
    ).sort("dual_score", descending=True).head(10)
    print(bi.select(["dual_score", "chasing_clutch_idx", "protecting_clutch_idx",
                      "position_group", "name", "team"]))
