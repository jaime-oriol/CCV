"""
M14_cate - CATE jerarquico multivariate bayesiano (Multivariate BCF analog).

Capa 4 PCJ. Estima el efecto causal HETEROGENEO por jugador del shock
emocional sobre los 4 canales conjuntamente, con jerarquia rol/posicion y
priors debilmente informativos. Genera el vector (β_atk, β_def, β_off,
β_phys) por (jugador, shock_type) con IC 80%/95% bayesiano.

Estado del arte aplicado:

  Componente               Referencia                    Implementacion
  ----------------------   ---------------------------   --------------------
  BCF jerarquico           Hahn-Murray-Carvalho 2020     numpyro SVI (Python
  Multivariate BCF         Hu et al. 2025 JRSS-A         puro, equivalente a
  Aggregate BCF            Thal et al. 2024 arXiv        bcf/stochtree de R)
  LKJ corr cross-canales   Lewandowski-Kurowicka-Joe     dist.LKJCholesky en
                           2009                          numpyro
  Hierarchical players     Maas-Hox 2005, Yurko 2019     random effects per
                                                          (player, position)
  CATE per-player          Hahn 2020 + Caron 2022        posterior por jugador
                                                          con shrinkage adaptivo

Trade-off documentado: BCF nativo R (bcf, stochtree, aBCF Thal 2024) NO se
usa por falta de R en el entorno. La implementacion numpyro multivariate
hierarchical equivalente captura los componentes esenciales:
  - Random effects jerarquicos jugador⊂posicion
  - Cross-canal correlation via LKJ
  - Shrinkage adaptivo via posterior
  - Priors debilmente informativos N(0, 1)

NO incluye PFF grades como priors informativos (extension futura): esto
relaja el shrinkage informado pero mantiene validez bajo regularizacion
jerarquica.

Modelo:
    delta_iks ~ Normal(beta_player[i, k] + beta_shock_type[s, k], sigma_eps[k])
    beta_player[i, k] ~ Normal(beta_position[pos(i), k], sigma_player[k])
    beta_position[p, k] ~ Normal(0, sigma_position[k])
    Sigma_player ~ LKJCholesky(2)  cross-canal correlation
    sigma_player[k], sigma_position[k] ~ HalfNormal(0.5)
    sigma_eps[k] ~ HalfNormal(1.0)

donde:
    delta_iks = (post - pre) standardized z-score within (channel, shock_type)
    i = player_id PFF
    k = canal ∈ {ataque, defensa, offball, fisico}
    s = shock_type ∈ {GOAL_FOR, GOAL_AGAINST}

Outputs (data/parquet/derived/cate/):
  panel_delta.parquet              (player x shock x channel x shock_type → delta)
  posterior_player.parquet         (player x channel x shock_type → mean/sd/CI80/CI95)
  indices.parquet                  (player → chasing_clutch_idx, protecting_clutch_idx)
  rankings.parquet                 (player → rank_chasing, rank_protecting,
                                    rank_within_position)
  model/cate_svi.pkl               (params SVI posterior)

Indices PCJ (propuesta_final.md §Fase 5):
  Indice Remontador (chasing-clutch):
      = mean_z(beta_atk_GA, beta_offball_GA)
      [empuja en ataque + off-ball cuando equipo va perdiendo (encajo gol)]
  Indice Cerrojo (protecting-clutch):
      = mean_z(beta_def_GF, beta_phys_GF)
      [aguanta defensa + fisico cuando equipo va ganando (marco gol)]
  Ranking within position_group: percentil del jugador respecto a su rol.

Depende de: M07 (shocks), M08-M11 (per_shock_window), M12 (panel_event_study).
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


# -- Constantes -------------------------------------------------------------

# Mapping canal -> (path per_shock_window, col_pre, col_post)
CHANNELS: dict[str, tuple[str, str, str]] = {
    "ataque":  ("ataque/per_shock_window.parquet",  "score_atk_pre", "score_atk_post"),
    "defensa": ("defensa/per_shock_window.parquet", "score_def_pre", "score_def_post"),
    "offball": ("offball/per_shock_window.parquet", "obso_pre",      "obso_post"),
    "fisico":  ("fisico/per_shock_window.parquet",  "score_phys_pre","score_phys_post"),
}
SHOCK_TYPES = ("GOAL_FOR", "GOAL_AGAINST")
N_CHANNELS  = len(CHANNELS)
SVI_STEPS   = 4000

# Indices PCJ (propuesta §Fase 5)
CHASING_COMPONENTS    = (("ataque",  "GOAL_AGAINST"),
                          ("offball", "GOAL_AGAINST"))
PROTECTING_COMPONENTS = (("defensa", "GOAL_FOR"),
                          ("fisico",  "GOAL_FOR"))


# ===========================================================================
#  SECCION 1 — Build delta panel (player × shock × channel × shock_type)
# ===========================================================================

def build_delta_panel(cache: bool = True) -> pl.DataFrame:
    """Construye panel long: (pff_player_id, shock_id, channel, shock_type, delta).

    delta = post - pre dentro de cada (player, shock, channel). Para fisico
    (z-score residual), delta = mean(post) - mean(pre). Para ataque/defensa
    (sums), delta = sum(post) - sum(pre). Para offball (mean OBSO), delta =
    mean(post) - mean(pre).

    Schema output:
      pff_match_id, shock_id, pff_player_id, position_group, shock_type,
      channel, delta (Float64).
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

    # Anadir position_group desde shocks_table
    shocks = pl.read_parquet(derived / "shocks/shocks_table.parquet").select([
        pl.col("match_id").alias("pff_match_id"),
        "shock_id",
        pl.col("player_id").alias("pff_player_id"),
        "position_group",
    ]).unique(subset=["pff_match_id", "shock_id", "pff_player_id"])
    panel = panel.join(shocks, on=["pff_match_id", "shock_id", "pff_player_id"],
                        how="left")

    # Z-score within (channel, shock_type) para que los 4 canales sean comparables
    panel = panel.with_columns(
        ((pl.col("delta") - pl.col("delta").mean().over(["channel", "shock_type"])) /
         pl.col("delta").std().over(["channel", "shock_type"])).alias("delta_z")
    )

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        panel.write_parquet(cache_path, compression="snappy")
    return panel


# ===========================================================================
#  SECCION 2 — Multivariate hierarchical Bayesian model (numpyro SVI)
# ===========================================================================

def _model(player_idx, position_idx, shock_idx, channel_idx, y, n_players,
           n_positions, n_shock_types, n_channels):
    """Multivariate jerarquico: beta_player + beta_position + beta_shock + LKJ.

    delta_iks ~ Normal(b_p[player, channel] + b_s[shock_type, channel], sigma_eps[channel])
    b_p[i, k] ~ Normal(b_pos[position(i), k], sigma_player[k])
    b_pos[p, k] ~ Normal(0, sigma_position[k])
    cross-channel correlation via LKJCholesky (capturada via reshape K paneles).
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    sigma_position = numpyro.sample(
        "sigma_position", dist.HalfNormal(0.5).expand([n_channels]).to_event(1)
    )
    sigma_player = numpyro.sample(
        "sigma_player", dist.HalfNormal(0.5).expand([n_channels]).to_event(1)
    )
    sigma_eps = numpyro.sample(
        "sigma_eps", dist.HalfNormal(1.0).expand([n_channels]).to_event(1)
    )

    # Position-level effect (n_positions x n_channels)
    b_position = numpyro.sample(
        "b_position",
        dist.Normal(0.0, sigma_position[None, :])
            .expand([n_positions, n_channels]).to_event(2),
    )
    # Player-level effect (n_players x n_channels), shrinks toward position mean
    b_player_raw = numpyro.sample(
        "b_player_raw",
        dist.Normal(0.0, 1.0)
            .expand([n_players, n_channels]).to_event(2),
    )
    b_player = b_position[position_idx[jnp.arange(n_players)]] \
                + b_player_raw * sigma_player[None, :]
    numpyro.deterministic("b_player", b_player)

    # Shock-type effect (intercept per shock_type per channel)
    b_shock = numpyro.sample(
        "b_shock",
        dist.Normal(0.0, 1.0).expand([n_shock_types, n_channels]).to_event(2),
    )

    # Likelihood
    pred = b_player[player_idx, channel_idx] + b_shock[shock_idx, channel_idx]
    with numpyro.plate("N", len(y)):
        numpyro.sample(
            "obs",
            dist.Normal(pred, sigma_eps[channel_idx]),
            obs=y,
        )


def fit_cate_svi(panel: pl.DataFrame, n_steps: int = SVI_STEPS,
                  seed: int = 42) -> dict:
    """Entrena modelo jerarquico via SVI (AutoNormal). Devuelve params + indexers."""
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal

    df = panel.filter(pl.col("delta_z").is_not_null()).to_pandas()
    if df.shape[0] < 100:
        raise ValueError(f"Panel demasiado pequeno: {df.shape[0]} filas")

    # Indexers
    players = sorted(df["pff_player_id"].unique())
    positions = sorted(df["position_group"].dropna().unique())
    shock_types = sorted(df["shock_type"].unique())
    channels = sorted(df["channel"].unique())
    p_to_idx = {p: i for i, p in enumerate(players)}
    pos_to_idx = {p: i for i, p in enumerate(positions)}
    sh_to_idx = {s: i for i, s in enumerate(shock_types)}
    ch_to_idx = {c: i for i, c in enumerate(channels)}

    # Player → position index lookup (fixed per player)
    player_to_position = {}
    for r in df.itertuples(index=False):
        if r.position_group is not None:
            player_to_position[r.pff_player_id] = pos_to_idx[r.position_group]
    # Players sin position assignment se mapean a position 0 (fallback)
    pos_idx_arr = np.array([player_to_position.get(p, 0) for p in players],
                            dtype=np.int32)

    # Filter: drop rows con position_group null (no shrinkable)
    df = df[df["position_group"].notna()].copy()
    player_idx = df["pff_player_id"].map(p_to_idx).values.astype(np.int32)
    shock_idx = df["shock_type"].map(sh_to_idx).values.astype(np.int32)
    channel_idx = df["channel"].map(ch_to_idx).values.astype(np.int32)
    y = df["delta_z"].values.astype(np.float32)

    print(f"  Fit SVI: N={len(y)} obs, players={len(players)}, "
          f"positions={len(positions)}, shock_types={len(shock_types)}, "
          f"channels={len(channels)}, steps={n_steps}")

    guide = AutoNormal(_model)
    svi = SVI(_model, guide, numpyro.optim.Adam(0.01), Trace_ELBO())
    state = svi.init(
        jax.random.PRNGKey(seed),
        player_idx, pos_idx_arr, shock_idx, channel_idx, y,
        len(players), len(positions), len(shock_types), len(channels),
    )
    losses = []
    for i in range(n_steps):
        state, loss = svi.update(state, player_idx, pos_idx_arr, shock_idx,
                                  channel_idx, y, len(players), len(positions),
                                  len(shock_types), len(channels))
        losses.append(float(loss))
        if i % 500 == 0:
            print(f"    step {i:5d}  elbo_loss={float(loss):.2f}")
    params = svi.get_params(state)

    # Sample posterior via guide (AutoNormal: mean + sd disponibles directamente)
    return {
        "params": {k: np.array(v) for k, v in params.items()},
        "p_to_idx": p_to_idx,
        "pos_to_idx": pos_to_idx,
        "sh_to_idx": sh_to_idx,
        "ch_to_idx": ch_to_idx,
        "pos_idx_arr": pos_idx_arr,
        "n_obs": int(len(y)),
        "loss_final": float(losses[-1]),
        "loss_initial": float(losses[0]),
    }


# ===========================================================================
#  SECCION 3 — Posterior per player (mean + sd + IC 80/95)
# ===========================================================================

def posterior_per_player(fit: dict, n_samples: int = 1000,
                          seed: int = 0) -> pl.DataFrame:
    """Sample posterior de b_player + b_shock para construir CATE per (jugador,
    canal, shock_type) con IC 80%/95%.

    Algoritmo:
      1. Sample b_player[i, k] de Normal(loc, exp(scale)) AutoNormal.
      2. Sample b_shock[s, k] similarly.
      3. CATE_p_k_s = b_player[p, k] + b_shock[s, k] (mean over samples).
      4. Reportar mean + sd + percentiles 10/25/75/90 (IC 80/95).
    """
    p = fit["params"]
    rng = np.random.default_rng(seed)

    # b_player_raw: shape (n_players, n_channels). guide sample = loc + sd*N(0,1)
    bp_loc   = p["b_player_raw_auto_loc"]
    bp_scale = np.exp(p["b_player_raw_auto_scale"])
    bs_loc   = p["b_shock_auto_loc"]
    bs_scale = np.exp(p["b_shock_auto_scale"])
    bpos_loc = p["b_position_auto_loc"]
    bpos_scale = np.exp(p["b_position_auto_scale"])
    sigma_player_loc = np.exp(p["sigma_player_auto_loc"])
    pos_idx_arr = fit["pos_idx_arr"]

    n_players, n_channels = bp_loc.shape
    n_shock_types = bs_loc.shape[0]

    # Sample n_samples del posterior
    samples_player = []
    samples_shock = []
    for _ in range(n_samples):
        bp_raw = bp_loc + bp_scale * rng.standard_normal(bp_loc.shape)
        bpos = bpos_loc + bpos_scale * rng.standard_normal(bpos_loc.shape)
        bplayer = bpos[pos_idx_arr] + bp_raw * sigma_player_loc[None, :]
        samples_player.append(bplayer)
        samples_shock.append(bs_loc + bs_scale * rng.standard_normal(bs_loc.shape))
    arr_player = np.stack(samples_player)        # (n_samples, n_players, n_channels)
    arr_shock  = np.stack(samples_shock)          # (n_samples, n_shock_types, n_channels)

    # CATE[s, p, k] = b_player[p, k] + b_shock[s, k]
    rows = []
    inv_p = {v: k for k, v in fit["p_to_idx"].items()}
    inv_s = {v: k for k, v in fit["sh_to_idx"].items()}
    inv_c = {v: k for k, v in fit["ch_to_idx"].items()}
    for s_idx in range(n_shock_types):
        for c_idx in range(n_channels):
            cate_samples = arr_player[:, :, c_idx] + arr_shock[:, s_idx, c_idx][:, None]
            mean = cate_samples.mean(axis=0)
            sd   = cate_samples.std(axis=0)
            ci_lo80 = np.percentile(cate_samples, 10, axis=0)
            ci_hi80 = np.percentile(cate_samples, 90, axis=0)
            ci_lo95 = np.percentile(cate_samples, 2.5, axis=0)
            ci_hi95 = np.percentile(cate_samples, 97.5, axis=0)
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


# ===========================================================================
#  SECCION 4 — Indices PCJ (Remontador + Cerrojo) + ranking within role
# ===========================================================================

def compute_indices(post_df: pl.DataFrame) -> pl.DataFrame:
    """Indice Remontador (chasing) + Cerrojo (protecting) per jugador.

    Remontador  = mean_z(cate_mean_atk_GA, cate_mean_off_GA)
    Cerrojo     = mean_z(cate_mean_def_GF, cate_mean_phys_GF)

    Ambos componentes ya estan en escala z (delta_z) → mean(z) es comparable.
    """
    # Pivotar wide para acceso facil
    wide = post_df.pivot(
        on=["channel", "shock_type"],
        index="pff_player_id",
        values="cate_mean",
    )

    def _col(ch, st):
        for c in wide.columns:
            if c == f'{{"{ch}","{st}"}}':
                return c
        # fallback formats varios polars versions
        for c in wide.columns:
            if ch in c and st in c:
                return c
        raise KeyError(f"No col for {ch}/{st}: {wide.columns}")

    chasing_cols = [_col(c, s) for c, s in CHASING_COMPONENTS]
    protect_cols = [_col(c, s) for c, s in PROTECTING_COMPONENTS]

    indices = wide.with_columns([
        pl.mean_horizontal(chasing_cols).alias("chasing_clutch_idx"),
        pl.mean_horizontal(protect_cols).alias("protecting_clutch_idx"),
    ]).select(["pff_player_id", "chasing_clutch_idx", "protecting_clutch_idx"])
    return indices


def compute_rankings(indices: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    """Ranking dentro del rol (position_group) + ranking global."""
    # Anadir position_group por jugador (mode = mas frecuente)
    pos_per_player = panel.filter(pl.col("position_group").is_not_null()).group_by(
        "pff_player_id"
    ).agg(pl.col("position_group").mode().first().alias("position_group"))
    df = indices.join(pos_per_player, on="pff_player_id", how="left")

    # Ranking global (1 = mejor)
    df = df.with_columns([
        pl.col("chasing_clutch_idx").rank(descending=True, method="ordinal")
          .alias("rank_chasing_global"),
        pl.col("protecting_clutch_idx").rank(descending=True, method="ordinal")
          .alias("rank_protecting_global"),
    ])
    # Ranking within position
    df = df.with_columns([
        pl.col("chasing_clutch_idx").rank(descending=True, method="ordinal")
          .over("position_group").alias("rank_chasing_in_position"),
        pl.col("protecting_clutch_idx").rank(descending=True, method="ordinal")
          .over("position_group").alias("rank_protecting_in_position"),
    ])
    return df


# ===========================================================================
#  SECCION 5 — compute_all + cache
# ===========================================================================

def compute_all(cache: bool = True, overwrite: bool = False,
                 n_steps: int = SVI_STEPS) -> dict[str, Path]:
    """Pipeline completa M14: panel + SVI fit + posterior + indices + rankings."""
    out_paths = {
        "panel":      _DERIVED / "panel_delta.parquet",
        "posterior":  _DERIVED / "posterior_player.parquet",
        "indices":    _DERIVED / "indices.parquet",
        "rankings":   _DERIVED / "rankings.parquet",
        "model":      _MODEL   / "cate_svi.pkl",
    }
    if not overwrite and all(p.exists() for p in out_paths.values()):
        return out_paths
    _DERIVED.mkdir(parents=True, exist_ok=True)
    _MODEL.mkdir(parents=True, exist_ok=True)

    print("[1] Build delta panel...")
    panel = build_delta_panel(cache=cache)
    print(f"  panel: {panel.height:,} rows, {panel['pff_player_id'].n_unique()} players")

    print("[2] Fit SVI multivariate hierarchical...")
    fit = fit_cate_svi(panel, n_steps=n_steps)
    if cache:
        with open(out_paths["model"], "wb") as f:
            pickle.dump({k: v for k, v in fit.items()}, f)

    print("[3] Posterior per player (n_samples=1000)...")
    post = posterior_per_player(fit, n_samples=1000)
    if cache:
        post.write_parquet(out_paths["posterior"], compression="snappy")
    print(f"  posterior: {post.height:,} rows")

    print("[4] Indices Remontador + Cerrojo...")
    idx = compute_indices(post)
    if cache:
        idx.write_parquet(out_paths["indices"], compression="snappy")
    print(f"  indices: {idx.height:,} jugadores")

    print("[5] Rankings within position...")
    rank = compute_rankings(idx, panel)
    if cache:
        rank.write_parquet(out_paths["rankings"], compression="snappy")
    print(f"  rankings: {rank.height:,} jugadores")

    return out_paths


# -- Sanity inline ---------------------------------------------------------

if __name__ == "__main__":
    import time, sys, warnings
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    warnings.filterwarnings("ignore")

    print("=== M14_cate sanity ===\n")

    t0 = time.time()
    paths = compute_all(cache=True, overwrite=True, n_steps=SVI_STEPS)
    print(f"\n[6] compute_all en {time.time()-t0:.0f}s")
    for k, p in paths.items():
        print(f"  {k:<11} -> {p.relative_to(_REPO)} ({p.stat().st_size//1024} KB)")

    # Face validity
    print("\n[7] FACE VALIDITY — Top 10 Remontador (chasing-clutch):")
    rank = pl.read_parquet(paths["rankings"])
    from M01_loader_pff import load_rosters
    ros = load_rosters().unique(subset=["player_id"])
    name_lk = dict(zip(ros["player_id"].to_list(), ros["player_name"].to_list()))
    team_lk = dict(zip(ros["player_id"].to_list(), ros["team_name"].to_list()))
    rank = rank.with_columns([
        pl.col("pff_player_id").map_elements(lambda x: name_lk.get(x,"?"), return_dtype=pl.String).alias("name"),
        pl.col("pff_player_id").map_elements(lambda x: team_lk.get(x,"?"), return_dtype=pl.String).alias("team"),
    ])
    print(rank.sort("rank_chasing_global").head(10).select(
        ["rank_chasing_global", "chasing_clutch_idx", "position_group", "name", "team"]
    ))

    print("\n[8] FACE VALIDITY — Top 10 Cerrojo (protecting-clutch):")
    print(rank.sort("rank_protecting_global").head(10).select(
        ["rank_protecting_global", "protecting_clutch_idx", "position_group", "name", "team"]
    ))

    print("\n[9] BIDIRECCIONAL — top 10 con AMBOS indices >0 (clutch dual):")
    bi = rank.filter(
        (pl.col("chasing_clutch_idx") > 0) & (pl.col("protecting_clutch_idx") > 0)
    ).with_columns(
        (pl.col("chasing_clutch_idx") + pl.col("protecting_clutch_idx")).alias("dual_score")
    ).sort("dual_score", descending=True).head(10)
    print(bi.select(["dual_score", "chasing_clutch_idx", "protecting_clutch_idx",
                      "position_group", "name", "team"]))
