"""M14b_pressure - Clutch sostenido bajo high-elim_prox via SHOCK DE ELIMINACION.

Cierra el gap del M14 original (que solo captura post-shock-de-gol):

CASO PARADIGMATICO Mbappe final WC22:
  - Argentina 2-0 min 36 (Messi PEN) → goal_shock GA cierra ventana min 46
  - Min 47-79 = NO en ventana shock-gol pero presion existencial maxima
  - Mbappe doblete min 80-81 (responde 33 min DESPUES del shock-Messi)
  - M14 NO lo capta (timing > ±10 min post-shock-Messi)

Solucion: definir SEGUNDO tipo de shock — la TRANSICION a high-elim_prox.

Definicion pre-registrada:
    PRESSURE_SHOCK = primer minuto del partido donde elim_prox_player > 0.7
                     AND minute > 60.

Por partido por equipo: 0-1 pressure shocks. Ventana pre/post ±10 min como
M07 goal shocks. M14b estima eta_pressure[player, channel] como respuesta
INDIVIDUAL al shock de eliminacion, neta de team/position/grade (analog M14).

Output:
    data/parquet/derived/cate/pressure_shocks.parquet  (M07-equivalente)
    data/parquet/derived/cate/pressure_panel.parquet   (delta_z per player x channel)
    data/parquet/derived/cate/pressure_response.parquet (eta + IC + posterior probs)
    data/parquet/derived/cate/model/pressure_nuts.pkl

M15 lo consume y expone:
    pressure_response_atk/def/off/phys (eta per channel)
    pressure_response_idx (mean across canales)
    p_pressure_clutch_positive
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

_REPO    = Path(__file__).resolve().parents[1]
_SRC     = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_DERIVED = _REPO / "data" / "parquet" / "derived"
_CATE    = _DERIVED / "cate"
_WP      = _DERIVED / "wp" / "per_minute.parquet"
_PFF_GRADES = _DERIVED / "preprocess" / "pff_grades.parquet"

# Trigger pre-registrado de pressure shock
ELIM_PROX_TRIGGER = 0.70    # elim_prox > 0.7 = "casi eliminados"
MIN_TRIGGER = 60            # solo despues de min 60 (evita early-game volatilidad)
WINDOW_MIN = 10             # +-10 min como M07 goal shocks

CHANNELS = {
    "ataque":  ("ataque/per_minute.parquet",  "score_atk_minute"),
    "defensa": ("defensa/per_minute.parquet", "score_def_minute"),
    "offball": ("offball/per_minute.parquet", "c_obso_mean"),       # counterfactual T1.2
    "fisico":  ("fisico/per_minute.parquet",  "score_phys"),
}
N_CHANNELS = len(CHANNELS)

NUTS_NUM_CHAINS  = 4
NUTS_NUM_WARMUP  = 1000
NUTS_NUM_SAMPLES = 1000


# ============================================================================
# 1. Build pressure shocks (analogo M07 pero trigger=elim_prox>0.7)
# ============================================================================
def build_pressure_shocks(cache: bool = True) -> pl.DataFrame:
    """Detecta el primer minuto del partido donde elim_prox del equipo supera
    el trigger pre-registrado, generando un SHOCK DE ELIMINACION per equipo.

    Output: (pff_match_id, shock_id, team_id, minute, elim_prox_at_shock)
    """
    cache_path = _CATE / "pressure_shocks.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    from M01_loader_pff import load_metadata
    md = load_metadata().select([
        pl.col("id").cast(pl.Int64).alias("pff_match_id"),
        "home_team_id", "away_team_id",
    ])
    wp = pl.read_parquet(_WP).rename({"match_id": "pff_match_id"}).join(
        md, on="pff_match_id", how="left")

    # Long format per (match, minute, team) con elim_prox del equipo
    home = wp.select([
        "pff_match_id", "minute", pl.col("home_team_id").alias("team_id"),
        pl.col("elim_prox_home").alias("elim_prox"),
    ])
    away = wp.select([
        "pff_match_id", "minute", pl.col("away_team_id").alias("team_id"),
        pl.col("elim_prox_away").alias("elim_prox"),
    ])
    long = pl.concat([home, away])
    # Filtrar minutos con trigger
    triggers = long.filter(
        (pl.col("elim_prox") >= ELIM_PROX_TRIGGER) &
        (pl.col("minute") >= MIN_TRIGGER)
    )
    # Para cada (match, team): primer minuto que dispara
    first_trigger = (triggers.group_by(["pff_match_id", "team_id"])
                              .agg([pl.col("minute").min().alias("trigger_min"),
                                    pl.col("elim_prox").max().alias("max_elim")]))
    first_trigger = first_trigger.with_row_index("shock_id").with_columns(
        (pl.col("shock_id") + 1).alias("shock_id"))
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        first_trigger.write_parquet(cache_path, compression="snappy")
    return first_trigger


# ============================================================================
# 2. Build delta panel (analogo M14 build_delta_panel pero con pressure shocks)
# ============================================================================
def build_pressure_delta_panel(cache: bool = True) -> pl.DataFrame:
    """Per (player, channel): delta = mean(canal | post-trigger) - mean(canal | pre-trigger).

    Para cada pressure shock identifica los jugadores en campo de su equipo
    durante la ventana ±WINDOW_MIN. Computes delta por (player, shock, channel).
    Aggregate per player → 1 delta_z per (player, channel).
    """
    cache_path = _CATE / "pressure_panel.parquet"
    if cache and cache_path.exists():
        return pl.read_parquet(cache_path)

    from M01_loader_pff import load_rosters
    from M03_preprocess import player_minutes

    triggers = build_pressure_shocks(cache=True)
    print(f"  pressure shocks detectados: {triggers.height}")

    # Para cada trigger: jugadores del equipo on field en los minutos pre+post
    rows_all_channels = []
    for ch, (rel_path, col) in CHANNELS.items():
        pm = pl.read_parquet(_DERIVED / rel_path).filter(pl.col(col).is_not_null())
        pm = pm.with_columns(
            ((pl.col("period") - 1) * 45 + pl.col("minute_in_period"))
                .cast(pl.Int64).alias("minute_global")
        )
        ch_rows = []
        for trig in triggers.iter_rows(named=True):
            mid = int(trig["pff_match_id"])
            team = int(trig["team_id"])
            tmin = int(trig["trigger_min"])
            sid = int(trig["shock_id"])
            # Jugadores del equipo en campo en el minuto del trigger
            try:
                pmin = player_minutes(mid)
                on_field = pmin.filter(
                    (pl.col("team_id") == team) &
                    pl.col("minute_in").is_not_null() &
                    (pl.col("minute_in") <= tmin) &
                    (pl.col("minute_out") >= tmin)
                )["player_id"].to_list()
            except Exception:
                continue
            if not on_field:
                continue
            # Ventana pre/post
            sub = pm.filter(
                (pl.col("pff_match_id") == mid) &
                pl.col("pff_player_id").is_in(on_field) &
                (pl.col("minute_global") >= tmin - WINDOW_MIN) &
                (pl.col("minute_global") <= tmin + WINDOW_MIN) &
                (pl.col("minute_global") != tmin)   # excluir minuto trigger
            ).with_columns(
                (pl.col("minute_global") > tmin).cast(pl.Int8).alias("post")
            )
            if sub.height == 0:
                continue
            agg = (sub.group_by(["pff_player_id", "post"])
                      .agg(pl.col(col).mean().alias("m")))
            wide = agg.pivot("post", index="pff_player_id", values="m").drop_nulls()
            wcols = wide.columns
            col_pre = next((c for c in wcols if c == "0"), None)
            col_post = next((c for c in wcols if c == "1"), None)
            if col_pre is None or col_post is None:
                continue
            wide = wide.with_columns(
                (pl.col(col_post) - pl.col(col_pre)).cast(pl.Float64).alias("delta")
            ).with_columns([
                pl.lit(sid).cast(pl.UInt32).alias("shock_id"),
                pl.lit(ch).alias("channel"),
            ]).select(["pff_player_id", "shock_id", "channel", "delta"])
            ch_rows.append(wide)
        if ch_rows:
            rows_all_channels.append(pl.concat(ch_rows))

    panel = pl.concat(rows_all_channels)
    # Aggregate per (player, channel): mean delta across shocks vividos
    panel = (panel.group_by(["pff_player_id", "channel"])
                  .agg([pl.col("delta").mean().alias("delta"),
                        pl.len().alias("n_pressure_shocks")]))
    # Z-score within channel
    panel = panel.with_columns(
        ((pl.col("delta") - pl.col("delta").mean().over("channel")) /
         pl.col("delta").std().over("channel")).alias("delta_z")
    )
    # Anadir position + team
    ro = (load_rosters().select(["player_id", "team_id", "position_group"])
          .unique("player_id").rename({
              "player_id": "pff_player_id",
              "team_id":   "pff_team_id",
          }))
    panel = panel.join(ro, on="pff_player_id", how="left")
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        panel.write_parquet(cache_path, compression="snappy")
    return panel


# ============================================================================
# 3. Bayesian model (analogo M14 simplificado: 1 eta per (player, channel))
# ============================================================================
def _model_pressure(player_idx, channel_idx, pff_grade_z, y,
                     n_players, n_teams, n_positions, n_channels,
                     player_to_team, player_to_position):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    sigma_team     = numpyro.sample("sigma_team",     dist.HalfNormal(0.5).expand([n_channels]).to_event(1))
    sigma_position = numpyro.sample("sigma_position", dist.HalfNormal(0.5).expand([n_channels]).to_event(1))
    sigma_eta      = numpyro.sample("sigma_eta",      dist.HalfNormal(0.5).expand([n_channels]).to_event(1))
    sigma_eps      = numpyro.sample("sigma_eps",      dist.HalfNormal(1.0).expand([n_channels]).to_event(1))
    gamma          = numpyro.sample("gamma",          dist.Normal(0.0, 1.0).expand([n_channels]).to_event(1))
    mu_global      = numpyro.sample("mu_global",      dist.Normal(0.0, 0.5).expand([n_channels]).to_event(1))

    b_team_raw = numpyro.sample("b_team_raw", dist.Normal(0, 1).expand([n_teams, n_channels]).to_event(2))
    b_team = numpyro.deterministic("b_team", b_team_raw * sigma_team[None, :])
    b_pos_raw = numpyro.sample("b_pos_raw", dist.Normal(0, 1).expand([n_positions, n_channels]).to_event(2))
    b_position = numpyro.deterministic("b_position", b_pos_raw * sigma_position[None, :])

    L_corr = numpyro.sample("L_corr", dist.LKJCholesky(n_channels, concentration=2.0))
    L = sigma_eta[:, None] * L_corr
    eta_raw = numpyro.sample("eta_raw", dist.Normal(0, 1).expand([n_players, n_channels]).to_event(2))
    eta = numpyro.deterministic("eta", jnp.matmul(eta_raw, L.T))

    b_context = numpyro.deterministic("b_context",
        gamma[None, :] * pff_grade_z[:, None]
        + b_team[player_to_team]
        + b_position[player_to_position])

    pred = (mu_global[channel_idx]
            + b_context[player_idx, channel_idx]
            + eta[player_idx, channel_idx])
    with numpyro.plate("N", len(y)):
        numpyro.sample("obs", dist.Normal(pred, sigma_eps[channel_idx]), obs=y)


def fit_pressure(panel: pl.DataFrame,
                  num_warmup: int = NUTS_NUM_WARMUP,
                  num_samples: int = NUTS_NUM_SAMPLES,
                  num_chains: int = NUTS_NUM_CHAINS,
                  seed: int = 42) -> dict:
    import jax
    import numpyro
    from numpyro.infer import MCMC, NUTS

    numpyro.set_host_device_count(num_chains)

    df = panel.filter(
        pl.col("delta_z").is_not_null() &
        pl.col("position_group").is_not_null() &
        pl.col("pff_team_id").is_not_null()
    ).to_pandas()

    grades = pl.read_parquet(_PFF_GRADES).select(["pff_player_id", "pff_grade_mean"])
    grades = grades.with_columns(
        ((pl.col("pff_grade_mean") - pl.col("pff_grade_mean").mean()) /
         pl.col("pff_grade_mean").std().fill_null(1.0))
        .fill_null(0.0).alias("pff_grade_z")
    ).select(["pff_player_id", "pff_grade_z"]).to_pandas()
    df = df.merge(grades, on="pff_player_id", how="left").fillna({"pff_grade_z": 0.0})

    players = sorted(df["pff_player_id"].unique())
    teams = sorted(df["pff_team_id"].dropna().unique())
    positions = sorted(df["position_group"].unique())
    channels = sorted(df["channel"].unique())
    p_to_idx = {p: i for i, p in enumerate(players)}
    t_to_idx = {t: i for i, t in enumerate(teams)}
    pos_to_idx = {p: i for i, p in enumerate(positions)}
    ch_to_idx = {c: i for i, c in enumerate(channels)}

    p_to_team, p_to_pos, p_to_grade = {}, {}, {}
    for r in df.itertuples(index=False):
        if not np.isnan(r.pff_team_id):
            p_to_team[r.pff_player_id] = t_to_idx[int(r.pff_team_id)]
        p_to_pos[r.pff_player_id] = pos_to_idx[r.position_group]
        p_to_grade[r.pff_player_id] = r.pff_grade_z
    player_to_team_arr = np.array([p_to_team.get(p, 0) for p in players], dtype=np.int32)
    player_to_position_arr = np.array([p_to_pos.get(p, 0) for p in players], dtype=np.int32)
    pff_grade_z_arr = np.array([p_to_grade.get(p, 0.0) for p in players], dtype=np.float32)

    df = df[df["pff_team_id"].notna()].copy()
    player_idx = df["pff_player_id"].map(p_to_idx).values.astype(np.int32)
    channel_idx = df["channel"].map(ch_to_idx).values.astype(np.int32)
    y = df["delta_z"].values.astype(np.float32)

    print(f"  NUTS pressure: N={len(y)}, players={len(players)}, "
          f"teams={len(teams)}, positions={len(positions)}, channels={len(channels)}")
    print(f"  warmup={num_warmup}, samples={num_samples}, chains={num_chains}")

    kernel = NUTS(_model_pressure, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                 num_chains=num_chains, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(seed),
             player_idx, channel_idx, pff_grade_z_arr, y,
             len(players), len(teams), len(positions), len(channels),
             player_to_team_arr, player_to_position_arr,
             extra_fields=("diverging", "accept_prob"))

    samples = mcmc.get_samples()
    extra = mcmc.get_extra_fields()
    diverging = np.asarray(extra.get("diverging", np.zeros(0)))
    accept_prob = np.asarray(extra.get("accept_prob", np.zeros(0)))
    n_div = int(diverging.sum()) if diverging.size else 0
    print(f"  divergencias HMC: {n_div}/{diverging.size}, "
          f"accept mean: {accept_prob.mean() if accept_prob.size else 0:.3f}")

    return {
        "samples":   {k: np.array(v) for k, v in samples.items()},
        "p_to_idx":  p_to_idx,
        "t_to_idx":  t_to_idx,
        "pos_to_idx": pos_to_idx,
        "ch_to_idx": ch_to_idx,
        "n_diverging": n_div,
    }


def compute_pressure_response(fit: dict) -> pl.DataFrame:
    s = fit["samples"]
    p_to_idx = fit["p_to_idx"]
    ch_to_idx = fit["ch_to_idx"]
    eta = s["eta"]
    idx_to_pid = {v: k for k, v in p_to_idx.items()}
    idx_to_ch = {v: k for k, v in ch_to_idx.items()}
    rows = []
    for i in range(eta.shape[1]):
        for k in range(eta.shape[2]):
            x = eta[:, i, k]
            rows.append({
                "pff_player_id": idx_to_pid[i],
                "channel": idx_to_ch[k],
                "pressure_eta_mean": float(x.mean()),
                "pressure_eta_sd":   float(x.std()),
                "pressure_eta_lo80": float(np.quantile(x, 0.10)),
                "pressure_eta_hi80": float(np.quantile(x, 0.90)),
                "p_pressure_positive": float((x > 0).mean()),
            })
    return pl.DataFrame(rows)


def compute_pressure_index(fit: dict) -> pl.DataFrame:
    s = fit["samples"]
    p_to_idx = fit["p_to_idx"]
    eta = s["eta"]
    idx_samples = eta.mean(axis=2)
    idx_to_pid = {v: k for k, v in p_to_idx.items()}
    rows = []
    for i in range(eta.shape[1]):
        x = idx_samples[:, i]
        rows.append({
            "pff_player_id": idx_to_pid[i],
            "pressure_response_idx":   float(x.mean()),
            "pressure_response_sd":    float(x.std()),
            "pressure_response_lo80":  float(np.quantile(x, 0.10)),
            "pressure_response_hi80":  float(np.quantile(x, 0.90)),
            "p_pressure_clutch_positive": float((x > 0).mean()),
        })
    return pl.DataFrame(rows)


def compute_all(overwrite: bool = False) -> dict[str, Path]:
    print("[M14b] Building pressure delta panel...")
    panel = build_pressure_delta_panel(cache=True)
    print(f"  panel: {panel.height} rows, {panel['pff_player_id'].n_unique()} jugadores")

    print("[M14b] Fit NUTS bayesian (4 chains x 2000 iter)...")
    fit = fit_pressure(panel)

    out_dir = _CATE / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pressure_nuts.pkl", "wb") as f:
        pickle.dump(fit, f)

    per_chan = compute_pressure_response(fit)
    per_chan.write_parquet(_CATE / "pressure_response_per_channel.parquet")

    idx = compute_pressure_index(fit)
    idx.write_parquet(_CATE / "pressure_response.parquet")

    print(f"\n[M14b] Saved:")
    print(f"  {out_dir / 'pressure_nuts.pkl'}")
    print(f"  pressure_response_per_channel.parquet ({per_chan.height})")
    print(f"  pressure_response.parquet ({idx.height})")
    return {
        "model": out_dir / "pressure_nuts.pkl",
        "per_channel": _CATE / "pressure_response_per_channel.parquet",
        "index": _CATE / "pressure_response.parquet",
    }


if __name__ == "__main__":
    import sys as _sys
    overwrite = "overwrite" in _sys.argv
    compute_all(overwrite=overwrite)
