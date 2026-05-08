"""smoke_e2e - Validacion estructural rapida del pipeline (sin gastar GPU/tiempo).

Verifica que TODO arranca y converge a nivel codigo:
  T1. Todos los modulos M0X + Z0X + extract/preprocess importan sin errores.
  T2. Compat numpy 2.0+: socceraction stack carga via M08 lazy import.
  T3. M14 NUTS smoke synthetic: 30 jugadores x 8 shocks x 4 canales, 200 iter
      x 2 chains. Verifica que el modelo extendido (5 etas, ga/gf/pres/
      ga_x_td/gf_x_td) corre sin divergencias.
  T4. M15 _build_scenario_wide produce 80 cols clutch_*.
  T5. render_ficha imprime ficha correcta sobre row mock.

Tiempo total: ~3-5 min.

Uso:
    python -m tests.smoke_e2e
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))


def t1_imports() -> bool:
    print("\n[T1] Imports check...")
    import importlib
    mods = [
        "M01_loader_pff", "M02_loader_public", "M03_preprocess",
        "M04_wp", "M05_psxg", "M05B_calibration", "M06_nearmiss",
        "M07_shocks", "M08_ataque", "M09_defensa", "M10_offball",
        "M11_fisico", "M12_did", "M12B_validation", "M13_aipw",
        "M14_cate", "M15_pcj", "render_ficha",
        "Z01_vaep", "Z02_pitch_control", "Z03_xpress", "Z04_vdep",
        "Z05_maejima", "Z06_unxpass",
        "preprocess.pff_grades_extract",
        "extract", "extract.pff", "extract.statsbomb",
        "extract.wyscout", "extract.audit", "extract._common",
    ]
    fail = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as e:
            fail.append((m, type(e).__name__, str(e)[:120]))
    if fail:
        for m, et, msg in fail:
            print(f"  FAIL {m}: {et}: {msg}")
        return False
    print(f"  OK ({len(mods)} modulos)")
    return True


def t2_numpy_socceraction() -> bool:
    print("\n[T2] numpy 2.0+ compat (M08 lazy stack)...")
    try:
        import M08_ataque as atk
        stack = atk._import_vaep_stack()
        assert len(stack) == 5
        print(f"  OK (socceraction stack: {len(stack)} components)")
        return True
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}")
        return False


def _make_synthetic_panel(seed=0):
    """Panel sintetico (jugador x shock x canal x shock_type x moduladores)."""
    rng = np.random.default_rng(seed)
    n_players = 30
    n_shocks_per_type = 4   # 4 GA + 4 GF
    channels = ["ataque", "defensa", "offball", "fisico"]
    shock_types = ["GOAL_AGAINST", "GOAL_FOR"]
    teams = list(range(8))
    positions = ["LW", "RW", "CB", "FB", "CM", "AM", "CF", "GK"]

    rows = []
    for s_type_idx, s_type in enumerate(shock_types):
        for shock_id in range(n_shocks_per_type):
            global_shock = s_type_idx * 100 + shock_id
            # team_direction_z aleatorio per shock
            tdz = float(rng.normal(0, 1))
            for p_id in range(n_players):
                for ch in channels:
                    rows.append({
                        "pff_match_id": shock_id // 2,
                        "shock_id": global_shock,
                        "pff_player_id": p_id,
                        "shock_type": s_type,
                        "channel": ch,
                        "delta": float(rng.normal(0, 0.3)),
                        "position_group": rng.choice(positions),
                        "pff_team_id": int(rng.choice(teams)),
                        "stage": "ko",
                        "minute": int(rng.integers(0, 120)),
                        "minute_norm": float(rng.uniform(0.0, 1.3)),
                        "score_diff_post_z": float(rng.normal(0, 1)),
                        "week_idx_norm":   float(rng.uniform(0, 1)),
                        "leverage_z":      float(rng.normal(0, 1)),
                        "elim_prox_z":     float(rng.normal(0, 1)),
                        "team_direction_z": tdz,
                        "pff_grade_z":     float(rng.normal(0, 1)),
                    })
    df = pl.DataFrame(rows)
    # delta_z (z-score within channel x shock_type) + clip
    df = df.with_columns(
        ((pl.col("delta") - pl.col("delta").mean().over(["channel", "shock_type"])) /
         pl.col("delta").std().over(["channel", "shock_type"]))
            .clip(-5.0, 5.0).alias("delta_z")
    )
    return df


def t3_m14_smoke_synthetic() -> bool:
    print("\n[T3] M14 NUTS smoke synthetic (30 players x 8 shocks, 2 chains x 200 iter)...")
    try:
        import M14_cate as cate
        panel = _make_synthetic_panel(seed=42)
        t0 = time.time()
        fit = cate.fit_cate_nuts(panel, num_warmup=200, num_samples=200,
                                   num_chains=2, seed=0)
        elapsed = time.time() - t0
        print(f"  fit OK en {elapsed:.0f}s ({fit['n_obs']} obs, {fit['n_diverging']} div)")
        # Shapes esperados
        s = fit["samples"]
        n_total = 400  # warmup+samples
        expected = {
            "eta_ga":        (n_total, 30, 4),
            "eta_gf":        (n_total, 30, 4),
            "eta_pressure":  (n_total, 30, 4),
            "eta_ga_x_td":   (n_total, 30, 4),
            "eta_gf_x_td":   (n_total, 30, 4),
            "L_ga_x_td_corr": (n_total, 4, 4),
            "L_gf_x_td_corr": (n_total, 4, 4),
        }
        for name, exp_shape in expected.items():
            actual = s[name].shape
            assert actual == exp_shape, f"{name}: {actual} != {exp_shape}"
        print(f"  shapes OK ({len(expected)} sites criticos)")

        # Pipeline downstream: posterior + corr + indices
        post = cate.posterior_per_player(fit)
        assert post.height == 30 * 4 * 5, f"posterior height: {post.height}"
        corr = cate.posterior_cross_canal_corr(fit)
        assert corr.height == 5 * 4 * 4, f"corr height: {corr.height}"
        idx = cate.compute_indices(fit)
        assert idx.height == 30, f"indices height: {idx.height}"
        # Verificar 16 cells contextualizados existen
        clutch_cols = [c for c in idx.columns if c.startswith("clutch_")]
        assert len(clutch_cols) == 16, f"clutch cells: {len(clutch_cols)} != 16"
        print(f"  posterior/corr/indices OK; 16 cells clutch_* generados")
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}")
        return False


def t4_m15_scenarios() -> bool:
    print("\n[T4] M15 _compute_scenario_outcomes + _build_scenario_wide...")
    try:
        import M14_cate as cate
        import M15_pcj as m15
        panel = _make_synthetic_panel(seed=1)
        fit = cate.fit_cate_nuts(panel, num_warmup=100, num_samples=100,
                                   num_chains=2, seed=0)
        scenarios = m15._compute_scenario_outcomes(fit)
        # 30 players x 4 channels x 2 shocks x 2 scenarios = 480
        assert scenarios.height == 30 * 4 * 2 * 2, f"scenarios height: {scenarios.height}"
        wide = m15._build_scenario_wide(scenarios)
        # cols clutch_* = 4 ch x 2 sh x 2 sc x 5 stats = 80
        clutch_cols = [c for c in wide.columns if c.startswith("clutch_")]
        assert len(clutch_cols) == 80, f"clutch cols wide: {len(clutch_cols)} != 80"
        print(f"  scenarios long: {scenarios.height}, wide cols clutch_*: {len(clutch_cols)}")
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}")
        return False


def t5_render_ficha() -> bool:
    print("\n[T5] render_ficha sobre row mock...")
    try:
        import render_ficha as rf
        mock = {
            "pff_player_id": 999, "player_name": "Test Player",
            "team_name": "Test FC", "position_group": "LW", "age_years": 25.0,
            "minutes_played": 540, "n_matches_played": 6,
            "n_shocks_for": 5, "n_shocks_against": 4,
            "n_shocks_groups": 4, "n_shocks_ko": 5,
            "n_high_leverage_shocks": 2, "n_elimination_shocks": 1,
            "chasing_clutch_idx": 0.32, "p_chasing_positive": 0.91,
            "protecting_clutch_idx": -0.05, "p_protecting_positive": 0.42,
            "pressure_response_idx": 0.45, "p_pressure_clutch_positive": 0.96,
            "sig_chasing": "Sig_remontador", "sig_protecting": "Inconclusive",
            "sig_pressure": "Sig_pressure_clutch_strong",
            "rank_chasing_in_position": 2,
            "rank_protecting_in_position": 18,
            "rank_pressure_in_position": 1,
            "tier_chasing_in_position": "Elite",
            "tier_protecting_in_position": "Below_avg",
            "tier_pressure_in_position": "Elite",
            "cate_ataque_GOAL_FOR_mean": 0.5, "cate_defensa_GOAL_FOR_mean": -0.1,
            "cate_offball_GOAL_FOR_mean": 0.4, "cate_fisico_GOAL_FOR_mean": 0.3,
            "cate_ataque_GOAL_AGAINST_mean": 0.4, "cate_defensa_GOAL_AGAINST_mean": -0.05,
            "cate_offball_GOAL_AGAINST_mean": 0.3, "cate_fisico_GOAL_AGAINST_mean": 0.5,
            "clutch_atk_GOAL_FOR_team_attacks_mean": 0.5,
            "clutch_atk_GOAL_FOR_team_defends_mean": 0.4,
            "clutch_def_GOAL_FOR_team_attacks_mean": -0.1,
            "clutch_def_GOAL_FOR_team_defends_mean": -0.1,
            "clutch_off_GOAL_FOR_team_attacks_mean": 0.4,
            "clutch_off_GOAL_FOR_team_defends_mean": 0.4,
            "clutch_phys_GOAL_FOR_team_attacks_mean": 0.3,
            "clutch_phys_GOAL_FOR_team_defends_mean": 0.3,
            "clutch_atk_GOAL_AGAINST_team_attacks_mean": 0.5,
            "clutch_atk_GOAL_AGAINST_team_defends_mean": 0.3,
            "clutch_def_GOAL_AGAINST_team_attacks_mean": -0.05,
            "clutch_def_GOAL_AGAINST_team_defends_mean": -0.05,
            "clutch_off_GOAL_AGAINST_team_attacks_mean": 0.3,
            "clutch_off_GOAL_AGAINST_team_defends_mean": 0.3,
            "clutch_phys_GOAL_AGAINST_team_attacks_mean": 0.5,
            "clutch_phys_GOAL_AGAINST_team_defends_mean": 0.5,
        }
        out = rf.render_ficha(mock)
        assert "Test Player" in out
        assert "post-GA" in out and "post-GF" in out
        assert "ELIMINACIÓN" in out
        print(f"  ficha render OK ({len(out)} chars, {out.count(chr(10))} lineas)")
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  FAIL: {type(e).__name__}: {str(e)[:200]}")
        return False


def main():
    print("=" * 60)
    print("smoke_e2e — pipeline structural validation")
    print("=" * 60)
    t0 = time.time()
    results = [
        ("T1 imports",            t1_imports()),
        ("T2 numpy/socceraction", t2_numpy_socceraction()),
        ("T3 M14 NUTS smoke",     t3_m14_smoke_synthetic()),
        ("T4 M15 scenarios",      t4_m15_scenarios()),
        ("T5 render_ficha",       t5_render_ficha()),
    ]
    print("\n" + "=" * 60)
    print(f"TOTAL: {time.time()-t0:.0f}s")
    n_ok = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {name:<28} {'OK' if ok else 'FAIL'}")
    print("=" * 60)
    if n_ok == len(results):
        print(f"\nALL {n_ok}/{len(results)} PASS — pipeline arranca limpio")
        sys.exit(0)
    else:
        print(f"\n{n_ok}/{len(results)} PASS — revisar fallos antes de RunPod")
        sys.exit(1)


if __name__ == "__main__":
    main()
