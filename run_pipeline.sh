#!/usr/bin/env bash
# run_pipeline.sh - Ejecuta el pipeline E2E completo en orden DAG.
#
# Diseño: cada etapa = 1 modulo Python invocado con su `compute_all`.
# Etapas independientes corren en paralelo (Z03||Z04||Z05||Z06).
# Cada etapa loguea a logs/<etapa>.log y aborta el pipeline si falla.
#
# Uso:
#   ./run_pipeline.sh                  # corrida completa
#   FORCE_CLEAN=1 ./run_pipeline.sh    # borra derived/* antes
#   SKIP_M10=1 ./run_pipeline.sh       # salta M10 (~4-5h, debug rapido)
#   SMOKE=1 ./run_pipeline.sh          # smoke E2E con N reducidos
#
# Variables de entorno opcionales:
#   PYTHON           : binario python a usar (default: python)
#   N_TRIALS_OPTUNA  : trials Optuna global (default: 30; smoke=5)
#   FORCE_CLEAN      : si =1 borra data/parquet/derived/* antes
#   SKIP_M10         : si =1 salta M10 (la etapa más cara, ~4-5h)
#   SKIP_M14         : si =1 salta M14 (NUTS HMC ~25min)
#   SMOKE            : si =1 reduce trials/steps para smoke rapido
#
# Estrategia de coste para RunPod $3-5:
#   - Pod CPU 16-32 cores ($0.10-0.20/h) > GPU para este pipeline (M14 NUTS
#     en CPU multi-chain es lo único GPU-acelerable, ~25min CPU vs ~5min GPU).
#   - Total estimado ~5-7h en CPU pod 16 cores.
#   - El cuello es M10 OBSO 25Hz × 64 partidos (~4-5h).

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
LOGS_DIR="logs"
DERIVED="data/parquet/derived"
N_TRIALS="${N_TRIALS_OPTUNA:-30}"

mkdir -p "$LOGS_DIR"

if [[ "${SMOKE:-0}" == "1" ]]; then
    echo "[smoke] modo SMOKE activado: trials=5, M04 SVI 1000 steps"
    N_TRIALS=5
    export SMOKE_M04_STEPS=1000
fi

if [[ "${FORCE_CLEAN:-0}" == "1" ]]; then
    echo "[clean] borrando derived/* + cache/vaep..."
    rm -rf "$DERIVED"/*
    rm -rf cache/vaep
fi

# ------------------------------------------------------------------------
# Helper: ejecuta una etapa con log + timing + error check.
# ------------------------------------------------------------------------
stage() {
    local name="$1"; shift
    local log="$LOGS_DIR/$name.log"
    local t0=$(date +%s)
    echo ""
    echo "=== [$name] $(date '+%H:%M:%S') === log: $log"
    if "$@" > "$log" 2>&1; then
        local elapsed=$(( $(date +%s) - t0 ))
        echo "    OK ($elapsed s)"
    else
        local elapsed=$(( $(date +%s) - t0 ))
        echo "    FAIL ($elapsed s)"
        echo ""
        echo "Ultimas 30 lineas de $log:"
        tail -30 "$log"
        exit 1
    fi
}

# ------------------------------------------------------------------------
# 1. Preprocesamiento auxiliar (PFF grades para priors M14)
# ------------------------------------------------------------------------
stage "00_pff_grades" "$PYTHON" -m src.preprocess.pff_grades_extract

# ------------------------------------------------------------------------
# 2. Pipeline lineal M03 -> M07
# ------------------------------------------------------------------------
stage "03_preprocess"  "$PYTHON" -c "
import sys; sys.path.insert(0,'src')
import M03_preprocess as m
m.cache_all_enriched(overwrite=False)
"

stage "04_wp"          "$PYTHON" -c "
import sys; sys.path.insert(0,'src')
import os
import M04_wp as m
n_steps = int(os.environ.get('SMOKE_M04_STEPS', '3000'))
X, info = m.build_training_matrix(cache=True)
print(f'  training matrix: {info}')
fit_path = m._MODEL / 'wp_regulation.pkl'
if fit_path.exists():
    fit = m.load_fit(fit_path)
else:
    Xtr, Xv = m.train_val_split(X, val_frac=0.2, seed=42)
    fit = m.fit_wp(Xtr, n_steps=n_steps)
    Xc = m.build_wc22_groups_calib_matrix()
    fit['temperature'] = m.fit_temperature(fit, Xc)
    m.save_fit(fit)
gctx = m.build_wc22_group_context()
m.cache_all_wp(fit, overwrite=False, group_ctx=gctx)
"

stage "05_psxg"        "$PYTHON" -c "
import sys; sys.path.insert(0,'src')
import os
import M05_psxg as m
n_trials = int(os.environ.get('N_TRIALS_OPTUNA', '60'))
train = m.build_training_shots(cache=True)
fit_path = m._MODEL / 'psxg_lgb.pkl'
if fit_path.exists():
    fit = m.load_fit(fit_path)
else:
    fit = m.fit_psxg(train, n_folds=5, seed=42, n_trials=n_trials)
    m.save_fit(fit)
m.cache_wc22_psxg(fit, overwrite=False)
"

stage "05b_calibration" "$PYTHON" -m src.M05B_calibration

stage "06_nearmiss"    "$PYTHON" -c "
import sys; sys.path.insert(0,'src')
import M06_nearmiss as m
m.build_near_miss_table(cache=True, overwrite=False)
"

stage "07_shocks"      "$PYTHON" -c "
import sys; sys.path.insert(0,'src')
import M07_shocks as m
m.build_shocks_table(cache=True, overwrite=False)
"

# ------------------------------------------------------------------------
# 3. M08 partA: train atomic-VAEP + apply WC22 (sin aggregate)
# ------------------------------------------------------------------------
stage "08a_train_vaep" "$PYTHON" -c "
import sys; sys.path.insert(0,'src')
import os
import M08_ataque as atk
n_trials = int(os.environ.get('N_TRIALS_OPTUNA', '30'))
train_df = atk.build_training_atomic(overwrite=False)
wc22_df  = atk.build_wc22_atomic(overwrite=False)
fit_path = atk._MODEL_DIR / 'vaep_atk_meta.pkl'
if fit_path.exists():
    fit = atk.load_models()
else:
    fit = atk.train_vaep_model(train_df, n_folds=5, seed=42,
                                tune=True, n_trials=n_trials)
    atk.save_models(fit)
atk.build_sb_to_pff_player_map(cache=True)
print('  M08 partA done; per_minute pendiente hasta despues de Z03-Z06')
"

# ------------------------------------------------------------------------
# 4. Z03 / Z04 / Z05 / Z06 building blocks SOTA — INDEPENDIENTES (paralelo)
# ------------------------------------------------------------------------
echo ""
echo "=== [Z03-Z06] arrancando 4 building blocks en paralelo $(date '+%H:%M:%S')"
"$PYTHON" -c "import sys; sys.path.insert(0,'src'); import Z03_xpress as m; m.compute_all(overwrite=False, n_trials=$N_TRIALS)" > "$LOGS_DIR/z03_xpress.log" 2>&1 &
PID_Z03=$!
"$PYTHON" -c "import sys; sys.path.insert(0,'src'); import Z04_vdep as m; m.compute_all(overwrite=False, n_trials=$N_TRIALS)" > "$LOGS_DIR/z04_vdep.log" 2>&1 &
PID_Z04=$!
"$PYTHON" -c "import sys; sys.path.insert(0,'src'); import Z05_maejima as m; m.compute_all(overwrite=False)" > "$LOGS_DIR/z05_maejima.log" 2>&1 &
PID_Z05=$!
"$PYTHON" -c "import sys; sys.path.insert(0,'src'); import Z06_unxpass as m; m.compute_all(overwrite=False, n_trials=$N_TRIALS)" > "$LOGS_DIR/z06_unxpass.log" 2>&1 &
PID_Z06=$!

# Wait + check
fail=0
for pid_var in PID_Z03 PID_Z04 PID_Z05 PID_Z06; do
    pid=${!pid_var}
    if wait "$pid"; then
        echo "    $pid_var ($pid) OK"
    else
        echo "    $pid_var ($pid) FAIL"
        fail=1
    fi
done
if [[ $fail == 1 ]]; then
    echo ""
    echo "Logs Z03-Z06 (ultimas 20 lineas cada):"
    for log in z03_xpress z04_vdep z05_maejima z06_unxpass; do
        echo "--- $log ---"
        tail -20 "$LOGS_DIR/$log.log"
    done
    exit 1
fi

# ------------------------------------------------------------------------
# 5. M08 partB: aggregate per_minute + per_shock_window (con un-xpass)
# ------------------------------------------------------------------------
stage "08b_aggregate" "$PYTHON" -c "
import sys; sys.path.insert(0,'src')
import importlib, M08_ataque as atk
importlib.reload(atk)
fit = atk.load_models()
wc22_df = atk.build_wc22_atomic(overwrite=False)
wc22_with_vaep = atk.apply_vaep_to_wc22(fit, wc22_df)
mapping = atk.build_sb_to_pff_player_map(cache=True)
per_min = atk.aggregate_per_player_minute(wc22_with_vaep, cache=True)
atk.aggregate_per_shock_window(per_min, mapping, cache=True)
"

# ------------------------------------------------------------------------
# 6. M09 defensa (lee Z03/Z04/Z05)
# ------------------------------------------------------------------------
stage "09_defensa" "$PYTHON" -c "
import sys; sys.path.insert(0,'src')
import M09_defensa as m
m.aggregate_per_player_minute(cache=True)
m.aggregate_per_shock_window(cache=True)
"

# ------------------------------------------------------------------------
# 7. M10 OBSO (CARO: ~4-5h, paralelizable a futuro)
# ------------------------------------------------------------------------
if [[ "${SKIP_M10:-0}" != "1" ]]; then
    stage "10_offball" "$PYTHON" -c "
import sys; sys.path.insert(0,'src')
import M10_offball as m
m.aggregate_per_player_minute(cache=True)
m.aggregate_per_shock_window(cache=True)
"
else
    echo "    [10_offball] SKIP_M10=1, saltado"
fi

# ------------------------------------------------------------------------
# 8. M11 fisico
# ------------------------------------------------------------------------
stage "11_fisico" "$PYTHON" -c "
import sys; sys.path.insert(0,'src')
import M11_fisico as m
m.build_raw_per_minute(cache=True)
m.cache_score_phys(overwrite=False, n_steps=4000)
m.aggregate_per_shock_window(cache=True)
"

# ------------------------------------------------------------------------
# 9. M12 DiD + M12B validation (paralelo: M12B independiente de M13)
# ------------------------------------------------------------------------
stage "12_did" "$PYTHON" -c "
import sys; sys.path.insert(0,'src')
import M12_did as m
m.compute_all(cache=True, overwrite=False)
"

stage "12b_validation" "$PYTHON" -m src.M12B_validation

# ------------------------------------------------------------------------
# 10. M13 AIPW (cuasi-experimento near-miss)
# ------------------------------------------------------------------------
stage "13_aipw" "$PYTHON" -c "
import sys; sys.path.insert(0,'src')
import M13_aipw as m
m.compute_all(cache=True, overwrite=False)
"

# ------------------------------------------------------------------------
# 11. M14 CATE NUTS HMC (~25min)
# ------------------------------------------------------------------------
if [[ "${SKIP_M14:-0}" != "1" ]]; then
    stage "14_cate" "$PYTHON" -c "
import sys; sys.path.insert(0,'src')
import M14_cate as m
m.compute_all(cache=True, overwrite=False,
              num_warmup=m.NUTS_NUM_WARMUP,
              num_samples=m.NUTS_NUM_SAMPLES,
              num_chains=m.NUTS_NUM_CHAINS)
"
else
    echo "    [14_cate] SKIP_M14=1, saltado"
fi

# ------------------------------------------------------------------------
# 12. M15 PCJ ensamblaje final
# ------------------------------------------------------------------------
stage "15_pcj" "$PYTHON" -m src.M15_pcj

# ------------------------------------------------------------------------
# Smoke final: tamano de outputs
# ------------------------------------------------------------------------
echo ""
echo "=== PIPELINE COMPLETO === $(date '+%H:%M:%S')"
echo ""
echo "Outputs:"
ls -lh outputs/pcj_table.parquet 2>/dev/null || echo "  (sin pcj_table.parquet)"
ls -lh outputs/pcj_aux/ 2>/dev/null || echo "  (sin pcj_aux/)"
echo ""
echo "Logs:"
ls -lh "$LOGS_DIR"/
