# PCJ: Perfil Clutch del Jugador

TFM Master Big Data Aplicado al Scouting Deportivo (Sport Data Campus).

Estimacion causal del efecto del shock emocional (gol a favor / gol en contra) sobre el comportamiento del jugador en ventanas pre vs post de ±10 min, medido en cuatro canales (Empuje Ofensivo, Solidez Defensiva, Inteligencia Espacial Off-ball, Pulso Fisico) sobre PFF FC World Cup Qatar 2022.

Output: ranking bidireccional de jugadores clutch (Indice Remontador + Indice Cerrojo) con intervalos de credibilidad bayesianos.

## Estructura

```text
src/
├── extract/                # extractores raw JSON -> parquet (lossless)
├── M01_loader_pff.py       # API PFF (events, tracking, metadata, rosters + vistas)
├── M02_loader_public.py    # API Wyscout + StatsBomb (polars nativo)
├── M03_preprocess.py       # direction, score state, minutos, enrich_events
├── M04_wp.py               # Win Probability bayesiana (numpyro SVI) + leverage + ET + pen
├── M05_psxg.py             # Post-shot xG (LightGBM + Optuna 60 trials, AUC 0.974 / WC22 0.976)
├── M06_nearmiss.py         # Near-miss (5 tipos, offside con 360 freeze-frame + spec curve)
├── M07_shocks.py           # Shocks emocionales (172 goles) + ventanas ±10min por jugador
├── M08_ataque.py           # Canal Empuje Ofensivo via atomic-VAEP (CatBoost + Optuna 30)
├── M09_defensa.py          # Canal Solidez Defensiva: score_def + VDEP + def_third + Pressing
├── M10_offball.py          # Canal Off-ball: OBSO + C-OBSO (PPCF Z02 + xG grid + tracking 25Hz)
├── (M11-M16)               # pipeline restante (ver docs/ARCHITECTURE.md)
├── Z01_vaep.py             # B01 atomic-VAEP (building block, usado por M08/M09)
└── Z02_pitch_control.py    # B02 PPCF Spearman 2018 vectorizado (building block, usado por M10)

notebooks/
└── M10_run.ipynb           # entry point para correr M10 a 25 Hz partido a partido
                            # (resumable: cachea cada partido aparte, retoma tras interrupcion)
```

Datos, documentacion interna del proyecto y outputs intermedios estan fuera del repo (`.gitignore`).

## Stack

Python (polars, pyarrow, scikit-learn, xgboost, catboost, lightgbm, numpyro/jax,
optuna, socceraction).
