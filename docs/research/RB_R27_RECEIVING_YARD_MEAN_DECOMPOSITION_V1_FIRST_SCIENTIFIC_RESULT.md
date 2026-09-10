# RB R27 Receiving-Yard Mean Decomposition V1 — First Scientific Result

**Status:** FIRST SCIENTIFIC RESULT PRESERVED — DO NOT RETUNE R27 V1

## Exact authority

- Study: `RB_R27_RECEIVING_YARD_MEAN_DECOMPOSITION_V1`
- Branch at execution: `research-rb-r27-receiving-yard-mean-decomposition-v1`
- Head SHA: `b7cfe5b2c450765208672fc6e2b017d0cb0c0dde`
- Frozen plan commit: `5333d7e1cc33dcb567d03c924c774afb6877e932`
- Frozen plan blob: `4ad4801dbe600238dd7f1090df0a6596ff8a6a46`
- Workflow run: `34423546037`
- Job: `102703879430`
- Artifact: `10132290573`
- Artifact name: `rb-r27-receiving-yard-mean-decomposition-v1`
- Artifact digest: `sha256:cd5c9e26efa44c47ef3374e4deefe9038dc504c155799b48548ad64746e10e45`
- Disposition: `R27_R26_OPPORTUNITY_TRANSLATION_MIXED_OR_FAIL`
- Frozen gates passed: **24/27**
- Structural Gates 1–12: **12/12 PASS**
- Production changed: **false**
- R22 changed: **false**
- Sportsbook inputs upstream: **0**
- Future outcomes used in features: **0**

## Exact scientific comparison

R27 V1 held receiving efficiency fixed and changed only the opportunity path:

- production baseline = `baseline_targets × production_ypt`
- R27 candidate = `exact R26 candidate targets × same production_ypt`

Pooled results:

- VACANCY_ACTIVE receiving-yard MAE: **-1.0967854669869492%**
- VACANCY_ACTIVE RMSE: **-1.3337252245802733%**
- VACANCY_ACTIVE absolute bias improved from `-2.899589306569956` to `-1.927584987996026`
- VACANCY_ACTIVE p90 absolute error: **+2.077705464816737%**
- VACANCY_ACTIVE target MAE: **-2.0869346310231074%**
- VACANCY_ACTIVE reception MAE: **-1.8026686460321795%**
- VACANCY_INCUMBENT receiving-yard MAE: **-0.7291009642442892%**
- VACANCY_RB1_INCUMBENT receiving-yard MAE: **+2.820339661055238%**
- VACANCY_RB2PLUS_INCUMBENT receiving-yard MAE: **-3.3577795492306994%**
- ALL-RB receiving-yard MAE: **-0.23517744112927508%**
- ALL-RB RMSE: **-0.28673056274011177%**
- WEEK1 receiving-yard MAE: **-3.3081859534149105%**
- Seasons with VACANCY_ACTIVE MAE improvement: **4/6**

VACANCY_ACTIVE receiving-yard MAE by season, candidate vs baseline percent change:

- 2020: **-1.5825051129687462%**
- 2021: **-2.35284400838629%**
- 2022: **-4.254481456055393%**
- 2023: **+4.915812026979438%**
- 2024: **-3.559057496015905%**
- 2025: **+2.869407749239672%**

## Failed frozen gates

- Gate 17 — `vacancy_active_p90_worsens_no_more_than_2pct`: **FAIL** (`+2.077705464816737%`)
- Gate 19 — `no_season_vacancy_mae_worsens_over_3pct`: **FAIL** (2023 `+4.915812026979438%`)
- Gate 20 — `rb1_and_rb2plus_each_worsen_no_more_than_1_5pct`: **FAIL** (RB1 `+2.820339661055238%`; RB2+ improved `-3.3577795492306994%`)

Gate 27, the predeclared material-support gate, **PASSED**: pooled VACANCY_ACTIVE receiving-yard MAE improved **1.0968%**, exceeding the required **0.50%**.

## Scientific interpretation authorized by the frozen result

The R26 opportunity mechanism has real receiving-yard support: it improved targets, receptions, pooled vacancy receiving-yard MAE/RMSE, Week 1, and the material Gate 27. R27 V1 nevertheless does **not** qualify for mean integration because the fixed production efficiency translation is not sufficiently stable across the high-error tail, 2023, and vacancy RB1 incumbents.

The especially useful decomposition is that vacancy RB1 targets/receptions improve while vacancy RB1 receiving yards worsen. That is consistent with the next unresolved error source living in receiving-efficiency translation rather than being evidence that the qualified R26 opportunity mechanism should be discarded.

This result authorizes a separately frozen **R27B strict-prior receiving-efficiency study on top of fixed R26 opportunity**. It does not authorize changing R27 V1 gates, changing the R26 opportunity mechanism, changing R22, or changing production.

## Non-retuning rule

R27 V1 is closed as observed. Do not rerun it with altered gates, cohorts, efficiency formulas, R26 logic, season exclusions, or post-result thresholds. Mechanical lineage remains preserved separately. Any new receiving-efficiency hypothesis must be named, frozen, implemented, locked, and evaluated as a new study.
