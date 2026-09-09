# RB R26 — Vacancy-Gated R9 Retrospective V1 Implementation Lock

Status: **LOCKED BEFORE CANDIDATE EXECUTION**
Date: 2026-09-09
Parent frozen plan: `docs/research/RB_R26_VACANCY_GATED_R9_RETROSPECTIVE_V1_FROZEN_PLAN.md`
Production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`

This document fixes implementation details that were not numerically specified in the parent plan. No R26 candidate outcomes have been inspected before this lock.

## Fold map

Use exactly one immediately-prior regular season as fitted training data for each retrospective test season:

- train 2019 -> test 2020
- train 2020 -> test 2021
- train 2021 -> test 2022
- train 2022 -> test 2023
- train 2023 -> test 2024
- train 2024 -> test 2025

No expanding-window alternative, pooled multi-season fit, or fold deletion may be selected after results.

## R8 residual fit

For each training season:

1. Build the same football-only finite target-entitlement baseline used by the R25 historical evaluator.
2. Build strict-prior R8 identity features with history start fixed at 2013.
3. For each RB/FB player-game with positive modeled and actual RB-room target mass, define:
   `within_residual_target = log(actual_rb_within_share + EPS) - log(baseline_rb_within_share + EPS)`.
4. Clip the training residual to `[-2, 2]` exactly as historical R8.
5. Fit `StandardScaler -> Ridge(alpha=20.0)` using the exact R8 feature vector.
6. At inference, clip raw R8 residual predictions to `[-1, 1]` exactly as historical R8.

No feature search, alpha search, clip search, or refit after test outcomes.

## R9 reliability

Use the original R9 training-only rolling-origin calibration blocks exactly:

- validation weeks 5-8, trained on weeks strictly before 5;
- validation weeks 9-12, trained on weeks strictly before 9;
- validation weeks 13-17, trained on weeks strictly before 13.

Concatenate those OOF predictions. Compute the zero-intercept calibration slope:

`dot(raw_pred, actual_residual) / dot(raw_pred, raw_pred)`

and clip once to `[0, 1]`. The resulting scalar is the fold's R9 reliability multiplier. Week 18, when present, may contribute to the full training-season R8 fit but does not change the historical R9 OOF block definition.

## Vacancy gate

Use the already-audited canonical state contract:

- current roster status in `{ACT, INA}`;
- prior roster status in `{ACT, INA}`;
- Week 1 compares to final regular-season prior-season roster snapshot;
- later weeks compare to the latest earlier same-season regular-season roster snapshot;
- `VACANCY_ACTIVE = room_exits_n >= 1`;
- no same-week historical depth is required or used.

## Candidate application

For `VACANCY_ACTIVE == 0`, candidate target entitlement equals baseline exactly.

For `VACANCY_ACTIVE == 1`:

`r9_score = log(baseline_within_rb_share + EPS) + reliability * clipped_R8_residual`

Softmax the score only inside the current RB/FB room and rescale to the exact pre-existing RB-room target mass.

Catch-rate conversion remains the current production/baseline conversion. R26 V1 changes target entitlement only; it does not introduce the R23 catch-conversion candidate.

Receiving-yard point means remain exactly baseline. No R22 transformation, asset, coefficient, seed, or production runtime is modified.

## Evaluation

The exact 20 gates in the parent frozen plan remain authoritative. This lock adds no new promotion gate, removes no gate, and changes no threshold.

All 2020-2025 results are labeled **retrospective mechanism evidence only**. A retrospective pass can authorize only a separately frozen 2026 prospective shadow.