# QB Team Pass Opportunity Schedule/Rest D1 — Frozen Development Plan

## Purpose

Test the **only V1 source-eligible new information family**, `SCHEDULE_REST_CONTEXT`, as a development-only predictor of the upstream M89 team-pass-opportunity residual.

This is not a generic QB mean search. It is a single-family, single-model test authorized by:

- `QB_OPPORTUNITY_CHAIN_DECOMPOSITION_V1` disposition `TEAM_PASS_OPPORTUNITY_PRIMARY_DIAGNOSTIC`;
- `QB_TEAM_PASS_OPPORTUNITY_SOURCE_AUDIT_V1` disposition `SCHEDULE_REST_CONTEXT = SOURCE_ELIGIBLE_FOR_PREDICTIVE_PREREGISTRATION`.

Production cannot change in D1.

## Canonical lineage

- Parent source-audit result commit: `1bc8ae263f5961248af152a6700a01e0bdf26146`
- Source-audit run: `34533408818`
- Source-audit job: `103059297754`
- Source-audit artifact: `10174433662`
- Source-audit digest: `sha256:47b57a1398e184c0ca1da29e72fa3be1fbf64bbf8c98b8d17e1da539873e2515`
- Opportunity-chain run: `34523313743`
- Opportunity-chain artifact: `10170531084`
- Opportunity-chain digest: `sha256:75cd32198caf7d9cbf193d5769e5762acce53352f2b2626cd92a9b117e75dafd`

## Immutable feature source

Use only the source-audit artifact file:

`schedule_rest_target_feasibility_2024_2025.csv`

Allowed feature columns are exactly:

1. `home`
2. `rest_days_minus_7 = rest_days - 7`
3. `opponent_rest_days_minus_7 = opponent_rest_days - 7`
4. `rest_diff`
5. `short_week`
6. `long_rest`
7. `thursday`
8. `monday`

No score, result, spread, total, moneyline, odds, or other sportsbook field may be loaded or derived.

No interactions, thresholds, alternate encodings, feature subsets, weekday categories, travel variables, or post-result feature additions are authorized in D1.

## Target and baseline

Load the immutable opportunity-chain casebook from Run `34523313743`.

Pregame baseline factor:

`pred_D = mc_team_expected_dropbacks`

Postgame diagnostic target:

`actual_D = pass_opportunities`

Development residual target:

`team_pass_opportunity_residual = actual_D - pred_D`

Target-game outcomes are labels only and may never enter feature construction.

## Frozen model

One model only:

- `StandardScaler`
- `Ridge(alpha=20.0, fit_intercept=True)`
- target = `actual_D - pred_D`

No alpha search, nonlinear model, ensemble, interaction search, clipping search, or alternate residual target.

## Development split

2025 target outcomes are prohibited from candidate selection.

Use 2024 only:

- fit: Weeks 1-9
- development holdout: Weeks 10-18

Rows must remain the exact M89-aligned 2024 cohort available in the immutable parent casebook. No player/team/row exclusions may be added after results are visible.

## Candidate propagation

For each development row:

`candidate_D = pred_D + predicted_D_residual`

Keep the existing M89 pregame factors unchanged:

- `pred_C = mc_pass_attempts_per_dropback`
- `pred_S = mc_qb_pass_att_share`
- `pred_ypa`

Then:

`candidate_attempts = candidate_D * pred_C * pred_S`

For a mean-only integration proxy that preserves all existing downstream M89/M90 corrections unchanged:

`candidate_pass_yards = football_synthesis + (candidate_attempts - pred_attempts) * pred_ypa`

This is a research propagation only. It does not modify the production M89/M90 artifact or code.

## Frozen metrics

On the 2024 Weeks 10-18 development holdout report baseline vs candidate:

### Team pass opportunity
- MAE
- RMSE
- bias
- correlation
- p90 absolute miss

### QB attempts
- MAE
- RMSE
- bias
- correlation
- 8+ absolute miss rate
- 10+ absolute miss rate

### QB passing yards
- MAE
- RMSE
- bias
- correlation
- p90 absolute error
- 75+ miss rate
- 100+ miss rate

Also report:

- mean and p90 absolute model correction in team pass opportunities;
- Ridge coefficients after standardization;
- 10,000-resample paired bootstrap probability that passing-yard MAE improves, seed `5611`.

## Integrity gates

Scientific interpretation stops unless all pass:

1. exact immutable source-audit feature artifact used;
2. exact immutable opportunity-chain casebook used;
3. zero sportsbook/result features;
4. zero 2025 target rows used in fitting, feature selection, gate choice, or scoring;
5. exact eight frozen features only;
6. one Ridge alpha `20.0` only;
7. no production files changed;
8. pregame baseline attempt identity reconciles within `1e-6`;
9. candidate modifies only the team-pass-opportunity factor before propagation;
10. 2024 Weeks 1-9 fit and Weeks 10-18 holdout are non-empty and key-unique.

## Development survivor gates

`SCHEDULE_REST_D1_DEVELOPMENT_SURVIVOR` requires **all**:

1. team-pass-opportunity MAE improves by >= `0.25` opportunities;
2. QB attempt MAE improves by >= `0.10` attempts;
3. QB passing-yard MAE improves by >= `0.75` yards;
4. QB passing-yard RMSE is non-worse;
5. QB passing-yard correlation is non-worse;
6. QB p90 absolute passing-yard error is non-worse;
7. QB 100+ yard miss rate does not increase;
8. QB 10+ attempt miss rate does not increase;
9. paired bootstrap `P(pass-yard MAE gain > 0) >= 0.70`;
10. all integrity gates pass.

If any survivor gate fails, disposition is:

`SCHEDULE_REST_D1_FAIL_NO_CONFIRMATION`

and no 2025 confirmation is authorized.

## If D1 survives

Without looking at 2025 outcomes:

- refit the exact same scaler + Ridge(alpha=20) on **all 2024 rows**;
- freeze the exact scaler statistics and Ridge coefficients/intercept into the D1 artifact;
- create a separate confirmation branch/plan;
- score the frozen all-2024 candidate on untouched 2025 only.

Passing D1 itself does not authorize production integration.

## Stopping rule

Run this exact candidate once.

Do not:

- retry penalty or fourth-down families in D1;
- add travel/time-zone/weather/injury/referee variables;
- retune alpha;
- search transformations or caps;
- inspect 2025 outcomes to rescue a failed development result;
- alter the survivor gates after results are visible.
