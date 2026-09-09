# RB R26F — 2020 Week 1 Vacancy Failure Forensic Atlas V1 — Frozen Plan

Status: FROZEN BEFORE NEW 2020 SUBGROUP OUTCOME SLICING
Date frozen: 2026-09-09
Production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Research governance: `docs/research/RESEARCH_COMPONENT_PRESERVATION_DOCTRINE.md`
Parent evidence: R26 V1 + R26E qualification

## Purpose

R26E qualified 17/18 Week-1 protections. The frozen R26 mechanism improved Week-1 vacancy-incumbent receptions MAE in 5/6 seasons and improved pooled Week-1 MAE/RMSE/bias/p90, targets, RB1, RB2+, and global RB safety. The sole failure was 2020, where vacancy-incumbent receptions MAE worsened 8.97% despite large bias correction.

R26F is a **no-refit forensic atlas**. It must determine what pregame/prediction-state mechanism makes 2020 harmful while the same frozen R26 behavior helps 2021-2025.

R26F may not exclude 2020, relabel it an outlier, weaken R26E gate 11, or change R9/R26 predictions.

## Immutable parent evidence

R26 V1:
- run `34356222339`
- artifact `10106271075`
- digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`

R26E:
- run `34364872300`
- artifact `10109398212`
- digest `sha256:4d136cf20bccebbdd874b24392d95669d11ad01e55b4ac4af6cdee06e7e99651`
- disposition `WEEK1_COMPONENT_MIXED_OR_FAIL_NO_SHADOW`
- sole failed gate: no Week-1 season worsens >2%, due 2020 +8.97% MAE

No predictions may be regenerated.

## Primary forensic population

All R26 V1 rows satisfying:
- `week == 1`
- `vacancy_active == 1`
- `continuing_same_team == 1`

Keep all seasons 2020-2025 visible. 2020 is the failure case; 2021-2025 are comparison seasons.

## Frozen forensic questions

### A. Room-total versus within-room allocation
For every vacancy-active Week-1 team:
- aggregate actual RB targets/receptions for labeled players;
- aggregate baseline and candidate RB targets/receptions;
- compare room-total absolute error;
- separately sum player-level absolute errors.

Question: did 2020 fail because R26 moved the **room total** incorrectly, or because it redistributed a roughly-correct pool to the wrong players?

### B. Role allocation
Compare R26 effect for:
- RB1 incumbents (`rb_rank == 1`)
- RB2+ incumbents (`rb_rank >= 2`)

Question: is 2020 damage concentrated in lead backs, secondary backs, or both?

### C. Vacancy intensity / room churn
Predeclare source-state groups:
- `room_exits_n == 1`
- `room_exits_n >= 2`
- `room_entrants_n == 0`
- `room_entrants_n >= 1`
- exits greater than entrants / equal / fewer than entrants

Question: does R26 overreact when Week-1 roster turnover is especially broad?

### D. Prior-depth observability
Compare:
- `prior_depth_available == 1`
- `prior_depth_available == 0`

This is diagnostic only. Same-week historical depth remains forbidden.

### E. Prediction movement magnitude
Using only baseline/candidate predictions, before outcome grading:
- calculate absolute target movement `abs(candidate_targets - baseline_targets)`;
- calculate absolute reception movement;
- define source-only quartiles using the pooled 2020-2025 Week-1 vacancy-incumbent movement distribution;
- compare outcome effect by quartile.

Question: is 2020 failure an overcorrection concentrated in the largest R26 moves?

No fixed numeric threshold may be selected afterward from these quartiles for a child router without a new frozen plan.

### F. Direction / sign crossing
Predeclare:
- target projection increased vs decreased vs unchanged;
- reception projection increased vs decreased vs unchanged;
- candidate/baseline role-order flip within a team-week where calculable.

Question: did R26 improve average bias by moving opportunity upward overall but place those increases on the wrong incumbent?

### G. R9 residual direction and magnitude
Use frozen `r9_calibrated_residual` only; no refit.
Report:
- positive / negative / zero residual;
- source-only quartiles of absolute residual magnitude.

Question: are harmful 2020 player-games concentrated in unusually large identity corrections?

### H. Cross-season replication of any harmful state
Any state proposed as a future guard must not be justified by 2020 alone.

For each predeclared dimension, report the same state in 2021-2025. A future child-router hypothesis is eligible only if a harmful ordering visible in 2020 is also directionally supported in **at least two of the five successful seasons**, with at least 15 labeled player-games in the relevant state across those supporting seasons combined.

This prevents a 2020-specific rescue rule.

## Metrics

Player level:
- n
- baseline/candidate receptions MAE
- baseline/candidate targets MAE
- RMSE
- bias
- p90 absolute error
- effect delta absolute error = candidate AE - baseline AE

Room level:
- n team-weeks
- baseline/candidate room-total target absolute error
- baseline/candidate room-total reception absolute error
- summed player absolute error
- difference between player-allocation error change and room-total error change

## Forensic dispositions

R26F cannot authorize shadow or production.

Possible dispositions:
- `WEEK1_FAILURE_MECHANISM_IDENTIFIED_REPLICATED`
  - a predeclared harmful state explains a material portion of 2020 damage and has supporting directional evidence outside 2020 under the replication rule;
- `WEEK1_2020_FAILURE_LOCALIZED_NOT_REPLICATED`
  - 2020 failure can be described but no guard has cross-season support;
- `WEEK1_FAILURE_MECHANISM_UNRESOLVED`
  - no coherent predeclared mechanism localizes the damage;
- `WEEK1_FORENSIC_INTEGRITY_FAILURE`
  - immutable evidence/integrity fails.

A replicated forensic mechanism only permits a separately frozen child-candidate design. It does not alter R26E's no-shadow disposition.

## Supported parent components that remain locked

Do not change:
- R9 identity mechanics;
- R26 finite RB target-pool conservation;
- broad Week-1 vacancy signal as a supported parent component;
- production-exact non-vacancy behavior;
- non-RB exactness;
- receiving-yard means;
- R22;
- sportsbook separation;
- strict-prior timing/leakage protections;
- 2020 negative evidence;
- R26/R26D/R26E scientific dispositions.

## Prohibited actions

- no R9 refit or coefficient adjustment;
- no alternate Week-1 season gate;
- no dropping 2020;
- no COVID-era exclusion rule merely because 2020 failed;
- no subgroup chosen after looking at subgroup outcomes;
- no sportsbook input;
- no current-season depth source work;
- no production write;
- no receiving-yard or R22 change.
