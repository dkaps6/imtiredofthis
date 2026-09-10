# RB R27C — Vacancy RB1 / 2023 Receiving-Yard Forensic V1 Frozen Plan

Status: `FROZEN BEFORE ROW-LEVEL FORENSIC EXECUTION / DIAGNOSTIC ONLY / NO CANDIDATE`

## Parent scientific evidence

This study consumes the immutable first valid R27B V2 result only.

- R27B V2 result-record commit: `4a885207785255e7ffa2b1c33a35ddcaedfe6668`
- R27B V2 workflow run: `34428917229`
- Job: `102720004328`
- Artifact: `10134023092`
- Artifact name: `rb-r27b-v2-novel-efficiency-context`
- Artifact digest: `sha256:1df1a14c9900b890fe86e04269849dc90d76fc03a9215ad8612c1820dee5341f`
- V2 implementation lock/head: `2c86520c84fbdd5a18aad3f4b88373e5f8c17051`
- Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`

## Purpose

Explain, without fitting or promoting a new model, why exact R26 vacancy opportunity translation behaves differently for vacancy RB1 incumbents than RB2+ incumbents and why 2023 is an adverse cohort.

The study is explicitly hypothesis-generating. It may identify a new football mechanism worth a separately frozen future experiment, but it cannot authorize routing, thresholds, feature selection, production changes, R26 changes, R22 changes, or a new receiving-yard mean candidate.

## Core questions

1. When R26 changes RB1 targets relative to B0, where does B1 yard error improve and where does it worsen?
2. Is RB1 failure primarily associated with target-volume translation, production YPT level, realized target quality, or specific vacancy states?
3. Does V2's YPT correction point in the right direction but with insufficient magnitude, point in the wrong direction, or help only selected football environments?
4. Why can V2 improve vacancy p90 while slightly worsening the 30+ yard miss rate?
5. What is structurally different about 2023 versus the other five test seasons?
6. Which pre-specified context families appear explanatory for RB1/2023 behavior? Ablations are explanation-only and cannot select a future candidate.
7. Does RB2+ success occur because opportunity deltas are smaller, production YPT is better calibrated, target quality is different, or the contextual correction behaves differently?

## Frozen cohorts

Primary:
- `VACANCY_RB1_INCUMBENT`: `vacancy_active == 1`, `vacancy_incumbent == 1`, `role == RB1`
- `VACANCY_RB2PLUS_INCUMBENT`: `vacancy_active == 1`, `vacancy_incumbent == 1`, `role == RB2+`

Secondary:
- all `VACANCY_ACTIVE`
- `2023 VACANCY_ACTIVE`
- `2023 VACANCY_RB1_INCUMBENT`
- non-2023 vacancy RB1 incumbent
- `week1` vacancy rows as a contextual reference only

No cohort may be deleted because it looks unfavorable.

## Frozen metrics

For B0, B1 and C1 where applicable:
- n
- MAE
- RMSE
- signed bias
- p90 absolute error
- 30+ yard absolute-error rate
- underprediction rate
- mean predicted receiving yards
- mean actual receiving yards

Incremental diagnostics:
- `B1_AE - B0_AE` per row: opportunity translation gain/loss
- `C1_AE - B1_AE` per row: context correction gain/loss
- target delta: `candidate_targets - baseline_targets`
- YPT correction: `c1_ypt - production_ypt`
- realized efficiency residual: `actual_ypt - production_ypt` on rows with actual_targets >= 1
- correction sign agreement with realized efficiency residual
- correction magnitude error versus realized efficiency residual
- actual target count and actual receiving-yard distribution

## Frozen descriptive bins

These bins are diagnostic only and are frozen before detailed forensic results.

### R26 target delta
- `<= -1.0`
- `(-1.0, -0.25]`
- `(-0.25, 0.25)`
- `[0.25, 1.0)`
- `>= 1.0`

### Candidate target volume
- `< 2`
- `[2, 4)`
- `[4, 6)`
- `>= 6`

### Production YPT
- `< 5.0`
- `[5.0, 6.5)`
- `[6.5, 8.0)`
- `>= 8.0`

### V2 applied YPT correction
- `<= -1.0`
- `(-1.0, -0.25]`
- `(-0.25, 0.25)`
- `[0.25, 1.0)`
- `>= 1.0`

### Actual targets
- `0`
- `1–2`
- `3–4`
- `5–6`
- `7+`

### Actual YPT, only when actual_targets >= 1
- `< 3`
- `[3, 6)`
- `[6, 9)`
- `[9, 12)`
- `>= 12`

## Frozen football-context comparisons

For RB1, RB2+, 2023 RB1, and non-2023 RB1 report distributions/means of:
- player air yards per target prior
- player YAC per reception prior
- player screen target rate prior
- player explosive-20 target rate prior
- team RB targets per official pass attempt prior
- team RB air yards per target prior
- team RB YAC per reception prior
- team RB screen target rate prior
- opponent RB air yards allowed per target prior
- opponent RB YAC allowed per reception prior
- opponent RB catch rate allowed prior
- opponent RB explosive-20 allowed per target prior
- opponent RB screen target rate faced prior
- production YPT
- production catch rate
- baseline targets
- candidate targets
- R26 target delta
- V2 applied YPT correction

For each continuous feature, report mean, median, p25 and p75 by cohort. No post-result threshold routing is allowed.

## Frozen ablation use

The three already-produced V2 explanation-only ablations may be evaluated descriptively:
- player target-shape only
- team/QB RB environment only
- opponent RB context only

For each primary/secondary cohort report MAE delta versus B1. These ablations are not eligible candidates, cannot be promoted, and cannot be used to choose a winning feature subset from this same evidence.

## 30+ miss forensic

For vacancy-active and vacancy RB1 incumbent rows:
- count rows crossing from <30 AE under B1 to >=30 AE under C1
- count rows crossing from >=30 AE under B1 to <30 AE under C1
- list aggregate characteristics of each crossing set
- compare error direction and target delta
- report whether p90 improvement is broad compression or driven by a limited subset

No individual player anecdote can determine the next mechanism.

## 2023 forensic

Compare 2023 versus pooled 2020–2022 + 2024–2025 for vacancy RB1 and all vacancy rows on:
- B0/B1/C1 errors
- target delta
- candidate target volume
- production YPT
- realized YPT residual
- correction sign agreement
- all 13 novel context variables
- vacancy-state composition
- actual targets and actual receiving yards

Also report whether 2023's failure is concentrated in a small number of high-error games or broad across the distribution using median AE delta, p75 AE delta, p90 AE delta and share of rows worsened.

## Integrity / anti-leakage rules

- Consume exact preserved V2 artifact by ID/digest; do not rerun or refit V2.
- No sportsbook inputs.
- No future data introduced into pregame features; actual outcomes may be used only as labels for retrospective forensic scoring.
- No new model fit.
- No hyperparameter tuning.
- No threshold optimization.
- No production mutation.
- No R26 mutation.
- No R22 mutation.
- No candidate projection for deployment.

## Frozen forensic dispositions

This study has no PASS-for-production state.

Allowed terminal labels:
- `R27C_FORENSIC_COMPLETE_MECHANISM_HYPOTHESIS_IDENTIFIED`
- `R27C_FORENSIC_COMPLETE_NO_SINGLE_MECHANISM_IDENTIFIED`
- `R27C_MECHANICAL_OR_INTEGRITY_FAILURE_NO_FORENSIC_CONCLUSION`

A mechanism hypothesis is allowed only if it is supported by multiple pre-specified cohort comparisons and is stated as a hypothesis requiring a separate newly frozen experiment.

## Next-step boundary

After R27C completes, any proposed R27D or later candidate must be designed and frozen from the forensic evidence before fitting/scoring it. R27C itself cannot be retrofitted into production.
