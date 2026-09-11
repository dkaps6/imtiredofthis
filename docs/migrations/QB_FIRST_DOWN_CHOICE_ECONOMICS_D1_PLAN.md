# QB First-Down Choice Economics D1 — Frozen Development Plan

## Purpose

Test exactly one pregame mechanism for the newly localized shared QB/receiver opportunity problem:

> When first-down passing is relatively more valuable than first-down designed rushing for both the offense and the upcoming opponent-defense matchup, does the offense choose a pass-origin play on first down more often than the strictly-prior baseline expects?

This is the only D1 candidate. There is no feature search, model zoo, alternate weighting, threshold sweep, sportsbook input, or production change.

## Parent lineage

Source audit:

- branch: `research-qb-first-down-choice-economics-source-v1`
- source result commit: `e2f5d100e9284e78489b487d82f6c55683bd04e1`
- source run: `34544776077`
- source job: `103094922496`
- source artifact: `10178612429`
- source artifact name: `qb-first-down-choice-economics-source-v1`
- digest: `sha256:5a4928458d079f79edbaedfb1ef1ccf3f47f2d1bb76dbb39d3dfb01fa6fbe48a`
- disposition: `FIRST_DOWN_CHOICE_ECONOMICS_SOURCE_QUALIFIED`

Mechanism discovery:

- parent state-attribution run: `34544457497`
- parent artifact: `10178499888`
- parent disposition: `FIRST_DOWN_SHARED_PRIMARY_DIAGNOSTIC`

## Why 2023 is used first

The first-down mechanism was localized using 2024-2025 M89/shared-receiver data. Therefore D1 does **not** open 2024 or 2025 first-down outcomes.

D1 uses only 2023 target outcomes, with 2022 plus strictly-prior 2023 history. Within 2023:

- development fit: Weeks 1-9;
- untouched D1 holdout: Weeks 10-18.

No 2024/2025 target PBP, first-down residual, QB outcome, WR target/reception outcome, or M89 target outcome may be read by this evaluator.

## Frozen first-down play semantics

Exactly the source-audit semantics:

- regular season;
- `down == 1`;
- valid possession and defense teams;
- `qb_dropback == 1` OR `rush_attempt == 1`;
- exclude `two_point_attempt == 1` when field exists;
- exclude `no_play == 1` when field exists.

`PASS_ORIGIN = qb_dropback == 1`, which includes sacks and QB scrambles under corrected M89 opportunity semantics.

`DESIGNED_RUN = rush_attempt == 1 AND qb_dropback != 1`; QB kneels are excluded when the field is available.

Actual target first-down pass-origin rate is used only as the outcome after all pregame values for that target are fixed.

## Frozen baseline first-down DBR

D1 evaluates an incremental correction to the same strictly-prior offense/opponent-defense reference concept used by the parent down/distance decomposition.

For each 2023 target team-week:

1. offense history = last 8 completed games strictly before target week;
2. opponent-defense history = last 8 completed defensive games strictly before target week;
3. league first-down DBR = all eligible first-down opportunity plays strictly before the target week;
4. offense first-down DBR and opponent-defense first-down DBR are each shrunk toward the strictly-prior league rate with **4 league-equivalent games**, using the parent game's-count shrinkage form;
5. baseline D1 DBR = equal-weight mean of the two shrunk values.

If one side lacks a finite value, use the other finite value; if both lack a finite value, use the strictly-prior league rate.

The baseline is clipped to `[0.05, 0.95]` only as the same mechanical probability bound used by the parent decomposition.

No target-game play enters its own baseline.

## Frozen choice-edge score

D1 consumes only the four predeclared descriptors from the qualified source artifact:

- `off_fd_epa_pass_minus_run`
- `def_fd_epa_pass_minus_run`
- `off_fd_success_pass_minus_run`
- `def_fd_success_pass_minus_run`

For each target week independently, percentile-rank each descriptor across all scheduled teams using average ranks and `pct=True`. Higher always means pass-origin football has been relatively more valuable than designed rushing.

Define:

`CHOICE_EDGE = mean(four weekly percentile ranks) - 0.5`

No descriptor receives a learned or manually different weight. No alternate score may be tested after results.

## Frozen one-parameter candidate

On 2023 Weeks 1-9 only, define:

`residual = actual_first_down_pass_origin_rate - baseline_first_down_dbr`

Fit exactly one non-negative no-intercept coefficient:

`beta_raw = sum(CHOICE_EDGE * residual) / sum(CHOICE_EDGE^2)`

`beta = max(0, beta_raw)`

No regularization, intercept, nonlinear term, feature interaction, hyperparameter, or alternative coefficient is allowed.

For every 2023 Weeks 10-18 holdout row:

`candidate_d1_dbr = clip(baseline_first_down_dbr + beta * CHOICE_EDGE, 0.20, 0.80)`

The `[0.20, 0.80]` clip is frozen as a mechanical plausibility bound before outcomes are opened.

## Frozen holdout metrics

On 2023 Weeks 10-18 only, report baseline and candidate:

- N;
- MAE;
- RMSE;
- bias;
- correlation with actual first-down DBR;
- p50 / p75 / p90 absolute error;
- mean predicted rate;
- mean actual rate.

Also report:

- `beta_raw` and frozen non-negative `beta`;
- candidate minus baseline correction mean / mean-absolute / p90-absolute;
- Spearman and Pearson correlation of the candidate correction with the **baseline holdout residual**;
- sign agreement of correction with baseline holdout residual;
- week-by-week MAE gain and number of Weeks 10-18 with positive MAE gain;
- paired bootstrap support that holdout candidate MAE is lower than baseline MAE.

Bootstrap contract:

- 2,000 paired row-resamples;
- seed `310`;
- support = fraction of resamples where `baseline_MAE - candidate_MAE > 0`.

No bootstrap tuning is permitted.

## Frozen scientific advancement gates

D1 advances only if **all** are true on the 2023 Weeks 10-18 holdout:

1. source artifact integrity: exact 544 2023 target team-weeks, no duplicate `(season, week, team)` keys, four required descriptors finite;
2. target PBP alignment is exact and all holdout scheduled team-weeks have a finite actual first-down DBR;
3. all baseline values are strictly prior and finite;
4. `beta > 0`;
5. candidate first-down DBR MAE gain >= `0.0050`;
6. candidate RMSE is non-worse than baseline;
7. absolute candidate bias is non-worse than absolute baseline bias;
8. candidate p90 absolute error is non-worse than baseline p90 absolute error;
9. correction vs baseline holdout residual Spearman >= `0.15`;
10. at least `6` of the `9` Weeks 10-18 have positive candidate MAE gain;
11. paired-bootstrap support for positive MAE gain >= `0.90`;
12. no 2024/2025 target PBP or target outcome is read;
13. no QB/WR target outcome or parent residual is read;
14. zero sportsbook/game-market inputs;
15. zero production changes.

The thresholds above may not change after results are visible.

## Frozen disposition

Exactly one of:

- `FIRST_DOWN_CHOICE_ECONOMICS_D1_ADVANCES`
- `FIRST_DOWN_CHOICE_ECONOMICS_D1_FAIL_NO_CONFIRMATION`
- `MECHANICAL_OR_INTEGRITY_FAIL_NO_SCIENCE`

A clean scientific failure is final for this exact score/coefficient family. Do not retune the four weights, replace percentile ranks with z-scores, add an intercept, loosen the gates, swap model families, or inspect 2024/2025 to rescue it.

## If D1 advances

Only then may a separate confirmation/integration migration be frozen.

That later migration must keep the D1 score and beta unchanged and may evaluate 2024-2025 against:

- exact parent first-down DBR residuals;
- corrected M89 team pass opportunities / QB attempts;
- immutable WR target/reception residual cohorts;
- passing-yard consequences as a downstream guardrail.

Because 2024-2025 were used for mechanism discovery, they are not pristine untouched confirmation seasons. Any eventual production promotion would still require explicit prospective-2026 monitoring/certification.
