# M95R — Expanded-Panel Dynamic Population-Prior Candidate

## Primary question

Can a pregame workload-population prior adapt M95F stable-workhorse 20+ probabilities across seasons and game environments without recreating M95K's cross-season instability or changing the M94C/M95F backbone outside the stable-workhorse tail?

## Frozen inputs / population

- Exact reconstructed stable-workhorse panel from M95Q for 2020-2022.
- Already-authoritative exact M95P stable-workhorse traces for 2023-2025.
- Primary comparable scope: Weeks 13-18.
- 2025 full-season stable trace is a secondary diagnostic only.
- Vacancy/role-transition rows remain outside this candidate and remain governed by the separate M95I research branch.
- M94C central carries are unchanged.
- M95F is the fixed 20+ probability backbone.

## Precommitted R candidate

The candidate is a conservative residual calibration layer in log-odds space:

`logit(P20_R) = logit(P20_M95F) + clipped_dynamic_delta`

The coefficient on the M95F backbone is fixed at 1.0. Only the additive dynamic delta is fit.

Pregame-only dynamic inputs are frozen to exactly five M95P-supported workload-regime variables:

1. league season-to-date lead-RB 20+ rate;
2. league prior-four-week lead-RB 20+ rate;
3. team prior-four lead-RB 20+ rate;
4. team prior-four lead-RB 25+ rate;
5. team prior-four mean lead-RB carries.

No feature combinations may be added or removed after results are exposed.

Training preprocessing is computed from the training seasons only. Missing values use the training median and features are standardized using training means/standard deviations.

The additive delta is fit with a single precommitted ridge objective:

- ridge lambda = **10.0**;
- no hyperparameter search;
- intercept unpenalized;
- feature coefficients penalized;
- final additive delta hard-clipped to **[-0.75, +0.75] log-odds** so the population prior cannot overwhelm the M95F football backbone.

## Strict temporal evaluation

No random pooled split.

- 2023 target: train on 2020-2022 only.
- 2024 target: train on 2020-2023 only.
- 2025 target: train on 2020-2024 only.

The primary evaluation uses W13-18 stable-workhorse rows in each target season. A secondary 2025-full diagnostic uses the model trained on 2020-2024 late-season rows.

## Primary metrics

Calibration first:

- Brier score;
- log loss;
- mean probability vs actual event rate;
- absolute calibration gap.

Discrimination second:

- ROC AUC.

Outputs must include season-by-season results, pooled rolling OOF results, probability-shift regime slices, event counts, rolling fitted coefficients, and a casebook of the largest candidate changes.

## Frozen fail-closed advancement gates

R may advance only if **all** are true on the 2023-2025 primary rolling OOF panel:

1. pooled Brier improves;
2. pooled log loss improves;
3. pooled AUC does not regress by more than 0.02;
4. Brier improves in at least 2 of 3 target seasons;
5. absolute calibration gap improves in at least 2 of 3 target seasons;
6. no target season Brier regression exceeds 0.01;
7. no target season absolute-calibration-gap regression exceeds 2.5 percentage points.

If the candidate fails these gates, the scientific disposition is `M95R_RETAIN_DIAGNOSTIC_DO_NOT_PROMOTE`. Do not tune the same architecture against the exposed target outcomes.

## 25+ rule

25+ remains a pooled/event-count diagnostic only in M95R. No 25+ dynamic candidate is fit or selected from sparse seasonal tails.

## Integrity rules

- No sportsbook inputs.
- No target-week/postgame variables in the candidate.
- Feature search = 0.
- Hyperparameter search = 0.
- M94C central projection change = 0.
- Non-stable RB change = 0.
- Vacancy change = 0.
- Production change = 0.
- 2023, 2024, and 2025 are not pristine prospective confirmation seasons; even a passing M95R result would still require genuinely prospective/untouched confirmation before promotion.
