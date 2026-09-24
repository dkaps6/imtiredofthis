# TE Target-Quality Efficiency V1 — Frozen Predictive Plan

Date: 2026-09-24

Status: FROZEN BEFORE PREDICTIVE SCORING

Parent source result:
`TE_TARGET_QUALITY_NGS_SOURCE_SUPPORTS_FROZEN_PREDICTIVE_STUDY`

## Question

Does strict-prior NGS receiving target-quality information improve TE receiving-yard mean prediction out of sample relative to the frozen historical football projection?

This is a mean-translation study. It is separate from Width V2 probability/distribution work.

## Immutable football baseline

Use the exact frozen PR #549 production-order TE receiving-yard football authority:
- source run `34722725629`
- compact artifact `10307242156`
- source SHA `f04a8a775f4a56fe282cb292f6a52bd509bc8f24`

Cohort:
- position TE
- market `rec_yards`
- seasons 2024 and 2025
- exact frozen row identity
- baseline mean = frozen `ensemble_proj`

Do not rebuild the baseline from live provider history.

## Feature construction

NGS receiving rows are outcomes of prior games, but are legal features only when strictly before the target week.

For each target TE player-game, construct from that player's prior NGS receiving games only:

For each base metric:
- `avg_separation`
- `avg_cushion`
- `avg_intended_air_yards`
- `avg_expected_yac`
- `avg_yac_above_expectation`
- `percent_share_of_intended_air_yards`
- NGS receptions / targets

Frozen rolling summaries:
- last1
- mean3
- mean8
- mean3-minus-mean8

Also include:
- number of prior NGS games available, capped only as a numeric count (no threshold search).

No target-game NGS row may enter its own features.

No opponent target-game outcome, sportsbook line, market odds, model residual history, or target-game participation may be a feature.

## Model

One model only:

- standardized Ridge
- alpha = 50.0
- median imputation fit on training fold only
- no hyperparameter search
- no feature subset search
- no interaction search
- no nonlinear rescue model

Prediction target:
`actual_rec_yards - frozen_ensemble_proj`

Candidate football mean:
`frozen_ensemble_proj + predicted_residual`

No clipping/capping/search is authorized in V1. If extreme corrections break the gates, V1 fails rather than adding a post-hoc cap.

## Blind folds

Two symmetric directions:

1. fit 2024 -> blind test 2025
2. fit 2025 -> blind test 2024

2020-2023 NGS may be used only as strict-prior feature history for the 2024/2025 rows. They are not added as target rows in V1.

2026 outcomes are not used anywhere in fitting or scientific disposition.

## Primary gates

Both blind directions must satisfy all:

1. baseline row identity exact against frozen PR #549 authority;
2. test rows >= 250;
3. candidate MAE strictly lower than frozen baseline MAE;
4. candidate RMSE non-worse;
5. absolute bias non-worse;
6. p90 absolute error non-worse;
7. 30+ yard absolute misses non-increase;
8. correction-vs-realized-residual Pearson correlation > 0;
9. no target-week leakage;
10. sportsbook inputs = 0.

Pooled 2024+2025 candidate MAE must also improve.

No subgroup rescue is allowed if either blind direction fails.

## Interpretation

- both folds pass all gates -> `TE_TARGET_QUALITY_EFFICIENCY_V1_REPLICATES`
- one fold passes, one fails -> `TE_TARGET_QUALITY_EFFICIENCY_V1_NONREPLICATING_SIGNAL`
- neither passes -> `TE_TARGET_QUALITY_EFFICIENCY_V1_FAIL`

Only the replicated disposition authorizes a separately frozen Week-3 deployment/shadow design.

## Week-3 relevance

If and only if the historical blind study replicates, build the 2026 Week-3 feature state from NGS games strictly through Week 2. Do not inspect Week-3 outcomes. Any production mutation still requires an explicit deployment plan and certification.

## Standing protections

- do not alter TE-R5P entitlement;
- do not refit PR #627 current-snap coefficients;
- do not change Width V2;
- do not use sportsbook lines upstream;
- do not tune on 2026 W1/W2 outcome error;
- no post-result feature search.
