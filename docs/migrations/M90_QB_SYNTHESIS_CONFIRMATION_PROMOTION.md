# Migration 90 — QB Synthesis Confirmation / Promotion

## Purpose

M90 is a confirmation/promotion migration only. It does not reopen the M89 catastrophic-game casebook, the established 63/60 single-largest-completion split, the persistent-failure taxonomy, or any previously closed QB feature family.

M89 produced a football-only synthesis candidate that improved the corrected 2024–2025 common cohort from 57.638995 MAE to 55.060118 MAE, with lower RMSE, higher correlation, fewer total 100+ yard misses, and 99.62% bootstrap probability of improvement. M90 tests whether the exact recipe is temporally stable before promotion.

## Frozen candidate

- Football-only pregame synthesis feature set: exactly the M89 football-only feature set.
- Model family: Ridge regression residual correction.
- Ridge alpha: 20.0.
- Residual correction cap: ±45 passing yards.
- No sportsbook variables in the football model.
- No postgame casebook variables in prediction.
- No hyperparameter, feature, threshold, or cap tuning in M90.

The market-assisted M89 candidate may be reproduced as a secondary decision-layer benchmark, but it is not eligible to become the football-only projection.

## Confirmation design

Primary new confirmation is a one-year-earlier temporal rotation:

1. Build a corrected 2022 current-stack walk-forward using strictly prior 2021 history.
2. Build the exact M90/M89 pregame feature trace for 2022.
3. Fit the frozen synthesis recipe on 2022 only.
4. Evaluate prospectively on corrected 2023 using no 2023 labels during fitting.
5. Reproduce the already-frozen M89 2023-fit → 2024/2025 evaluation as a consistency check only; 2024/2025 are not new confirmation data.

This design tests whether the recipe, rather than one specific 2023 coefficient fit, generalizes across an earlier chronological train/evaluation rotation.

## Primary promotion gates on 2023 rotated confirmation

All must pass for football-only promotion eligibility:

1. Corrected 2022 and 2023 data-integrity contracts pass.
2. 2023 MAE improves by at least 1.0 yard versus the corrected 2023 base.
3. 2023 RMSE is non-worse than base.
4. 2023 correlation is non-worse than base.
5. 2023 total 100+ yard misses do not increase.
6. Paired bootstrap probability that synthesis improves absolute error is at least 0.90.
7. No sportsbook or postgame features enter the football model.

Directional 100+ under/over counts are mandatory diagnostics. They are not retroactive tuning targets. M90 must report whether improvement simply swaps one catastrophic direction for the other.

## Dispositions

- `PROMOTE_M89_FOOTBALL_SYNTHESIS` — all primary rotated-confirmation gates pass; M89 football-only synthesis is eligible to become the clean QB passing-yard architecture for 2026 production integration.
- `DO_NOT_PROMOTE_M89_SYNTHESIS` — any primary gate fails; freeze the pre-M89 corrected base and do not tune around the failure in M90.

## Anti-loop rules

- No new casebook taxonomy work.
- No repeat of the single-largest-completion study.
- No reopening M56–M88 failed feature families.
- No model zoo, selector search, threshold search, or residual-cap search.
- M90 ends broad QB point-projection research either way; subsequent QB work is production integration and distribution/tail calibration, not another open-ended mean-MAE search.
