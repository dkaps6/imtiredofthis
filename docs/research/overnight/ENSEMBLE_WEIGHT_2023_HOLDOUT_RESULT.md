STATUS: RESEARCH ONLY — NOT PROMOTED. No production/model/weight/threshold
change. Genuine train/holdout weight fit -- NOT combined with the
concurrent fair-probability translator fix (PR #546); one-variable-at-a-time
ablation, per agreement in Issue #535.

# Ensemble Weight 2023 Holdout Result

Checkpoint 14 found `rec_yards`/`receptions`/`rush_rec_yards` have no fitted
weight entry in `data/model_ensemble_weights.csv` at all, silently falling
back to 100% MC even where `ml_proj` is measurably more accurate. Fitting
weights on the same 2024-2025 rows graded against Vegas would be circular
(flagged by GPT-5.6, checkpoint 8, agreed). This builds a genuine
train/holdout design instead: fit weights on 2023 only, apply them blind to
2024-2025 (data the fit never saw), re-grade against Vegas.

## Weights fit on 2023 only (`scripts/research/fit_2023_ensemble_weights_v1.py`)

| market | mc_weight | ml_weight | state_weight | calibration_rows |
|---|---:|---:|---:|---:|
| rec_yards | 0.657 | 0.301 | 0.042 | 4,335 |
| receptions | 0.546 | 0.454 | 0.000 | 4,335 |
| rush_rec_yards | 0.507 | 0.468 | 0.025 | 4,991 |

(`rush_yards`/`pass_yards`/`rush_att` also refit as a consistency check —
not applied, since those markets already have legitimate production
weights; the fresh fits land in a similar direction to the existing
frozen weights, e.g. `rush_yards` mc=0.396/ml=0.560 fresh vs mc=0.557/
ml=0.443 frozen -- both favor `ml_proj`, magnitude differs.)

## Applied blind to the clean 2024-2025 cohort, STRONG tier

| market | model_mae before→after | vegas_mae | ROI before→after | win_rate before→after |
|---|---|---:|---|---|
| rec_yards | 21.603→20.810 | 19.812 | -3.36%→**-3.13%** | 51.3%→51.4% |
| receptions | 1.707→1.597 | 1.549 | -0.87%→**-0.80%** | 53.9%→54.0% |
| rush_rec_yards | 33.040→30.456 | 27.661 | -4.42%→**-6.10%** | 51.0%→50.1% |
| ALL_MARKETS | 18.482→17.941 | 16.742 | -2.77%→**-2.92%** | 52.0%→52.0% |

## Honest read

**MAE improves genuinely and materially, out-of-sample, on rows the
fitting process never saw.** This is real signal, not overfitting -- the
weights were frozen before 2024-2025 was ever touched, the same discipline
already validated for the QB M89 recipe.

**ROI is mixed, and one market gets meaningfully worse.** `rec_yards` and
`receptions` improve slightly; `rush_rec_yards` gets materially worse
(-4.42%→-6.10%) despite a large MAE improvement (33.0→30.5); the
aggregate ALL_MARKETS ROI ticks slightly negative overall (-2.77%→-2.92%)
despite meaningfully better raw accuracy (18.48→17.94).

This is a direct, concrete demonstration of why the concurrent
fair-probability work (Issue #535 checkpoints 9-11) matters more than this
result alone: **better point-accuracy does not reliably translate to
better ROI when the probability/edge-selection layer underneath is still
the disclosed `Normal(mean, component_sd)` approximation.** A more
accurate mean can shift which side of a line a bet falls on, or how far
past the STRONG threshold it lands, in ways that interact with an
already-known-overconfident probability translator unpredictably. The
concurrent PR #546 result (component_sd → empirical MC probabilities)
shows real Brier/log-loss/ROI improvement on the *existing* mean stack;
this result shows real MAE improvement on the *existing* probability
stack. Neither alone closes the gap. Per agreement with GPT-5.6, these are
being tested one variable at a time, not combined yet.

## Files

`scripts/research/fit_2023_ensemble_weights_v1.py`,
`.github/workflows/backtest-ensemble-weight-2023-fit-v1.yml` (no
governance restriction on this workflow surface, unlike QB -- left
committed as reusable research infra rather than removed).
