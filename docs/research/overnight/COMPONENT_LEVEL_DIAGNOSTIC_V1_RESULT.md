STATUS: RESEARCH ONLY — NOT PROMOTED. No production/model/weight change.

# Component-Level Diagnostic V1

Does MC, ML, or State individually beat Vegas, or are all three
individually mediocre and the blend is just averaging weak pieces?
Runs on the already-committed identity-clean non-QB cohort
(clean_v1_full_stack_vegas_benchmark_detail.csv, PR #541/#542) -- no new
CI run, no new data build.

## 1. MAE by component, market, season

|   season | market         |    n |   mc_proj_mae |   ml_proj_mae |   state_proj_mae |   ensemble_proj_mae |   line_mae |
|---------:|:---------------|-----:|--------------:|--------------:|-----------------:|--------------------:|-----------:|
|     2024 | pass_yards     |  429 |        65.731 |        62.068 |           62.403 |              60.799 |     58.917 |
|     2024 | rec_yards      | 3021 |        21.576 |        21.639 |           22.549 |              21.576 |     19.969 |
|     2024 | receptions     | 2863 |         1.737 |         1.639 |            1.739 |               1.737 |      1.574 |
|     2024 | rush_rec_yards | 1937 |        31.253 |        28.861 |           30.485 |              31.253 |     26.211 |
|     2024 | rush_yards     | 1426 |        23.297 |        21.267 |           23.331 |              21.202 |     19.004 |
|     2025 | pass_yards     |  383 |        59.791 |        59.902 |           59.522 |              58.472 |     54.424 |
|     2025 | rec_yards      | 2832 |        20.967 |        21.487 |           22.501 |              20.967 |     19.662 |
|     2025 | receptions     | 2839 |         1.624 |         1.579 |            1.674 |               1.624 |      1.49  |
|     2025 | rush_rec_yards |  654 |        36.311 |        30.966 |           33.584 |              36.311 |     29.502 |
|     2025 | rush_yards     | 1331 |        21.903 |        20.137 |           21.915 |              20.203 |     19.082 |

**Overall MAE**: mc_proj=18.736, ml_proj=17.965, state_proj=18.886, ensemble_proj=18.291, vegas_line=16.564.

**Best individual component**: `ml_proj` (MAE=17.965). Still loses to Vegas (> 16.564).

**Does the ensemble beat its own best component?** NO (ensemble=18.291 vs best component=17.965).

## 2. Ensemble weights actually applied

|       |   ensemble_weight_mc |   ensemble_weight_ml |   ensemble_weight_state |
|:------|---------------------:|---------------------:|------------------------:|
| count |           17715      |           17715      |              17715      |
| mean  |               0.8948 |               0.0812 |                  0.0239 |
| std   |               0.2194 |               0.1651 |                  0.1094 |
| min   |               0.2088 |               0      |                  0      |
| 25%   |               1      |               0      |                  0      |
| 50%   |               1      |               0      |                  0      |
| 75%   |               1      |               0      |                  0      |
| max   |               1      |               0.5613 |                  0.5241 |

Std dev of weights across rows: {'ensemble_weight_mc': 0.2194, 'ensemble_weight_ml': 0.1651, 'ensemble_weight_state': 0.1094}. Weights vary meaningfully row to row.

Mean weight by market:

| market         |   ensemble_weight_mc |   ensemble_weight_ml |   ensemble_weight_state |
|:---------------|---------------------:|---------------------:|------------------------:|
| pass_yards     |               0.2096 |               0.2682 |                  0.5222 |
| rec_yards      |               1      |               0      |                  0      |
| receptions     |               1      |               0      |                  0      |
| rush_rec_yards |               1      |               0      |                  0      |
| rush_yards     |               0.557  |               0.443  |                  0      |

### Root cause traced

| market         | ensemble_status                | ensemble_method                                   |
|:---------------|:-------------------------------|:--------------------------------------------------|
| pass_yards     | {'calibrated': 812}            | {'promoted_nonnegative_oos_linear_blend_v2': 812} |
| rec_yards      | {'uncalibrated_mc_only': 5853} | {'mc_fallback_no_oos_weights': 5853}              |
| receptions     | {'uncalibrated_mc_only': 5702} | {'mc_fallback_no_oos_weights': 5702}              |
| rush_rec_yards | {'uncalibrated_mc_only': 2591} | {'mc_fallback_no_oos_weights': 2591}              |
| rush_yards     | {'calibrated': 2757}           | {'nonnegative_oos_linear_blend_v2': 2757}         |

`data/model_ensemble_weights.csv` has fitted weights for: ['pass_yards', 'rush_att', 'rush_yards']. It has **no entry at all** for: ['rec_yards', 'receptions', 'rush_rec_yards']. For those missing markets, `apply_ensemble()` correctly and safely falls back to MC-only by explicit design (`data/backtests/component_predictions.csv`, the calibration accumulation file `fit_market_weights()` reads from, does not exist in this repo at all) -- this is not a modeling bug, it's an incomplete pipeline step: nobody has ever run weight-fitting for these markets. The code to do it already exists and works (proven by pass_yards/rush_att/rush_yards). Completing it is closing an existing gap, not building something new.


## 3. Component correlation (redundancy check)

|            |   mc_proj |   ml_proj |   state_proj |   actual |
|:-----------|----------:|----------:|-------------:|---------:|
| mc_proj    |     1     |     0.964 |        0.96  |    0.842 |
| ml_proj    |     0.964 |     1     |        0.962 |    0.85  |
| state_proj |     0.96  |     0.962 |        1     |    0.839 |
| actual     |     0.842 |     0.85  |        0.839 |    1     |

High pairwise correlation among mc_proj/ml_proj/state_proj (independent of their correlation with actual) would mean the three components are largely measuring the same thing -- blending them adds little regardless of how the weights are set.

## 4. In-sample oracle ceiling (diagnostic bound, not a proposal)

Best possible STATIC (non-negative, sum-to-1) blend fit in-sample on this exact data (159 rows with missing state_proj dropped for this computation only): MAE=17.583 at weights mc=0.35/ml=0.60/state=0.05, vs the realized frozen-ensemble MAE=18.291 and Vegas MAE=16.564.

This is an UPPER BOUND on what reweighting alone could ever achieve (fit and evaluated on the same data -- classic in-sample overfitting, not a real out-of-sample estimate). If this ceiling still doesn't approach Vegas, reweighting cannot be the fix by itself. If it does approach or beat Vegas, that's a signal the components carry more real signal than the current static weighting extracts -- worth a genuine held-out weight-fitting study, not a reason to change production weights from this number directly.

