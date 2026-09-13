# Empirical-MC STRONG Gate Algebra Diagnosis V1

**STATUS: RESEARCH/DIAGNOSIS ONLY. NO PRODUCTION, MODEL, WEIGHT, OR THRESHOLD CHANGE.**

## Scope

This audit answers the still-open question from Issue #535 checkpoints 22/28/29: after replacing the legacy `Normal(mean, component_sd)` translator with the actual reconstructed Monte Carlo outcome distribution, why does `STRONG_EDGE` still fire on ~86.5% of the identity-clean historical cohort?

Source is the frozen historical fair-probability reconstruction run `34712931786`, artifact `10304625925`, file `grade/empirical_fair_prob_detail.csv` (17,715 same-row graded props, exactly 2,000 MC draws per row, sportsbook excluded from the football simulation). No model probability or threshold is fit in this audit.

Frozen gate being diagnosed:

- `STRONG_EDGE` iff `best_ev >= 0.05` **and** `prob_edge >= 0.03`;
- `prob_edge = best_model_p - best_market_p` where `best_market_p` is the two-way no-vig probability;
- `best_ev` uses the selected side's actual American price, hence the **raw vig-bearing break-even probability**.

## Result 1 — the 3-point probability-edge gate is completely non-binding on this cohort

Across all 17,715 rows:

| condition | rows | coverage |
|---|---:|---:|
| `best_ev >= 5%` | 15,324 | 86.503% |
| `prob_edge >= 3pp` | 16,449 | 92.854% |
| both = STRONG | 15,324 | 86.503% |
| EV pass but probability-edge fail | **0** | **0.000%** |
| probability-edge pass but EV fail | 1,125 | 6.351% |

This is true in every market. The current historical empirical-MC STRONG rule therefore reduces exactly to **`best_ev >= 5%`** on the observed cohort. The second condition does not remove a single EV-qualified row.

The algebra explains why. Let `q_raw` be the selected side's raw vig-bearing implied probability. For valid American odds,

`EV = p_model / q_raw - 1`,

so `EV >= 5%` requires

`p_model >= 1.05 * q_raw`.

The probability-edge condition requires

`p_model >= q_no_vig + 0.03`.

On this cohort the median selected-side values are:

- raw implied probability: **0.53488**;
- no-vig probability: **0.50000**;
- EV>=5% required model probability: **0.56163**;
- edge>=3pp required model probability: **0.53000**.

The EV requirement is the stricter mathematical threshold on **99.819%** of rows. Because the EV calculation is referenced to the vig-bearing price while `prob_edge` is referenced to the de-vigged market probability, the 5% EV hurdle already subsumes a 3-point no-vig edge in effectively the entire historical cohort.

### Per-market overlap

| market | rows | EV>=5% | edge>=3pp | STRONG | EV-pass / edge-fail |
|---|---:|---:|---:|---:|---:|
| pass_yards | 812 | 80.05% | 89.78% | 80.05% | 0 |
| rec_yards | 5,853 | 83.63% | 91.80% | 83.63% | 0 |
| receptions | 5,702 | 89.62% | 94.41% | 89.62% | 0 |
| rush_rec_yards | 2,591 | 90.08% | 94.48% | 90.08% | 0 |
| rush_yards | 2,757 | 84.69% | 91.26% | 84.69% | 0 |

**Disposition:** `PROB_EDGE_3PP_AS_INDEPENDENT_STRONG_GUARD = FALSIFIED` on this historical empirical-MC cohort. This is an algebra/mechanics statement, not a proposal to tune or remove the threshold.

## Result 2 — the over-firing is being driven by grossly overconfident empirical model probabilities, not by the no-vig calculation

For the 15,324 STRONG rows:

- mean selected-side model probability: **75.36%**;
- median selected-side model probability: **74.75%**;
- mean selected-side no-vig market probability: **50.03%**;
- mean selected-side raw break-even probability: **53.22%**;
- mean model-predicted EV: **+42.17%**;
- realized win rate: **52.04%**;
- realized flat-stake ROI: **-2.31%**.

The probability forecasts are therefore not merely a little aggressive. The gate is being handed probabilities that imply enormous theoretical edge while the realized hit rate remains close to the market break-even rate.

A rank diagnostic tells the same story. On all 17,715 decided rows, selected-side `best_model_p` has **AUC 0.5173** for realized win/loss — only slightly above random ranking. Fixed probability bins are also nearly flat in realized win rate: forecasts from ~0.60 through >0.95 mostly realize at only ~51-55% wins. For example:

| model-p bin | rows | mean model p | realized win rate | mean predicted EV | realized ROI |
|---|---:|---:|---:|---:|---:|
| 0.60-0.65 | 2,023 | 62.54% | 50.96% | +18.68% | -3.67% |
| 0.70-0.75 | 2,101 | 72.51% | 53.36% | +36.03% | -0.42% |
| 0.80-0.85 | 1,690 | 82.44% | 51.60% | +53.35% | -4.51% |
| 0.90-0.95 | 1,270 | 92.44% | 53.31% | +70.83% | -1.85% |
| 0.95-1.00 | 1,055 | 97.60% | 54.88% | +81.24% | +1.56% |

This audit does **not** fit an outcome-based recalibration map from those bins. They are diagnostic evidence only.

## Result 3 — distribution under-width is part of the problem, but not the whole problem

PR #548's prospectively held-out prior-season widening test already established that the base empirical MC distributions are under-dispersed. Applying its frozen widening factors reduced STRONG coverage from:

- 2024: **86.19% -> 78.90%**;
- 2025: **86.88% -> 80.73%**.

Calibration improved materially, but roughly four of every five rows still classified STRONG. Therefore:

1. legacy `component_sd` misuse is a real, separate defect and is now closed by PR #553 for the legacy translator;
2. base empirical MC under-dispersion is also real and explains part of the empirical overconfidence;
3. **neither explains the remaining selection rate by itself**;
4. the dominant current mechanical fact is that the selected-side model probabilities are far too extreme relative to realized discrimination, and the current two-condition STRONG gate collapses to one effective EV condition.

## What is and is not authorized from this result

Authorized conclusion:

> On the identity-clean historical empirical-MC cohort, `prob_edge >= 3pp` provides no independent protection once `EV >= 5%` is satisfied, while model probabilities are dramatically overconfident relative to realized outcomes. STRONG over-firing therefore survives removal of `component_sd` because the empirical distribution/probability layer itself remains too sharp and the gate's two nominal safeguards are algebraically redundant on observed prices.

Not authorized:

- no new STRONG threshold;
- no threshold optimization from historical ROI;
- no production gate change;
- no use of sportsbook information upstream;
- no claim that a specific calibration transform is promotion-ready.

The next defensible downstream step, if explicitly authorized, is a **prospectively frozen probability-calibration study** using genuine season holdout (fit calibration on one season, blind-test the other) while preserving the football projection and sportsbook separation. It should compare calibrated empirical probabilities to the untouched current gate, not tune the gate directly to ROI.

## Reproducibility

Script: `scripts/research/diagnose_empirical_strong_gate_algebra_v1.py`

Input:

```text
historical-fair-probability-reconstruction-v1
run 34712931786
artifact 10304625925
grade/empirical_fair_prob_detail.csv
```

Outputs committed with this diagnosis:

- `docs/research/overnight/empirical_strong_gate_overlap_summary_v1.csv`
- `docs/research/overnight/empirical_model_p_calibration_bins_v1.csv`

No production/model/weight/threshold change.
