# PRISM — RECEIVING MASS DECOMPOSITION V1

**Date:** 2026-11-03
**Status:** MEASUREMENT. Nothing promoted. No production behaviour changed.
**Follows:** `PRISM_ALPHA_TIER_RESULTS_V1.md` §4, which found Ja'Marr Chase's *target share*
correctly projected (0.2485 vs 0.2424 actual) while V21 found his *receptions* projected at 4.49
against a market implying 6+. That proved the elite-receiver gap is **at least two defects**.
This decomposes the second one.

---

## 1. Receiving mass is systematically under-projected

Same player set on both sides — board rows joined to their own graded actuals, so this is a
projection bias, not a roster-coverage artifact.

| market | team-weeks | model | actual | bias | 95% CI |
|---|---|---|---|---|---|
| **receptions** | 84 | 15.52 | 18.21 | **−2.70 (−14.8%)** | [−3.80, −1.62] |
| **rec_yards** | 86 | 170.47 | 203.45 | **−32.99 (−16.2%)** | [−46.86, −18.95] |

73.8% of team-weeks are under-projected on receptions.

---

## 2. The receptions bias is ADDITIVE, not compression

Regressing `actual ~ a + b·model_proj`:

| market | slope b | intercept a | mean bias |
|---|---|---|---|
| **receptions** | **1.0027** | +0.52 | −0.53 |
| rec_yards | 1.1249 | +3.10 | −6.99 |
| rush_yards | 1.2038 | +0.22 | −6.25 |
| pass_yards | 1.0640 | −16.77 | **+2.14** |

Slope ≈ 1.00 on receptions, and the bias is flat across every level:

| model_proj tier | n | model | actual | bias |
|---|---|---|---|---|
| <1.5 | 81 | 1.27 | 1.80 | −0.53 |
| 1.5–3 | 288 | 2.22 | 2.73 | −0.51 |
| 3–4.5 | 230 | 3.63 | 4.14 | −0.50 |
| 4.5+ | 46 | 5.47 | 6.26 | −0.79 |

**Every priced receiver is about half a reception low, regardless of his level.** This is a
different defect from the alpha-tier compression (which is multiplicative and tier-dependent).
`rec_yards` and `rush_yards` show *both* an offset and genuine compression (slope 1.12, 1.20).

`discrete_count_alignment` was checked and exonerated: `target_mean_gap = 0.0000` on all 496
Week 8 receptions rows — it preserves the mean.

---

## 3. The cause: the ensemble discards its only unbiased component

Component biases on receptions, graded:

| component | mean | bias | MAE |
|---|---|---|---|
| `mc_proj` | 2.61 | **−0.762** | 1.702 |
| **`ml_proj`** | **3.27** | **−0.105** | 1.708 |
| `state_proj` | 2.91 | −0.485 | 1.814 |
| `ensemble_proj` = `model_proj` | 2.84 | −0.530 | **1.663** |

Ensemble weights, `nonnegative_oos_linear_blend_v2`:

| market | w_mc | w_ml | w_state |
|---|---|---|---|
| receptions | **0.658** | 0.341 | 0.001 |
| rec_yards | **0.739** | 0.224 | 0.036 |
| rush_yards | 0.572 | 0.428 | 0.000 |
| pass_yards | 0.210 | 0.269 | 0.520 |
| **rush_rec_yards** | **1.000** | 0.000 | 0.000 |

**The ML component is nearly unbiased and receives 34% of the weight; the most biased component
receives 66%.** Sweeping the blend:

| w_ml | bias | MAE |
|---|---|---|
| 0.00 | −0.757 | 1.702 |
| 0.50 | −0.431 | **1.658 (min)** |
| **1.00** | **−0.105** | 1.708 |
| current (0.34) | −0.522 | 1.663 |

**The ensemble is sitting near the MAE optimum, and the MAE optimum is a biased estimator.**
Moving to pure ML removes 80% of the bias for a 2.7% MAE increase.

This is the failure mode AGENTS.md names explicitly:

> *"Averaged MAE is disallowed as the primary metric for any individual-player distribution
> claim. A model can have every mean right and every floor and ceiling wrong and score
> identically."*

The ensemble was calibrated on an MAE-like objective and bought a −0.53 reception bias with it.

**Separately: `rush_rec_yards` has no calibration at all** — `mc_fallback_no_oos_weights`,
w_mc = 1.000 — and carries the worst bias on the board, **−17.49**. That is an uncalibrated
market running in production.

---

## 4. The complication that forbids a naive fix

Side accuracy on receptions (model pick vs line, corrected direction):

| component | over-rate | hit rate |
|---|---|---|
| `mc_proj` (most biased) | 23.7% | **54.73%** |
| `ensemble_proj` | 32.7% | 54.11% |
| **`ml_proj` (least biased)** | 54.7% | **52.36%** |

**Removing the bias makes side accuracy worse.** The reason is visible in the levels:

```
model 2.853   <   line 3.204   <   actual 3.376
```

**The market is also low** — by −0.165 on receptions and −4.97 on `rec_yards`. The model's
under-projection is partly tracking a real market-versus-reality gap, so its downward bias
accidentally aligns with a market inefficiency. That is coincidence, not skill, and it is
fragile: it depends on the market's own bias persisting.

---

## 5. What this means

**Projection accuracy and betting side-selection are in direct conflict here**, and AGENTS.md
settles which wins:

> *"This repository builds individual-player projections. The named player in his current
> football situation is the primary scientific unit."*

By the repository's own first principle the **bias should be removed** and side selection
treated as a separate downstream concern that may need its own correction. Optimising the
projection for betting side accuracy is exactly the market-contamination AGENTS.md §3 forbids,
arriving through the back door of a calibration objective.

**Open questions this does not answer:**
- Why is `mc_proj` biased −0.76 on receptions? The MC path is where the finite-opportunity
  allocation lives, and the RB lane found the same simulator over-allocating rush mass then
  haircutting it. Receiving may have the mirror defect. **Untested.**
- Would a bias-corrected ensemble hold up prospectively, or is the ML component's low bias a
  2024–2026 artifact?
- `rush_rec_yards` running uncalibrated at −17.49 bias needs its own treatment and should not
  be bundled with this.

---

## 6. Next

Preregister a receiving-mass study with **bias as the primary metric and MAE as a reported
guardrail, not the objective** — the inverse of how the current ensemble was fit. Arms should
include a bias-corrected ensemble weighting and a direct test of whether the MC allocation
step is the source.

Do **not** change ensemble weights before that study. The current weights are frozen/promoted
calibration (AGENTS.md §5), and the side-accuracy complication in §4 means a naive bias fix
would degrade a visible metric while improving the one that matters.
