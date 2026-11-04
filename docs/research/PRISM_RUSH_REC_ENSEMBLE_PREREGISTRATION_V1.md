# PRISM — `rush_rec_yards` ENSEMBLE CALIBRATION, PREREGISTRATION V1

**Frozen:** 2026-11-04, **before any arm is scored.**
**Status:** PREREGISTRATION. No result here. Nothing promoted.
**Target artifact:** a new row in `data/model_ensemble_weights.csv` — a **canonical production
artifact** under AGENTS.md §5. This is therefore a **production change**, gated accordingly.

---

## 0. This fit was already run once and REJECTED — why it is being re-run

**`docs/research/overnight/ENSEMBLE_WEIGHT_2023_HOLDOUT_RESULT.md` already did this.** It fit
2023-only weights for `rec_yards`, `receptions` **and `rush_rec_yards`**, applied them blind to
2024–2025, and got:

| market | MAE before→after | ROI before→after | promoted? |
|---|---|---|---|
| `rec_yards` | 21.603→20.810 | −3.36%→−3.13% | **yes** |
| `receptions` | 1.707→1.597 | −0.87%→−0.80% | **yes** |
| **`rush_rec_yards`** | **33.040→30.456** | **−4.42%→−6.10%** | **NO** |

`rush_rec_yards` had the **largest MAE gain of the three** and was rejected on **main-line
STRONG-tier ROI**. Weights mc 0.507 / ml 0.468 / state 0.025, 4,991 calibration rows.

**Two V23 results invalidate that rejection gate**, and this is stated plainly rather than
quietly dropped:

1. **The main line is a near-perfect median in every volume quartile** —
   `PRISM_ALT_LINE_ANCHOR_RESULTS_V1.md` §2, realized over-rate 0.476–0.508 across all four
   quartiles of all five markets. **Main-line ROI is unbeatable by construction**, so it cannot
   function as a promotion gate for a projection change: an arm that improves the projection and
   leaves main-line ROI flat-to-worse is the *expected* outcome, not a failure.
2. **The selection layer that built the STRONG tier is miscalibrated by up to +0.448 on exactly
   this market's high-volume rows** — the model asserts a 3.9% chance of the over where reality is
   48.7%. The ROI reading was taken *through* that layer.

**The original result document says this itself:** *"better point-accuracy does not reliably
translate to better ROI when the probability/edge-selection layer underneath is still the disclosed
`Normal(mean, component_sd)` approximation."* It deferred the call rather than settling it.

**So this is a re-run of a settled fit against a corrected decision criterion.** The criterion is
now **projection accuracy and level**, which is what AGENTS.md's PLAYER PROJECTION FIRST PRINCIPLE
makes the primary object. **Main-line ROI is explicitly not a gate here (see K7).**

**What is new since: the diagnosis.** `rush_rec_yards` has **no row at all** in
`data/model_ensemble_weights.csv`, so it ships `uncalibrated_mc_only` /
`mc_fallback_no_oos_weights` — **raw Monte Carlo, zero weight on ML and State, both of which are
computed and present.** Measured on 2024+2025, n=2,573:

| component | mean | mean/actual | MAE |
|---|---|---|---|
| `mc_proj` **(= what ships)** | 40.956 | **0.6570** | 32.598 |
| `ml_proj` | 59.327 | **0.9517** | **29.385** |
| `state_proj` | 52.347 | 0.8398 | 31.265 |
| `actual` | 62.335 | — | — |

**ML is already close to correct on level and lowest on MAE, and it receives zero weight.**

## 1. Research question

**Does fitting the already-promoted `nonnegative_oos_linear_blend_v2` method for
`rush_rec_yards` improve the projection on a blind holdout, in level and in per-player error,
without harming any other market?**

Secondary, measured but **not promotable here**: **how much of the level error does the method's
own convex, intercept-free constraint make unfixable?**

## 2. Arms

- **A0 — naive control: current production.** `ensemble_proj = mc_proj`. The thing to beat.
- **A1 — the promotion candidate.** `scripts.modeling.ensemble_v2.fit_market_weights` **unmodified**,
  fit on **2024 only**, applied blind to **2025**. Same method, same function, same normalisation as
  the five existing rows.
- **A2 — reference only, NOT promotable.** `ml_proj` alone. Measures what the blend leaves behind.
- **A3 — reference only, NOT promotable.** Non-negative least squares, intercept-free, weights
  **not** renormalised to sum to 1. Measures what the sum-to-one constraint costs.

**A2 and A3 cannot be promoted by this study at any result.** Changing the ensemble *method*
affects all five existing rows and needs its own preregistration. They exist to size the
limitation, not to be shipped.

## 3. Sample, split and lineage

| | |
|---|---|
| fit | **2024** (`clean_v1_full_stack_vegas_benchmark_detail.csv`, `rush_rec_yards`, 1,937 rows) |
| blind holdout | **2025** (654 rows), untouched by the fit |

**Lineage differs from its siblings and that is disclosed.** `rec_yards`/`receptions` carry
`fit_2023_only_blind_holdout_2024_2025`. **No 2023 component data survives in this repository** —
`component_predictions_2023.csv` was produced in a workflow run and never committed, an
AGENTS.md §5a casualty. This row will therefore read `fit_2024_only_blind_holdout_2025`.

All bootstrapping is **clustered over players**, never over rows.

## 4. Gates

- **M1.** No sportsbook variable is a feature or a weight target. Target is `actual`; features are
  `mc_proj`, `ml_proj`, `state_proj`. Asserted by name against the forbidden list; **failure voids
  the run.**
- **M2.** No 2025 row enters the fit. Asserted on the fitting frame's season values; **failure
  voids the run.**
- **M3.** The consumer applies the new row **with no code change.** Verified by calling
  `apply_ensemble` with the amended weights and requiring `ensemble_status == "calibrated"` on
  `rush_rec_yards` rows.
- **M4 — positive control.** After the edit, the `pass_yards`, `rush_att`, `rush_yards`,
  `rec_yards` and `receptions` rows must be **byte-identical** to their current values, and
  `apply_ensemble` must return bit-identical `ensemble_proj` for every non-`rush_rec_yards` market.
  **`pass_yards` is the standing positive control: an arm that moves it is disqualified.**
- **S1.** A1 beats A0 on blind-2025 **per-player mean absolute error**, 95% CI excluding 0,
  clustered over players. *(Reported per player and per quartile, not as a pooled average —
  AGENTS.md disallows averaged MAE as the primary metric for a player-level claim.)*
- **S2.** A1 reduces `|mean(proj)/mean(actual) − 1|` versus A0 on blind 2025.
- **S3.** A1 worsens no line-quartile cell's MAE by more than **5% relative**.
- **S4.** Production gates pass after the edit: `python -m compileall scripts engine tests`,
  `python scripts/utils/audit_repo.py --strict`, `pytest -q`.

## 5. Kill conditions

**K1** M1–M4 breached → **run void, no row written.**
**K2** S1 fails → no blind improvement; **do not write the row.**
**K3** S2 fails → the blend does not fix the level; report it and **do not write the row** —
the whole football case for this change is the level.
**K4** S3 breached → name the quartile and arm; **do not write the row.**
**K5** S4 fails → **revert the row immediately**, in the same session.
**K6** fewer than **40** usable calibration rows → below `MIN_CALIBRATION_ROWS`; nothing fits.
**K7 — NOT a kill condition: main-line ROI.** It was the 2023 rejection gate and it is
**deliberately dropped**, for the two reasons in §0. **This is the single most important thing a
future reader must know about this promotion**: it was promoted on football accuracy, over a prior
rejection on betting ROI, because that ROI lane was subsequently proven unbeatable by construction
and its selection layer proven miscalibrated. **If anyone later shows main-line ROI is beatable,
this promotion must be re-examined.**

## 6. Prediction

**S1 and S2 both pass, and A1 only partially closes the level gap** — I expect roughly
0.657 → **0.78–0.84** of actual, not → 0.95. The convex constraint forces a ~30%-low MC into the
blend, and the regression minimises squared error, which trades level bias for variance reduction.
The 2023 fit landed mc 0.507 / ml 0.468, and a 2024 fit should land nearby.

**A2 (`ml_proj` alone) beats A1 on both MAE and level.** That is the real finding of this study and
**it is not promotable here.** If it holds, the honest next step is a preregistered study of the
ensemble *method* — specifically whether the sum-to-one constraint should be relaxed — across all
five existing rows, with `pass_yards` as the positive control.

**Named risks, weighted:**

1. **The strongest risk: that this improves MAE and level and still changes nothing that matters.**
   The V23 betting layer anchors on the **line**, not on our projection, so a better
   `rush_rec_yards` mean **does not change a single bet placed off the decision card.** The
   justification is the football projection itself under AGENTS.md's first principle, not money.
   **I should not let a passing S1 imply a betting gain.** I rate this the likeliest way this
   result gets over-read.
2. **A 2024-only fit may put more weight on `mc_proj` than the 2023 fit did**, since 2024 is 1,937
   rows against 4,991 and the seasons' component quality may differ. If mc ≥ 0.6 the level gain is
   materially smaller than §6 projects.
3. **654 blind rows across ~114 players** is thin for a player-clustered CI. S1 may land ns on
   sheer width even if the point estimate is good.
4. **The level fix may be in the wrong moment.** V23 showed the production compression is in the
   *spread* across volume, not only in the mean. A uniform reweighting lifts the whole
   distribution and may leave the quartile spread intact — which S3 will show.

My record: **~11 prediction hits, ~21 misses**, the named risk landing in nearly every study.

## 7. Scope

`rush_rec_yards` ensemble weights only. **One row added to
`data/model_ensemble_weights.csv`; no code modified**, because the consumer is already generic
(`_weight_map` keys on the lowercased market string). No odds fetched. No pricing path, no
simulation, no Bayesian, no rules, no QB synthesis, no RB P3 route, no 0.95 cap touched.

A pass licenses **exactly** the one row plus its documentation. It does **not** license a method
change (A2/A3), a `receptions` reparameterisation, or any ticket construction.
