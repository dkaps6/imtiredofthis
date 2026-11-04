# PRISM — SHRUNK VOLUME-CONDITIONAL CORRECTION, PREREGISTRATION V1

**Frozen:** 2026-11-04, **before any arm is fitted or scored.**
**Status:** PREREGISTRATION. No result here. Nothing promoted.
**Parent:** `PRISM_VOLUME_CONDITIONAL_RESULTS_V1.md`

---

## 0. What this follows from

The parent study established that **the compression spread IS correctable** by recalibrating
`ensemble_proj` against its own output — `receptions` fell **0.3247 → 0.0558**, an 83% cut, with
per-player MAE non-inferior and no quartile worse than +1.2%. It refuted my main structural worry.

**It failed on one thing: over-correction.** Every arm overshot the level (`receptions`
0.709 → 1.06) and the spread flipped sign. `receptions` missed promotion **by about 1 percentage
point of level**.

The parent explicitly refused to shrink the correction to clear that bar, because choosing a
coefficient after seeing which gate missed is what preregistration prevents. **This is that study
done properly.**

## 1. Research question

**Does a shrunk volume-conditional correction, with the shrinkage chosen without touching the test
set, survive a genuinely independent season?**

## 2. The selection procedure — fully specified, and it never sees 2026

Both the **arm** and the **shrinkage λ** are chosen by **leave-one-season-out inside 2024+2025
only**:

1. Grid: **λ ∈ {0.0, 0.1, 0.2, …, 1.0}**. λ=0 is no correction, λ=1 is the parent's full correction.
   `corrected = proj + λ · (f(proj) − proj)`.
2. Arms: **V1 linear**, **V2 tiered**, **V3 isotonic**, as in the parent.
3. For each (arm, λ): fit on 2024 → score 2025, fit on 2025 → score 2024, **average the two
   per-player MAEs**.
4. **Select the single (arm, λ) minimising that averaged per-player MAE**, per market.

**The criterion is per-player MAE and nothing else.** It is deliberately *not* |level − 1|, which
is the gate that failed — selecting on the failed gate would be circular. If the MAE-optimal λ
also fixes the level, that is evidence; if it does not, this fails honestly.

**2026 is touched exactly once, to score the already-chosen (arm, λ).** No tuning against it.

## 3. The baseline must match what production actually ships — and naively it would not

**This is the correction most likely to be missed and it would invalidate the whole study.**

In the 2024/25 benchmark, `ensemble_proj == mc_proj` for `rec_yards`, `receptions` and
`rush_rec_yards` — those markets ran `uncalibrated_mc_only` at the time, so the recorded ensemble
is **raw Monte Carlo at ~0.70 of actual**. The 2026 boards are **calibrated** and sit near
0.84–0.89.

A correction fitted on the uncalibrated baseline and applied to the calibrated one would
**over-correct 2026 by construction**, and I would have read that as the method failing.

**So the fit baseline is reconstructed to match production:** apply the promoted weights from
`data/model_ensemble_weights.csv` to the `mc_proj`/`ml_proj`/`state_proj` columns of the 2024/25
benchmark, and fit the correction on **that**. Asserted as gate M5.

## 4. Scope — `pass_yards` is excluded, and that fixes a dead end

The parent could not verify its positive control: `pass_yards` had **35 blind players** against a
40-player bar, so K5 fired and nothing could promote regardless of merit. **A control that can
never be evaluated is a design defect, not a safeguard.**

`pass_yards` is therefore **out of scope**. It already carries a volume-conditional correction in
production (the M89/M90 synthesis, features including `pred_attempts`) and has **14.4pp** of spread
against 32–54pp elsewhere. It does not need this one.

The control becomes **verifiable by construction**: the correction is per-market and emits no
`pass_yards` coefficient, so `pass_yards` output must be **bit-identical**. Asserted as M4.

Markets in scope: **`rec_yards`, `receptions`, `rush_yards`, `rush_rec_yards`.**

## 5. Gates — scored on 2026 (weeks 1/2/6/7/8 boards joined to nflverse actuals)

- **M1.** No sportsbook variable in any arm. Conditioner `ensemble_proj`, target `actual`.
  Asserted by name; **failure voids the run.**
- **M2.** No 2026 row in any fit or in the λ/arm selection. Asserted; **failure voids the run.**
- **M3.** At λ=0 every arm reproduces the baseline exactly (≤1e-12).
- **M4.** `pass_yards` emits no coefficient and is bit-identical.
- **M5.** The fit baseline is the **reconstructed calibrated** ensemble (§3), asserted by comparing
  its level against the recorded `ensemble_proj` level and requiring them to differ for the three
  mc-only markets.
- **S1 — spread.** On 2026, |spread| reduced by **≥40% relative** to the uncorrected baseline,
  median-of-ratios (the robust reading the parent designated as governing).
- **S2 — non-inferiority.** Per-player MAE delta vs baseline must **not** exclude zero on the worse
  side, bootstrapped over players.
- **S3 — both ends.** No projection quartile worse than **+5% relative** MAE.
- **S4 — level.** `mean(corrected)/mean(actual)` within **±5pp of 1.0**.
- **S6 — the shrinkage must be real.** The selected **λ < 1.0**. If LOSO picks λ=1.0 then no
  shrinkage was learned and this is just the parent study re-run; report it as such.

## 6. Kill conditions

**K1** M1–M5 breached → **run void.**
**K2** S1 fails → the spread does not transfer to an independent season; the parent's 83% was
in-sample-adjacent and must be withdrawn as a forward claim.
**K3** S4 fails → over-correction survives shrinkage, and the level error is not a tuning problem.
**K4** S3 breached → the low end is damaged, same as every scalar; not promoted.
**K5** fewer than **40 players** or **100 rows** in a market's 2026 sample → not promoted, reported.
**K6** selected λ = 0.0 → LOSO says the best correction is **no correction**; the lane closes.
**K7 — NOT a gate: main-line ROI.** Carried forward.

## 7. Prediction

**λ lands between 0.3 and 0.7 and `receptions` passes all four gates.** The parent's overshoot was
roughly 6pp of level on a 29pp move, so something near a two-thirds correction should land close to
1.0, and `receptions` already passed S1/S2/S3 at full strength.

**I expect `rush_yards` to fail S3 again.** It was the one market whose low quartile was damaged at
full strength (+9.9%), and shrinkage reduces the damage proportionally rather than removing it.

**Named risks, weighted:**

1. **§3 is the study.** If the reconstructed baseline is wrong — wrong weights, wrong column
   semantics, a market where the recorded ensemble was already calibrated — then the fit and the
   test are on different scales and every result is meaningless. **I rate this the single most
   likely way this study produces a confidently wrong answer rather than a null.**
2. **2026 is five weeks, not a season**, and three of those weeks (1, 2) are early-season when
   current-form evidence is thinnest. A correction validated on that may not hold in weeks 9–18.
3. **LOSO across two seasons is two folds.** λ selected from an average of two numbers is itself
   noisy, and the grid is coarse enough that adjacent λ values may be statistically identical.
4. **Selecting on MAE may systematically under-shrink for the level gate.** MAE rewards variance
   reduction, which favours small λ; the level needs a specific λ. These can disagree, and if they
   do, S4 fails and that is the honest answer rather than a reason to change the criterion.

My record: **~16 hits, ~27 misses**, the named risk landing in nearly every study. **Weight risk 1.**

## 8. Scope

A per-market recalibration layer over `ensemble_proj` for four non-QB markets. No component model,
no simulation, no rules, no Bayesian, no QB synthesis, no RB P3 route, no 0.95 cap, no ensemble
weight, no sportsbook variable. A pass licenses the layer and its coefficients and nothing else.
