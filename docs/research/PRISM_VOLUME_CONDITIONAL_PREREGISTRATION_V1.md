# PRISM — VOLUME-CONDITIONAL RECALIBRATION, PREREGISTRATION V1

**Frozen:** 2026-11-04, **before any arm is scored.**
**Status:** PREREGISTRATION. No result here. Nothing promoted.
**Scope if it passes:** a per-market recalibration applied to `ensemble_proj`.
A **football-only** change to the projection itself — the first one this programme has attempted.

---

## 0. Why this and not another scalar

Three studies now converge on the same shape.

**V22**: compression in **10 of 10** role × market cells, monotone in every one.
**V23 anchor study**: `model_proj / vegas_line` by line quartile on the 2026 boards —

| market | Q0 low | Q1 | Q2 | Q3 high | spread |
|---|---|---|---|---|---|
| `rec_yards` | 1.366 | 1.035 | 0.961 | 0.827 | **53.9pp** |
| `rush_yards` | 1.221 | 1.070 | 0.817 | 0.790 | 43.1pp |
| `receptions` | 1.116 | 0.922 | 0.854 | 0.791 | 32.5pp |
| **`pass_yards`** | 1.077 | 1.026 | 1.012 | 0.933 | **14.4pp** |

**V23 ensemble-method study**: relaxing the sum-to-one constraint fixed the mean
(`rush_yards` level 0.750 → 1.030) and **broke the low end** — Q0/Q1 MAE worse by
+8.7% to +12.9% in every market while Q2/Q3 improved. Adding an intercept (C3)
did not help either, and its fitted intercepts were **positive** (+3.485
`rec_yards`, +2.953 `rush_yards`), i.e. it added to the low end that was already
too high.

**So the error is a slope and both scalars failed in the same way: they move the
whole distribution.** The correction has to shrink the low end and expand the
high end at once.

**The fourth pointer is the control itself.** `pass_yards` is the only market
with a volume-conditional correction already in production — the M89/M90
synthesis, whose features include `pred_attempts` — and it is the only market
without the defect (14.4pp against 32–54pp).

## 1. Research question

**Does a volume-conditional recalibration of `ensemble_proj` reduce the
compression spread without costing per-player accuracy?**

## 2. The conditioner is the model's own output — and that is a §3 requirement

**The volume proxy must be football-only.** `vegas_line` is the obvious
volume signal and it is **FORBIDDEN**: AGENTS.md §3 bars sportsbook variables
from the football projection, and this is the football projection.

The conditioner is therefore **`ensemble_proj` itself**. Recalibrating a model
against its own output is standard (Platt, isotonic) and uses no market data.
Quantile cut points come from the **fit sample only**.

## 3. Arms

Per market, fit **2024**, blind **2025**.

- **V0 — incumbent.** `ensemble_proj` with the promoted weights, uncorrected. The control.
- **V1 — linear recalibration.** `a + b · proj`, per market. The minimal model that
  can express a slope. **Not the same as the failed C3**, which fitted an intercept
  on top of free component weights; this fits `(a, b)` directly against `actual`.
- **V2 — tiered multiplier.** A separate multiplicative factor per **projection
  quartile**, cut on the fit sample. The least constrained expression of
  "volume-conditional" and the most likely to overfit.
- **V3 — isotonic.** Monotone non-parametric regression of `actual` on `proj`.
  Cannot invert the ordering; clamps outside the fit range.

## 4. Gates

- **M1 — all arms.** No sportsbook variable enters any arm. Conditioner is
  `ensemble_proj`; target is `actual`. Asserted by name; **failure voids the run.**
- **M2 — all arms.** No 2025 row in any fit. Asserted; **failure voids the run.**
- **M3 — all arms.** At `a=0, b=1` (identity) every arm must reproduce V0 exactly.
- **M4 — POSITIVE CONTROL, absolute.** **`pass_yards` must not be damaged.** If an
  arm worsens its per-player MAE with a CI excluding zero, **that arm is dead for
  every market.** `pass_yards` already carries a volume-conditional correction, so
  a second one making it worse means the arm is doing something other than
  de-compressing.
- **S1 — PRIMARY: compression spread.** Define
  `spread = mean(proj/actual | top proj quartile) − mean(proj/actual | bottom proj quartile)`
  on blind 2025. An arm must reduce **|spread|** by at least **40% relative** versus
  V0. *This is the defect; it is the primary metric, not MAE.*
- **S2 — non-inferiority on per-player MAE.** The arm's per-player MAE delta vs V0
  must **not** exclude zero on the worse side, bootstrapped over players.
- **S3 — per-quartile, both ends.** No projection quartile's MAE may worsen by more
  than **5% relative**. **This is the gate the previous scalars failed**, and it is
  the one that decides whether this is genuinely different from them.
- **S4 — level.** `mean(proj)/mean(actual)` must land within **±5pp of 1.0**.

## 5. Kill conditions

**K1** M1–M3 breached → **run void.**
**K2** **M4 breached → that arm is dead for every market.** No partial promotion.
**K3** S1 fails for every arm → the spread is not correctable by recalibration on
the model's own output, and the next move is a feature-level fix, not a calibration one.
**K4** S3 breached for an arm → **it is the same failure as the scalars** and it is
not promoted, however good its spread looks.
**K5** fewer than **40 players** or **100 rows** in a market's blind sample → not promoted.
**K6** V2 beats V1 and V3 on the fit but not the holdout → overfitting; prefer the
constrained arm.
**K7 — NOT a gate: main-line ROI.** Carried forward from the `rush_rec_yards` and
ensemble-method preregistrations, same reasoning.

## 6. Prediction

**V1 and V3 pass S1; V2 overfits.** A linear slope and an isotonic map can both
shrink-low/expand-high, which is what the defect needs; a free per-quartile
multiplier has four parameters against a noisy target and 2024 is one season.

**S3 is where this is decided and I am genuinely unsure.** Every previous arm
died there. The difference now is that V1/V3 can move the two ends in *opposite*
directions, which C2 and C3 structurally could not — but it is also possible
that the low-end "over-projection" is mostly the zero-inflation of these markets
rather than a slope, in which case shrinking the low end will help the mean and
hurt the MAE exactly as before.

**Named risks, weighted:**

1. **The biggest: `proj/actual` is a ratio with a near-zero denominator at the low
   end.** 10% of `rec_yards` rows have `actual = 0`. The spread metric may be
   dominated by those rows and not mean what it looks like. I will report the
   spread on a trimmed basis as well, and if the two disagree the trimmed one governs.
2. **Conditioning on the model's own output cannot fix a feature-level error.** If
   the model is wrong about *which* players are high volume, recalibration only
   relabels the error. S1 could pass while real per-player accuracy is unchanged —
   which is exactly what S2 and S3 are for.
3. **Regression to the mean is correct for MSE.** Some of this "compression" is the
   optimal shrinkage any good estimator applies. De-compressing necessarily adds
   variance, and S3 is where that bill arrives.
4. **One season of fit.** 2024 only, per market, with 36 players for `pass_yards`.

My record: **~15 hits, ~25 misses**, with the named risk landing in nearly every
study. **Weight risk 1 and 3.**

## 7. Scope

A recalibration layer applied to `ensemble_proj`, per market. **No component model,
no simulation, no rules, no Bayesian, no QB synthesis, no RB P3 route, no 0.95 cap,
no sportsbook variable.** A pass licenses the layer plus its coefficients; it does
**not** license touching the ensemble weights, which the previous study closed.
