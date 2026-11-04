# PRISM — ENSEMBLE METHOD: SHOULD THE SUM-TO-ONE CONSTRAINT BE RELAXED? PREREGISTRATION V1

**Frozen:** 2026-11-04, **before any arm is scored.**
**Status:** PREREGISTRATION. No result here. Nothing promoted.
**Scope of the change if it passes:** `scripts/modeling/ensemble_v2.py` **and all six rows** of
`data/model_ensemble_weights.csv`. This is the largest production change contemplated in this
programme, and the gates are set accordingly.

---

## 0. What prompted this, and it is already measured

`PRISM_RUSH_REC_ENSEMBLE_RESULTS_V1.md` §4. `fit_market_weights` is
`LinearRegression(fit_intercept=False, positive=True)` followed by `_normalise_nonnegative`,
which divides the fitted coefficients by their sum.

**For `rush_rec_yards` the raw coefficients sum to 1.2315** — the regression wants to scale the
components **up 23%**, and the normalisation discards exactly that. Measured on blind 2025:

| arm | level (mean/actual) | per-player MAE vs the promoted blend |
|---|---|---|
| promoted, normalised | 0.7773 | — |
| same fit, **unnormalised** | **0.9572** | +0.7343 [−0.5183, +1.9558] — CI spans 0 |
| `ml_proj` alone | 0.9392 | +0.4751 [−0.5856, +1.5122] — CI spans 0 |

**And the pattern is not confined to one market.** Component levels on 2024+2025:

| market | `mc_proj`/actual | `ml_proj`/actual | promoted blend reaches |
|---|---|---|---|
| `rec_yards` | 0.7147 | 1.0123 | — |
| `receptions` | 0.6992 | 1.0183 | — |
| `rush_yards` | 0.5885 | 0.9436 | **0.7458** |
| `rush_rec_yards` | 0.6570 | 0.9517 | 0.7773 |
| `pass_yards` | — | — | ~1.0 |

**Monte Carlo runs 30–41% low in every non-QB market, ML is near correct, and a convex blend
cannot escape the interval between them.** `rush_yards`' promoted blend still only reaches 0.746.

## 1. Research question

**Should the ensemble be an unconstrained non-negative linear combination rather than a convex
one — and is any gain from relaxing the constraint, or merely from refitting?**

## 2. Arms

Per market, fit on 2024, blind on 2025.

- **C0 — incumbent / naive control.** The weights exactly as they stand in
  `data/model_ensemble_weights.csv` today. **2025 is a genuine blind holdout for every existing
  row**: `pass_yards`/`rush_att`/`rush_yards` are `all_2024_oos_frozen_for_2025`,
  `rec_yards`/`receptions` are `fit_2023_only`, `rush_rec_yards` is `fit_2024_only`.
- **C1 — refit, still normalised.** Current method, refit on 2024. **Isolates "refit" from
  "method change" — without this arm the study cannot attribute its own result.**
- **C2 — refit, unnormalised.** `LinearRegression(fit_intercept=False, positive=True)`, weights
  used **as fitted**. **The primary candidate.**
- **C3 — refit, unnormalised, with intercept.** Tests whether the level error is **additive**
  rather than **multiplicative**. Secondary candidate; a win here points at M89's residual design
  rather than at reweighting.

## 3. Gates

- **M1 — all arms.** No sportsbook variable is a feature or a target. Target `actual`, features
  `mc_proj`/`ml_proj`/`state_proj`. Asserted by name; **failure voids the run.**
- **M2 — all arms.** No 2025 row enters any fit. Asserted; **failure voids the run.**
- **M3 — implementation sanity.** With weights forced to sum to 1 and intercept 0, C2 must
  reproduce C1 to ≤1e-12. Confirms the only difference between them is the constraint.
- **M4 — POSITIVE CONTROL, and it is absolute.** `pass_yards` is the one market with a promoted
  synthesis, ~1.0 level and no compression. **If C2 (or C3) degrades `pass_yards` per-player MAE
  with a CI excluding zero, that arm is disqualified for EVERY market**, however well it does
  elsewhere. This is the standing rule in this programme and it is not negotiable here.
- **S1 — superiority, per market.** C2 beats C0 on blind per-player MAE, CI excluding 0, clustered
  over players. **Reported, NOT required — see §4.**
- **S1N — non-inferiority, per market.** C2's per-player MAE delta vs C0 must **not** exclude zero
  on the worse side. **This is required.**
- **S2 — level, per market.** C2 must reduce `|mean(proj)/mean(actual) − 1|` by at least
  **5 percentage points** versus C0.
- **S3 — attribution, per market.** C2 must beat **C1** on level by at least 5pp. **If C1 already
  captures the gain, this is a refit and not a method change, and the method must not be touched.**
- **S4 — per line-quartile.** C2 worsens no quartile's MAE by more than **5% relative**.
- **S5 — production.** `compileall`, `audit_repo --strict`, and `pytest -q` with **no new failure**
  versus the branch baseline of 4 pre-existing failures.

## 4. The promotion rule, stated before any number is seen

**I expect level to improve a lot and per-player MAE to be roughly flat.** That is what happened
for `rush_rec_yards`, where unnormalised vs normalised had a CI spanning zero on MAE and an 18pp
level gain. **So I am stating the decision rule now rather than choosing a metric after the fact.**

**Promote a market's C2 weights if and only if: M1–M4 pass, AND S1N (non-inferior on per-player
MAE), AND S2 (≥5pp level gain), AND S3 (the gain is the constraint, not the refit), AND S4.**

**S1 superiority is reported and is not required.** The justification: a systematic 25–40% level
bias is an error a projection should not carry, and non-inferiority on per-player error means we
are not paying for its removal. **If this rule feels convenient after the fact, it is on record
before the fact.**

## 5. Kill conditions

**K1** M1–M3 breached → **run void.**
**K2** **M4 breached → the arm is dead for every market.** No partial promotion, no exception.
**K3** S3 fails → the gain is the refit; **do not touch the method.** Promote nothing here and open
a separate refit study instead.
**K4** S1N fails for a market → that market **not promoted**; others may still be.
**K5** S2 fails for a market → no material level gain there; not promoted.
**K6** C2's weights sum **>1.5** or **<0.7** for any market → the fit is extrapolating a scale
change larger than I will ship without its own study. **Report, do not promote that market.**
**K7** S5 regresses → **revert in the same session.**
**K8 — NOT a kill condition: main-line ROI.** Carried forward from
`PRISM_RUSH_REC_ENSEMBLE_PREREGISTRATION_V1` K7, same reasoning: V23 proved that lane unbeatable
by construction and its selection layer miscalibrated. **If anyone later shows main-line ROI is
beatable, this and the `rush_rec_yards` promotion must both be re-examined.**

## 6. Prediction

**C2 passes S2 and S3 decisively in the four non-QB yardage/reception markets** — levels should go
from 0.70–0.78 to 0.92–0.97 — **and lands non-significant on S1 per-player MAE in most of them.**
The promotion therefore rests on §4's rule, which is why §4 exists.

**`pass_yards` is where I expect trouble and it is the one that can kill everything.** Its
components are already near level, so the unconstrained fit has little to gain and some variance to
add; if its weights sum meaningfully above 1 it will over-project and M4 fires. **I rate this the
single likeliest way the study ships nothing.**

**C3 (intercept) beats C2 on level but loses on per-player MAE**, because an intercept adds a
constant to low-volume players who do not need it. If C3 wins outright, the correct read is that
the level error is additive and the answer is a residual model, not a reweighting.

**Named risks, weighted:**

1. **M4 / `pass_yards`.** As above. The likeliest killer.
2. **The constraint may be doing real work I am about to remove.** Normalising to 1 is a strong
   regulariser; without it the fit can chase scale in-sample and over-project out-of-sample.
   2024→2025 is one season-pair, and a 23% upscale learned on 2024 may be a 2024 fact. **K6 is the
   guard and it may well fire.**
3. **Level and MAE may genuinely disagree about what is better.** If C2 improves level and
   *worsens* per-player MAE significantly, §4's rule correctly refuses it — and I should not then
   argue the metric was wrong.
4. **`rush_att` has no rows in this dataset** (the benchmark carries five markets, not `rush_att`).
   That row cannot be evaluated and **must be left exactly as it is**, not refit blind.

My record: **~13 prediction hits, ~23 misses**, with the named risk landing in nearly every study.
**Weight risk 1 and 2 heavily.**

## 7. Scope

Ensemble weights and, if and only if every gate passes, the normalisation step in
`scripts/modeling/ensemble_v2.py`. **`rush_att` is untouched** (risk 4). No odds fetched. No
pricing path, simulation, Bayesian, rules, QB synthesis, RB P3 route or 0.95 cap touched.

**If the method changes, `_normalise_nonnegative` must stay available and must stay the behaviour
for any market without an unconstrained row**, so a partial promotion cannot silently change a
market this study did not clear.
