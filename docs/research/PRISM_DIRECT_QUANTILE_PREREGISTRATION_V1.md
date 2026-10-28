# PRISM — Direct per-tail quantile fit: PREREGISTRATION (V1)

**Frozen before any scoring.** Committed before the model is fit or evaluated. If the
result contradicts the hypothesis, that is the finding.

## 0. Why, and what went wrong last time

`PRISM_PLAYER_DISPERSION_RESULTS_V1` established on naive 2022 holdout data that
production's intervals contain **50.0%** of outcomes against a **60%** target —
systematically too narrow, worst for the best players. Scaling a single σ by the
player's projected mean raised well-calibrated players from **41.7% → 52.1%** and cut
high-mean floor bias from **+9.4 → −3.9**.

It still failed the frozen rule, and the failure was specific: a symmetric normal
centred on the mean **forces p20 and p80 equidistant**, so widening fixed the floor for
77.1% of players while breaking the ceiling for 68.8% (high-mean ceiling bias
−2.7 → +10.5). The empirical distribution puts the floor further from the mean than the
ceiling. A preregistered lognormal matched on the same mean and σ gave asymmetry in the
right direction but narrowed the central interval, dropping coverage back to 0.500.

**Diagnosis: the blocker is the distribution family, not the σ function.**

This study removes the family. It fits the two tails **directly and independently**, so
the per-tail asymmetry is free rather than implied by any parametric shape.

## 1. Hypothesis

**H1.** Fitting p20 and p80 directly, each as its own function of the player's projected
mean, calibrates individual floors and ceilings better than production's constant-σ
normal — and does so at **both** tails simultaneously, which the symmetric form could
not.

**H0.** Freeing the tails does not achieve two-tailed improvement either, in which case
the per-player floor/ceiling problem is not solvable from the projected mean alone and
the lane stops rather than acquiring more features.

## 2. Data, frozen

- **Fit:** 2020, 2021, 2022. All already inspected; fitting on inspected data is sound,
  only the holdout must be naive.
- **Primary holdout: 2017, 2018, 2019 — never inspected.** The only decisive evaluation.
- **Reported, NOT decisive:** 2023, 2024, 2025 (contaminated by prior work).
- The holdout is **earlier** than the fit. Disclosed deliberately: these are the only
  naive seasons left, and because every arm predicts from a player's **own
  within-season prior mean**, the target is a structural relationship rather than an era
  level. Testing it on a different era is a harder test, not an easier one.
- Season length is handled per era: regular season runs to **week 17** before 2021 and
  **week 18** from 2021.
- Market: WR/TE **receiving yards**. **No sportsbook variable at any stage.**

## 3. Eligibility and gate, frozen (unchanged from the prior study)

- ≥ **20** prior targets, ≥ **3** prior games with a target, week ≥ **4**, ≥ **1**
  target in week *W*;
- anomaly gate: `DNP_INACTIVE` excluded from prior and outcome, `PARTIAL_GAME` excluded
  from outcomes, `UNRESOLVED_CONTEXT` fails closed; counts reported;
- `MIN_HOLDOUT_GAMES = 8` for per-player coverage, carried forward explicitly this time
  rather than left unfrozen, with sensitivity at 6 and 10 reported.

## 4. Models, frozen

`mu_hat` is **identical in every arm** — the player's own strictly-prior
`prior_ypt × prior_tgt_per_game`. `centrality` is his strictly-prior share of team
targets.

- **Q0 — production equivalent:** `p20 = mu_hat − 0.8416·26.0`,
  `p80 = mu_hat + 0.8416·26.0`
- **Q1 — direct per-tail fit:** `p20 = a₂₀ + b₂₀·mu_hat`, `p80 = a₈₀ + b₈₀·mu_hat`,
  each estimated by minimising its **own** pinball loss. The two tails get separate
  intercepts and separate slopes, so asymmetry is unconstrained.
- **Q2 — plus role:** adds `c·centrality` to each tail.

Estimator: pinball-loss minimisation via `scipy.optimize`. No regularisation tuning, no
feature search. These three arms are the entire candidate space; nothing may be added
after scoring begins.

## 5. Evaluation, frozen — per-player, never averaged

**Averaged MAE remains disallowed as a primary metric**: it is insensitive to dispersion
by construction, which is how the constant survived unexamined.

- **Headline:** fraction of players **well-calibrated** — the share of his holdout games
  inside his predicted p20–p80 is within **±0.10** of **0.60**.
- Full **distribution** of per-player coverage: median, IQR, deciles. Not a mean.
- Per-player **pinball loss at p20 and p80**, as the fraction of players improved vs Q0.
- **Floor/ceiling bias by mean tercile**, which must flatten. Q0's table on 2022 was
  floor −6.3 / +5.2 / +9.4 and ceiling +5.7 / +5.9 / −2.7.

## 6. Decision rule, frozen before scoring

**H1 is supported only if ALL FOUR hold on the 2017–2019 naive holdout:**

1. the fraction of well-calibrated players **increases** versus Q0;
2. per-player pinball loss improves for a **majority** of players at **p20 AND p80** —
   the rule the symmetric form failed;
3. in the **high-mean tercile**, `|floor_bias|` **decreases** AND `|ceiling_bias|` does
   **not increase**. This is deliberately stricter than last time, because the previous
   arm bought its floor improvement by wrecking the ceiling;
4. `p20_hat < p80_hat` on **every** row (no crossed quantiles).

**If any fails, H1 is NOT supported.** No arm re-specified, no subgroup searched, no
player carved out, no re-run with different windows. Q1 versus Q2 is decided by the same
rules; if Q1 suffices the simpler form wins.

## 7. What this study does NOT do

- does not change production; a merge requires separate explicit user authorisation;
- does not alter `mu` — the point projection is untouched in every arm;
- does not revisit the opponent side, settled as weak in
  `PRISM_OPPONENT_SIGNAL_NULLS_V1`;
- does not use the player's own measured SD, which is only weakly persistent
  (`r ≈ 0.25`);
- does not address rushing, receptions, or alignment collapse.
