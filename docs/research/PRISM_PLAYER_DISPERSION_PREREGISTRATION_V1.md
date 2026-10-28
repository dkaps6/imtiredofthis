# PRISM — Player-specific dispersion: PREREGISTRATION (V1)

**Frozen before any scoring.** Committed before the model is fit or evaluated. If a
result below contradicts the hypothesis, that is the finding and it will be reported
as such.

## 0. Why

`scripts/pricing_v2.py:244` is `sigma = SIGMA[market]` — **26.0 for every receiver in
the league**, with a normal CDF, so every player's floor and ceiling are equidistant
from his mean and no player property enters dispersion at all. Its origin predates
available git history; it is an undocumented default, and 26.0 is close to the league
median realized SD (27.1), so it was most likely set as a league average and never
revisited.

Measured cost (`PRISM_PLAYER_DISPERSION_IS_A_CONSTANT_V1`): realized per-player SD
spans 10.0–53.8; 29 of 121 receivers are >30% under-dispersed; McLaurin (mean 58.2,
SD 19.0) and Wilson (mean 59.2, SD 51.5) are 1 yard of mean apart and priced
identically, with true floors of 46.6 and 15.2. The bias is systematic: production
over-disperses role players and **compresses the stars**.

This matters because floor and ceiling *are* the over/under. A dispersion error is a
pricing error even when the mean is perfect.

## 1. Hypothesis

**H1.** A dispersion that is a function of the player's own **projected mean** and
**role centrality** calibrates individual players' floors and ceilings better than a
per-market constant.

**H0.** The constant is adequate; player-specific dispersion does not improve
per-player quantile calibration out of sample. If H0 holds, the constant stays.

## 2. Why NOT the player's own measured volatility

Per-player volatility is real but **weak**: positive in 5 of 5 persistence tests at
`r(SD) = +0.15 … +0.57`, far below the mean's `r = +0.59 … +0.74`, and unstable between
years. Roughly a quarter of the observed spread is signal, so **a player's own measured
SD must not be used raw** — Wilson's 51.5 is mostly estimation noise on 15 games.

What is reliable: SD rises monotonically with the mean (21.7 / 28.3 / 35.9 by tercile)
while the mean is strongly persistent, and centrality predicts relative volatility
(`r = −0.656`) and floor ratio (`r = +0.620`). Both inputs are therefore taken from the
**persistent** part of player identity, not from his noisy realized spread.

## 3. Data, frozen

- **Fit:** 2020–2021.
- **Primary holdout:** **2022**, forward in time and **never inspected**. This is the
  only decisive evaluation.
- **Reported, NOT decisive:** 2023, 2024, 2025. Distributional summaries for these
  seasons have already been examined in prior work, so they are contaminated for a
  study whose outcome is those very quantiles. They are reported for era-stability only
  and **cannot** change the verdict.
- Source: `nflreadpy.load_player_stats`. Markets: WR/TE **receiving yards**.
- **No sportsbook variable at any stage.**

## 4. Eligibility, frozen

- player: ≥ **20** prior targets and ≥ **3** prior games with a target;
- week `W` ≥ **4**; player must have ≥ **1** target in week `W`;
- `sigma` floored at **5.0** yards so no arm can emit a degenerate distribution.

## 5. Anomaly gate (AGENTS.md, non-negotiable)

Every player-week is state-labelled **before** entering evidence: `DNP_INACTIVE`
(zero targets and zero snaps) excluded from both prior and outcome; `PARTIAL_GAME`
(snaps below half the player's own median) excluded from outcomes; `UNRESOLVED_CONTEXT`
fails closed. Counts reported; nothing silently dropped.

## 6. Models, frozen

`mu_hat` is **identical in every arm** — the player's own strictly-prior
`prior_ypt × prior_tgt_per_game` — so the only thing that varies is dispersion.
`centrality` is his strictly-prior share of team targets.

- **S0 — production:** `sigma = 26.0`
- **S1 — mean-scaled:** `sigma = a + b·mu_hat`
- **S2 — mean + role:** `sigma = a + b·mu_hat + c·centrality`

An **intercept is mandatory**: CV falls as the mean rises (0.79 / 0.70 / 0.58 by
tercile), so the relationship is sublinear and `sigma = k·mu` is already falsified.

**Estimator, frozen:** OLS of `|y − mu_hat|` on the arm's terms, then
`sigma_hat = fitted · sqrt(pi/2)` (the normal mean-absolute-deviation identity).
Absolute rather than squared residuals, for robustness to the long tail.

**Distribution family: normal in every arm**, so this study tests dispersion *only* and
never confounds it with shape. A lognormal variant of the winning arm is reported as a
preregistered secondary and **cannot** change the verdict.

No other features. No feature search. Nothing may be added after scoring begins.

## 7. Evaluation, frozen — per-player, never averaged

**Averaged MAE is explicitly disallowed as a primary metric.** It is mathematically
insensitive to dispersion: a model can have every player's mean right and every floor
and ceiling wrong and score identically to one that gets both right. That is precisely
how a single constant survived unexamined.

- **Headline:** the **fraction of players well-calibrated**, where a player is
  well-calibrated if the share of his holdout games falling inside his predicted
  central 60% interval (p20–p80) is within **±0.10** of 0.60.
- The **full distribution** of per-player coverage is reported — median, IQR, deciles —
  not a mean.
- Per-player **pinball loss at p20 and p80**, reported as the fraction of players
  improved versus S0.
- The systematic **floor/ceiling bias by mean tercile** must flatten; the current table
  is floor −4.3 / +0.6 / +6.1 and ceiling +7.8 / +2.4 / −4.7.

## 8. Decision rule, frozen before scoring

**H1 is supported only if ALL of these hold on the 2022 clean holdout:**

1. the fraction of well-calibrated players **increases** versus S0;
2. per-player pinball loss improves at **both** p20 and p80 for a **majority** of
   players;
3. the floor/ceiling bias by mean tercile **reduces in magnitude** — specifically the
   star-compression term (high-mean floor bias) moves toward zero;
4. every player receives a finite, positive `sigma` without relying on the floor for
   more than 5% of rows.

**If any fails, H1 is NOT supported.** No subgroup will be searched, no arm
re-specified, no player carved out, and the study will not be re-run with different
windows to find a winner. S1 versus S2 is decided by the same rules; if S1 suffices,
the simpler form wins.

## 9. What this study does NOT do

- does not change production; a merge requires separate explicit user authorization;
- does not alter `mu` — the point projection is untouched in every arm;
- does not address distribution **shape**/skew beyond the reported secondary;
- does not use the player's own measured SD, per section 2;
- does not touch the opponent side, which is settled as weak
  (`PRISM_OPPONENT_SIGNAL_NULLS_V1`);
- does not resolve alignment collapse or SLOT unreachability, which remain separate.
