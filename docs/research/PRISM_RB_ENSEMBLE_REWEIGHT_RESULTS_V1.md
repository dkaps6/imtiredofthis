# PRISM — RB `rush_yards` ensemble reweight (fix B): RESULTS (V1)

Preregistration: `PRISM_RB_ENSEMBLE_REWEIGHT_PREREGISTRATION_V1.md` (frozen `af63785b`).

## Headline

**K2 fired. Nothing is promoted.** The naive equal-weight control passed the gates too,
which per the frozen rules means the finding is "the incumbent weights are stale", not
"ML is right".

Two results matter more than the candidate comparison:

1. **The ensemble machinery structurally cannot express the fix.** NNLS, given recent
   data, asks to scale `mc_proj` up by **1.52×**. `_normalise_nonnegative` divides that
   away and returns `(1.0, 0, 0)` — keeping MC's −16 bias at full strength.
2. **Fixing the level buys zero decision improvement.** Pure ML lands the bias at
   **+0.26** (from −8.54) and side accuracy does not move at all: **42/79 → 42/79**.

## 1. Candidates on weeks 6–7 (`rush_yards`, fit n=89, test n=79)

| candidate | weights (mc/ml/state) | bias | MAE | side acc |
|---|---|---|---|---|
| **C0** incumbent | 0.557 / 0.443 / 0 | −8.54 | **23.66** | 53.2% |
| **C1** refit on wk1–2 | **1.0 / 0 / 0** | −15.53 | 24.55 | 51.9% |
| **C2** pure ML | 0 / 1 / 0 | **+0.26** | 24.34 | 53.2% |
| **C3** equal (control) | ⅓ / ⅓ / ⅓ | −6.42 | 23.91 | 53.2% |

Improvement in |bias| versus C0, paired bootstrap on identical rows (4,000 draws):

| candidate | |bias| gain | 95% CI | MAE Δ | side Δ |
|---|---|---|---|---|---|
| C1 | **−6.99** | [−8.65, −5.36] | +0.89 | −1.3pp |
| C2 | **+8.28** | **[−4.12, +10.31]** | +0.68 | **+0.0pp** |
| C3 | +2.11 | [+0.91, +3.07] | +0.25 | +0.0pp |

**C2 has the right level and fails G1 anyway**: its CI straddles zero at n=79. C3's
smaller shift is *more* statistically reliable than C2's larger one — and C3 passing
G1–G3 is exactly what **K2** is written to catch.

**K3 also fires for C2:** its bias flips sign between test weeks — week 6 **+4.1**,
week 7 **−3.3**. Unstable at this n.

**C1 is the result that indicts the method.** The *promoted* fitting procedure, applied to
the most recent data available, moved `rush_yards` to **100% MC** and made bias
*worse* (−15.53). Squared-error NNLS keeps choosing MC because MC has the best MAE of
the three despite the worst bias — good discrimination, wrong level. Refitting with the
existing methodology is not a fix; it is the mechanism that produced the problem.

## 2. The structural finding: the normalizer discards the correction

`fit_market_weights` runs `LinearRegression(fit_intercept=False, positive=True)` and then
`_normalise_nonnegative` divides the coefficients by their sum.

Raw NNLS coefficients on weeks 1–2: **[1.5165, 0, 0], sum = 1.5165.**

The fitter *detected the level defect and asked for a 52% scale-up on MC.* Normalizing to
sum 1 throws that away and yields `(1.0, 0, 0)`. `rush_rec_yards` shows the same thing:
raw **[1.1883, 0, 0.1474], sum = 1.3357**.

The preregistration recorded this as an *expectation*; it is now measured. Because the
blend is a strict convex combination with no intercept, **the ensemble has no mechanism to
correct a component's level drift except by reweighting toward a less-biased component.**
That is a code-level limitation, not a weights problem, and no choice of weights fixes it.

## 3. The finding that matters most for betting: the level is not the decision

C2 nails the level and changes nothing that would be acted on.

- C0 and C2 pick **different sides on 26 of 79 rows**;
- on those 26, C2 is right **13** times and C0 is right **13** times — a dead heat;
- hits are **42/79 for C0, C2 and C3 alike**. The identical figure is a genuine
  coincidence of a coin-flip split, not a bug; it was verified row by row.

C2 also picks OVER on **58/79** rows while actuals went over on only **37/79**. An
unbiased *mean* is not a calibrated *side*, because the line is close to efficient
(vegas MAE **22.78** still beats our best candidate's 23.66).

**On this evidence, correcting the RB rushing level does not improve a single bet
decision.** That is the honest answer to "will this make Sunday better", and it is
independent of which candidate is chosen.

## 4. `rush_rec_yards` (secondary, not a promotion candidate)

No incumbent weight row exists, so C0 is the uncalibrated 100%-MC fallback (bias −15.97).
C2 improves bias to −3.74, **+12.23 [+1.50, +15.24]** — significant — and improves MAE by
1.20. But side accuracy falls 41.5% → 39.6%, so **G2 blocks it**. Side accuracy is below
50% for every candidate on this market, which is its own problem and is not a weights
question. K3 also fires: week 6 **+3.1** → week 7 **−10.8**.

## 5. Disposition

- **nothing promoted.** `data/model_ensemble_weights.csv` is untouched;
- candidate weights written to `data/research/rb_ensemble_reweight/` as research only,
  following the `fit_2023_ensemble_weights_v1.py` precedent;
- **K2** fired (`rush_yards`), **K3** fired (C2, both markets), **G2** blocked
  (`rush_rec_yards`);
- the `_normalise_nonnegative` scale-discard is banked as a **code-level finding** and is
  the first thing to resolve before any future ensemble refit;
- the fit/test separation was honest but **not blind** — `ml_proj`'s +0.79 bias had been
  measured on the test weeks, as the preregistration disclosed up front.

## 6. What this leaves

Fix A (allocation) cannot move a level. Fix B (reweighting) cannot move the level either,
because the machinery divides the correction away — and when the level *is* forced right,
no decision changes. **Both candidate fixes from the parent diagnosis are now closed
negative.**

The finding still standing is upstream of both: the Bayesian layer over-shrinks RB rush
shares by ~9pp (`−0.0933, 95% CI [−0.1292, −0.0575]`, 15/16 teams, back-count controlled)
while PlayerForm's blended share is on target. That is a player-identity correction — how
much evidence does *this* back have — and it is the only live lead.

No sportsbook variable entered any weight or fit; lines were the benchmark in the
side-accuracy metric only. Betting remains **CLOSED**. No odds fetch performed.
