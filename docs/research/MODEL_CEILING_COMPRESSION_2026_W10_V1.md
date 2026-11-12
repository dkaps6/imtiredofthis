# The model compresses ceilings, and that is why the gates only ever emit near-certainties

> **CORRECTION 2026-11-13 00:05Z -- this document overstates its finding.** The
> gaps below compare a current projection against season aggregates that were
> not adjusted for opportunity. (1) The Jayden Daniels rushing rows are NOT
> compression: his carries fell from 5/7/9 to 4/2/4 after an elbow injury, and
> the model's 21.0 matches his last three games (10, 23, 16). (2) Nabers' 121
> and 165 came on 54 and 47 team pass attempts; per attempt at tonight's
> projected 29.8, his mean is about 70 against the model's 55.9, so the gap is
> roughly -14, not -25. The direction of the finding stands for Nabers and
> Likely; its size does not. See the addendum in `BLENDED_LEG_ESTIMATOR_V1.md`.

**Board:** 2026 week 10 TNF (NYG @ WAS), run 43327080728, real `model_proj`/`model_sd`
**Evidence:** `data/research/tnf_2026_wk10_gated_legs.csv`, `data/market_track_record/boards/2026_wk10.csv`

## The complaint that triggered this

The nine-gate builder, run honestly on a priced board, emitted two legs out of
503 candidate rungs, and both were trivialities -- a quarterback over 124 passing
yards, a tight end over 9.5 receiving. The operator's objection was that this is
not a model forming an opinion, and that HIGH RISK / LOTTERY tiers are supposed
to represent plausible-but-unlikely football, not to be certified as +EV.

Both parts of that objection are correct, and the second one exposed the first.

## Finding 1 -- means are low for high-ceiling players

`model_proj` against the player's own 2026 production, same week, same market:

| player | market | actual mean | model_proj | gap |
|---|---|---|---|---|
| Malik Nabers | rec_yards | 81.0 | 55.9 | **-25.1** |
| Cam Skattebo | rush_yards | 69.7 | 47.3 | **-22.4** |
| Cam Skattebo | rush_rec_yards | 82.3 | 60.6 | -21.7 |
| Jayden Daniels | rush_yards | 42.0 | 21.0 | **-21.0** |
| Isaiah Likely | rec_yards | 56.0 | 43.7 | -12.3 |

Low-ceiling, stable-role players are fine: Diggs +1.2, Mooney -0.8, Williams
-1.3, Fields -2.9. The error is concentrated in exactly the players whose
distribution has a long right tail.

## Finding 2 -- the tails are about half as wide as reality

| player | market | own p90 | model p90 | gap | own sd | model_sd |
|---|---|---|---|---|---|---|
| Jayden Daniels | rush_yards | 86.0 | 38.6 | **-47.4** | 36.4 | 13.1 |
| Malik Nabers | rec_yards | 134.2 | 90.7 | **-43.5** | 55.5 | 25.9 |
| Cam Skattebo | rush_yards | 102.5 | 74.8 | -27.7 | 30.2 | 20.6 |
| Isaiah Likely | rec_yards | 95.5 | 73.7 | -21.8 | 34.2 | 22.3 |

Nabers has recorded 112, 121, 165 and 111 receiving yards in eight games. The
model's 90th percentile for him is 90.7.

`mc_dispersion_state` on this board reads `candidate_not_promoted` with
`mc_dispersion_k = 1.0`, i.e. no dispersion correction is applied in production.

## Why this makes the gates emit only trivia

G3 requires `model_proj / threshold >= 1.50`. With a compressed mean AND a
compressed tail:

* every rung deep enough to pass G3 is one the thin distribution says is
  near-certain, so it prices at -850 to -2400;
* every rung with real payout sits in the tail the model has collapsed, so it
  fails G3 or reads as a 6% shot;
* therefore the survivor set is structurally "heavy favourites only", which the
  builder's docstring recorded as a property of the market rather than of the
  model.

The model says Nabers over 99.5 is a **6.5%** shot. The market says **24.4%**.
His own game log says **4 of 8**. The model is the outlier, and it is the
component we control.

## Distribution validation

Gamma matched to the published (`model_proj`, `model_sd`) reproduces the model's
own `fair_prob` at the main line to a mean absolute gap of **0.027** (max 0.063)
across 23 player/markets, so gamma is a faithful stand-in for the simulator's
shape and the tail numbers above are the model's own, not an artefact of the fit.

An earlier attempt to calibrate a lognormal to the mean plus `fair_prob` was
discarded: with a fixed mean it needs absurd dispersion to match a low
over-probability at a line above the mean (implied sd 90.0 against `model_sd`
20.6 for Skattebo), which is a property of that parameterisation, not of the
model.

## Finding 3 -- same-game legs are strongly correlated, and independence understates tickets

Diggs over 29.5 and Fields over 14.5 both cleared in **6 of 6** games they both
played. The independent product of their shrunk probabilities is 48.8%.

The empirical joint rate is small-sample and must not be read as a probability,
but the direction is not in doubt, and it means a coherent same-game narrative
ticket is more likely than the product of its legs. A book's SGP quote claws
some of this back; the naive product is not a price anyone can take.

## Consequence for the shrinkage constant

The measured PRISM shrinkage toward market (0.159) was fit where the model is
**over**-confident (picked side claimed 63.8%, delivered 51.7%). Applied in the
tail it moves probabilities the wrong way: Nabers over 99.5 goes from the
model's 6.5% to 21.5% purely by being dragged to the market price. That is not
calibration, it is adopting the market's number and relabelling it.

**A single global shrinkage constant is not valid across the probability range.**
Tail legs must not be priced through it. Until that is fit per band, the
defensible estimate for a tail leg is the player's own rate with an explicit
small-sample penalty, stated as such.

## What to do

1. Do not promote the mc dispersion candidate blindly -- it addresses width, but
   the mean error on ceiling players is separate and larger.
2. Investigate why ceiling means are low. Candidates: strict-prior windows that
   lean on 2025 for a 2026 breakout, shrinkage toward position priors that
   punishes outliers, opponent adjustment overpowering current role.
3. Fit the shrinkage per probability band before any tail leg is priced through it.
4. Treat G3 as a filter only. (Done: see
   `build_alt_cards_v1.leg_value` and `MODEL_CEILING_COMPRESSION` above.)
5. Nabers is the prospective test. The model says 6.5% over 99.5; his log says
   50%; the market says 24.4%. Grade it.
