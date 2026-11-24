# FanDuel alt ladders are generic, but player steadiness does not beat their margin (exploratory, weeks 9-10)

**Date:** 2026-11-20.

**Status:** EXPLORATORY. Two weeks, not preregistered. Nothing here is a rule.

**Rows:** `data/research/alt_ladder/fanduel_alt_rungs_wk9_10_with_volatility_v1.csv`.

## Prior-work check

- `PRISM_ALT_PRICING_VALIDATION_2026_W9.md`: the model's `model_sd` is too narrow, and no alt
  profitability test existed before week 9.
- `PRISM_W10_COMPLETE_GRADING_V1.md` §3: every cushion band lost in week 10.
- **No study had tested whether FanDuel's ladder is player-specific,** or whether player
  steadiness creates a gap.

## Result

**1. The ladder is generic.**
- Market plus rung/main-line ratio alone explains **95.3%** of FanDuel's alt price variation
  (5,444 rungs, 261 players).
- Every player gets essentially the same curve around his main line.
- **The operator's hypothesis is right on the structure.**

**2. Player steadiness did not produce a beatable gap in weeks 9-10.** Hit minus implied
probability, FanDuel alt OVER rungs:

| zone | steadiness measure | steady / secure | mid | volatile / unstable |
|---|---|---|---|---|
| floors (< 0.85x line) | yards CV | -5.3pp (ROI -6.9%) | -4.2pp | -7.2pp |
| floors | role CV (targets/carries/attempts) | **-9.7pp (ROI -12.4%)** | -2.1pp | -4.4pp |
| ceilings (> 1.15x line) | yards CV | -1.7pp | -2.4pp | -2.6pp |
| ceilings | role CV | -3.4pp | -2.4pp | -1.0pp |

**Every cell loses.** The margin built into each rung is larger than the player differences
these measures capture. Secure-role floors were the worst cell. These are mostly stars with high
lines, and that may be week-specific; at n of about 550 over two weeks it is not interpretable.

## What it means

- FanDuel prices alt ladders off the main line with one curve.
- **A player-specific distribution is therefore the right place for a model to look.** But
  neither simple steadiness measure is enough to beat the margin.
- Alt OVER rungs also sit on the side of the market that loses every week (main-line overs lost
  7 of 7 weeks).
- Our one validated edge is an **under**, and FanDuel's alt ladders are overs only.
- **Retest only as a frozen preregistration on weeks not yet graded.** Weeks 9-10 are used up.
