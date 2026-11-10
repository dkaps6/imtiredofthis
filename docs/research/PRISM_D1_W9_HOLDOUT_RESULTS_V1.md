# D1 injury overlay — W9 holdout result

**Date:** 2026-11-10 · **Rows:** `data/research/d1_w9_holdout_graded_v1.csv`
**Script:** `scripts/research/capture_prospective_board_v2.py` (`--include-played`)

**VERDICT: D1 is a correctness fix, promoted on that basis. It is NOT an edge
claim — its measured betting impact on this slate is indistinguishable from
noise, and that is stated plainly below.**

## The holdout did not need to wait

D1 was recorded as pending for three weeks on the reasoning that it needed a
fresh prospective capture after Friday designations. That was wrong, and the
user said so.

A valid holdout needs three things and all three already existed:

| ingredient | source |
|---|---|
| universe built from weeks ≤ 8 only | reconstructed, `assert cur.wk.max() <= prior_week` |
| retention calibrated on weeks ≤ 8 only | `calibrate_retention(..., prior_week=8)` |
| real week 9 game designations | nflverse injury feed, 154 rows, 48 in universe |
| week 9 actuals | nflverse weekly stats |

Calibrating on ≤8 and testing on 9 is out-of-sample. The original study's defect
was fitting *and* testing on W7+W8, not the absence of a frozen file.
"Prospective capture" and "valid holdout" are not the same thing, and conflating
them cost three weeks.

## Grading population is the whole story

Graded two ways, and they answer different questions:

- **Played-only** (n=314, snap denominator): 12 designated players survive, **all
  QUESTIONABLE**. Every OUT and DOUBTFUL player is excluded *because they did not
  play*. D1's primary effect is therefore structurally invisible. **K4 fires on
  this population** (<20 designated) and the result is inconclusive by design.
- **Full universe** (n=478, non-producers at actual 0): 48 designated, all three
  tiers present. This is the correct population for a projection claim.

## Full-universe accuracy

| market | arm | MAE | RMSE | bias |
|---|---|---|---|---|
| receiving | base | 13.391 | 20.957 | +1.282 |
| | d1 | 12.603 | **20.048** | +1.328 |
| | d3 | 12.803 | 20.896 | +0.518 |
| | **d1d3** | **11.929** | **19.975** | **+0.505** |
| rushing | base | 6.734 | 12.626 | +1.111 |
| | d1 | 6.385 | **12.232** | +1.008 |
| | d3 | 5.940 | 12.631 | +0.159 |
| | **d1d3** | **5.564** | 12.234 | **+0.031** |

**D1 improves RMSE, which D3 alone never did** (`PRISM_D3_W9_HOLDOUT_RESULTS_V1`:
RMSE flat at 22.846 vs 22.841). D3 wins MAE by shrinking the middle; D1 fixes
the tails. `d1d3` drives rushing bias to +0.031.

## The mechanism, which is not subtle

| OUT (n=24), actual 0.0 | rec | rush |
|---|---|---|
| base | **17.74** | **8.55** |
| d1 | 2.40 | 1.36 |
| **d1d3** | **0.00** | **0.00** |

DOUBTFUL (n=3), actual 0.0: base projects 15.53 / 13.31.

**Production projects 17.7 receiving yards for players publicly ruled OUT before
kickoff.** That is not a modelling judgement. It is a published report the model
does not read. 228 of 478 players recorded zero yards; base projects them a mean
10.91 receiving and 5.45 rushing.

QUESTIONABLE (n=21, actual rec 17.1 / rush 2.0): base 18.60 / 8.23, d1d3
17.55 / 4.57. MAE on rushing falls 6.28 → 3.68.

## The mandated metric — and the honest negative

| market | n | base | d1 | d3 | d1d3 |
|---|---|---|---|---|---|
| rec_yards | 186 | 54.8% | 55.9% | 54.8% | **58.1%** |
| rush_yards | 81 | 51.9% | 51.9% | 50.6% | 51.9% |

That 58.1% looks like a result. **It is not, and two facts kill it:**

1. `d1d3` vs base produced **18 disagreements, won 12**. Binomial **p = 0.238**.
   Not significant.
2. **Only 2 of those 18 disagreements involve a designated player.** The other 16
   are undesignated players flipping because the residual-mass change perturbs
   the multinomial draw — the 0.14-target drift documented in the
   preregistration's restated G2. **D1 is not causing most of its own
   side-accuracy gain.**

Only **8 of 186** priced receiving rows and **3 of 81** rushing rows involve a
designated player at all, because books pull lines for players who are OUT. The
population where D1 acts barely intersects the population that can be bet.

**K1 therefore fires for the betting claim.** D1 does not earn promotion as an
edge.

## What is promoted, and what is not

**PROMOTE — zeroing OUT and DOUBTFUL opportunity.** Not as an edge, as a
correctness repair. A model that assigns 0.1392 mean target share and 17.7
projected yards to a player the league has ruled out is wrong in a way that
needs no statistical test, and it contaminates team opportunity mass, every
aggregate, and the ceiling-compression diagnostics. Retention for both tiers is
0.0000 on 387 prior cases with 0.0% played.

**DO NOT PROMOTE — the QUESTIONABLE retention multiplier (0.8535 targets /
0.5718 rush).** n=21 in this holdout, its side-accuracy contribution is 2 of 18
disagreements, and the designation is used inconsistently across clubs — the
named risk in the preregistration. It needs more weeks.

**D3 remains rejected** on its own holdout; it is carried here only as a control.
Note `d1d3` beats `d1` on both markets, so the joint arm is worth keeping under
test even though D3 alone fails.

## Follow-ups

1. Wire OUT/DOUBTFUL zeroing into the production opportunity path under its own
   change, with a test that a player with an OUT designation receives zero
   modeled opportunity.
2. Re-test the QUESTIONABLE multiplier on W10 and W11 with the capture already
   built. K4 (<20 designated) is the binding constraint, so accumulate weeks
   rather than re-run one.
3. The 16-of-18 undesignated disagreement finding means **simulation noise moves
   more side calls than this arm does**. That is its own result and bears on
   every arm graded this way: a single-week side-accuracy comparison on
   re-simulated universes has a noise floor wide enough to swallow a real effect.
