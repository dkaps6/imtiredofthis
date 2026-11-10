# D3 receiving/rushing overlay — W9 prospective holdout result

**Date:** 2026-11-10 · **Rows:** `data/research/d3_w9_holdout_graded_v1.csv`
**Holdout:** `data/market_track_record/w9_prospective_frozen_v1.csv`, captured
**2026-11-04T00:41:10Z**, before any week 9 game. 478 players, 30 teams, reading
player stats no later than week 8. Arms frozen: `baseline`, `D3`. `D1 overlay`
was listed pending at capture and remains untested.

**VERDICT: D3 IS NOT PROMOTED.**

This closes the item carried since V22: *"`D1--D3` for receiving is ready and
awaits the W9 holdout."* The holdout was clean, the test ran, the answer is no.

## Graded population

314 players present in the frozen board who also recorded week 9 offensive snaps
(`load_snap_counts`, `offense_snaps > 0`). Players who did not play are excluded;
players who played and recorded nothing count as real zeros.

## Accuracy — D3 wins, for the wrong reason

| market | arm | MAE | RMSE | bias | mean proj | mean actual |
|---|---|---|---|---|---|---|
| receiving | baseline | 14.035 | 22.841 | -3.308 | 17.35 | 20.66 |
| receiving | **D3** | **13.604** | 22.846 | **-3.984** | 16.67 | 20.66 |
| rushing | baseline | 6.197 | 12.388 | -1.362 | 7.88 | 9.25 |
| rushing | **D3** | **5.424** | 12.414 | **-2.346** | 6.90 | 9.25 |

Per-player, D3 is closer on 63.5% of receiving players and 82.0% of rushing
players.

That looks like a win and is not one:

- **RMSE is unchanged** (22.846 vs 22.841; 12.414 vs 12.388 — marginally worse
  in both). D3 improves the typical case and does nothing for the tails.
- **Bias worsens in both markets.** Both arms already under-project
  (actual 20.66 vs 17.35 receiving), and D3 projects *lower still*. It wins MAE
  by shrinking toward zero across a population dominated by low-production
  players, on a slate where the realized mean was above both arms.
- This is the same high-end ceiling compression measured prospectively in
  `PRISM_ALT_CEILING_AND_COVERAGE_2026_W9.md` (TOP-quartile bias -12.5 receiving,
  -7.3 rushing). D3 moves in the wrong direction on it.

## The mandated metric — side accuracy against the line

The handoff forbids averaged MAE as the primary metric for an individual-player
distribution claim and requires side accuracy against the line for a decision
claim. Using week 9 board lines (`boards/2026_wk09.csv`):

| market | baseline | D3 | n | disagreements |
|---|---|---|---|---|
| rec_yards | 55.3% | 55.9% | 170 | **5** (D3 3/5, baseline 2/5) |
| rush_yards | **52.7%** | **52.7%** | 74 | **0** |

Breakeven at -110 is 52.4%.

**In rushing, D3 does not flip a single side call.** In receiving it flips five
of 170, and 3-of-5 is indistinguishable from noise. On the metric that governs
promotion, the two arms are the same model.

## Why this is the right call despite the MAE gain

A 0.43-yard receiving and 0.77-yard rushing MAE improvement that (a) leaves RMSE
flat, (b) worsens an already-negative bias, and (c) changes zero rushing and five
receiving decisions is a shrinkage artifact, not a football improvement. Promoting
it would deepen the ceiling-compression defect that is currently the largest known
source of missed high-usage plays, in exchange for a metric the repository
explicitly disallows as a promotion basis.

## Status of the sibling arms

- **D1 overlay:** never testable. It depends on the `injuries.csv` schema repair
  (`build_injuries_weekly` must emit its declared 11 columns, and
  `audit_2026_production_readiness.py` must assert it so a file without `status`
  fails closed). That repair has not landed, so D1 did nothing at capture and was
  frozen as pending. **D1 remains untested, not rejected.**
- **D2:** not in the frozen arms.

## What a future D-arm must show

Not a lower MAE. On a clean prospective holdout, with the snap-count denominator:

1. side accuracy against the line that beats baseline by more than the
   disagreement count can explain, and
2. no worsening of the bias term, and
3. improvement at the top usage quartile specifically, where the current model
   is measurably compressed.
