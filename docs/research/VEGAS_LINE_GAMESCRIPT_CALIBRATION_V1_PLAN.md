# Vegas Line -> Actual Game Script Calibration V1 — Frozen Plan

**STATUS: FROZEN BEFORE ANY CANDIDATE OUTPUT. RESEARCH ONLY. NO PRODUCTION CHANGE.**

## Motivation

`docs/research/MARKET_IMPLIED_GAME_SCRIPT_V1_PLAN.md` (PR #558) tested a
narrower, conditional question: does the market's spread/total add
incremental predictive value for team plays/pass-rate *on top of* our own
historical-tendency baseline? It found no incremental value there.

That is a different question from the one asked here. This plan tests
Vegas's closing spread/total **on its own terms, unconditionally**: how
accurately does the market's own posted line describe what actually
happened in the game — the final combined score (scoring environment) and
the final margin (competitiveness/blowout-vs-close)? This has nothing to
do with our model, our baseline, or our player projections. It is a direct
calibration check of the market itself.

If Vegas's own line turns out to be well-calibrated on game script, that is
a distinct, useful fact even though #558 found no incremental lift over our
baseline — because #558 only asked "does it help beyond what we already
capture," not "is the market accurate in absolute terms." A well-calibrated
market signal could still be worth conditioning specific bet selection on
(a separate, later design question), independent of whether it improves a
regression fit against our own historical baseline.

## Design

**Data**: nflverse schedule (`nflreadpy.load_schedules`), regular season
only, 2023-2025, restricted to completed games (`home_score`/`away_score`
present). Per game, uses only what is posted pregame
(`spread_line`, `total_line`) plus the actual final outcome
(`home_score`, `away_score`). No player-level or team-week data at all —
this is intentionally the simplest possible pipeline, since there is no
baseline to build and no rolling history to compute (and therefore no
leakage risk from cross-season history windows, unlike PR #558).

**Sign convention** (same as PR #558, re-verified here independently):
nflverse's `spread_line` is POSITIVE when the home team is favored.
`predicted_margin_home = spread_line` (home team's expected margin, signed).
`actual_margin_home = home_score - away_score`.
`predicted_total = total_line`. `actual_total = home_score + away_score`.

**Three measurements, all unconditional (no fitting against our own model
anywhere in this plan)**:

1. **Direct calibration (no fit)**: pooled and per-season correlation, MAE,
   and bias (`mean(actual - predicted)`) of `total_line` vs `actual_total`
   and of `spread_line` vs `actual_margin_home`. This is the primary
   answer to "is the market's own number accurate."

2. **Out-of-sample linear recalibration (3-fold, one holdout season at a
   time)**: fit `actual ~ a + b * predicted` on the two non-held-out
   seasons, apply blind to the held-out season, for both total and margin.
   Reports fitted slope/intercept and whether recalibration improves MAE
   over using the raw line as-is. A slope near 1 and intercept near 0
   with negligible MAE improvement from recalibration means the raw line
   is already an unbiased, well-calibrated predictor; a slope far from 1
   means the market is systematically over/under-confident in some
   correctable, exploitable way. This is a genuine holdout (no cross-season
   feature leakage is possible here since there is no engineered history
   feature — each game's predictors and outcome are independent rows).

3. **Binned/operational game-script accuracy** (pooled across all 3
   seasons, no fitting): buckets by `abs(spread_line)` (0-3, 3-7, 7-10,
   10-14, 14+) reporting (a) mean actual `abs(actual_margin_home)` per
   bucket — does actual competitiveness increase monotonically with the
   posted spread magnitude; (b) favorite win rate per bucket — how often
   the team Vegas favored actually won outright; and a matching total-based
   table binning by `total_line` reporting mean `actual_total` per bucket.
   This directly answers "if the spread/total says X, how often did the
   game actually look like X."

## What this is not

- Not a claim about profitability, ROI, or betting edge — this measures
  whether the market's own posted numbers describe reality, not whether
  betting against/with them profits after vig.
- Not a repeat of PR #558 — that PR's baseline-vs-market comparison and
  this plan's unconditional market-accuracy question are independent; a
  negative result in one does not predict the result of the other.
- Not a production change. No modification to `rules_v2.py`,
  `simulation_rules.py`, pricing, or any live path — diagnostic script,
  tests, and this doc only.
- Not a claim that player-prop projections should condition on this
  result — if the market proves well-calibrated on game script, using that
  to prioritize which props to trust is a separate, later design question
  this plan does not attempt.

## Collaboration

Per standing practice this session, posting this plan to Issue #535 and
inviting GPT-5.6 to review before/alongside the result, same as PRs #557
and #558.
