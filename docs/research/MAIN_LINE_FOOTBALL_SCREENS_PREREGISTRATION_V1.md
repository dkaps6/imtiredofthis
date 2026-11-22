# Do football screens pick the winning main-line bets within each category? PREREGISTRATION V1

**Date:** 2026-11-22, frozen before any screen below is scored. **Research only.**

## The question, in the operator's words

> "Sure the other ones might be negative ROI but that doesn't mean we lost every single bet each
> week in those other categories... how do we cherry pick the right players out of those other
> categories."

`SELECTIVE_PICK_IDENTIFICATION_RESULTS_V1.md` tried six statistical pickers:
- edge size;
- side;
- price;
- spread;
- the model's record on the player;
- a classifier.

**It did not try the football screens a handicapper uses,** the same family the alt-line gates
use:
- on the field;
- stable role;
- no teammate shock;
- game script;
- matchup.

This tests those screens on main lines, within each market and side.

## Data

- **2026:** FanDuel main lines, weeks 1, 2, 6-10, graded
  (`data/research/selective_pick/prism_2026_w1_2_6_10_mainline_graded_slim_v1.csv`), with PRISM
  `model_proj`.
- **2024-2025:** the full-stack benchmark (`non_qb_detail.csv`, ensemble `proj`, mostly
  DraftKings), plus pass_yards from the base archive (`mc_proj`).

## The pick and the screens (fixed, nothing fit)

**The pick:** the model's side, OVER if proj > line and UNDER if below. All features below are
strictly prior to the game week (nflverse weekly stats and snap counts, same season).

| screen | rule |
|---|---|
| **F1 field** | avg offensive snap share over his last 3 games ≥ 0.50 (QBs pass automatically) |
| **F2 available** | played (snap > 0) in each of his team's last 2 games |
| **F3 stable role** | last-3 share of team opportunity (carries for rushing, targets for receiving) within ±30% of his season-to-date share. Needs ≥ 4 prior games. |
| **F4 clean disagreement** | \|proj - line\| / line between 0.08 and 0.35 |
| **F5 script** | **rushing** UNDER: team not favored by 7+. **rushing** OVER: team favored by 3+. **Receiving** (rec_yards, receptions) OVER: team not favored by 10+. **Receiving** UNDER: team not an underdog by 7+. pass_yards OVER: not favored by 10+. pass_yards UNDER: not an underdog by 7+. |
| **F6 matchup** | the opponent's prior yards allowed per game to the position group, as a percentile in the same market and season. OVER needs ≥ 0.40 (not a top-40% defense). UNDER needs ≤ 0.60. |

- rush_rec_yards is excluded: it mixes two roles.
- **SCREENED** = all six pass.

## What is reported

For each market x side, **SCREENED vs every model pick in that cell**, and for each season:
- n;
- hit;
- ROI at the posted price;
- weeks positive.

## Pass / fail (per market x side cell)

- **HOLDS:** SCREENED ROI > 0 in 2026 **and** in the pooled 2024-25 data (both seasons
  individually > -2%), with n ≥ 40 in each.
- **IMPROVES BUT NOT PROFITABLE:** SCREENED ROI beats the cell's own unscreened ROI in both
  2026 and 2024-25, but at least one of them is ≤ 0.
- **NO EFFECT:** anything else.

Up to 8 cells are scored, so one false HOLDS by chance is plausible. A HOLDS cell is only bet
small and tracked.

## Prediction

- The screens raise hit rate by 1-3 points within most cells.
- They clear breakeven in at most one or two cells, most likely rushing UNDER.
- **Overs stay negative.**
- If nothing holds, the screens are a tool for avoiding bad bets, not for finding winners.
