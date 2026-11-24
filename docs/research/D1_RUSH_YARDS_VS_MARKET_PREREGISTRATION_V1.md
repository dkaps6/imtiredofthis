# D1 carry shares → RB rushing-yards projection vs the FanDuel line: PREREGISTRATION V1

**Date:** 2026-11-19, frozen before scoring. **Research only.**

## Question

D1 (all-block allocative prior + team budget 1.0) cuts RB expected-carries error by 19-21%.
**Does that survive into rushing YARDS**, the market being bet, once multiplied by team volume
and efficiency? And how does it compare with PRISM's projection and the book's line?

**Prior-work check:**
- The "objective S" in `PRISM_ROLE_STATE_FRONTIER_RESULTS_V2.md` used a reduced-form
  projector (share × team volume × efficiency) on **receiving** markets only.
- No study has scored **rushing** props with a share-model swap, and D1 is new.
- This test uses the same reduced-form logic.
- YPC follows the settled result that efficiency barely persists:
  `PRISM_ROLE_SHRINKAGE_FIX_RESULTS_V1.md` and `REVERSE_ENGINEER_2026_W1_10_V1.md` §5.

## Design

**Rows:** FanDuel RB `rush_yards` main-line offers, 2026 weeks 6-10, one row per player-week.
- The player must be in the pregame universe. The board name is mapped to a gsis id through
  that week's roster: normalized name + team.
- The player must have taken an offensive snap. Voids are dropped.

**Projection:** `share x team_rush_attempts x ypc`, all strictly prior.
- **share:** C0 (R0) or D1 (R1), from the frozen 2026-holdout tables.
- **team_rush_attempts:** the team's rolling mean of team carries over its previous 8 games,
  crossing seasons.
- **ypc:** 0.79 x league RB YPC + 0.21 x the player's own YPC. Both are computed from the 2025
  season plus 2026 weeks before the target week. A player with no carries uses the league
  value.
- **Only the share differs between R0 and R1.**

**Benchmarks:** PRISM `model_proj` and the FanDuel line, on the same rows.

## Metrics

- **Primary:** the MAE gain in rushing yards, R0 minus R1, with a 95% bootstrap CI clustered by
  game. **Pass** if the CI excludes 0.
- **Reported:**
  - MAE and bias of R0, R1, PRISM and the line;
  - side accuracy (projection vs line → over/under correct) for R0, R1 and PRISM;
  - the ROI of betting R1's side at the posted price. **Underpowered:** at n ≈ 250-300, the
    minimum detectable side difference is about 8pp, and it is stated with the result.

## Prediction

- R1 beats R0 on MAE, by less than the carries gain in relative terms, because YPC noise
  dilutes it.
- R1 closes part of PRISM's gap to the line.
- Side accuracy is not resolvable at this n.

**Named risk:** YPC and game-script noise swamp the carry gain, so the primary CI includes 0.
