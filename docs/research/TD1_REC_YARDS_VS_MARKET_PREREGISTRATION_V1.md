# TD1 target shares → receiving yards and receptions vs the FanDuel line: PREREGISTRATION V1

**Date:** 2026-11-19, frozen before scoring. **Research only.**

## Why

- TD1 cut WR+TE **target** error by 7-8% and removed the WR1 under-projection
  (`TARGET_ALLOCATIVE_PRIOR_BUDGET_RESULTS_V1.md`).
- `PRISM_JOINT_D1_D2_D3_RESULTS_V1.md` showed a share fix can **worsen yards**, because
  yards-per-target differs by position.
- `PRISM_ALPHA_COMPRESSION_BOARD_CHECK_V1.md` §3 showed that correcting alpha compression with a
  fitted multiplier made sides worse.
- The rushing analogue (`D1_RUSH_YARDS_VS_MARKET_RESULTS_V1.md`) was null on yards.

So the share gain has to be checked on the priced quantities.

**Prior-work check:**
- The role-state frontier V2 S objective scored rec_yards + receptions sides for the λ
  role-rank anchor. It used a reduced-form projector, with no pregame universe and no budget
  stage.
- TD1 has not been scored on yards or sides.

## Design

**Rows:** FanDuel `rec_yards` and `receptions` main-line offers, 2026 weeks 6-10, for WR, TE
and RB.
- The player must be in the pregame universe and have played (voids dropped).
- The board name maps to gsis through that week's roster (normalized name + team), as in the
  rushing test.

**Projection:** all strictly prior.
- **Expected targets:** `share x team_targets_hat`.
  - **share:** C0 or TD1, from the frozen 2026-holdout tables.
  - **team_targets_hat:** the team's rolling mean of team targets over its previous 8 games,
    crossing seasons.
- **rec_yards:** expected targets x YPT. YPT = 0.73 x the league YPT for his position + 0.27 x
  his own YPT, from 2025 plus 2026 weeks before the target week. 0.27 is the measured
  week-to-week stability of yards per target.
- **receptions:** expected targets x catch rate. Catch rate = 0.80 x position league + 0.20 x
  own; catch rate barely persists (r about 0.2).
- **Only the share differs between the two arms.**

## Metrics

- **Primary:** the `rec_yards` MAE gain, C0 minus TD1, with a game-clustered 95% CI. **Pass** if
  the CI excludes 0.
- **Reported:**
  - the receptions MAE gain;
  - bias and median residual;
  - the WR1 / alpha subset;
  - side accuracy and ROI on the projection's side, with the MDE;
  - PRISM `model_proj` and the line as benchmarks.

## Prediction

- TD1 improves rec_yards MAE by a little, likely not significantly at n of a few hundred.
- Its alpha projections rise toward the line.
- Side accuracy is not resolvable.

**Named risk:** as with D2, moving targets toward WRs raises yards where receiving yards are
already high, so MAE gets worse.
