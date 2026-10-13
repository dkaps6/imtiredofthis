# PLAYER EFFICIENCY TAIL STABILITY V1 — FROZEN HISTORICAL CONTRACT

Date frozen: 2026-10-13

Status: **FROZEN BEFORE SCORING. RESEARCH-ONLY.**

Parent:
- `docs/research/PLAYER_EFFICIENCY_TAIL_STABILITY_V1_ANTIRETEST_AUDIT.md`

## Question

Does strictly-prior individual per-opportunity efficiency volatility predict
next-game absolute efficiency surprise?

## Data / chronology

Target seasons:
- 2023
- 2024
- 2025

Target weeks:
- 5 through 18

Source:
- nflverse / nflreadpy weekly player statistics;
- nflverse schedule authority for game identity;
- canonical repository stable historical player identity.

No 2026 outcomes are read.

For every target row, every feature game must have week < target week in the
same season. Cross-season carryover is not used in V1. Team changes are allowed
because the hypothesis is explicitly player-specific; stable player identity is
the grouping authority.

## Frozen cells and event support

1. `RB_RUSH_YPC`
   - positions: RB/HB/FB
   - efficiency: rushing_yards / rushing_attempts
   - eligible game requires >=4 rushing attempts

2. `WR_TARGET_YPT`
   - positions: WR/LWR/RWR/SWR
   - efficiency: receiving_yards / targets
   - eligible game requires >=3 targets

3. `TE_TARGET_YPT`
   - position: TE
   - efficiency: receiving_yards / targets
   - eligible game requires >=3 targets

4. `RB_TARGET_YPT`
   - positions: RB/HB/FB
   - efficiency: receiving_yards / targets
   - eligible game requires >=2 targets

History:
- last 8 eligible strictly-prior games;
- minimum 4 prior eligible games.

No support floor may be changed after results are visible.

## Frozen feature

Primary advancing feature:
`prior_efficiency_sd8`

Descriptive only:
`prior_efficiency_mad8`

Also recorded as controls:
- prior efficiency mean;
- prior opportunity mean.

## Frozen target

`efficiency_surprise = abs(target_game_efficiency - prior_efficiency_mean)`

Target-game opportunity count is used only to determine whether the target
efficiency denominator clears the frozen cell support floor. It is not a
predictor.

## Primary statistics

For each cell and season independently:

1. raw Spearman correlation:
   `prior_efficiency_sd8` vs `efficiency_surprise`;

2. partial Spearman correlation after rank-residualizing both variables against:
   - prior efficiency mean;
   - prior opportunity mean.

Clustered bootstrap:
- 1,000 reps;
- game-cluster bootstrap;
- player-cluster bootstrap;
- frozen deterministic seeds.

Support gate per cell-season:
- >=250 rows;
- >=60 distinct players;
- >=50 distinct games.

Scientific support for a season requires:
- raw rho > 0;
- raw game-cluster 95% CI lower bound > 0;
- raw player-cluster 95% CI lower bound > 0;
- partial rho > 0;
- partial game-cluster 95% CI lower bound > 0;
- partial player-cluster 95% CI lower bound > 0.

A cell is confirmed only if the same frozen rule passes independently in ALL
three seasons 2023, 2024, and 2025.

No majority-season rescue.

## Integrity gates

Must all pass:
- exact target seasons only;
- target weeks remain 5-18;
- zero feature chronology violations;
- stable player identity coverage >=95%;
- no target-game statistic is used as a feature;
- sportsbook inputs = 0;
- OddsAPI calls = 0;
- model residuals = 0;
- fitted predictive models = 0;
- candidate mean changes = 0;
- production changes = 0.

## Interpretation

PASS means only that player-specific efficiency volatility is a replicated
uncertainty signal and may advance to a separately frozen V2 that removes
finite-opportunity sampling noise and evaluates a mean-identical distribution
with proper scores.

FAIL means the exact volatility family closes for that cell. No retune or
subgroup rescue.
