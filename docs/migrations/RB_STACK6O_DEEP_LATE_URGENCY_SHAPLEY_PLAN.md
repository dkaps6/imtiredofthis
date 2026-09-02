# RB STACK6O — Deep-Late Urgency Shapley

## Purpose

STACK6N showed that a universal deep-late penalty contains real signal but is too blunt: it materially helps false-high P3 team-pool games while damaging false-low and ordinary games. STACK6O tests whether the `deep_late` state itself is too coarse.

This is an exact, no-fit attribution study only.

## Frozen lineage

- Parent: STACK6N `DEEP_LATE_HISTORY_NOT_RETAINABLE`
- Parent results commit: `be2b917c21402472e42b3f9e6c2387ef8e379ee2`
- M94C artifact: run `33353485070`
- STACK6H error-bin artifact: run `33632678179`
- No sportsbook inputs.

## Population

- 2025 regular season
- W6–18 primary evaluation
- Same 388 team-games as STACK6J/N

## Parent baseline

Continue the conditional occupancy scaffold used by STACK6J–N:

- actual target-game lead / neutral / trail play shares are grading-only oracle occupancy,
- parent M94C lead / neutral / trail rushing tendencies remain unchanged,
- only the `deep_late` subset of trailing plays is decomposed.

Frozen occupancy baseline MAE: `5.518381962346741`.

Frozen perfect deep-late-only MAE from STACK6N: `5.121110810459461`.

Frozen total deep-late headroom: `0.39727115188728046` attempts of MAE.

## Football-natural urgency cells

`deep_late` parent definition remains:

- score differential <= -9,
- Q4+.

It is partitioned exactly into four mutually exclusive cells using no outcome search:

1. `two_score_early_q4`
   - deficit 9–16 points,
   - more than 7:30 remaining in regulation.
2. `three_plus_early_q4`
   - deficit >=17 points,
   - more than 7:30 remaining.
3. `two_score_late_q4`
   - deficit 9–16 points,
   - 7:30 or less remaining.
4. `three_plus_late_q4`
   - deficit >=17 points,
   - 7:30 or less remaining.

The 7:30 split is the fixed midpoint of the fourth quarter, not a tuned threshold. A 17-point deficit is the fixed three-possession boundary.

## Oracle substitution

For every subset of the four urgency cells, replace only those cells' generic parent trailing-rate contribution with the actual target-game rush contribution:

`cell_rushes / actual_offensive_plays`.

All uncorrected deep-late cells retain:

`cell_play_share × parent_generic_trailing_rate`.

Lead, neutral, and non-deep-late trailing contributions are unchanged.

The full four-cell correction must reproduce the STACK6N perfect deep-late oracle exactly.

## Shapley attribution

Evaluate all 16 correction subsets and compute exact four-player Shapley values for MAE recovery. This makes attribution independent of correction order.

Populations:

- `ALL_W6_18`
- `POOL_OVER_5`
- `POOL_UNDER_5`
- `POOL_ABS_5`
- `NON_EXTREME_ABS_LT3`

## Required integrity

Before interpretation:

- four urgency shares must sum to target deep-late share within `1e-9`,
- empty subset MAE must equal `5.518381962346741` within `1e-9`,
- all-four subset MAE must equal `5.121110810459461` within `1e-9`,
- all-four recovery and Shapley sum must equal `0.39727115188728046` within `1e-9`,
- 388 W6–18 rows,
- no fit/search/sportsbook usage.

## Diagnostic disposition

This stack authorizes only the next research direction.

Call `URGENCY_CELL_DOMINANT` only if the same top overall cell satisfies:

- overall Shapley recovery >= `0.12` attempts,
- overall fraction of deep-late headroom >= `35%`,
- positive Shapley recovery in `POOL_OVER_5`,
- and its `POOL_OVER_5` fraction >= `40%`.

Otherwise disposition is `DEEP_LATE_URGENCY_DISTRIBUTED`.

Regardless of result:

- no production change,
- no player recomposition,
- no predictive model authorization.
