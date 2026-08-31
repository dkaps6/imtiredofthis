# Migration 94 — RB Team Rushing Opportunity / Football-Only Game Script

## Why M94 exists

M91 established that the current RB model compresses true high-workload games. M92 then showed that, especially in 20+ carry games, team rushing-volume error is at least as important as backfield-share error. M93/M93B confirmed that backfield concentration is real but insufficient by itself.

M94 therefore isolates the next layer:

> How many total rushing opportunities should this offense have in this game before those opportunities are allocated to individual backs?

This remains football-only research. Sportsbook spreads, totals, moneylines, player props, and any other betting-market variables are excluded.

## Frozen inputs

M94 reuses the frozen M91 artifact rather than rebuilding the historical backtest. It consumes:

- 2024 and 2025 component predictions
- leakage-safe team-week history
- the historical schedule/opponent map
- recovered defensive box context when available

The M91 player projection, RB allocation, rushing efficiency, and receiving projection stay frozen except for the single M94 intervention: replacing the baseline team rushing total with a football-only candidate team rushing total and then preserving each player's existing share of that team total.

## Pregame feature families

Only information available before the target game is used. Rolling one-, three-, and five-game context is constructed for both the offense and its opponent, including currently available historical fields such as:

- recent offensive plays
- dropback rate / implied rush rate
- PROE
- neutral pace
- offensive and defensive success rates
- pressure allowed/generated
- defensive rush EPA
- defensive pass EPA
- explosive-play allowance
- pass-attempt conversion context
- defenders in the box
- light-box rate
- heavy-box rate
- opponent pace and offensive playcalling environment
- home/away
- the existing M91 projected team rushing total

The opponent's offense is included because possession count and tempo can affect how many offensive opportunities a team receives even before explicit score-state modeling is added.

## Temporal design

### Development

- 2024 Weeks 1-12: training
- 2024 Weeks 13-18: model-family holdout

Three deliberately limited model families are compared on the untouched 2024 holdout:

- ridge regression
- shallow gradient boosting
- constrained random forest

The winning family is selected only from 2024 holdout accuracy.

### Validation

The selected model family is refit using all 2024 target-season observations, then frozen and evaluated on the complete 2025 season. No 2025 result is used to choose the family or tune coefficients.

## Explicit game-script diagnostics

M94 also trains football-only diagnostic classifiers for:

- probability of a 30+ team-rush game
- probability of a 20-or-fewer team-rush game

These classifiers are diagnostic only in M94 and do not feed the rushing-volume candidate. Their AUC shows how much information the current pregame football context contains about high- and low-rush scripts without sportsbook input.

If these diagnostics are promising but the volume regressor remains weak, a later M94B may add richer historical score-state/playcalling observations (leading/trailing/neutral rush rates) to the cached historical foundation rather than repeatedly downloading them.

## RB translation

For every 2025 team-week:

1. preserve the M91 ML player's share of projected team rush attempts;
2. replace only the M91 projected team rush total with the M94 candidate total;
3. recompute RB carries;
4. hold M91 implied rushing efficiency fixed;
5. recompute rushing yards;
6. hold the receiving component fixed when recomputing rushing + receiving yards.

This cleanly measures the amount of RB error recoverable from better team rushing opportunity alone.

## Scoreboard

M94 reports team-volume performance for:

- all team-games
- 20 or fewer rushes
- 21-29 rushes
- 30+ rushes
- 35+ rushes

It also reports RB performance for:

- all RBs
- 0-5 carries
- 6-10 carries
- 11-14 carries
- 15+ carries
- 20+ carries
- 25+ carries
- 60%+ bell-cow games

The old all-player rushing-yard scoreboard remains a regression guard.

## Advance criteria

M94 is research-only and is not promoted directly to production. `ADVANCE_TEAM_VOLUME_SIGNAL` requires positive 2025 validation gains in:

- all-team rushing-volume MAE
- 30+ team-rush games
- all-RB rushing-attempt MAE
- all-RB rushing-yard MAE
- 20+ carry rushing-attempt MAE
- the legacy all-player rushing-yard guard

A failure does not mean game script is unimportant. It means the current cached football context is insufficient, in which case the next step is richer score-state/playcalling data rather than coefficient tuning.
