# RB Rush + Receiving Conservation V2 — Frozen Mean Test

Date: 2026-09-24

Status: FROZEN BEFORE HISTORICAL OUTCOME SCORING

## Why this exists

The 2026 Week-2 production board exposes a football-construction inconsistency outside the Week-1 P3 path:

For RBs with all three production markets present,

`rush_rec_yards` should describe the same realized quantity as:

`rush_yards + rec_yards`.

Week 1's promoted P3 conservation path enforces that identity. Outside Week 1, the current generic calibrated ensemble allows the three market means to move independently.

Pregame Week-2 structural audit, performed without using outcomes:
- comparable RBs: 33
- exact/algebraically conserved within 1e-6: 1
- combo below component sum: 32
- mean combo-minus-components gap: -9.9631 yd
- mean absolute gap: 9.9631 yd
- max absolute gap: 29.2915 yd

This plan does **not** use the Week-2 realized outcomes to define the candidate. The candidate is fixed by the outcome identity itself.

## Question

Historically, does the algebraically coherent mean

`candidate_rush_rec = production_rush_yards_mean + production_rec_yards_mean`

predict actual RB rush+receiving yards better than the independently calibrated production `rush_rec_yards` mean?

## Authority

Use the exact preserved PR #549 production-order historical trace:
- run: `34722725629`
- artifact: `10307242156`
- source SHA: `f04a8a775f4a56fe282cb292f6a52bd509bc8f24`
- trace: `full_stack_projection_trace_base.csv`

No live provider rebuild.

## Cohort

- position = RB
- markets required for same player-game:
  - `rush_yards`
  - `rec_yards`
  - `rush_rec_yards`
- seasons 2020-2025
- exact identity key:
  `season, week, team, opponent, player_clean_key, game_id`

Only player-games with finite `ensemble_proj` and `actual` for all three markets qualify.

## Integrity gates

Before scoring:
1. each player-game-market key unique;
2. actual rush+receiving identity must hold:
   `actual_rush_rec == actual_rush + actual_rec` within 1e-9;
3. no sportsbook lines/odds used;
4. no 2026 outcomes used;
5. no fitted coefficient, blend, clipping rule, threshold search, or subgroup router.

If actual identity does not hold, fail mechanically and diagnose before science.

## Arms

Baseline:
- frozen PR #549 `ensemble_proj` for `rush_rec_yards`.

Candidate:
- frozen PR #549 `ensemble_proj(rush_yards) + ensemble_proj(rec_yards)`.

There are no fitted parameters.

## Metrics

Report pooled and per-season:
- n
- MAE
- RMSE
- signed bias
- p90 absolute error
- count of absolute errors >=30 yd
- candidate-minus-baseline absolute error paired mean
- share of player-games candidate closer than baseline
- baseline combo-minus-component projection gap distribution.

## Promotion gate for a subsequent production-integration test

Candidate must satisfy all:
1. pooled 2020-2025 MAE strictly improves;
2. 2024 MAE strictly improves;
3. 2025 MAE strictly improves;
4. pooled RMSE non-worse;
5. pooled absolute bias non-worse;
6. pooled p90 absolute error non-worse;
7. pooled 30+ yard misses non-increase;
8. candidate closer rate > 50%;
9. no season has MAE degradation >0.50 yd.

If all pass:
`RB_RUSH_REC_CONSERVATION_V2_MEAN_QUALIFIED`

If not:
`RB_RUSH_REC_CONSERVATION_V2_MEAN_FAIL`

A mean PASS does not itself change production. It authorizes one separately frozen draw-level integration study that must preserve:
- standalone rush-yard final mean;
- standalone rec-yard final mean;
- pathwise `rush_rec = rush + rec`;
- current downstream sportsbook separation.

## Prohibited rescue

After result:
- no weighted sum search;
- no coefficient on receiving yards;
- no player/depth/volume router;
- no Week-2-outcome-derived exception;
- no M96 reopening;
- no sportsbook feature.
