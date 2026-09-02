# RB STACK6K — Within-State Rushing-Tendency Oracle

## Status

Frozen no-fit diagnostic. No production change. No fitted predictive candidate is authorized in STACK6K.

## Why STACK6K exists

Current lineage has localized the P3 team-RB-pool problem in stages:

1. STACK6H: total team rushing dominates RB-share error.
2. STACK6I: effective rushing rate dominates offensive-play-volume error.
3. STACK6J: correct lead/neutral/trail occupancy recovers only `0.6850728182` total-rush MAE, or `21.14%` of STACK6I's `3.2406943942` perfect-rate headroom.

STACK6J disposition was `STATE_OCCUPANCY_PARTIAL`. The remaining question is whether the larger portion of rush-rate error comes from **rushing tendency / play selection conditional on the realized game states**.

STACK6K answers that question without fitting a new model.

## Frozen lineage

M94C:
- run `33353485070`
- SHA `7e85dd0e836982fdfbea080ea69c0149d2e186e3`
- artifact ID `9744399103`

STACK6H current P3 bins:
- run `33632678179`
- artifact ID `9847470353`

STACK6I:
- run `33637785836`
- artifact ID `9849508420`
- perfect effective-rush-rate recovery `3.2406943942`

STACK6J:
- run `33638294183`
- job `100274619508`
- SHA `660c489bd3545d086e0de2f8d359b206d447ab3f`
- artifact ID `9849711171`
- occupancy recovery `0.6850728182`
- disposition `STATE_OCCUPANCY_PARTIAL`

## Population

- 2025 regular season
- Weeks 6-18
- same 388 team-games as STACK6I/J
- P3 error bins from STACK6H
- no sportsbook selection

## Truth bridge

M94C's canonical target is weekly-stat `actual_team_rush_att`, while its game-state labels come from PBP.

The M94C artifact also contains:
- `actual_off_plays`
- `actual_rush_att_pbp`

Define realized PBP rushing rate:

`Q_pbp = actual_rush_att_pbp / actual_off_plays`

Before interpreting the oracle, report the bridge between `actual_rush_att_pbp` and weekly `actual_team_rush_att` on W6-18:
- MAE
- RMSE
- bias
- correlation
- exact-match rate
- rate with absolute difference > 1
- rate with absolute difference > 2

Frozen bridge integrity requirement:
- MAE <= `0.50` rush attempt;
- correlation >= `0.98`;
- fraction with absolute difference > 2 <= `0.05`.

Failure means `STACK6K_TRUTH_BRIDGE_FAILURE_DO_NOT_INTERPRET`.

## Frozen arms

Let:
- `B_hat` = M94C baseline team-rush prediction
- `P_hat` = M94C predicted offensive plays
- alpha = `0.75`
- `s_actual,state` = actual target-game lead/neutral/trail play share
- `r_hat,state` = strictly-prior M94C shrunk rushing tendency in that state

### BASE_M94C_TOTAL_RUSH

M94C candidate unchanged.

### ORACLE_STATE_OCCUPANCY

Exact STACK6J arm:

`0.25*B_hat + 0.75*P_hat*Σ(s_actual,state * r_hat,state)`

### ORACLE_OCC_PLUS_TENDENCY

Replace the occupancy-weighted state rushing tendency with the target game's realized PBP rushing rate while keeping predicted plays and baseline blend frozen:

`0.25*B_hat + 0.75*P_hat*Q_pbp`

Because

`Q_pbp = Σ(s_actual,state * r_actual,state)`

for states actually observed in the target game, this arm gives perfect realized state-weighted play selection without inventing a counterfactual rush rate for states with zero target-game plays.

Target-game PBP is grading truth only.

## Required metrics

For all three arms:
- n
- MAE
- RMSE
- signed bias
- Pearson correlation
- recovery vs base

For `ORACLE_OCC_PLUS_TENDENCY`, also report:

`incremental_tendency_recovery = MAE(ORACLE_STATE_OCCUPANCY) - MAE(ORACLE_OCC_PLUS_TENDENCY)`

Frozen remaining STACK6I rate headroom after occupancy:

`remaining_rate_headroom = 3.2406943942 - 0.6850728182 = 2.5556215760`

Report:

`tendency_fraction_of_remaining = incremental_tendency_recovery / 2.5556215760`

Repeat the incremental recovery in:
- `POOL_OVER_5`
- `POOL_UNDER_5`
- `POOL_ABS_5`
- `NON_EXTREME_ABS_LT3`

## Frozen attribution rule

- `WITHIN_STATE_TENDENCY_DOMINANT` if:
  - incremental tendency recovery >= `1.00` attempt;
  - tendency fraction of remaining headroom >= `0.50`;
  - incremental recovery is positive in both `POOL_OVER_5` and `POOL_UNDER_5`.

- `WITHIN_STATE_TENDENCY_MATERIAL` if:
  - incremental recovery >= `0.50`;
  - incremental recovery is positive in both extreme directions;
  - but the dominant gate above is not met.

- otherwise `WITHIN_STATE_TENDENCY_NOT_PRIMARY`.

No threshold may change after results are exposed.

## Integrity requirements

1. no fitted models;
2. no hyperparameter/feature/threshold search;
3. no sportsbook inputs;
4. reconstruct STACK6J occupancy arm to numerical tolerance;
5. reproduce M94C W6-18 base MAE `6.2034547805`;
6. reproduce STACK6J occupancy MAE `5.5183819623`;
7. PBP/weekly truth bridge passes frozen requirements;
8. target-game PBP used only in oracle grading.

## Required outputs

1. `stack6k_integrity.csv`
2. `stack6k_truth_bridge.csv`
3. `stack6k_overall_scores.csv`
4. `stack6k_bin_scores.csv`
5. `stack6k_team_trace.csv`
6. `stack6k_disposition.csv`

## Next-step rule

- `WITHIN_STATE_TENDENCY_DOMINANT`: next research should localize which football situations drive the tendency miss (neutral early downs, leading clock-control, trailing rush persistence, etc.) before fitting a candidate.
- `WITHIN_STATE_TENDENCY_MATERIAL`: retain tendency as one contributor and decompose further before fitting.
- `WITHIN_STATE_TENDENCY_NOT_PRIMARY`: do not chase generic historical state rush rates; investigate interaction/non-state mechanics.

P3 remains the RB point champion.
