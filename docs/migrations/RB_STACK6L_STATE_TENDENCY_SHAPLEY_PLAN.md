# RB STACK6L — Lead / Neutral / Trail Tendency Shapley Attribution

## Status

Frozen no-fit diagnostic. No production change. No fitted predictive candidate is authorized in STACK6L.

## Why STACK6L exists

STACK6K established that realized rushing tendency after state occupancy is the dominant remaining M94C rush-rate mechanism:

- incremental tendency MAE recovery after occupancy: `2.0680540592`
- fraction of remaining STACK6I rush-rate headroom: `80.92%`
- frozen disposition: `WITHIN_STATE_TENDENCY_DOMINANT`

The next question is which score state is responsible: lead, neutral, or trail.

A simple one-at-a-time comparison is order-dependent because MAE is nonlinear. With only three states, STACK6L will evaluate all 2^3 correction subsets and compute exact Shapley attribution of the 6K tendency recovery.

## Frozen lineage

M94C:
- run `33353485070`
- artifact ID `9744399103`

STACK6H P3 error bins:
- run `33632678179`
- artifact ID `9847470353`

STACK6J occupancy result:
- run `33638294183`
- occupancy MAE `5.5183819623`

STACK6K tendency result:
- run `33638739235`
- job `100276119195`
- SHA `2a6775bacaa1e6b7d1ec5a0b0f5ca302c66a7417`
- artifact ID `9849896540`
- occupancy+tendency MAE `3.4503279032`
- incremental tendency recovery `2.0680540592`

## Population

- 2025 regular season
- Weeks 6-18
- same 388 team-games
- current P3 bins: `POOL_OVER_5`, `POOL_UNDER_5`, `POOL_ABS_5`, `NON_EXTREME_ABS_LT3`
- no sportsbook selection

## Target-game PBP reconstruction

Use the existing M94B score-state definition and PBP builder:

- lead: score differential > +3
- neutral: -3 through +3
- trail: score differential < -3

For each team-game reconstruct:
- actual offensive plays
- actual total PBP rush attempts
- lead/neutral/trail plays
- lead/neutral/trail rushes
- lead/neutral/trail play shares
- state-specific realized rush rates where observed

Target-game PBP is grading truth only.

## Source integrity bridge

Before interpreting attribution, the fresh PBP reconstruction must reproduce the M94C artifact labels on W6-18:

- max abs difference in `actual_off_plays` <= `1e-9`
- max abs difference in `actual_rush_att_pbp` <= `1e-9`
- max abs difference in each lead/neutral/trail play share <= `1e-9`

If not, disposition is `STACK6L_PBP_REPRODUCTION_FAILURE_DO_NOT_INTERPRET`.

## Frozen subset oracle

Start from the STACK6J occupancy arm, which already supplies actual state occupancy while retaining M94C's strictly-prior state rush tendencies.

For a corrected-state subset `S`, define the occupancy-weighted structured rush-rate contribution for each state:

- if state is NOT in S:
  `actual_state_play_share × pregame_state_rush_rate_shrunk`

- if state IS in S:
  `actual_state_rushes / actual_off_plays`

The corrected state contribution is therefore well-defined even when the state has zero target plays: zero state rushes / total offensive plays = zero contribution. No counterfactual state rush rate is invented.

For each of all 8 subsets of `{lead, neutral, trail}`:

`ORACLE_S = 0.25*baseline_team_rush_att + 0.75*pred_off_plays*sum(state contributions)`

Required identities:
- empty subset reproduces STACK6J occupancy MAE `5.5183819623`;
- all-three subset reproduces STACK6K occupancy+tendency MAE `3.4503279032`.

## Exact Shapley attribution

Define value:

`v(S) = MAE(empty subset) - MAE(S)`

For each state, compute the exact three-player Shapley value across all subset orderings.

The three Shapley values must sum to the full STACK6K incremental tendency recovery to numerical tolerance.

Report for each state:
- Shapley MAE recovery
- fraction of total tendency recovery
- Shapley recovery in `POOL_OVER_5`
- Shapley recovery in `POOL_UNDER_5`
- Shapley recovery in `POOL_ABS_5`
- Shapley recovery in `NON_EXTREME_ABS_LT3`

Also report direct one-state-corrected subset scores for football interpretability.

## Frozen attribution rule

Let the state with the largest positive overall Shapley recovery be `TOP_STATE`.

Disposition is `<TOP_STATE>_TENDENCY_DOMINANT` if:
- TOP_STATE Shapley recovery >= `0.75` attempt;
- TOP_STATE fraction of total tendency recovery >= `0.45`;
- TOP_STATE Shapley recovery is positive in both `POOL_OVER_5` and `POOL_UNDER_5`.

Otherwise disposition is `MULTI_STATE_TENDENCY`.

No threshold may change after results are exposed.

## Required outputs

1. `stack6l_integrity.csv`
2. `stack6l_subset_scores.csv`
3. `stack6l_shapley.csv`
4. `stack6l_team_trace.csv`
5. `stack6l_disposition.csv`

## Next-step rule

- `NEUTRAL_TENDENCY_DOMINANT`: next decompose neutral rushing into early-down / late-down and clock/game-phase mechanics before fitting.
- `LEAD_TENDENCY_DOMINANT`: next isolate clock-control, win-probability/late-game and possession effects.
- `TRAIL_TENDENCY_DOMINANT`: next isolate comeback-state rush persistence, QB scramble/design effects and pass-vs-run response.
- `MULTI_STATE_TENDENCY`: build a common pregame run-intent/tendency architecture rather than a state-specific patch.

P3 remains the RB point champion.
