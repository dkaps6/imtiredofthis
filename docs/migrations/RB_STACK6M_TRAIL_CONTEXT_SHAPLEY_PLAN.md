# RB STACK6M — Trail-Context Shapley Attribution

## Status

Frozen no-fit diagnostic. No production change. No predictive candidate is authorized in STACK6M.

## Why STACK6M exists

STACK6L identified M94C's coarse trailing-state rushing tendency as the largest state-level contributor:

- direct trail-only correction recovery: `1.0153694874` attempts MAE
- trail Shapley recovery: `1.0249065737`
- overall tendency share: `49.56%`
- P3 pool-over-5 tendency share: `65.06%`
- frozen disposition: `TRAIL_TENDENCY_DOMINANT`

A single `trail` state currently combines very different football situations. STACK6M tests whether the trail correction is concentrated in deficit severity and game phase.

## Frozen lineage

M94C:
- run `33353485070`
- artifact ID `9744399103`

STACK6H P3 bins:
- run `33632678179`
- artifact ID `9847470353`

STACK6L:
- corrected run `33639241740`
- job `100277828082`
- SHA `3adc8b0c616d2a56b5005f2421431c84254ddcb1`
- artifact ID `9850103613`
- empty occupancy MAE `5.5183819623`
- trail-only corrected MAE `4.5030124750`
- direct trail recovery `1.0153694874`

## Population

- 2025 regular season
- Weeks 6-18
- same 388 team-games
- P3 bins: `POOL_OVER_5`, `POOL_UNDER_5`, `POOL_ABS_5`, `NON_EXTREME_ABS_LT3`
- no sportsbook selection

## Frozen trail contexts

Trailing remains score differential < -3 from the offense's perspective.

Split trail plays into four mutually exclusive football contexts:

1. `CLOSE_EARLY`
   - deficit 4 through 8 points
   - quarters 1-3
2. `DEEP_EARLY`
   - deficit 9+ points
   - quarters 1-3
3. `CLOSE_LATE`
   - deficit 4 through 8 points
   - quarter 4 or later
4. `DEEP_LATE`
   - deficit 9+ points
   - quarter 4 or later

The 8-point boundary keeps all one-possession-with-conversion deficits in the close group; 9+ is a true multi-score deficit.

## PBP reconstruction

For each team-game and trail context, reconstruct:
- context offensive plays
- context rush attempts
- context share of all offensive plays
- context realized rushing contribution = context rushes / all offensive plays

The four context play shares must sum to the M94C target-game `trail_play_share` to numerical tolerance.

## Frozen subset oracle

Start from the STACK6J perfect-occupancy arm:

- lead contribution remains `actual lead share × pregame lead rush-rate tendency`;
- neutral contribution remains `actual neutral share × pregame neutral rush-rate tendency`;
- each uncorrected trail context contributes `context play share × pregame coarse trail rush-rate tendency`;
- each corrected trail context contributes `actual context rushes / actual_off_plays`.

For every one of the 16 subsets of the four trail contexts:

`ORACLE_S = 0.25*baseline_team_rush_att + 0.75*pred_off_plays*(lead + neutral + trail-context contributions)`

Required identities:
- empty subset MAE = STACK6J occupancy MAE `5.5183819623`;
- all four corrected contexts MAE = STACK6L trail-only MAE `4.5030124750`.

## Exact Shapley attribution

Define:

`v(S) = MAE(empty) - MAE(S)`

Compute exact four-player Shapley values across all 16 subsets.

The four Shapley values must sum to direct trail recovery `1.0153694874` to numerical tolerance.

Report each context's:
- Shapley MAE recovery
- fraction of direct trail recovery
- Shapley value in `POOL_OVER_5`
- `POOL_UNDER_5`
- `POOL_ABS_5`
- `NON_EXTREME_ABS_LT3`

Also report context-level play volume and actual rushing rate for football interpretation.

## Frozen attribution rule

Let `TOP_CONTEXT` be the context with largest positive overall Shapley recovery.

Disposition `<TOP_CONTEXT>_DOMINANT` if:
- TOP_CONTEXT Shapley recovery >= `0.30` attempt;
- TOP_CONTEXT fraction of direct trail recovery >= `0.40`;
- TOP_CONTEXT Shapley value is positive in both `POOL_OVER_5` and `POOL_UNDER_5`.

Otherwise disposition `TRAIL_CONTEXT_DISTRIBUTED`.

No threshold may change after results are exposed.

## Integrity requirements

1. no fitting/search/sportsbook inputs;
2. fresh PBP trail share reproduces M94C trail share to max absolute error <= `1e-9`;
3. context shares sum to reconstructed trail share to <= `1e-9`;
4. empty-subset MAE reproduces `5.5183819623`;
5. all-context MAE reproduces `4.5030124750`;
6. Shapley sum reproduces `1.0153694874`;
7. target-game context truth is grading only.

## Next-step rule

- `CLOSE_EARLY_DOMINANT`: model one-score trailing run persistence before late-game urgency.
- `DEEP_EARLY_DOMINANT`: model early multi-score response / pass-vs-run adaptation.
- `CLOSE_LATE_DOMINANT`: model late one-possession clock, timeout and win-probability behavior.
- `DEEP_LATE_DOMINANT`: model late multi-score pass-heavy/scramble behavior and garbage-time run persistence.
- `TRAIL_CONTEXT_DISTRIBUTED`: coarse trailing-state tendency is broadly insufficient; build a shared continuous score/time run-intent architecture rather than one context patch.

P3 remains the RB point champion.
