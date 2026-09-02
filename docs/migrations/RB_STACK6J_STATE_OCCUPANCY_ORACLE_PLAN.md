# RB STACK6J — M94C State-Occupancy Oracle

## Status

Frozen no-fit diagnostic. No production change. No fitted predictive candidate is authorized in STACK6J.

## Why STACK6J exists

STACK6H localized the current P3 RB-pool bottleneck primarily to total team rushing. STACK6I then localized the M94C total-rush error primarily to effective rushing rate per offensive play rather than offensive play volume:

- M94C W6-18 total-rush MAE: `6.2034547805`
- perfect offensive plays recovery: `+0.9614142644`
- perfect effective rush rate recovery: `+3.2406943942`
- frozen STACK6I disposition: `RUSH_RATE_DOMINANT`

M94C's structured rush-rate mechanism is explicitly:

`pred_off_plays × Σ(predicted state share × pregame state-conditioned rush tendency)`

across lead / neutral / trail states, then blended 75% with the frozen baseline team-rush forecast.

STACK6J asks whether the rush-rate miss is primarily caused by **wrong lead/neutral/trail occupancy**. It does not fit or tune a new model.

## Frozen lineage

M94C authoritative artifact:

- run `33353485070`
- SHA `7e85dd0e836982fdfbea080ea69c0149d2e186e3`
- artifact ID `9744399103`

Current P3 error-bin authority:

- STACK6H run `33632678179`
- artifact ID `9847470353`

STACK6I authoritative result:

- run `33637785836`
- job `100272894018`
- SHA `343312c47be50bd450b0cea69361aeea3e2f52fa`
- artifact ID `9849508420`
- disposition `RUSH_RATE_DOMINANT`

## Population

Primary:

- 2025 regular season
- Weeks 6-18
- same 388 team-games used by STACK6I

Current-P3 context bins carried from STACK6H:

- `POOL_OVER_5`
- `POOL_UNDER_5`
- `POOL_ABS_5`
- `NON_EXTREME_ABS_LT3`

No sportsbook selection.

## Frozen M94C reconstruction

Let:

- `B_hat` = M94C `baseline_team_rush_att`
- `P_hat` = M94C `pred_off_plays`
- `s_hat_lead`, `s_hat_neutral`, `s_hat_trail` = M94C predicted state play shares
- `r_hat_lead`, `r_hat_neutral`, `r_hat_trail` = strictly-prior M94C `gs_team_*_rush_rate_shrunk`
- alpha = frozen M94C blend `0.75`

Rebuild the structured arm:

`STRUCTURED_REBUILT = P_hat × Σ(s_hat_state × r_hat_state)`

It must reproduce M94C `structured_team_rush_att` to numerical tolerance.

Rebuild the candidate:

`BASE_REBUILT = 0.25 × B_hat + 0.75 × STRUCTURED_REBUILT`

It must reproduce M94C `candidate_team_rush_att` to numerical tolerance.

## Frozen oracle arm

Target-game state shares are grading truth only:

- `s_lead` = actual M94C `lead_play_share`
- `s_neutral` = actual M94C `neutral_play_share`
- `s_trail` = actual M94C `trail_play_share`

The state-occupancy oracle is:

`ORACLE_STATE_OCCUPANCY = 0.25 × B_hat + 0.75 × P_hat × Σ(s_actual_state × r_hat_state)`

Everything except state occupancy remains frozen:

- predicted offensive plays remain M94C's prediction;
- pregame state-conditioned rushing tendencies remain unchanged;
- baseline component remains unchanged;
- 75/25 blend remains unchanged.

This isolates how much total-rush error is recoverable solely by putting the offense in the correct lead/neutral/trail mix.

## Required metrics

For `BASE_M94C_TOTAL_RUSH` and `ORACLE_STATE_OCCUPANCY`:

- n
- MAE
- RMSE
- signed bias
- Pearson correlation
- MAE recovery vs base

Also report:

- predicted vs actual lead-share MAE/corr
- predicted vs actual neutral-share MAE/corr
- predicted vs actual trail-share MAE/corr
- occupancy oracle results in all four P3 error bins above.

## Frozen attribution rule

STACK6I perfect-rush-rate MAE recovery is frozen at `3.2406943942` attempts.

Define:

`occupancy_headroom_fraction = STACK6J occupancy MAE recovery / 3.2406943942`

Disposition:

- `STATE_OCCUPANCY_MATERIAL` if overall occupancy recovery is at least `1.00` total-rush attempt **and** recovery is positive in both `POOL_OVER_5` and `POOL_UNDER_5`.
- `STATE_OCCUPANCY_NOT_PRIMARY` if overall occupancy recovery is less than `0.50` **or** recovery is nonpositive in either `POOL_OVER_5` or `POOL_UNDER_5`.
- otherwise `STATE_OCCUPANCY_PARTIAL`.

No threshold may change after results are exposed.

## Integrity requirements

1. no fitted models;
2. no feature/hyperparameter/threshold search;
3. no sportsbook inputs;
4. structured reconstruction must reproduce M94C `structured_team_rush_att` to numerical tolerance;
5. candidate reconstruction must reproduce M94C `candidate_team_rush_att` to numerical tolerance;
6. W6-18 base score must reproduce STACK6I exactly;
7. actual target-game state shares are used only in the oracle grading arm.

## Required outputs

1. `stack6j_integrity.csv`
2. `stack6j_state_share_scores.csv`
3. `stack6j_overall_scores.csv`
4. `stack6j_bin_scores.csv`
5. `stack6j_team_trace.csv`
6. `stack6j_disposition.csv`

## Next-step rule

- `STATE_OCCUPANCY_MATERIAL` -> investigate the margin/state-share mapper as the next predictive target.
- `STATE_OCCUPANCY_PARTIAL` -> retain state occupancy as one contributor but continue decomposing within-state rushing tendency before fitting.
- `STATE_OCCUPANCY_NOT_PRIMARY` -> do not spend the next cycle on another game-state mapper; move directly to within-state rushing-tendency / play-selection mechanics.

P3 remains the RB point champion.
