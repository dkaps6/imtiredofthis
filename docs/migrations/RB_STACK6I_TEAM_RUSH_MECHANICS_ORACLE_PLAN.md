# RB STACK6I — Total-Team-Rush Mechanics Oracle Decomposition

## Status

Frozen no-fit diagnostic. No production change. No fitted predictive candidate is authorized in STACK6I.

## Why STACK6I exists

STACK6H established decisively that the remaining current-P3 team-RB-pool error is dominated by **total team rushing opportunity**, not RB share of total rushing:

- P3 team-RB-pool W6-18 MAE: `5.741946`
- perfect total-team rush attempts, P3-implied RB share preserved: MAE `2.261455` (recovery `3.480491`)
- perfect RB share, M94C total-team rush prediction preserved: MAE `5.181572` (recovery `0.560374`)
- frozen disposition: `TOTAL_TEAM_RUSHING_DOMINANT`

M94/M94B/M94C already modeled total team rushing, so STACK6I does not fit another total-rush model. It decomposes the current frozen M94C total-rush forecast into the two multiplicative quantities that can generate its error:

1. offensive play volume;
2. effective rushing rate per offensive play.

The purpose is to decide which internal mechanic deserves the next genuinely new source/model investigation.

## Frozen lineage

M94C authoritative artifact:

- run `33353485070`
- SHA `7e85dd0e836982fdfbea080ea69c0149d2e186e3`
- artifact `migration-94c-rb-game-environment`
- artifact ID `9744399103`

Current P3 / STACK6 casebook:

- run `33549529203`
- SHA `4db6a46bd1a911d27d0957d5992d43633c3075ce`
- artifact `rb-stack6-secondary-role-model`
- artifact ID `9816904835`

STACK6H current-frontier result:

- run `33632678179`
- SHA `268362dc386e7f3952a86e6d44e714c3379d1d2e`
- artifact `9847470353`
- disposition `TOTAL_TEAM_RUSHING_DOMINANT`

## Population

Primary population:

- 2025 regular season
- Weeks 6-18
- team-games present in frozen M94C 2025 team trace and current P3/STACK6 casebook

Frozen current-P3 pool-error bins are also carried forward:

- `POOL_OVER_5`: P3 RB pool minus actual RB carries >= +5
- `POOL_UNDER_5`: <= -5
- `POOL_ABS_5`: absolute pool residual >= 5
- `NON_EXTREME_ABS_LT3`: absolute pool residual <3

No sportsbook selection.

## Exact factorization

For each team-game:

- `T_hat` = M94C `candidate_team_rush_att`
- `T` = M94C `actual_team_rush_att`
- `P_hat` = M94C `pred_off_plays`
- `P` = M94C `actual_off_plays`

Define the **effective** predicted and actual team-rush rates:

- `Q_hat = T_hat / P_hat`
- `Q = T / P`

`Q` is deliberately named effective rush rate rather than PBP rush rate because M94C's weekly-stat total-rush truth can differ slightly from PBP rush attempts in some games. The factorization is nevertheless exact:

- `T_hat = P_hat × Q_hat`
- `T = P × Q`

Rows with missing or nonpositive play denominators are integrity failures; no imputation.

## Frozen oracle arms

### BASE_M94C_TOTAL_RUSH

`T_hat = P_hat × Q_hat`

### ORACLE_PLAYS

Replace offensive play volume only:

`T_oracle_plays = P × Q_hat`

This measures recoverable total-rush error if offensive play volume were perfect while M94C's effective run/pass mix remained unchanged.

### ORACLE_RUSH_RATE

Replace effective rush rate only:

`T_oracle_rate = P_hat × Q`

This measures recoverable total-rush error if run/pass allocation per play were perfect while M94C's offensive-play forecast remained unchanged.

### ORACLE_BOTH

`P × Q = T`

Identity/integrity check only.

## Required metrics

For BASE, ORACLE_PLAYS, ORACLE_RUSH_RATE, ORACLE_BOTH:

- n
- MAE
- RMSE
- signed bias
- Pearson correlation vs actual total team rush attempts
- MAE recovery versus BASE

Also report:

- `P_hat` vs `P`: MAE/RMSE/bias/correlation
- `Q_hat` vs `Q`: MAE/RMSE/bias/correlation
- same oracle MAE/recovery within `POOL_OVER_5`, `POOL_UNDER_5`, `POOL_ABS_5`, and `NON_EXTREME_ABS_LT3`.

## Frozen attribution rule

- `PLAY_VOLUME_DOMINANT` if ORACLE_PLAYS recovery exceeds ORACLE_RUSH_RATE recovery by >= `0.50` total-rush attempt overall **and** ORACLE_PLAYS recovery is >= ORACLE_RUSH_RATE recovery in both `POOL_OVER_5` and `POOL_UNDER_5`.
- `RUSH_RATE_DOMINANT` if the reverse holds by >= `0.50` overall and in both extreme directions.
- otherwise `MIXED_TOTAL_RUSH_MECHANICS`.

No threshold may change after results are exposed.

## Integrity requirements

1. no fitted models;
2. no hyperparameter/feature/threshold search;
3. no sportsbook inputs;
4. `P_hat × Q_hat` must reproduce `candidate_team_rush_att` to numerical tolerance;
5. `P × Q` must reproduce `actual_team_rush_att` to numerical tolerance;
6. W6-18 BASE score must reproduce the frozen M94C team-total-rush score on the same population;
7. current P3 pool-error bins are grading/context only.

## Required outputs

1. `stack6i_integrity.csv`
2. `stack6i_component_scores.csv`
3. `stack6i_overall_oracle_scores.csv`
4. `stack6i_bin_oracle_scores.csv`
5. `stack6i_team_trace.csv`
6. `stack6i_disposition.csv`

## Next-step rule

STACK6I authorizes no predictive change.

- `PLAY_VOLUME_DOMINANT` -> next research must seek genuinely new pregame information about target-game offensive possession/play count beyond the already-tested generic pace/history features.
- `RUSH_RATE_DOMINANT` -> next research must localize the error inside M94C's game-state/run-tendency mechanics before fitting anything new.
- `MIXED_TOTAL_RUSH_MECHANICS` -> do not force a single old family; decompose state/possession interactions further.

P3 remains the RB point champion.