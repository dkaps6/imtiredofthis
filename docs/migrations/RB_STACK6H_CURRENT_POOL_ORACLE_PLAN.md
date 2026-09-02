# RB STACK6H — Current P3 Team-RB-Pool Oracle Decomposition

## Status

Frozen no-fit diagnostic. No production change. No candidate model or coefficient fitting is authorized in STACK6H.

## Why STACK6H exists

STACK6E/STACK6F established that the remaining false-high secondary-back problem is usually an upstream **team RB carry-pool** error rather than simple RB1/RB2 redistribution. STACK6F showed that a compact strictly-prior history model improves pool MAE/RMSE but does not rank target games strongly enough. STACK6G then rejected two plausible target-week discontinuity mechanisms (QB1 rushing-regime change and playcaller change).

Prior work already tested both total team rushing and RB-room share as predictive models:

- M94/M94B/M94C modeled total team rushing opportunity/game state.
- M95E modeled `team rush attempts × RB-room share × player RB-room share` and direct absolute share.

STACK6H does **not** retest those model families. It asks a narrower current-frontier question:

> On the exact current 2025 P3 team-RB-pool population, which latent component now owns more of the remaining error: total team rushing volume or RB share of team rushing?

This determines which component is still worth attacking with genuinely new information.

## Frozen artifacts / lineage

P3 allocation/current casebook:

- STACK6 run `33549529203`
- artifact `rb-stack6-secondary-role-model`
- artifact ID `9816904835`
- SHA `4db6a46bd1a911d27d0957d5992d43633c3075ce`

M94C total-team rushing decomposition:

- M94C run `33353485070`
- artifact `migration-94c-rb-game-environment`
- artifact ID `9744399103`
- SHA `7e85dd0e836982fdfbea080ea69c0149d2e186e3`

STACK6F current pool scoring reference:

- run `33578446070`
- SHA `a72361a328533101f670e672314e21fa1b8672f4`
- W6-18 P3 pool MAE `5.741946`

## Population

Primary population:

- 2025 regular season
- Weeks 6-18
- team-games present in both frozen M94C team trace and current P3/STACK6 casebook
- no selection by sportsbook data

Secondary descriptive bins, frozen before computation:

- P3 predicted RB pool minus actual RB carries >= +3 (`POOL_OVER_3`)
- >= +5 (`POOL_OVER_5`)
- <= -3 (`POOL_UNDER_3`)
- <= -5 (`POOL_UNDER_5`)
- absolute residual >= 5 (`POOL_ABS_5`)
- absolute residual <3 (`NON_EXTREME_ABS_LT3`)

## Definitions

For each team-game:

- `T_hat` = frozen M94C candidate **total team rush attempts** (`candidate_team_rush_att`).
- `T` = actual total team rush attempts from the frozen M94C team truth (`actual_team_rush_att`).
- `R_hat` = current P3 team RB carry pool, sum of `parent_att` across the current STACK6/P3 casebook.
- `R` = actual team RB/HB/FB carries from the current canonical RB truth used by STACK6F semantics.
- `S_hat = R_hat / T_hat` = P3-implied RB share of predicted total team rushes.
- `S = R / T` = actual RB share of actual total team rushes.

Rows with nonpositive/invalid denominators are integrity failures rather than imputed.

## Frozen oracle arms

### BASE_P3_POOL

`R_hat = T_hat × S_hat`

### ORACLE_TOTAL_RUSH

Replace only total team rush volume with truth; preserve P3-implied RB share:

`R_oracle_total = T × S_hat`

This measures the recoverable error if total team rushing opportunity were perfect but RB-vs-QB/non-RB share remained as projected.

### ORACLE_RB_SHARE

Replace only RB share with truth; preserve M94C predicted total team rush volume:

`R_oracle_share = T_hat × S`

This measures recoverable error if the RB-room share of team rushing were perfect but total team rush opportunity remained as projected.

### ORACLE_BOTH

`R_oracle_both = T × S = R`

Used only as an integrity identity check; it is not a candidate.

## Required metrics

For BASE, ORACLE_TOTAL_RUSH, and ORACLE_RB_SHARE on W6-18:

- n
- MAE
- RMSE
- signed bias
- Pearson correlation vs actual RB carries

Recoverable MAE:

- `total_rush_mae_recovery = BASE_MAE - ORACLE_TOTAL_RUSH_MAE`
- `rb_share_mae_recovery = BASE_MAE - ORACLE_RB_SHARE_MAE`

Also report the same MAE/recovery quantities for each frozen residual bin.

## Frozen attribution rule

No statistical significance claim is required. This is an oracle bottleneck attribution.

- If total-rush MAE recovery exceeds RB-share recovery by >= `0.50` carry overall **and** is >= the RB-share recovery in both `POOL_OVER_5` and `POOL_UNDER_5`, classify `TOTAL_TEAM_RUSHING_DOMINANT`.
- If RB-share MAE recovery exceeds total-rush recovery by >= `0.50` carry overall **and** is >= the total-rush recovery in both `POOL_OVER_5` and `POOL_UNDER_5`, classify `RB_SHARE_DOMINANT`.
- Otherwise classify `MIXED_TEAM_POOL_BOTTLENECK`.

These thresholds are frozen before the oracle results are computed.

## Integrity requirements

- no fitted models
- no hyperparameter search
- no threshold/feature search
- no sportsbook inputs
- actual total rush / actual RB share are oracle grading variables only
- exact join coverage reported
- `ORACLE_BOTH` must reproduce actual team RB carries to numerical tolerance
- BASE P3 pool must reproduce the current STACK6F P3 pool score within small numerical tolerance; otherwise stop and diagnose truth/universe mismatch before interpreting attribution

## Required outputs

1. `stack6h_integrity.csv`
2. `stack6h_overall_oracle_scores.csv`
3. `stack6h_bin_oracle_scores.csv`
4. `stack6h_team_trace.csv`
5. `stack6h_disposition.csv`

## Next-step rule

STACK6H does not authorize a predictive model by itself.

- `TOTAL_TEAM_RUSHING_DOMINANT` -> search only for genuinely new information that ranks target-game total team rushing opportunity beyond M94C/STACK6F history.
- `RB_SHARE_DOMINANT` -> search only for genuinely new information affecting RB-vs-QB/non-RB rushing mass beyond M95E and STACK6G.
- `MIXED_TEAM_POOL_BOTTLENECK` -> do not force either old family; investigate whether a joint latent game-volume/state representation or an unmodeled source class is needed.

P3 remains champion unless a later frozen predictive candidate clears its own retention gates.