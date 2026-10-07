# PLAYER OPPORTUNITY VOLUME VS SHARE DECOMPOSITION V1 — FROZEN CONTRACT

Date: 2026-10-07  
Branch: `research-player-opportunity-volume-share-decomposition-v1`

## Purpose

Decompose the residual individual-player workload compression that survived the
historical ACT-only availability parity diagnostic into two distinct mechanisms:

1. **team opportunity volume error**, and
2. **individual player share/allocation error**.

This is diagnostic only. It does not fit a correction, choose a threshold,
change production, or use sportsbook information.

## Frozen parent

Parent authority:

- `HISTORICAL_AVAILABILITY_PARITY_DIAGNOSTIC_V1`
- run `37687979574`
- artifact `11511927864`
- digest `sha256:6bcbf63f8dfc5b2b0362e16fbd612653d6f813e5c2d9aec79e9d49c98cb0303b`

Use the parent's **ACT-only individual-player opportunity rows**. Do not return
to the baseline ACT+INA universe for scientific interpretation.

The prediction side is frozen before any completed-game volume/share is read.

## Player-level quantities

### QB pass attempts

For each ACT-only QB player-game:

- predicted team official pass attempts =
  `predicted_team_opportunity_mean`;
- predicted QB attempt share =
  `final_player_probability`;
- predicted player attempts =
  `expected_opportunities_from_probability`;
- actual team official pass attempts = completed-game team PBP official pass
  attempts;
- actual QB attempts = frozen parent `actual_opportunities`;
- actual QB attempt share =
  `actual_qb_attempts / actual_team_official_pass_attempts`.

### RB carries

For each ACT-only RB/FB carry row:

- predicted team rush attempts =
  `predicted_team_opportunity_mean`;
- predicted player carry share =
  `final_player_probability`;
- predicted player carries =
  `expected_opportunities_from_probability`;
- actual team rush attempts = completed-game team PBP rush attempts;
- actual player carries = frozen parent `actual_opportunities`;
- actual player carry share =
  `actual_player_carries / actual_team_rush_attempts`.

### RB / WR / TE targets

For each ACT-only receiving target row:

- predicted team official pass attempts =
  `predicted_team_opportunity_mean`;
- predicted target probability per official pass attempt =
  `final_player_probability`;
- predicted player targets =
  `expected_opportunities_from_probability`;
- actual team official pass attempts = completed-game team PBP official pass
  attempts;
- actual player targets = frozen parent `actual_opportunities`;
- actual target share per official pass attempt =
  `actual_player_targets / actual_team_official_pass_attempts`.

## Completed-game team-volume authority

Actual team opportunity volume is outcome-side grading evidence and is loaded
**only after the ACT-only prediction parent has been frozen**.

Use nflverse PBP for 2026 Weeks 1-4:

- official pass attempt:
  `pass_attempt == 1 AND sack != 1`;
- team rush attempt:
  `rush_attempt == 1`;
- regular season only;
- team key = canonical `posteam`.

No actual team volume/share may enter any upstream projection or allocation.

## Frozen no-fit decomposition

For every player row define:

### Baseline expected opportunity

`baseline = predicted_team_volume * predicted_player_share`

This must equal the parent's
`expected_opportunities_from_probability` within floating-point tolerance.

### Oracle-team-volume diagnostic

`oracle_team = actual_team_volume * predicted_player_share`

Only the team volume is replaced by the realized completed-game value.

### Oracle-player-share diagnostic

`oracle_share = predicted_team_volume * actual_player_share`

Only the player's realized share is substituted.

### Full oracle identity

`full_oracle = actual_team_volume * actual_player_share`

This must equal actual individual opportunity within tolerance whenever
actual team volume > 0.

The two oracle stages are diagnostics only. They are not candidates and may
never be routed into production.

## Required outputs

1. `player_opportunity_volume_share_rows.csv`
2. `player_opportunity_volume_share_summary.csv`
3. `player_opportunity_volume_share_high_workload.csv`
4. `player_opportunity_volume_share_summary.json`

Each row must contain:

- season/week/team/opponent/event/player/player key/position;
- opportunity type;
- predicted team volume;
- actual team volume;
- predicted player share;
- actual player share;
- baseline expected opportunity;
- oracle-team-volume opportunity;
- oracle-player-share opportunity;
- actual player opportunity;
- baseline/oracle errors and absolute errors;
- linked yard/count error from the frozen parent;
- parent actual-opportunity bin;
- sportsbook-input flag = false.

## Primary comparisons

By position + opportunity type report:

- row count;
- baseline MAE/bias/RMSE;
- oracle-team MAE/bias/RMSE;
- oracle-share MAE/bias/RMSE;
- baseline minus oracle-team MAE improvement;
- baseline minus oracle-share MAE improvement;
- fraction of baseline MAE removed by team-volume oracle;
- fraction removed by player-share oracle;
- predicted vs actual team-volume MAE/bias;
- predicted vs actual player-share MAE/bias;
- correlation between opportunity error and linked yard/count error.

Report the same decomposition for:

- actual individual opportunity > 0;
- frozen high-workload bin:
  - QB pass attempts: `41_PLUS`
  - RB carries: `15_PLUS`
  - RB/WR/TE targets: `09_PLUS`.

## Scientific interpretation

The result must distinguish:

- `TEAM_VOLUME_DOMINANT`: oracle team volume removes materially more player
  opportunity error than oracle player share;
- `PLAYER_SHARE_DOMINANT`: oracle player share removes materially more error
  than oracle team volume;
- `MIXED_VOLUME_AND_SHARE`: both are meaningful and neither clearly dominates;
- `DECOMPOSITION_INVALID`: integrity or lineage failure.

These are descriptive result labels only; no numeric cutoff is frozen and no
automatic promotion follows.

## Integrity gates

Fail closed if:

- parent artifact/run/digest lineage is not the frozen ACT-only authority;
- parent contains sportsbook input;
- parent contains Week-5 RB room shadow;
- parent contains any status other than the ACT-only reconstruction scope;
- actual team volumes are loaded before parent predictions are frozen;
- actual team volume is <=0 on a scored player row;
- actual player share is outside [0,1] beyond floating tolerance;
- baseline arithmetic does not reproduce
  `expected_opportunities_from_probability`;
- full-oracle arithmetic does not reproduce actual player opportunity;
- any parameter is fit;
- any threshold is selected;
- any sportsbook data is used.

## Prohibited

- no production change;
- no new target/carry/pass-attempt coefficient;
- no outcome-selected share transform;
- no generic position recalibration;
- no paid OddsAPI;
- no weakening of closed research;
- no use of oracle values as model inputs.

## Next-step rule

Only after this decomposition may a new player-level mechanism be considered.

If player share is dominant, the next research must target strictly-prior
individual workload/share state while conserving team volume.

If team volume is dominant, the next research must stay at team opportunity
volume and must not masquerade as player-specific science.

If mixed, isolate the larger residual mechanism first and keep the other frozen.
