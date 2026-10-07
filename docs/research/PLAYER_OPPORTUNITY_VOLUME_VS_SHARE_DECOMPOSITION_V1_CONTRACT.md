# PLAYER OPPORTUNITY VOLUME VS SHARE DECOMPOSITION V1 — FROZEN CONTRACT

Date: 2026-10-07
Branch: `research-player-opportunity-volume-vs-share-decomposition-v1`

## Purpose

Decompose residual individual-player opportunity error after ACT-only historical
availability parity into two components:

1. team opportunity volume error
2. individual player share error

Diagnostic only. No fitting, threshold selection, or production promotion.

## Frozen population

- 2026 Weeks 1-4
- ACT-only historical parity universe
- QB pass attempts
- RB carries
- RB targets
- WR targets
- TE targets
- full football universe, not sportsbook-conditioned

Use the ACT-only player opportunity rows from the frozen availability parity
artifact as the model-side population.

## Model quantities

For every player-game use the already-recorded:

- predicted_team_opportunity_mean
- final_player_probability
- predicted_opportunities
- expected_opportunities_from_probability

Deterministic expected opportunity:

`model_expected = predicted_team_opportunity_mean * final_player_probability`

## Actual team volume

Use completed 2026 Weeks 1-4 regular-season nflverse PBP only after loading the
frozen model rows.

The comparator must match the canonical simulator's exact volume semantics:

- QB pass attempts: official pass attempts =
  pass_attempt=1, sack!=1, two_point_attempt!=1
- target allocation volume: team dropbacks = qb_dropback=1 within canonical
  offensive plays, excluding two-point attempts
- carry allocation volume: canonical non-dropback offensive plays =
  off_play(qb_dropback=1 OR rush_attempt=1) minus dropback plays, excluding
  two-point attempts

This distinction is binding. The explicit simulator samples
`pass_att ~ Binomial(plays, dropback_rate)` for target allocation and uses
`rush_att = plays - pass_att`; therefore official pass attempts are not the
correct team-volume comparator for target allocation.

No sportsbook source.

## Actual player share

- QB: player official pass attempts / team official pass attempts
- RB carry: player carries / canonical team non-dropback opportunity volume
- RB/WR/TE target: player targets / canonical team dropback opportunity volume

Rows with zero realized team volume have undefined actual share and fail closed
for share diagnostics.

## Counterfactual diagnostics

For each row:

- model = predicted team volume * predicted player share
- actual-volume diagnostic = actual team volume * predicted player share
- actual-share diagnostic = predicted team volume * actual player share
- full identity = actual team volume * actual player share

The full identity must reproduce actual player opportunity within 1e-10.

## Required outputs

1. player_opportunity_volume_share_rows.csv
2. player_opportunity_volume_share_group_summary.csv
3. player_opportunity_volume_share_high_workload_summary.csv
4. player_opportunity_volume_share_decomposition_summary.json

## Required group diagnostics

By position and opportunity type report:

- rows
- model MAE and bias
- actual-volume diagnostic MAE and bias
- actual-share diagnostic MAE and bias
- MAE improvement from each diagnostic
- fraction of model MAE removed by each diagnostic
- predicted vs actual team-volume MAE, using the exact simulator-matched
  volume definition for each opportunity family
- predicted vs actual player-share MAE
- correlation between share error and individual opportunity error
- high-workload-bin bias under model, actual-volume diagnostic, and
  actual-share diagnostic
- per-week results

## Integrity gates

Fail closed if:

- ACT-only parent artifact is missing or identity population drifts
- actual team-volume source reads beyond Weeks 1-4
- sportsbook fields appear
- full identity gap exceeds 1e-10
- deterministic model expectation differs from
  expected_opportunities_from_probability beyond 1e-10
- finite-MC sampled allocation parity exceeds the already-certified tolerance
- any parameter is fit
- any correction is promoted

## Interpretation

Use descriptive conclusions only:

- TEAM_VOLUME_DOMINANT
- PLAYER_SHARE_DOMINANT
- MIXED_VOLUME_AND_SHARE
- LOW_RESIDUAL_OR_INCONCLUSIVE

No label is an automatic promotion gate.

## Prohibited

- no parameter fitting
- no threshold selection
- no production promotion
- no paid OddsAPI
- no sportsbook input
- no reopening availability science
- no changes to protected QB/RB/M38/WR-R15/TE-R5P science
- no use of realized volume/share as production features

## Next-step rule

If the player-share diagnostic removes materially more individual error than the
team-volume diagnostic, continue only with pregame individual role/share state.

If team volume dominates, continue only with pregame team play-volume/pass-rate
state.

If both matter, keep them separate and do not collapse them into one fitted
correction.
