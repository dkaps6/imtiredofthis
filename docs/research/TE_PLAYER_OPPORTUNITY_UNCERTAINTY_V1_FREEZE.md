# TE Player Opportunity Uncertainty V1 — Frozen Diagnostic

Date frozen: 2026-10-14

Status: **FROZEN BEFORE SCORING / RESEARCH ONLY / NO PRODUCTION CHANGE**

## Parent evidence

Exact parent:
- `TE_PLAYER_PERSISTENCE_MIXED_MECHANISM`
- run `37631011196`
- artifact `11486082927`
- digest `sha256:1b208531a8cf34735760081b7a232c4cd164a551c325f518d33428a50d7da8ab`

That parent established, on 2,005 scoreable rows / 121 TEs:
- signed opportunity persistence: pooled rho `+0.249982`;
- opportunity-difficulty persistence: pooled rho `+0.393593`;
- efficiency-difficulty persistence: pooled rho `+0.224049`;
- signed efficiency persistence failed.

This experiment advances only the strongest surviving individual signal:
**strictly-prior same-TE opportunity difficulty**.

It does not reopen:
- TE-R5P Width V2;
- global TE width scaling;
- direct historical residual mean correction;
- generic TE YPT correction;
- target-depth width;
- position-vs-defense boosts.

## Question

Does a named TE's own strictly-prior opportunity difficulty produce a better
mean-neutral uncertainty forecast for his next TE-R5P opportunity error than a
contemporaneous pooled TE uncertainty baseline?

The hypothesis grain is the stable player identity.

Position-level pooling is used only as the baseline comparator / fallback, not
as the candidate signal.

## Frozen source

Use only:
`te_player_mechanism_persistence_rows.csv`

from canonical run `37631011196` / artifact `11486082927`.

Required columns:
- season
- week
- team
- player_clean_key
- prior_games
- prior8_abs_opportunity_error_mean
- current_opportunity_error

Target seasons are exactly 2024 and 2025.

No 2026 outcomes are read.

## Frozen distributions

For each target row define:

`player_mae = prior8_abs_opportunity_error_mean`

Because for a zero-mean Normal distribution:

`E|X| = sigma * sqrt(2/pi)`

map MAE to scale without fitting:

`sigma_player = max(1e-6, player_mae * sqrt(pi/2))`

### Baseline

For each exact target `(season, week)`, compute:

`week_pool_mae = median(player_mae across all eligible TEs in that same pregame week)`

and:

`sigma_baseline = max(1e-6, week_pool_mae * sqrt(pi/2))`

This uses only already-strict-prior player-history features that were available
before the target outcomes. It does not use target-game results.

### Candidate

Use `Normal(0, sigma_player)` for that TE's opportunity-error distribution.

### Baseline comparator

Use `Normal(0, sigma_baseline)`.

Both distributions are centered at zero opportunity error. Therefore this test:
- changes no target mean;
- changes no receiving-yard mean;
- changes no TE-R5P entitlement;
- fits no coefficient;
- changes no production projection.

The only difference is individual-vs-pooled uncertainty width.

## Frozen scoring

For each row score the realized `current_opportunity_error` with:
- Normal CRPS;
- central 80% interval coverage;
- central 90% interval coverage.

Primary paired quantity:

`crps_gain = baseline_crps - candidate_crps`

Positive favors individual uncertainty.

## Frozen support

Each season independently must have:
- >= 400 eligible rows;
- >= 50 distinct TEs.

Pooled:
- >= 1,000 eligible rows;
- >= 80 distinct TEs.

## Frozen player-first gates

A scientific confirmation requires ALL:

1. support passes;
2. candidate mean CRPS < baseline mean CRPS in 2024;
3. candidate mean CRPS < baseline mean CRPS in 2025;
4. pooled mean CRPS gain > 0;
5. player-cluster bootstrap 95% CI lower bound for pooled CRPS gain > 0;
6. pooled 80% absolute coverage gap from nominal is non-worse;
7. pooled 90% absolute coverage gap from nominal is non-worse;
8. high-difficulty quartile (Q4 by `player_mae`) candidate CRPS < baseline CRPS;
9. low-difficulty quartile is reported but is not required to improve;
10. zero source/integrity violations.

Bootstrap:
- cluster = `player_clean_key`;
- paired rows stay together;
- 5,000 replicates;
- seed `20261014`.

The aggregate TE metrics are guardrails. The primary scientific object is the
named player's strictly-prior difficulty state and the paired player-clustered
forecast comparison.

## Frozen dispositions

If integrity fails:
`TE_PLAYER_OPPORTUNITY_UNCERTAINTY_INTEGRITY_FAILURE`

If support fails:
`TE_PLAYER_OPPORTUNITY_UNCERTAINTY_SOURCE_LIMITED`

If every scientific gate passes:
`TE_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`

Otherwise:
`NO_ACTIONABLE_TE_PLAYER_OPPORTUNITY_UNCERTAINTY`

## Anti-rescue

After results are visible do not:
- change the last-8/min-4 parent feature;
- change target seasons;
- replace median week pool with another pooled baseline;
- add shrinkage weights;
- cap/floor player difficulty beyond the numerical `1e-6` scale floor;
- tune a multiplier;
- rescue only TE1/TE2 subsets;
- pool seasons to waive a failed season;
- use sportsbook inputs;
- use 2026 outcomes;
- convert a null into a direct target-mean correction.

A positive result only authorizes a separately frozen integration/shadow design
inside the real target-allocation / receiving-yard distribution stack.

Candidate models fit: **0**  
Sportsbook inputs: **0**  
OddsAPI calls: **0**  
2026 outcomes: **0**  
Projection changes: **0**  
Production changes: **0**
