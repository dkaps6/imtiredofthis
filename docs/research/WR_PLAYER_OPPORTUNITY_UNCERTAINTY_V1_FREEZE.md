# WR Player Opportunity Uncertainty V1 — Frozen Diagnostic

Date frozen: 2026-10-14

Status: **FROZEN BEFORE SCORING / RESEARCH ONLY / NO PRODUCTION CHANGE**

## Parent evidence

Exact parent:
- `WR_PLAYER_PERSISTENCE_MIXED_MECHANISM`
- run `37634044900`
- artifact `11487359185`
- digest `sha256:13bb2df4d239f9f2832370520c3d31a08ed4f39c782766fbfb96295edc11c4e3`

That parent established, on 3,224 scoreable rows / 211 WRs:
- signed opportunity persistence: pooled rho `+0.169932`;
- opportunity-difficulty persistence: pooled rho `+0.318305`;
- efficiency-difficulty persistence: pooled rho `+0.283308`;
- signed efficiency persistence failed.

This experiment advances only the strongest surviving individual opportunity signal:
**strictly-prior same-WR opportunity difficulty**.

It does not reopen:
- WR-R3 combined calibration;
- WR-R3 direct bias shrink;
- target-depth distribution transforms;
- WR YPT mean correction;
- generic WR width scaling;
- defense-v-position or direct matchup multipliers.

## Question

Does a named WR's own strictly-prior opportunity difficulty produce a better
mean-neutral uncertainty forecast for his next promoted-authority opportunity
error than a contemporaneous pooled WR uncertainty baseline?

The hypothesis grain is stable player identity.

Position pooling is the baseline comparator / sparse-player fallback, not the
candidate signal.

## Frozen source

Use only:
`wr_player_mechanism_persistence_rows.csv`

from canonical run `37634044900` / artifact `11487359185`.

Required columns:
- season
- week
- team
- player_clean_key
- wr_rank
- prior_games
- prior8_abs_opportunity_error_mean
- current_opportunity_error

Target seasons are exactly 2023 and 2024.

No 2026 outcomes are read.

## Frozen distributions

For each target row define:

`player_mae = prior8_abs_opportunity_error_mean`

Using the zero-mean Normal identity
`E|X| = sigma * sqrt(2/pi)`:

`sigma_player = max(1e-6, player_mae * sqrt(pi/2))`

### Baseline

For each exact target `(season, week)`, compute:

`week_pool_mae = median(player_mae across all eligible WRs in that same pregame week)`

and:

`sigma_baseline = max(1e-6, week_pool_mae * sqrt(pi/2))`

This baseline uses only already-strictly-prior features available before the
target outcomes.

### Candidate

Use `Normal(0, sigma_player)` for the named WR's opportunity-error distribution.

### Comparator

Use `Normal(0, sigma_baseline)`.

Both distributions are centered at zero opportunity error. Therefore this test:
- changes no target mean;
- changes no receiving-yard mean;
- changes no M38/WR-R15 entitlement;
- fits no coefficient;
- changes no production projection.

## Frozen scoring

Per row:
- Normal CRPS;
- central 80% interval coverage;
- central 90% interval coverage.

Primary paired quantity:

`crps_gain = baseline_crps - candidate_crps`

Positive favors named-player uncertainty.

## Frozen support

Each season independently:
- >= 700 eligible rows;
- >= 100 distinct WRs.

Pooled:
- >= 2,000 eligible rows;
- >= 150 distinct WRs.

## Frozen player-first gates

Scientific confirmation requires ALL:

1. support passes;
2. candidate mean CRPS < baseline mean CRPS in 2023;
3. candidate mean CRPS < baseline mean CRPS in 2024;
4. pooled mean CRPS gain > 0;
5. player-cluster bootstrap 95% CI lower bound for pooled CRPS gain > 0;
6. pooled 80% absolute coverage gap from nominal is non-worse;
7. pooled 90% absolute coverage gap from nominal is non-worse;
8. high-difficulty quartile (Q4 by `player_mae`) candidate CRPS < baseline CRPS;
9. WR1 and WR2+ slices are reported descriptively but cannot rescue a failed pooled gate;
10. zero source/integrity violations.

Bootstrap:
- cluster = `player_clean_key`;
- paired rows remain together;
- 5,000 replicates;
- seed `20261014`.

Position-wide metrics remain guardrails. The scientific object is the named
WR's own strictly-prior difficulty state.

## Frozen dispositions

Integrity failure:
`WR_PLAYER_OPPORTUNITY_UNCERTAINTY_INTEGRITY_FAILURE`

Insufficient support:
`WR_PLAYER_OPPORTUNITY_UNCERTAINTY_SOURCE_LIMITED`

All scientific gates pass:
`WR_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`

Otherwise:
`NO_ACTIONABLE_WR_PLAYER_OPPORTUNITY_UNCERTAINTY`

## Anti-rescue

After results are visible do not:
- change last-8/min-4 parent history;
- change seasons;
- replace the same-week median pooled comparator;
- fit shrinkage weights or scale multipliers;
- add caps/floors beyond numerical `1e-6`;
- rescue by WR1/WR2+ subset;
- pool seasons to waive a failed season;
- use sportsbook inputs;
- use 2026 outcomes;
- convert a null into a target-mean correction.

A positive result authorizes only a separately frozen integration/shadow study
inside the real WR opportunity / receiving-yard distribution stack.

Candidate models fit: **0**
Sportsbook inputs: **0**
OddsAPI calls: **0**
2026 outcomes: **0**
Projection changes: **0**
Production changes: **0**
