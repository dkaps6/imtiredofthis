# PRISM Pass-Catcher × QB-Regime Target-CPOE Signal V1 — Frozen Scoring Plan

Date: 2026-10-17

Status: **FROZEN BEFORE TARGET-CPOE OUTCOME SCORING / RESEARCH ONLY**

Parent source:
- run `39087841064` SUCCESS;
- artifact `12121277666`;
- disposition `PRISM_PASS_CATCHER_QB_REGIME_TARGET_CPOE_STATE_V1_SOURCE_READY_ALL`.

## Question

For a named WR/TE whose current QB regime is known pregame:

> Does that receiver's strictly-prior target-CPOE state **with this QB** predict his next-game target CPOE better than the receiver's ordinary strictly-prior target-CPOE state?

This is a pair-conditioned individual conversion-quality state question.

It does not modify target entitlement, target depth, receiving yards, receptions, touchdowns, or any sportsbook-facing output.

## Scientific unit

Primary unit:
**named player-game**, conditional on the target game containing at least one finite native-CPOE target event for that player.

Rows without a finite target-game CPOE event are not converted to zero.

## Frozen state eligibility

Use only source rows with `mechanism_source_ready == true`, which already require:

- stable player identity;
- source-valid selected current QB;
- latest up to 8 strictly-prior target-bearing receiver games selected before CPOE filtering;
- >=4 prior target-bearing receiver games;
- >=16 finite-CPOE ordinary targets;
- >=2 same-QB prior receiver games;
- >=8 same-QB finite-CPOE targets;
- zero chronology violation.

No gate may be weakened after scoring.

## One exact candidate

Exactly **one** candidate is scored.

Baseline:
`prior8_mean_cpoe`

Candidate:
`same_qb_mean_cpoe`

No fitted coefficient, shrinkage, interpolation, blend, alternate window, or subgroup rule.

The point of this direct replacement test is to ask whether current-QB identity defines a more predictive conversion-quality regime for the named receiver.

## Outcome

From regular-season nflverse/nflreadpy PBP with the same target semantics as the source audit:

For each eligible named player-game:
- same player;
- same team;
- target-game target events only;
- finite native `cpoe`;
- `actual_target_cpoe = mean(cpoe)`.

Receiving yards, receptions, touchdowns, target depth and betting markets are not scored.

## Development and confirmation

- 2024 = development;
- 2025 = independent confirmation;
- WR and TE adjudicated separately;
- WR+TE pooling cannot qualify either family.

## Frozen metrics

For every family-season:
- scoreable player-games;
- distinct players;
- baseline MAE;
- candidate MAE;
- paired MAE delta = candidate - baseline;
- baseline RMSE;
- candidate RMSE;
- baseline p90 absolute error;
- candidate p90 absolute error;
- tie rate;
- candidate-closer rate among **non-tied rows only** as a diagnostic;
- baseline signed bias;
- candidate signed bias;
- median equal-player mean MAE delta.

The non-tied closer rate is diagnostic only, not a qualification gate. Structural ties are expected when the same QB owns the receiver's entire recent history.

For 2025 independent confirmation:
- 5,000-replicate player-cluster bootstrap CI of paired MAE delta;
- 5,000-replicate team-week-cluster bootstrap CI of paired MAE delta;
- seed `20261017`.

## Frozen scoring-support floors

Per season:

WR:
- >=650 scoreable player-games;
- >=90 distinct WRs.

TE:
- >=275 scoreable player-games;
- >=40 distinct TEs.

These are scoring-support floors only and do not modify the stricter source contract.

## Qualification gates

A family qualifies only if **all** gates pass.

2024 development:
1. support floor passes;
2. candidate MAE < baseline MAE;
3. candidate RMSE <= baseline RMSE;
4. candidate p90 absolute error <= baseline p90 absolute error;
5. median equal-player MAE delta < 0.

2025 independent confirmation:
6. support floor passes;
7. candidate MAE < baseline MAE;
8. candidate RMSE <= baseline RMSE;
9. candidate p90 absolute error <= baseline p90 absolute error;
10. median equal-player MAE delta < 0;
11. player-cluster 95% CI upper bound for paired MAE delta < 0;
12. team-week-cluster 95% CI upper bound for paired MAE delta < 0;
13. chronology violations = 0.

Dispositions:

Both qualify:
`PRISM_PASS_CATCHER_QB_REGIME_TARGET_CPOE_SIGNAL_V1_QUALIFIED_ALL`

Exactly one qualifies:
`PRISM_PASS_CATCHER_QB_REGIME_TARGET_CPOE_SIGNAL_V1_QUALIFIED_PARTIAL`

Neither qualifies:
`PRISM_PASS_CATCHER_QB_REGIME_TARGET_CPOE_SIGNAL_V1_CLOSED`

## Interpretation boundary

A pass means only that the named receiver's current-QB-conditioned historical CPOE state contains replicated next-game CPOE information beyond his ordinary receiver state.

It does **not** prove:
- more targets;
- more receptions;
- more receiving yards;
- more touchdowns;
- a bettable edge.

Any market translation would require a separate anti-retest audit and separately frozen test.

## Anti-rescue

After scoring do not:
- fit a baseline/same-QB blend;
- change the 8-game window;
- lower 4 games / 16 ordinary targets / 2 same-QB games / 8 same-QB targets;
- use >=3-QB-game-only rescue;
- use QB-change-only rescue;
- use WR1/TE1 carveouts;
- pool WR+TE;
- switch to completion rate, YPT, target depth or catchable-ball charting as a rescue;
- translate this failed exact signal to receiving yards;
- add opponent variables;
- use sportsbook inputs;
- use 2026 outcomes.

## Integrity

- candidate variants: 1;
- coefficients fit: 0;
- thresholds searched: 0;
- target entitlement changed: false;
- target depth changed: false;
- receiving yards scored: false;
- receptions scored: false;
- touchdowns scored: false;
- sportsbook inputs: 0;
- OddsAPI calls: 0;
- 2026 outcomes: 0;
- production changed: false.
