# PRISM Pass-Catcher × QB-Regime Target-Depth Signal V1 — Frozen Scoring Plan

Date: 2026-10-17

Status: **FROZEN BEFORE TARGET-DEPTH OUTCOME SCORING / RESEARCH ONLY**

Parent source:
- source run `39087349029` SUCCESS;
- artifact `12120839869`;
- disposition `PRISM_PASS_CATCHER_QB_REGIME_TARGET_DEPTH_STATE_V1_SOURCE_READY_ALL`.

## Question

For a named WR/TE whose current QB regime is known pregame:

> Does that player's strictly-prior target depth **with this QB** predict his next-game target depth better than the player's ordinary strictly-prior target-depth history?

This is an opportunity-type / target-quality question.

It does not modify target entitlement, receiving efficiency, yards, receptions, touchdowns or any sportsbook-facing output.

## Scientific unit

Primary unit:
**named player-game**, conditional on the target game containing at least one finite-air-yard target event for that player.

The conditionality is explicit: target depth is undefined when the player receives no target with finite `air_yards`. Those rows are not converted into zero depth.

## Frozen state eligibility

Use only source rows with `mechanism_source_ready == true`, which already require:

- stable player identity;
- source-valid selected current QB;
- latest up to 8 strictly-prior same-team target-bearing receiver games;
- >=4 prior receiver games;
- >=10 finite-air-yard prior targets;
- >=2 same-QB prior receiver games;
- >=4 same-QB finite-air-yard prior targets;
- zero chronology violation.

No support gate may be weakened after scoring.

## One exact candidate

There is exactly **one** candidate.

Baseline:
`prior8_mean_air_yards`

Candidate:
`same_qb_mean_air_yards`

No coefficient, shrinkage, interpolation or blend is fit.

This is intentionally a direct test of whether the current QB changes the named receiver's target-depth state.

Target-share/target-count projections are not loaded and cannot change.

## Outcome

From regular-season nflverse/nflreadpy PBP using the same target-event semantics as the source audit:

For each eligible named player-game:
- use only target-game target events for that player/team;
- require finite `air_yards`;
- `actual_target_depth = mean(air_yards)`.

No receiving-yard, reception or touchdown outcome is scored.

## Development and confirmation

- 2024 = development;
- 2025 = independent confirmation;
- WR and TE adjudicated separately;
- no WR+TE pooling can qualify a family.

## Frozen metrics

For each family-season:

- player-game rows;
- distinct players;
- baseline MAE;
- candidate MAE;
- MAE delta = candidate - baseline;
- baseline RMSE;
- candidate RMSE;
- baseline p90 absolute error;
- candidate p90 absolute error;
- candidate-closer rate;
- tie rate;
- baseline signed bias;
- candidate signed bias;
- median equal-player mean MAE delta.

For 2025 confirmation only:
- 5,000-replicate player-cluster bootstrap CI of paired MAE delta;
- 5,000-replicate team-week-cluster bootstrap CI of paired MAE delta;
- seed `20261017`.

## Frozen support floors

Per season:

WR:
- >=750 scoreable player-games;
- >=100 distinct WRs.

TE:
- >=350 scoreable player-games;
- >=50 distinct TEs.

These are scoring-support floors only; source gates remain the stricter source authority and are not modified.

## Qualification gates

A family qualifies only if **every** gate passes.

Development 2024:
1. support floor passes;
2. candidate MAE < baseline MAE;
3. candidate RMSE <= baseline RMSE;
4. candidate p90 absolute error <= baseline p90 absolute error;
5. candidate-closer rate > 0.50;
6. median equal-player MAE delta < 0.

Independent confirmation 2025:
7. support floor passes;
8. candidate MAE < baseline MAE;
9. candidate RMSE <= baseline RMSE;
10. candidate p90 absolute error <= baseline p90 absolute error;
11. candidate-closer rate > 0.50;
12. median equal-player MAE delta < 0;
13. player-cluster 95% CI upper bound for paired MAE delta < 0;
14. team-week-cluster 95% CI upper bound for paired MAE delta < 0;
15. chronology violations = 0.

Dispositions:

Both WR and TE qualify:
`PRISM_PASS_CATCHER_QB_REGIME_TARGET_DEPTH_SIGNAL_V1_QUALIFIED_ALL`

Exactly one qualifies:
`PRISM_PASS_CATCHER_QB_REGIME_TARGET_DEPTH_SIGNAL_V1_QUALIFIED_PARTIAL`

Neither qualifies:
`PRISM_PASS_CATCHER_QB_REGIME_TARGET_DEPTH_SIGNAL_V1_CLOSED`

## Interpretation boundary

A pass means only that QB identity contains replicated information about the **type/depth of targets** a named pass catcher receives.

It does not prove:
- more targets;
- more catches;
- more receiving yards;
- more touchdowns;
- a bettable edge.

Any translation from a qualified target-depth state into receiving yards or another market must be separately frozen and tested later, with opportunity quantity held distinct from efficiency.

## Anti-rescue

After scoring do not:
- blend baseline and same-QB depth with a fitted weight;
- use >=3-only same-QB history;
- change the 8-game window;
- lower the 4-game / 10-target / 2-QB-game / 4-QB-target gates;
- carve out QB changes;
- carve out WR1/TE1;
- pool WR+TE;
- use deep-target rate or depth SD as a rescue candidate;
- translate to YPT/YPR/receiving yards in the same run;
- add opponent variables;
- use sportsbook inputs;
- use 2026 outcomes.

## Integrity

- candidate variants: 1;
- coefficients fit: 0;
- thresholds searched: 0;
- target entitlement changed: false;
- receiving yards scored: false;
- receptions scored: false;
- touchdowns scored: false;
- sportsbook inputs: 0;
- OddsAPI calls: 0;
- 2026 outcomes: 0;
- production changed: false.
