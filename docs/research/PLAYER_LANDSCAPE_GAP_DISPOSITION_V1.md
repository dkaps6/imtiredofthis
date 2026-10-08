# PLAYER LANDSCAPE GAP DISPOSITION V1

Date: 2026-10-07
Branch: `research-player-output-component-decomposition-v1`

## Purpose

Reconcile the open gaps from `PLAYER_LANDSCAPE_TRANSMISSION_AUDIT_V1` against
existing research before opening another model lane.

This is anti-reinvention governance only.

## Resolved gap families

### Route participation / YPRR

Status:
`SOURCE_PARITY_BLOCKED_FOR_HISTORICAL_PREDICTIVE_TEST`

Canonical PlayerForm only populates `route_rate` / `yprr` when a real
`routes` or `routes_run` field exists. Canonical nflverse weekly player
stats do not provide a reliable historical total-routes-run field, and the repo
explicitly prohibits substituting nflverse participation `route` labels for
player routes run.

This applies to RB, WR, and TE historical walk-forward testing.

RB already has an explicit prior disposition:
`LIVE_RB_ROUTE_VOLUME_CONFIRMED_HISTORICAL_WEEKLY_PARITY_NOT_CLEARED`.

No fake route-history bridge is authorized.

### WR-R3 same-player persistence

Status:
`TESTED_AND_CLOSED`

The old overnight note saying the combined candidate was unbuilt became stale.

The actual frozen combined candidate later ran:
- branch `research-wr-r3-combined-calibration`
- run `34726509088`
- artifact `10309091155`
- final disposition:
  `NO_ACTIONABLE_WR_R3_COMBINED_CALIBRATION`

The candidate improved all six season MAEs descriptively, but failed its frozen
hard gates, including the >=1% 2025 MAE improvement requirement and the
20-yard-miss non-worsening guard.

Do not rebuild or retune it.

### Coverage-v2 team-level man/zone

Status:
`TESTED_NEAR_NULL__DO_NOT_REOPEN_AS_SAME_FEATURE`

The production feature-ablation harness already tested the team-level coverage
family. The recovered result was essentially neutral for receiving yards and
receptions.

### Player-level WR-CB exposure

Status:
`SOURCE_PARITY_BLOCKED`

Current-slate WR-CB exposure exists, but no reproducible historical assignment
ground truth exists in the current free stack. Do not infer historical
assignments from nflverse participation fields.

### Generic game script / betting-market confirmation

Status:
`TESTED_NO_ACTIONABLE_PREGAME_STATE`

Existing work established:
- game-level market variables add essentially no incremental improvement to team
  play-count / dropback-rate prediction beyond the historical baseline;
- Vegas line is a real but noisy descriptor of realized game script;
- the subsequent pregame confirmation-likelihood experiment returned:
  `NO_ACTIONABLE_PREGAME_CONFIRMATION_STATE` for margin and total and
  `NO_COMBINED_PROMOTION`.

This does not justify using player-prop or betting-market information upstream.

### Simple football matchup transmission

Status:
`EXACT_V1_FORMULAS_TESTED_AND_CLOSED`

The frozen Football Matchup Transmission integrations all failed:
- RB opponent pass-rate-faced;
- WR true PROE;
- TE opponent pass-success allowed.

The architecture gap remains real, but these exact formulas may not be
resurrected or retuned.

## What remains scientifically unresolved

The current 2026 Weeks 1-4 individual-player replay proves persistent output
error, but the landscape audit alone does not tell us whether each market's
remaining final-output error is primarily:

1. **opportunity / workload**
2. **per-opportunity efficiency**
3. both

That distinction should be resolved before opening another matchup or efficiency
mechanism.

## Next authorized diagnostic

`PLAYER_OUTPUT_COMPONENT_DECOMPOSITION_V1`

Use the already-completed 2026 Weeks 1-4 ACT-only player population and exact
final player projections.

No new model fit.
No feature selection.
No sportsbook inputs.
No production mutation.
