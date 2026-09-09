# RB R26K Week-1 Allocation Mechanism Atlas V1 — Frozen Plan

Status: FROZEN BEFORE R26K OUTCOME SLICING
Date: 2026-09-09
Parent production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Research parent predictions: R26 run `34356222339`, artifact `10106271075`, digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`
Source-state parent: R26J run `34374987828`, artifact `10113466373`, digest `sha256:7d886f228487e052ee1617cc5f42b974cd9da6a9d448f3a86b1de82cb4fc4f46`

## Question

R26J established, without outcome access, that 2020 Week-1 vacancy rooms were structurally distinct from 2021-2025. R26K asks:

> Which predeclared, football-coherent R26J source states are associated with the R26 within-room allocation failure, and does that harmful mechanism replicate in comparable rooms outside 2020?

R26K is a diagnostic/mechanism atlas only. It does not create a new projection, router, shadow candidate, or production model.

## Governance

The component-preservation doctrine applies.

Preserve from R26:
- vacancy as useful football state;
- R9 receiving identity mechanics;
- non-vacancy baseline exactness;
- fixed RB receiving pool / RB-room conservation;
- non-RB exactness;
- production receiving-yard means;
- R22 tail behavior;
- sportsbook separation;
- strict-prior source rules.

R26K may not:
- refit R9;
- regenerate R26 predictions;
- change R26D meaningful-exit thresholds;
- exclude 2020;
- alter any production model/runtime;
- add sportsbook inputs;
- add same-week historical depth;
- convert a diagnostic result directly into shadow or production authority.

## Immutable evidence

R26K must verify exact parent artifact digests before execution.

R26 supplies frozen player-level Week-1 baseline/R26 predictions and target-game labels for retrospective grading.

R26J supplies frozen pregame-only Week-1 vacancy-room source state. R26J itself selected zero `actual_*` fields and passed all source-integrity gates.

The room join key is exactly `(season, week, team)`.

Target seasons are exactly 2020-2025, Week 1 only.

## Primary predeclared football states

These definitions are frozen before R26K outcome scoring. They are intentionally simple football states, not tuned thresholds searched against outcomes.

### K1 — LARGE_COMPLEX_ROOM

`current_room_n >= 5 AND continuing_n >= 2 AND entrants_n >= 2`

Interpretation: a large backfield containing multiple holdovers and multiple new claimants.

### K2 — VETERAN_PRESSURE_FLAT_HIERARCHY

`veteran_entry_n >= 1 AND baseline_top_room_share <= 1/3`

Interpretation: at least one established veteran enters a room whose baseline receiving hierarchy has no clear >33.3% leader.

### K3 — HIGH_VACATED_LOAD_FLAT_HIERARCHY

`sum_exit_last8_targets_pg >= 3.0 AND baseline_top_room_share <= 1/3`

Interpretation: at least three combined recent targets/game leave the room while the surviving/current baseline hierarchy is flat.

### K4 — MANY_CLAIMANTS_HIGH_VACATED_LOAD

`continuing_n >= 2 AND entrants_n >= 2 AND sum_exit_last8_targets_pg >= 3.0`

Interpretation: meaningful recent receiving work becomes available while at least four plausible current-room claimants exist across incumbents and entrants.

### K5 — VETERAN_PRESSURE_HIGH_VACATED_LOAD

`veteran_entry_n >= 1 AND sum_exit_last8_targets_pg >= 3.0`

Interpretation: meaningful recent receiving work is vacated while an established veteran arrives.

No threshold above may be changed after R26K results are visible.

## Secondary atomic diagnostics

Report but do not authorize a child design by themselves:
- `current_room_n >= 5`
- `continuing_n >= 3`
- `entrants_n >= 3`
- `veteran_entry_n >= 1`
- `baseline_top_room_share <= 1/3`
- `baseline_room_hhi <= 0.27`
- `sum_exit_last8_targets_pg >= 3.0`
- `exit_history_coverage >= 0.95`

The R26J meaningful-exit definition remains unchanged and may be reported only as frozen context:
`max prior targets/game > 1 OR max prior RB-room target share >= 0.25`.

## Population and endpoints

Primary player population:
- Week 1;
- vacancy-active R26 rooms;
- continuing same-team RB/FB incumbents;
- actual receptions, baseline receptions, and R26 candidate receptions all finite.

Targets are a protected secondary endpoint using the same eligible rows when target values are finite.

For every primary state report:
- incumbent n;
- team-room n;
- baseline/candidate reception MAE;
- reception RMSE;
- reception bias;
- reception p90 absolute error;
- baseline/candidate target MAE;
- RB1 and RB2+ reception MAE;
- room-total reception absolute-error delta;
- summed player-level reception absolute-error delta;
- baseline vs R26 top-incumbent ordering accuracy where at least two labeled incumbents exist.

## Allocation-vs-room-total diagnosis

For each primary state, calculate team-room reception totals using the exact frozen R26 rows with finite actual/baseline/candidate receptions.

A state is considered `ALLOCATION_DOMINANT` when its summed player-level reception absolute-error delta is positive and either:
- room-total reception absolute-error delta is <= 0; or
- room-total delta is <= 25% of the positive summed-player delta.

This definition is frozen.

## Replicated harmful-state qualification

A primary K1-K5 state may authorize a later child-design study only if ALL of the following are true:

1. **2020 support:** at least 8 labeled vacancy-incumbent rows in the state.
2. **2020 harm:** R26 reception MAE is >2% worse than baseline in 2020.
3. **2020 materiality:** the state's signed summed incumbent reception-AE worsening contributes at least 20% of total 2020 R26 Week-1 vacancy-incumbent net worsening.
4. **Outside-2020 support:** at least 20 labeled incumbent rows across 2021-2025 in the state.
5. **Temporal replication:** at least 2 distinct seasons among 2021-2025 show positive mean reception-AE harm for the state.
6. **Outside-2020 pooled direction:** pooled 2021-2025 state reception MAE is worse than baseline (>0% relative worsening).
7. **Allocation mechanism:** pooled state room/player diagnosis is `ALLOCATION_DOMINANT` either in 2020 or pooled 2021-2025; report both.
8. **Integrity:** all immutable-parent, production-boundary, no-refit, no-new-prediction, sportsbook-zero protections pass.

No state can qualify by 2020 behavior alone.

## Ordering diagnostic

Within each `(season, week, team)` state room with >=2 labeled continuing incumbents:
- identify actual top incumbent by receptions, tie-broken deterministically by player key;
- identify baseline projected top incumbent;
- identify R26 projected top incumbent;
- report baseline and R26 top-incumbent accuracy.

This is diagnostic and not an independent child-design gate in V1.

## Dispositions

### `REPLICATED_ALLOCATION_FAILURE_MECHANISM_IDENTIFIED`
At least one K1-K5 state satisfies all eight frozen qualification requirements.

This authorizes design of a later R26L-style child candidate only. It does NOT authorize shadow or production.

### `2020_SPECIFIC_MECHANISM_NO_REPLICATED_ROUTER`
One or more K1-K5 states are materially harmful in 2020 but none satisfy outside-2020 replication.

No child router is authorized from R26K.

### `NO_COHERENT_MECHANISM_IDENTIFIED`
No K1-K5 state clears the frozen 2020 materiality requirements.

No child router is authorized.

## Required outputs

- `r26k_player_state_effects.csv`
- `r26k_primary_state_summary.csv`
- `r26k_primary_state_by_season.csv`
- `r26k_room_allocation_diagnostics.csv`
- `r26k_ordering_diagnostics.csv`
- `r26k_secondary_atomic_diagnostics.csv`
- `r26k_disposition.json`

## Maximum authority

R26K maximum authority is **child-candidate design authorization**.

Regardless of result:
- prospective shadow authorization = false;
- production promotion authorization = false;
- production parameters changed = false.
