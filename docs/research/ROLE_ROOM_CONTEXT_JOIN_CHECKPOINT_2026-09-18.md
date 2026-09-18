# Role + Room Context Join Checkpoint — 2026-09-18

## Disposition

`ROLE_ROOM_CONTEXT_JOIN_ENGINEERING_READY_PENDING_HISTORICAL_EXECUTION`

## Scope

Engineering/QA only. No target outcomes, sportsbook inputs, model fitting, production science changes, or Issue #535 work.

## What changed

Added `scripts/research/build_role_room_context.py` to combine the already-engineered strict-prior player usage-regime surface with the strict-prior team/position-room continuity surface at canonical player-game grain.

The join is deliberately `many_to_one` from:

`season, week, team, player_identity_key`

to room context keyed by:

`season, week, team, position`.

It fails closed on duplicate player-game keys, duplicate room keys, or row-count fanout.

## Qualification-ready diagnostics

The joined surface now emits:

- `stable_identity_flag`
- `pregame_context_eligible_flag`
- `usage_unknown_flag`
- `room_unknown_flag`
- `any_context_unknown_flag`
- `strict_prior_support_games`
- `room_join_state`

These fields are designed to feed the existing candidate-profile and signal-qualification builders without using target-game outcomes.

## Semantic boundary

Room continuity remains descriptive. It measures prior room concentration/continuity and does not assert that a particular player inherits departed opportunity.

Missing room context remains explicit missing/unknown state; it is not converted to zero or neutral football context.

## Tests

Added `tests/test_role_room_context.py` covering:

1. many-to-one join row preservation;
2. support/unknown accounting;
3. explicit missing-room behavior;
4. duplicate player-game fail-closed behavior;
5. duplicate room-key fail-closed behavior.

No GitHub Actions workflow currently runs directly on `research-football-context-program-v1`, so this checkpoint does not claim CI certification.

## Knowledge / next question

The player-regime and room-continuity features can now be evaluated together at the exact player-game grain needed for qualification. The next execution step is to materialize the canonical multi-season historical base, run both upstream materializers, create this joined table, then quantify coverage, strict-prior stability, unknown rates and redundancy before authorizing any predictive experiment.

The central football hypothesis remains unproven but now testable without leakage: historical player usage may deserve less authority when player-level usage regime and surrounding position-room continuity indicate a materially changed opportunity environment.
