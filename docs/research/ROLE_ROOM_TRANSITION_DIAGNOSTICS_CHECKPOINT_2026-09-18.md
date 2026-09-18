# Role/Room Transition Diagnostics — 2026-09-18

Disposition: `ROLE_ROOM_TRANSITION_DIAGNOSTICS_ENGINEERING_READY`

## What changed

Added `scripts/research/build_role_room_transition_diagnostics.py` and fail-closed tests. The builder consumes the strict-prior player+room join and produces outcome-free transition flags plus season/position prevalence summaries.

It characterizes five pregame-known transition dimensions: recent target-share movement, recent rush-share movement, target-room continuity loss, rush-room continuity loss, and team change. It also identifies joint player+room transitions, which are the most direct engineering representation so far of the hypothesis that an individual's old usage history can become stale when both his own role and the surrounding opportunity environment change.

## Scientific boundary

The default 0.05 usage-delta and 0.70 room-overlap cutoffs are **diagnostic definitions only**, not predictive thresholds. They must not be tuned against yards, receptions, carries, betting results, or sportsbook lines. No target outcomes are read by this builder.

Unknown context rows remain explicit and are excluded from transition-rate denominators. Duplicate player-period keys fail closed.

## What this enables

Once canonical multi-season history is materialized, this diagnostic can answer before any predictive experiment:

- how common role/room transition states actually are by position and season;
- whether adequate known-context coverage exists;
- whether joint transitions provide enough sample support for a future frozen cohort;
- which positions have enough transition evidence to justify further qualification work.

This is support/coverage evidence only. It does not establish projection lift.

## Next

Run canonical history through usage regime -> room continuity -> role/room join -> transition diagnostics -> stability/candidate-profile/qualification inventory. If transition cohorts are too sparse or unstable, fail them before predictive science. If sufficiently supported, freeze a separate experiment plan before reading target outcomes.

No production science changed. No predictive experiment ran. No paid odds pull occurred. Issue #535 was not touched. No failed closed family was reopened.
