# Role/Room Qualification Execution Lane — 2026-09-19

Disposition: `ROLE_ROOM_QUALIFICATION_PIPELINE_EXECUTABLE_PENDING_CANONICAL_HISTORY`

## What changed

A dedicated execution branch was cut from `research-football-context-program-v1` at `797fd17af902ab91ca3bfb38009019e4f6a95015` so the continuing context program and Claude/Maude work are not overwritten.

Added `scripts/research/run_role_room_qualification_pipeline.py`, which wires the already-frozen engineering surfaces into one deterministic, outcome-free sequence:

1. canonical historical player-game input;
2. strict-prior player usage regime;
3. strict-prior position-room continuity;
4. canonical many-to-one role/room join;
5. transition prevalence diagnostics;
6. adjacent-period strict-prior stability evidence;
7. qualification-ready candidate profiles.

Added `tests/test_role_room_qualification_pipeline.py` to lock the expected builder sequence and canonical integrity fields and to prevent odds/outcome arguments from entering the orchestration path.

## Why this matters

The program's immediate bottleneck is no longer missing feature definitions. It is historical execution. The new runner removes manual handoffs between the individual builders and makes the next canonical-history execution reproducible from one command while preserving the pregame-only scientific firewall.

The first candidate family includes player target/rush share history, room target/rush concentration, and returning-opportunity overlap. These are profiled for coverage, stable identity, explicit unknown state, strict-prior support and adjacent-period persistence before any predictive experiment is permitted.

## Scientific boundary

This lane does not read target-game outcomes, betting results or sportsbook inputs, and it does not fit a model. It cannot promote anything to production by itself.

A candidate may proceed to predictive science only after qualification evidence is produced and a separate experiment plan is frozen before outcomes are inspected.

## Current blocker / next action

There is still no Actions workflow attached directly to the football-context branch, and no canonical multi-season player-game CSV is committed to the repository (by design). The next execution step is to locate a retained canonical historical artifact; if none is retained, deterministically rehydrate the canonical historical base using the existing historical builders, then run this pipeline and classify the resulting signals.

No production science changed. No paid odds pull occurred. Issue #535 was not touched. No failed-closed family was reopened.
