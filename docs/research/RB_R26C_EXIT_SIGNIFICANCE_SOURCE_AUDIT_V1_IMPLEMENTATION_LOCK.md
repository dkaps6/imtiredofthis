# RB R26C Exit Significance Source Audit V1 — Implementation Lock

Status: **FROZEN BEFORE SOURCE AUDIT EXECUTION**
Date: 2026-09-09

This note resolves implementation details left qualitative in the parent source-audit plan. No R26C source results or target-game outcomes have been inspected before this lock.

## Readiness coverage criteria

In addition to the ten integrity gates in the frozen plan, `EXIT_SIGNIFICANCE_SOURCE_READY` requires:

1. At least **60%** of persisted exited-player rows have one or more strict-prior NFL player games (`prior_games > 0`).
2. At least **60%** of vacancy-active team-weeks have at least one exited RB/FB with one or more strict-prior NFL player games.
3. At least **99.5%** of vacancy-active team-weeks in the immutable parent R26 room-state artifact are reconstructed by this audit.

Prior-depth coverage is diagnostic only and is not a readiness requirement because historical lagged-depth availability is known to be incomplete and 2025 changed source semantics.

## Strict-as-of receiving identity implementation

The audit may use the already-production-safe `rb_receiving_identity_runtime_v1` identity state machinery. It must attach features with the runtime's strict as-of behavior (`allow_exact_matches=False`). Target-game and future rows may exist in the underlying historical source store used to construct the as-of state, but **zero target-game/future outcome values may be selected, merged, emitted, or used to compute any exited-player feature**. The audit will therefore report both:

- `target_game_outcomes_selected_or_used = 0`
- `strict_asof_identity = true`

This clarification preserves the frozen scientific requirement: no target-game outcome can influence the source-state measurement.

## Missing versus zero history

`prior_games` is reconstructed from the runtime's `log1p_prior_games`. A player with `prior_games == 0` is explicitly classified `NO_PRIOR_HISTORY`; zero-valued receiving rates for those rows are never interpreted as observed zero-usage history.

## Prior depth order parsing

`prior_depth_team` is parsed only if it contains an unambiguous integer ordinal. Values <= 0 or unparseable strings are `UNAVAILABLE`. No outcome-based remapping is allowed.
