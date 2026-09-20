# Historical Analog State Qualification V1

**Status:** `HISTORICAL_ANALOG_STATE_QUALIFICATION_V1_FROZEN_PRE_OUTCOME`

**Parent:** `d767a79ec058fe334d50cf9e51decd9a61255ef5`

## Why this is a distinct mechanism

The following football-context mechanisms are closed under their frozen V1 definitions and may not be rescued here:

- `ROLE_ROOM_CONCENTRATION_OPPORTUNITY_V1`: direct room-concentration mean/opportunity mechanism failed closed.
- `EVENT_REGIME_RELIABILITY_EXPERIMENT_V1`: no family achieved required temporal replication.
- `RETURNING_OPPORTUNITY_CONTINUITY_EXPERIMENT_V1`: no primary family cleared the frozen 2024 direct-opportunity gate.

Historical analog state is not another scalar room descriptor and is not a retune of those hypotheses. It asks whether the *joint pregame football state* has sufficiently comparable prior player-games to support a local historical prior, or whether the current game is structurally novel.

## Scientific question

Can a strict-prior, outcome-free representation of player role + room structure identify historically comparable player-games without target-game information, sportsbook data, or post-hoc outcome selection?

This document is qualification only. It does not authorize reading predictive outcomes or production integration.

## Canonical inputs

Reuse already-materialized canonical historical player-game and role/room context before acquiring anything new. Candidate state dimensions must be available before target kickoff and may include only strict-prior football context such as:

- position and team;
- prior player target/rush opportunity state;
- prior room top-1/top-2 concentration;
- returning target/rush opportunity overlap;
- explicit team-change / cold-start state;
- qualified player/room transition and churn state;
- prior support count and explicit unknown flags.

No target-game usage, target-game result, final score, target-game tracking learned after kickoff, market/odds field, or sportsbook-derived value is allowed.

## Analog construction

For each target player-game, candidate analogs must come only from chronologically earlier completed player-games. Same-row and future rows are forbidden.

V1 must use a deterministic, preregistered distance/similarity representation. Before any predictive outcome is read, the implementation must freeze:

1. exact feature list;
2. scaling/normalization fit using prior/training data only;
3. categorical mismatch penalties;
4. missing/unknown treatment;
5. candidate-pool eligibility;
6. neighbor count or radius convention;
7. tie-breaking;
8. minimum support required to publish an analog state.

The implementation may materialize outcome-free descriptors such as nearest-neighbor distance, effective analog count, distance-weighted support, and novelty/abstention flag. It may not materialize target-derived neighbor labels during qualification.

## Qualification gates

### 1. Leakage / chronology — hard gate

For every published target row, every analog row must satisfy `analog_kickoff < target_kickoff`. Zero future/same-game analog leakage is permitted.

Failure: `REJECTED_LEAKAGE`.

### 2. Identity / join integrity — hard gate

- stable identity coverage >= 0.99 for eligible player rows;
- duplicate published player-game keys = 0;
- canonical-base fanout = 0;
- ambiguous identity matches cannot silently pass.

Failure: `REJECTED_INTEGRITY`.

### 3. Pregame coverage — hard gate

For each intended position/domain family, at least 0.80 of otherwise eligible rows must either receive a valid analog state or an explicit `NO_ANALOG_SUPPORT` / `UNKNOWN_CONTEXT` state. Missingness may not be encoded as a favorable or neutral analog score.

### 4. Historical support — hard gate

A broadly usable family must have at least 2,000 eligible historical target player-games across at least 4 seasons. Any sparse subgroup proposed for later testing must contain at least 250 target rows across at least 3 seasons before outcome inspection.

### 5. Analog diversity — hard gate

For rows with published analog support, the nearest-neighbor set may not collapse to trivial identity recurrence. Report same-player, same-team, same-season and same-opponent shares. No more than 50% of all neighbor assignments may come from the same player identity unless a separate player-specific mechanism is frozen before outcomes.

### 6. Outcome-free novelty versus production state — hard gate

Quantify whether analog descriptors are reconstructible from canonical production PlayerForm-style opportunity inputs using 2019-2023 fit and 2024-2025 holdout, without reading outcomes.

For continuous analog descriptors:

- holdout R2 >= 0.90: `HIGHLY_RECONSTRUCTIBLE_REDUNDANT`;
- 0.75 <= R2 < 0.90: `REDUNDANCY_REVIEW`;
- R2 < 0.75: incremental information survives.

For binary novelty/abstention states, use the already-frozen classification convention: redundant only if holdout balanced accuracy >= 0.90 **and** event-class F1 >= 0.80.

Surviving redundancy does not imply predictive value.

### 7. Temporal stability — hard gate for experiment readiness

Report analog coverage, median nearest distance, effective analog count and novelty rate separately by season. A candidate cannot proceed if one season supplies >50% of all valid analog assignments or if the descriptor definition changes by season.

## Allowed dispositions

- `READY_FOR_FROZEN_EXPERIMENT`
- `ENGINEERING_READY_SOURCE_THIN`
- `DESCRIPTIVE_ONLY`
- `HIGHLY_RECONSTRUCTIBLE_REDUNDANT`
- `SOURCE_BLOCKED`
- `REJECTED_LEAKAGE`
- `REJECTED_INTEGRITY`

## Predictive sequencing if qualification passes

Only after an analog family reaches `READY_FOR_FROZEN_EXPERIMENT` may a separate predictive experiment be frozen. That future plan must specify one mechanism before reading outcomes, e.g. local analog opportunity prior or novelty-conditioned uncertainty. It must preserve 2019-2023 training, 2024 primary holdout and untouched 2025 replication unless a separately justified pre-outcome split is frozen.

No failed V1 family above may be used to rescue the analog experiment by post-hoc combination.

## Production rule

Qualification alone cannot change production. Any later predictive pass must also survive untouched temporal replication, leakage/identity QA, replay/integration checks, full-stack calibration/regression gates and existing repository governance before promotion.

## Hard prohibitions

- no sportsbook/odds input or paid odds pull;
- no Issue #535 changes;
- no target-game information in pregame state;
- no outcome inspection during qualification;
- no post-hoc distance, neighbor-count, threshold, subgroup or feature tuning;
- no production merge from qualification alone.

## Immediate execution task

Implement an outcome-free historical analog-state materializer and qualification audit using the already-canonical 2019-2025 historical/context data. Freeze exact analog geometry in code/tests before any predictive outcome is read, then report chronology integrity, support, diversity, season stability, and production-state redundancy.
