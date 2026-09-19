# Football Context Event/Regime Qualification V1

**Status:** `FOOTBALL_CONTEXT_EVENT_REGIME_QUALIFICATION_V1_FROZEN`  
**Frozen before any event-regime predictive experiment.**

## Purpose

Qualify football context variables whose meaning is an event or regime transition rather than a persistent player tendency. These signals must not be rejected merely because adjacent-week autocorrelation is low; an injury vacancy, team change, room turnover, or simultaneous player/room transition is expected to be episodic.

This contract is qualification only. It does not inspect predictive target lift and does not authorize production integration.

## Candidate event families

Initial candidates are restricted to information already available from strict-prior canonical context:

1. `team_change_flag` / new-team cold-start state;
2. joint player-usage + room-transition state;
3. target-room churn event;
4. rush-room churn event;
5. returning target-opportunity overlap as an event descriptor;
6. returning rush-opportunity overlap as an event descriptor.

No new paid source acquisition is authorized by this contract.

## Pregame rule

For target kickoff T, every event input must satisfy `information_timestamp < T` or be deterministically reconstructible only from completed prior games / already-known roster state. Target-game usage, target-game tracking, target-game injuries learned after kickoff, final score, and sportsbook outcomes are forbidden.

## Qualification dimensions

### 1. Identity and join integrity — hard gate

- stable-ID coverage >= 0.99 where player identity is applicable;
- duplicate published keys = 0;
- canonical-base fanout = 0;
- ambiguous fuzzy-name matches may not silently pass.

Failure disposition: `REJECTED_INTEGRITY`.

### 2. Event observability / explicit unknown semantics — hard gate

An event must be distinguishable from missing information. Unknown/cold-start states must be explicit.

For broadly applicable event families, pregame known-state coverage >= 0.80 is required for broad use. Lower-coverage signals may remain `ENGINEERING_READY_SOURCE_THIN` if the source semantics are valid and the intended use is explicitly sparse/abstaining.

### 3. Event support — hard gate

The positive event cohort must contain at least:

- 500 historical player-games overall for a broad cross-position event candidate; or
- 250 historical player-games for a position-specific candidate,

across at least 3 seasons, with no single season contributing > 50% of positive events.

This gate prevents a rare transition label from proceeding on anecdotal support.

### 4. Temporal precision — hard gate

The event label must change only when its underlying football state changes under the frozen definition. Mechanical persistence caused by forward-filling an event flag is not acceptable.

Audit:

- event onset count;
- consecutive-event-run distribution;
- first-known-week / cold-start behavior;
- season-boundary reset behavior;
- team-change alignment where applicable.

### 5. Novelty / redundancy versus production state — hard gate for experiment readiness

Compare the event signal with canonical pregame production opportunity/role inputs without reading outcomes.

A candidate cannot be `READY_FOR_FROZEN_EXPERIMENT` if it is effectively reconstructible from existing production state.

For continuous event descriptors, retain the existing holdout reconstructibility convention:

- R2 >= 0.90: redundant;
- 0.75 <= R2 < 0.90: review;
- R2 < 0.75: incremental information survives.

For binary/categorical events, use deterministic out-of-time reconstruction/classification and report balanced accuracy, precision, recall, F1 and prevalence. A candidate is considered highly reconstructible/redundant if holdout balanced accuracy >= 0.90 AND event-class F1 >= 0.80. Values below those thresholds do not prove predictive value; they only establish that incremental event information survives the redundancy gate.

### 6. Mechanism coherence — hard gate

Each candidate must name one intended football mechanism before predictive testing:

- stale-history detection;
- role-regime uncertainty;
- opportunity entitlement transition;
- personnel continuity uncertainty.

No candidate may map directly to a fixed yard/carry/target boost at qualification time.

### 7. Persistence — descriptive only, not a hard event gate

Adjacent-period persistence is reported where informative but is not a pass/fail requirement for event signals. Low persistence is expected for genuine one-time transitions.

## Qualification dispositions

- `READY_FOR_FROZEN_EXPERIMENT`
- `ENGINEERING_READY_SOURCE_THIN`
- `DESCRIPTIVE_ONLY`
- `SOURCE_BLOCKED`
- `REJECTED_INTEGRITY`

`READY_FOR_FROZEN_EXPERIMENT` requires all applicable hard gates above and a documented, non-redundant mechanism. It does not mean predictive lift exists.

## Scientific sequencing

If an event candidate qualifies:

1. freeze a separate predictive plan before reading target outcomes;
2. test the intended mechanism first;
3. for uncertainty/regime hypotheses, evaluate calibration/error dispersion rather than forcing a mean shift;
4. require a primary holdout and untouched temporal replication;
5. freeze global and event-cohort gates in advance;
6. event-subgroup improvement cannot rescue unacceptable global regression;
7. failed V1 hypotheses close and may not be retuned post hoc.

## Relationship to the failed room-concentration V1

The direct room-concentration opportunity experiment is closed as `MECHANISM_FAIL_CLOSED_V1`. This event qualification contract is not a rescue of that hypothesis. It addresses a distinct question: whether discrete regime-change information identifies games where the reliability/representativeness of historical player state changes.

## Immediate execution task

Materialize an event-regime qualification table from the already-built role/room context, quantify event prevalence/support/temporal precision, and perform outcome-free redundancy audits against canonical production state. Do not inspect predictive target outcomes until a candidate earns `READY_FOR_FROZEN_EXPERIMENT` and a separate plan is frozen.
