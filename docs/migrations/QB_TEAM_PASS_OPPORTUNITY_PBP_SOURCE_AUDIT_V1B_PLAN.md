# QB Team Pass Opportunity PBP Source Audit V1B — Frozen Plan

## Purpose

Re-audit only the two still-untested PBP information families from V1 using **family-specific football universes** for the same 95% source-coverage threshold.

This is a new source-audit migration. It does not reinterpret or overwrite V1. V1 remains canonical evidence that an all-PBP-row coverage denominator was too coarse for fields that are structurally undefined on kickoffs, timeouts and special-teams rows.

No target team-pass-opportunity residual, QB attempt residual, passing-yard residual, or sportsbook field may be opened in V1B.

## Lineage

- Parent architecture result: `9b6ef3e087b34d402c6e51a662d203f88898bab7`
- Parent A1 run: `34534125445`
- Parent A1 disposition: `M89_SYNTHESIS_OPPORTUNITY_REALLOCATION_NOT_SUPPORTED`
- Original source audit V1 run: `34533408818`
- Original source audit V1 artifact: `10174433662`
- V1 dispositions:
  - `PENALTY_DRIVE_EXTENSION = SOURCE_INELIGIBLE_SCHEMA_OR_COVERAGE`
  - `FOURTH_DOWN_AGGRESSION = SOURCE_INELIGIBLE_SCHEMA_OR_COVERAGE`
  - `SCHEDULE_REST_CONTEXT = SOURCE_ELIGIBLE_FOR_PREDICTIVE_PREREGISTRATION`

Schedule/rest has already been tested in D1 and failed. It is not part of V1B.

## Why V1B is mechanically distinct from loosening a threshold

The numerical coverage threshold remains exactly `95%`.

The only change is the denominator to which that threshold is applied:

- possession/offensive-play fields are evaluated on rows where an offensive possession team is defined;
- fourth-down decision fields are evaluated on actual fourth-down decision rows;
- sparse penalty-team fields are evaluated conditionally on penalty events when audited.

This is a source-semantics correction, not a target-outcome-driven threshold change. V1B is frozen before either family is compared with the target residual.

## Historical source scope

Use nflverse/nflfastR PBP for regular seasons 2023, 2024 and 2025.

Use the exact 884 M89 2024-2025 target identifiers only to audit strict-prior history availability. Do not load any M89 target outcome field beyond season/week/team/player key.

## Family A — PENALTY_DRIVE_EXTENSION

The only predictive concept authorized for later testing, if source-eligible, is **explicit drive extension by penalty**.

Safe raw source semantics:

- `first_down_penalty`: nflfastR explicit indicator that a penalty converted the first down;
- `posteam`: possession team;
- `defteam`: defense;
- `game_id`, `season`, `week`, `play_id`: keys.

Do **not** infer accepted/declined penalty status from the generic `penalty` flag.

No generic accepted-penalty-rate feature is authorized by V1B.

### Penalty relevant universe

`PENALTY_RELEVANT` = regular-season PBP rows with:

- nonblank `posteam`;
- nonblank `defteam`;
- week 1-18.

Core fields are evaluated for >=95% populated coverage only on this relevant universe.

Derived source-safe team-game quantities may include only:

- offensive `first_down_penalty` events per possession-team PBP row;
- opponent-defense allowed `first_down_penalty` events per possession-team PBP row;
- raw event counts and denominators.

This source audit does not calculate predictive rolling features against target outcomes.

## Family B — FOURTH_DOWN_AGGRESSION

Safe source semantics:

- `down` identifies fourth down;
- `fourth_down_converted` explicitly identifies a fourth-down conversion;
- `fourth_down_failed` explicitly identifies a failed fourth-down attempt;
- `play_type` identifies pass/run/punt/field_goal/no_play;
- `posteam` / `defteam` provide teams.

### Fourth-down relevant universe

`FOURTH_DECISION_RELEVANT` = rows with:

- `down == 4`;
- nonblank `posteam` and `defteam`;
- `play_type` populated;
- `play_type != no_play`.

A source-safe fourth-down go attempt is exactly:

`fourth_down_converted == 1 OR fourth_down_failed == 1`

A source-safe fourth-down decision opportunity is a relevant row with play type in:

- `pass`
- `run`
- `punt`
- `field_goal`
- `qb_kneel`
- `qb_spike`

Derived source-safe team-game quantities may include only:

- go attempts / fourth-down decision opportunities;
- fourth-down conversions / go attempts;
- fourth-down pass go attempts / go attempts;
- fourth-down run go attempts / go attempts;
- opponent-defense allowed versions of the same event counts.

No target residual is permitted in V1B.

## Frozen source gates

A family is `SOURCE_ELIGIBLE_FOR_PREDICTIVE_PREREGISTRATION` only if all applicable gates pass:

1. PBP loads for 2023, 2024 and 2025;
2. required raw fields exist in every season;
3. family-relevant universe is non-empty in every season;
4. all core non-sparse fields are populated on >=95% of the family-relevant universe in every season;
5. event indicators used as binary outcomes are populated on >=95% of the family-relevant universe in every season;
6. any conditional sparse field, if reported, has >=95% coverage on its event-specific denominator;
7. team-game keys are unique after aggregation;
8. >=95% of exact 2024-2025 M89 target identifiers have at least one strictly-prior eligible team game; prior-season history may serve Week 1;
9. no sportsbook/result field is required;
10. target-game outcome/PBP is never required for construction of a target pregame historical-rate feature;
11. family remains materially distinct from the no-retest ledger;
12. source semantics are explicit without guessing accepted/declined or fourth-down opportunity definitions.

## Dispositions

Each family receives exactly one:

- `SOURCE_ELIGIBLE_FOR_PREDICTIVE_PREREGISTRATION`
- `SOURCE_INELIGIBLE_SCHEMA_OR_COVERAGE`
- `DUPLICATE_OR_CLOSED_INFORMATION_FAMILY`
- `MECHANICAL_SOURCE_AUDIT_FAIL`

## If one or both pass

Freeze a separate D2 predictive development plan before opening any relationship with the parent team-pass-opportunity residual.

If both pass, D2 may test both independently under the same single model architecture. A combined model is allowed only if both independently clear their own frozen development gates.

## Stopping rule

- same 95% threshold, no post-result lowering;
- no schedule/rest retest;
- no new family added after results;
- no predictive fitting;
- no target residual correlation;
- no sportsbook data;
- no production changes.
