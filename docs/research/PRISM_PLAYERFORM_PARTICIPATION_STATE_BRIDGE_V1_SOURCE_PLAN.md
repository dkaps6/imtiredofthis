# PRISM PlayerForm Participation-State Bridge V1 — Frozen Source Plan

Date: 2026-10-18

Status: **FROZEN SOURCE BRIDGE ONLY / NO PROJECTION COMPARISON**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_PLAYERFORM_PARTICIPATION_STATE_BRIDGE_V1_SOURCE_AUDIT`

## 1. Player-first question

The source-ready participation-anomaly layer established that a positive-usage
game can be materially non-comparable to the named player's own established
participation role.

This bridge asks:

> Which exact named-player positive-participation anomaly rows are currently
> admitted into PlayerForm's historical identity as ordinary weekly evidence?

This is an implementation/source question.

It does not ask whether removing, replacing or downweighting a row improves a
future projection yet.

## 2. Immutable participation-anomaly parent

Use only:

- run `39160554692` — SUCCESS;
- artifact `12147549966`;
- artifact name
  `prism-player-participation-anomaly-state-v1-source-39160554692`;
- digest
  `sha256:e7753e981f0ecaaf23289280830fd32b9d733402c02ab9aaa0116fc929450449`;
- disposition
  `PRISM_PLAYER_PARTICIPATION_ANOMALY_STATE_V1_SOURCE_READY`;
- source file
  `player_participation_anomaly_state_rows.csv`.

Relevant states:
- `PARTICIPATION_COLLAPSE_INJURY_CORROBORATED`;
- `PARTICIPATION_COLLAPSE_UNRESOLVED`.

No thresholds are changed.

## 3. Exact PlayerForm implementation authority

Frozen current implementation:

- file: `scripts/player_form_v2.py`;
- Git blob SHA:
  `9246e4daf19f3f01cf61de3ad036b07873fd9561`.

The bridge must verify this exact blob before execution.

The relevant current semantics are:

### Weekly row admission

`_normalize_weekly` loads weekly player stats and retains a row when:

`targets + rushes + pass_att > 0`

with valid player identity and team.

A positive-usage injury-shortened game can therefore survive this gate.

### Historical aggregation

`_season_totals` aggregates every admitted weekly row for the player's season
identity.

Current PlayerForm does not ingest:
- `participation_anomaly_state`;
- `corroboration_week`;
- injury-corroborated partial-participation state;
- unresolved participation-anomaly state.

This audit must verify the current code has no such state-aware history filter.

## 4. Historical provider source

Use the same weekly-stat provider path as PlayerForm:

`nflreadpy.load_player_stats(seasons=[season], summary_level="week")`

Seasons:
- 2024;
- 2025.

Use the repository's exact `scripts.player_form_v2._normalize_weekly`
implementation to construct the PlayerForm-admitted weekly row ledger.

Do not reimplement the usage admission formula independently.

No projection is built.

## 5. Identity join

Primary:
- exact stable player ID:
  anomaly `stable_id` == PlayerForm `player_id`;
- exact season;
- exact week;
- exact canonical team.

Fallback is not authorized in V1.

Rows without a stable ID are disclosed as unsupported.

No name fuzzy matching.

## 6. Frozen bridge outputs

For every participation-anomaly parent row, persist:

- exact season/week/team/player identity;
- anomaly state;
- corroboration source/week;
- current and prior snap percentages;
- PlayerForm weekly-row present = true/false;
- PlayerForm position where available.

No target, yard, carry, reception, TD, fantasy or projection-error value is
needed in the output.

## 7. Core contamination questions

Report separately for:

### Injury-corroborated collapse
How many source-certified injury-corroborated positive-participation collapses
are admitted into PlayerForm as ordinary weekly history rows?

### Unresolved collapse
How many unresolved major participation collapses are admitted into PlayerForm?

An admitted unresolved row is not proof of injury.

It is proof that current PlayerForm treats an unexplained participation anomaly
as ordinary historical evidence.

## 8. Frozen source gates

Adjudicate 2024 and 2025 separately.

1. exact anomaly parent run/artifact/name/digest match;
2. exact PlayerForm code blob SHA match;
3. PlayerForm weekly provider loads successfully;
4. PlayerForm weekly admitted-row duplicate
   `season/week/team/player_id` identities = 0;
5. stable-ID coverage on anomaly rows >=99%;
6. stable-ID identity reconciliation from anomaly rows to the raw weekly-stat
   provider ledger >=98%;
7. injury-corroborated collapse rows admitted into PlayerForm > 0;
8. unresolved collapse rows admitted into PlayerForm > 0;
9. no anomaly row is joined across season/week/team mismatch;
10. current PlayerForm implementation has no
    `participation_anomaly_state`-aware history branch;
11. no performance outcome is scored;
12. no projection is generated;
13. no 2026 outcomes;
14. no sportsbook/OddsAPI;
15. no fitted coefficient/threshold;
16. production changed = false.

Do not weaken gates after execution.

## 9. Interpretation boundary

A SOURCE READY result establishes only:

> Current PlayerForm admits some player-games that the new player-specific state
> layer says are participation anomalies, without distinguishing them from
> ordinary historical evidence.

It does not prove what the correct replacement rule is.

## 10. If source-ready

Freeze a separate player-identity mechanism before scoring.

The next predictive question must be at named-player grain:

> When building this player's current identity, is a history made from the most
> recent **comparable participation observations** more faithful to his next-game
> role than a history that treats injury-corroborated partial games as ordinary
> evidence?

That future mechanism must:
- preserve DNP/no-snap as non-performance observations;
- obey corroboration as-of timing;
- leave unresolved collapses unresolved unless a separately frozen rule says
  otherwise;
- use the player's own history, not a position coefficient;
- evaluate affected named players directly;
- keep pooled/position summaries as guardrails only.

## 11. Integrity at freeze

- projection outcomes scored: **0**;
- projection generated: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_PLAYERFORM_PARTICIPATION_STATE_BRIDGE_V1_SOURCE_FROZEN`.


## 12. Pre-implementation source-gate amendment

Before implementation or source execution, gate 6 was clarified.

The bridge has two different questions:
- can the anomaly row be identity-reconciled to the raw weekly-stat provider?
- does the exact PlayerForm usage gate admit that row?

A positive-snap skill player may legitimately have zero targets, rushes and pass
attempts, so absence from the **admitted** PlayerForm ledger is not an identity
failure.

Therefore:
- identity coverage is measured against the raw provider ledger;
- PlayerForm admission remains an audited boolean outcome of the exact
  `_normalize_weekly` usage gate;
- no minimum overall admission percentage is imposed;
- the frozen requirement that both injury-corroborated and unresolved anomaly
  classes have admitted rows remains unchanged.

No provider values, bridge counts, or performance outcomes were inspected before
this clarification.
