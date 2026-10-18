# PRISM PlayerForm Participation-State Membership V2 — Frozen Source Plan

Date: 2026-10-18

Status: **FROZEN SOURCE MEMBERSHIP AUDIT ONLY / NO PROJECTION COMPARISON**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_PLAYERFORM_PARTICIPATION_STATE_MEMBERSHIP_V2_SOURCE_AUDIT`

## 1. Why V2 exists

V1 remains:

`PRISM_PLAYERFORM_PARTICIPATION_STATE_BRIDGE_V1_SOURCE_BLOCKED`.

V1 incorrectly used the raw weekly-stat table as though it were a complete
player identity ledger.

It is not.

This V2 does not weaken V1.

It changes the source question to the one actually required by the model:

> Given an already-certified stable player identity for an anomaly observation,
> is that exact observation present in the exact PlayerForm-admitted weekly
> history ledger?

That is a deterministic membership question.

## 2. Immutable anomaly identity authority

Use only:

- run `39160554692`;
- artifact `12147549966`;
- name `prism-player-participation-anomaly-state-v1-source-39160554692`;
- digest
  `sha256:e7753e981f0ecaaf23289280830fd32b9d733402c02ab9aaa0116fc929450449`;
- disposition
  `PRISM_PLAYER_PARTICIPATION_ANOMALY_STATE_V1_SOURCE_READY`;
- file `player_participation_anomaly_state_rows.csv`.

The anomaly source itself is roster-based and stable-ID backed.

Relevant states:
- `PARTICIPATION_COLLAPSE_INJURY_CORROBORATED`;
- `PARTICIPATION_COLLAPSE_UNRESOLVED`.

## 3. Exact PlayerForm implementation

Use the exact current implementation:

- `scripts/player_form_v2.py`;
- Git blob SHA
  `9246e4daf19f3f01cf61de3ad036b07873fd9561`.

Generate the admitted historical weekly ledger through the exact
`_normalize_weekly` function.

Do not reimplement the PlayerForm positive-usage rule in V2.

## 4. Membership key

For every anomaly row with stable ID:

`season + week + canonical team + stable_id`

Membership:

`playerform_row_present = exact key exists in _normalize_weekly output`

There is no fallback join.

No name matching.

No fuzzy identity.

No raw-weekly-provider completeness denominator.

## 5. What present / absent means

### Present

The exact anomaly observation is admitted into PlayerForm's weekly history.

Under the current code, it is then eligible to contribute to the player's
season totals/current identity without a participation-anomaly label.

### Absent

The exact anomaly observation is not admitted by PlayerForm's current weekly
history gate.

V2 does not need to explain why it is absent.

Absence is not an identity failure because identity is already certified by the
roster-based anomaly parent.

## 6. Static implementation audit

Verify the exact PlayerForm code:
- contains `_normalize_weekly`;
- contains `_season_totals`;
- contains its current positive-usage row gate;
- does not contain:
  - `participation_anomaly_state`;
  - `corroboration_week`;
  - injury-corroborated participation-collapse handling;
  - unresolved participation-collapse handling.

This proves that an admitted anomaly row is not currently state-labeled before
historical aggregation.

## 7. Frozen source gates

Adjudicate 2024 and 2025 separately.

1. exact anomaly artifact lineage matches;
2. exact PlayerForm blob SHA matches;
3. anomaly stable-ID coverage >=99%;
4. duplicate anomaly membership keys = 0;
5. duplicate PlayerForm admitted membership keys = 0;
6. injury-corroborated anomaly rows > 0;
7. injury-corroborated anomaly rows admitted by PlayerForm > 0;
8. unresolved anomaly rows > 0;
9. unresolved anomaly rows admitted by PlayerForm > 0;
10. current PlayerForm has no anomaly-state-aware history branch;
11. no projection outcome is scored;
12. no projection is generated;
13. no fitted coefficient;
14. no threshold search;
15. no 2026 outcomes;
16. no sportsbook/OddsAPI;
17. production changed = false.

Do not lower gates after execution.

## 8. Required output

For every anomaly row persist:
- season;
- week;
- team;
- stable ID;
- family;
- anomaly state;
- corroboration source/week;
- prior player snap baseline;
- current snap percentage;
- retention ratio;
- exact PlayerForm row membership;
- PlayerForm position where present.

Report by season and family:
- anomaly rows;
- injury-corroborated rows;
- injury-corroborated admitted rows;
- unresolved rows;
- unresolved admitted rows.

Family is reporting inventory only.

It does not define the rule.

## 9. If source-ready

Source readiness proves a real upstream contamination seam:

> current PlayerForm admits some named-player participation anomalies as ordinary
> historical evidence.

That authorizes one separately frozen **comparable-observation player identity**
experiment.

The future candidate must operate at the individual player level.

It may not be a position correction.

## 10. Next predictive architecture if source-ready

The next candidate should ask:

> For a named player whose history contains an injury-corroborated
> partial-participation observation, should that row update his ordinary current
> role identity the same way as a comparable full-participation observation?

The candidate must be frozen before any future-production score.

Hard boundaries:
- DNP/no-snap rows remain non-performance observations;
- injury-corroborated partial rows and unresolved collapse rows remain separate;
- next-week corroboration obeys as-of timing;
- no position coefficient;
- no post-result window/threshold tuning;
- current role and teammate context remain part of the player state;
- evaluate affected named players directly.

## 11. Integrity at freeze

- future player outcomes scored: **0**;
- projections generated: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_PLAYERFORM_PARTICIPATION_STATE_MEMBERSHIP_V2_SOURCE_FROZEN`.
