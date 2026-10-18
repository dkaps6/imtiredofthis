# PRISM Return-from-Injury Role Restoration V1 — Source Result

Date: 2026-10-18

Status: **SOURCE READY / PLAYER-STATE ONLY / NO PERFORMANCE RELATIONSHIP SCORED**

Model: **PRISM — Player Role & Individual Situation Model**

Frozen source plan:
`docs/research/PRISM_RETURN_FROM_INJURY_ROLE_RESTORATION_V1_SOURCE_PLAN.md`

Freeze:
`3eafb971c0a842750fa5af31a77a8b17adf890b0`

Canonical execution:
- run `39162437467` — **SUCCESS**;
- exact head `4e5edc98c95f25800f6f66633443aa79379a2944`;
- job `117778377817` — **SUCCESS**;
- artifact `12149417234`;
- artifact name `prism-return-from-injury-role-restoration-v1-source-39162437467`;
- digest `sha256:7d69111925b7ad48f16b07150f5e5fbfac5e592bd01430d03efd907ca778c51e`;
- artifact live / expired=false at result reconciliation.

Disposition:

`PRISM_RETURN_FROM_INJURY_ROLE_RESTORATION_V1_SOURCE_READY`

## Player-first question

After a named player's participation is interrupted by an injury-corroborated partial game or injury-related no-snap observation, what is the first subsequent positive-participation state relative to that **same player's own pre-interruption role**?

This source audit creates only the player-state label:
- `RETURN_PARTICIPATION_RESTORED`;
- `RETURN_PARTICIPATION_STILL_REDUCED`;
- `NO_RETURN_OBSERVED`.

It does not say what the state predicts yet.

No target, carry, reception, yard, touchdown, fantasy-point, projection-error or sportsbook outcome was used to define the state.

## Immutable parents

Observation-health state:
- run `39154145666`;
- artifact `12146625382`;
- digest `sha256:d74f2d3c37a88aa639527aedf719e7e08001d464c722613b46291cda0f74cbeb`.

Participation-anomaly state:
- run `39160554692`;
- artifact `12147549966`;
- digest `sha256:e7753e981f0ecaaf23289280830fd32b9d733402c02ab9aaa0116fc929450449`.

No provider rebuild was used.

## State semantics

Episode starts only through source-valid injury interruption:
1. `PARTICIPATION_COLLAPSE_INJURY_CORROBORATED`; or
2. `REPORTED_INJURY_NO_OFFENSIVE_SNAPS`.

Pre-interruption role baseline:
- same named player;
- same team;
- same season;
- positive offensive participation only;
- strictly earlier rows;
- latest up to four;
- minimum three;
- median offensive snap percentage.

Return:
- first later scheduled same-team positive-offensive-participation observation.

Inherited major-collapse boundary:
- return retention ratio <= 0.65; **and**
- absolute snap-percentage drop >= 0.20.

If both hold, the player is still reduced.
Otherwise the player's participation is classified restored relative to his own pre-injury role.

This does **not** mean medically 100%, identical route role, identical target role or identical efficiency.

## 2024 source support

- supported episodes: **309**;
- named players: **240**;
- restored: **162 episodes / 146 players**;
- still reduced: **46 / 42**;
- no return observed: **101 / 100**;
- median pre-episode snap pct: **0.605**;
- median restored return snap pct: **0.580**;
- median still-reduced return snap pct: **0.165**.

All frozen source gates passed.

## 2025 source support

- supported episodes: **342**;
- named players: **257**;
- restored: **187 episodes / 154 players**;
- still reduced: **44 / 38**;
- no return observed: **111 / 111**;
- median pre-episode snap pct: **0.562**;
- median restored return snap pct: **0.530**;
- median still-reduced return snap pct: **0.210**.

All frozen source gates passed.

## Meaning for PRISM

This source result exists because PRISM must understand **the player's current state**, not flatten all recent box-score observations into a trend.

A sequence containing:
- injury-limited participation;
- DNP/no-snap;
- and later restored participation

must not automatically become a decline trajectory.

The source is sufficient to distinguish the named player's return-participation state before any performance relationship is tested.

## What this result does NOT authorize

It does not authorize:
- assuming every restored-snap return restores targets/yards immediately;
- deleting all injury games;
- treating a DNP as zero production;
- converting restored participation directly into a receiving-yard multiplier;
- position-wide return coefficients;
- fitting to 2026 results;
- sportsbook-assisted state definition;
- production changes.

## Exact next scientific boundary

No predictive return-state result has been scored yet.

The next legal experiment must be frozen **before outcomes are scored** and ask a named-player question such as:

> Once this specific player has a source-certified `RETURN_PARTICIPATION_RESTORED` observation, should PRISM treat his current role as restored relative to his own pre-interruption role rather than continue carrying the injury interruption forward as decline evidence?

Any such contract must:
- keep the named player as the primary unit;
- use strictly-prior state only;
- preserve teammate competition and finite opportunity;
- separate participation restoration from target earning and efficiency;
- avoid another deletion/reweighting variant of already-closed history rules;
- evaluate individual players directly;
- use position/family summaries only as diagnostics;
- use no 2026 outcomes;
- use no sportsbook/OddsAPI upstream;
- fit/tune nothing after outcomes are seen.

## Integrity

- player performance relationships scored: **0**;
- projections generated: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- paid OddsAPI calls: **0**;
- production changed: **false**.
