# PRISM Player Observation Health-State V1 — Source Result

Date: 2026-10-18

Status: **SOURCE READY / RESEARCH ONLY / NO PREDICTIVE SCORING**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_PLAYER_OBSERVATION_HEALTH_STATE_V1_SOURCE_AUDIT`

Frozen source contract:
`docs/research/PRISM_PLAYER_OBSERVATION_HEALTH_STATE_V1_SOURCE_PLAN.md`

Amended pre-implementation freeze:
`731541d42c82b94cf4261d31d12202ad0911e8e8`

Canonical execution:
- run `39154145666` — **SUCCESS**;
- exact head `2e80a60d8290efd20dacecca849cb271368cde1d`;
- job `117753750451` — **SUCCESS**;
- artifact `12146625382`;
- artifact name `prism-player-observation-health-state-v1-source-39154145666`;
- digest `sha256:d74f2d3c37a88aa639527aedf719e7e08001d464c722613b46291cda0f74cbeb`;
- strict repository audit: **PASS**.

Disposition:

`PRISM_PLAYER_OBSERVATION_HEALTH_STATE_V1_SOURCE_READY`

## Question

Can PRISM source-validly label a completed historical player-game observation
before that observation is allowed to become future role/trajectory evidence?

This is deliberately different from predicting target-game availability.

It asks whether the historical row itself was:
- ordinary participation with no weekly injury report;
- participation while on the weekly injury report;
- reported injury plus no offensive snaps;
- unresolved no-report/no-offensive-snap state;
- a return after a reported no-offensive-snap observation.

No performance relationship was scored.

## Source authority

Historical seasons:
- 2024 REG;
- 2025 REG.

Sources:
- `nflreadpy.load_schedules`;
- `nflreadpy.load_rosters_weekly`;
- `nflreadpy.load_snap_counts`;
- `nflreadpy.load_injuries`.

The schedule amendment was frozen before implementation so bye-week roster rows
could not be misclassified as player-game non-participation.

## Identity / source integrity

Schedule:
- 1,088 regular-season team-weeks across 2024-2025.

Weekly roster:
- post-dedup duplicate identities: **0**;
- 2024 stable-ID coverage: **100.0000%**;
- 2025 stable-ID coverage: **99.9343%**.

Snap source:
- source stable ID is not exposed, so the frozen deterministic team + normalized
  player-name fallback was used under uniqueness gates;
- fallback ambiguity rows: **0**;
- scoped snap rows: **13,951**;
- matched rows: **13,949**;
- overall match coverage: **99.9857%**;
- positive-participation field coverage: **100%**.

Injury source:
- stable ID field: exact `gsis_id`;
- stable-ID coverage: **100%**;
- scoped rows: **3,533**;
- matched rows: **3,532**;
- overall match coverage: **99.9717%**;
- fallback ambiguity rows: **0**;
- joined injury-information coverage: **99.9717%**.

Only one scoped injury row failed roster identity:
- DeAndre Hopkins, TEN, 2024 Week 8.

That single exception remains disclosed and did not violate the frozen >=95%
injury identity gate.

## 2024

- scoped roster player-games: **13,466**;
- distinct scoped players: **996**;
- positive snap rows: **6,579**;
- positive-snap -> roster join: **99.9696%**;
- injury rows: **1,742**;
- injury -> roster join: **99.9426%**;
- joined injury-information coverage: **100%**;
- bye rows retained: **0**.

Observation states:
- `NO_REPORT_PARTICIPATED`: **5,479**;
- `REPORTED_INJURY_PARTICIPATED`: **1,098**;
- `REPORTED_INJURY_NO_OFFENSIVE_SNAPS`: **643**;
- `NO_REPORT_NO_OFFENSIVE_SNAPS_UNRESOLVED`: **6,246**.

Return-after-reported-nonparticipation rows:
**201**.

Every frozen source gate passed.

## 2025

- scoped roster player-games: **13,689**;
- distinct scoped players: **972**;
- positive snap rows: **6,551**;
- positive-snap -> roster join: **100%**;
- injury rows: **1,791**;
- injury -> roster join: **100%**;
- joined injury-information coverage: **99.9442%**;
- bye rows retained: **0**.

Observation states:
- `NO_REPORT_PARTICIPATED`: **5,508**;
- `REPORTED_INJURY_PARTICIPATED`: **1,043**;
- `REPORTED_INJURY_NO_OFFENSIVE_SNAPS`: **748**;
- `NO_REPORT_NO_OFFENSIVE_SNAPS_UNRESOLVED`: **6,390**.

Return-after-reported-nonparticipation rows:
**232**.

Every frozen source gate passed.

## Position-family state support

All four PRISM families have substantial source support in both seasons.

2024:
- QB: 1,867 rows / 130 players / 16 return rows;
- RB: 3,223 / 233 / 48;
- WR: 5,440 / 422 / 88;
- TE: 2,936 / 215 / 49.

2025:
- QB: 1,948 / 131 / 27;
- RB: 3,372 / 235 / 44;
- WR: 5,401 / 400 / 112;
- TE: 2,968 / 209 / 49.

## What this proves

PRISM can now distinguish a prior **observation-state quality label** from the
player's actual football output.

In particular:
- a reported injury + no-offensive-snap week does not need to masquerade as an
  ordinary zero-usage role observation;
- a no-report/no-snap row can remain explicitly unresolved instead of being
  called healthy role decline;
- return-after-reported-nonparticipation state is source-reconstructable;
- ordinary participation and injury-reported participation can remain separate
  descriptive states.

This directly supports the permanent operating rule:

> **Anomalies are questions, not trends.**

## What it does not prove

V1 does **not** label:
- injury severity;
- early exits;
- snap limitation;
- healthy/full-strength status;
- role decline;
- target-game availability;
- future performance effect.

It does not authorize a low-snap threshold.

It does not authorize automatically dropping every injury-report game from
history.

## Next legal question

Freeze a separate source/bridge audit over an already-qualified named-player
role mechanism.

The clean first candidate is the existing WR/TE Target Share Trajectory lineage:

> Does its frozen recent-two target-share state sometimes use a prior team-game
> in which the named player had no source-certified offensive participation,
> and can that contamination be identified exactly before any outcome score?

That next audit must remain source-only first.

If source support exists, a later separately frozen integration test may compare:
- the already-qualified trajectory rule;
- versus the same exact trajectory rule with contaminated recent-state evidence
  neutralized.

No trajectory coefficient/window/threshold may be retuned.

## Integrity

- chronology violations: **0**;
- predictive relationships scored: **0**;
- outcome production fields scored: **0**;
- models fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- sportsbook inputs used: **0**;
- OddsAPI calls: **0**;
- production changed: **false**.
