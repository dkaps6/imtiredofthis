# PRISM Receiving Opportunity-Earning / Participation Exposure V1 — Frozen Source Plan

Date: 2026-10-18

Status: **FROZEN SOURCE AUDIT ONLY / NO PREDICTIVE SCORING**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_RECEIVING_OPPORTUNITY_EARNING_PARTICIPATION_EXPOSURE_V1_SOURCE_AUDIT`

## 1. Player-first question

PRISM projects named players.

The comparable-observation identity experiment established that completely
deleting an injury-corroborated partial game is not a reliable solution.

That does not make the partial game an ordinary full-game role observation.

A partial game contains two separate football facts:

1. **participation exposure** — how much of the offense the player actually
   participated in;
2. **opportunity earning** — how often he earned receiving opportunities while
   he participated.

This V1 asks only:

> Can PRISM source-validly pair the named player's exact weekly receiving
> opportunity with his exact offensive participation exposure, so later
> research can distinguish "played less" from "earned less while playing"?

No future outcome relationship is scored here.

## 2. Anti-retest boundary

This is distinct from:

### Stage 1 active-role participation
`PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`

That answers:
> Will this named player have any offensive role?

It does not estimate receiving-opportunity earning intensity conditional on
participation.

### Stage 2 conditional allocation
`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL`

That preserves existing finite canonical entitlement and conditions it on the
active set.

It does not reconstruct whether a prior low target total was caused by limited
participation exposure.

### Target Share Trajectory
Do not reopen or retune its recent-share formula.

### Comparable-observation receiving identity V1
`PRISM_COMPARABLE_OBSERVATION_RECEIVING_OPPORTUNITY_IDENTITY_V1_CLOSED`

Do not rescue the exact full-row deletion action.

### Routes/YPRR
Offensive snaps are **not routes**.

This source audit must never label snap exposure as route participation.

## 3. Immutable observation-state authority

Use:

- run `39154145666`;
- artifact `12146625382`;
- name `prism-player-observation-health-state-v1-source-39154145666`;
- digest
  `sha256:d74f2d3c37a88aa639527aedf719e7e08001d464c722613b46291cda0f74cbeb`;
- file `player_observation_health_state_rows.csv`.

Required state fields:
- season;
- week;
- team;
- stable_id;
- family;
- positive_participation;
- offense_snaps;
- offense_pct;
- injury_reported;
- observation_state.

Seasons:
- 2024;
- 2025.

## 4. Immutable participation-anomaly authority

Use:

- run `39160554692`;
- artifact `12147549966`;
- name `prism-player-participation-anomaly-state-v1-source-39160554692`;
- digest
  `sha256:e7753e981f0ecaaf23289280830fd32b9d733402c02ab9aaa0116fc929450449`;
- file `player_participation_anomaly_state_rows.csv`.

This adds:
- same-player prior snap baseline;
- snap retention;
- injury-corroborated major participation collapse;
- unresolved major participation collapse;
- corroboration timing.

No anomaly threshold is changed.

## 5. Exact receiving-opportunity source

Use the current PlayerForm weekly-stat path:

`nflreadpy.load_player_stats(seasons=[season], summary_level="week")`

and the exact current:

`scripts.player_form_v2._normalize_weekly`.

PlayerForm blob authority:

`9246e4daf19f3f01cf61de3ad036b07873fd9561`.

Required fields from the normalized weekly ledger:
- season;
- week;
- team;
- stable player ID;
- position/family;
- targets;
- team_targets.

Do not rebuild target counts from another provider.

## 6. Scope

Families:
- RB;
- WR;
- TE.

Rows:
- certified positive offensive participation;
- scheduled 2024 / 2025 regular-season player-games;
- stable player identity.

A player may have zero targets and still belong in the participation-exposure
panel.

If PlayerForm's positive-usage gate omits a positive-snap player who had zero
targets/rushes/pass attempts, preserve the observation-state row and set:
- `targets = 0`;
- team target total from the exact team/week weekly-stat ledger.

Do not treat source absence as missing player identity.

## 7. Exact join hierarchy

Primary identity:
- season;
- week;
- canonical team;
- stable ID.

No fuzzy names.

No player-name fallback.

The roster/snap state parent is the identity authority.

## 8. Frozen descriptive state

For every source-supported positive-participation player-game persist:

### Full-game receiving opportunity
`target_share_game = targets / team_targets`

when team_targets > 0.

### Offensive participation exposure
- `offense_snaps`;
- `offense_pct`.

### Opportunity earning per offensive snap
`targets_per_offensive_snap = targets / offense_snaps`

when offense_snaps > 0.

This is exactly what it says:
**targets per offensive snap**.

It is not:
- targets per route;
- target rate per pass route;
- first-read rate;
- route participation.

### Participation-adjusted team-target exposure diagnostic

Also persist:

`team_targets_times_snap_share = team_targets * offense_pct`

and, when positive:

`targets_per_snap_scaled_team_target = targets / (team_targets * offense_pct)`.

This is a descriptive diagnostic only. It approximates the player's target
earning relative to team target volume and offensive snap exposure.

It is **not** authorized as a projection formula by this source audit.

## 9. Required anomaly-state linkage

Attach, when available:
- participation anomaly state;
- player prior snap baseline;
- snap retention ratio;
- corroboration source;
- corroboration week.

Ordinary/noncollapse rows remain distinguishable from:
- injury-corroborated collapses;
- unresolved collapses.

No performance field may change the anomaly label.

## 10. Source gates

Adjudicate 2024 and 2025 separately.

1. exact health-state artifact lineage matches;
2. exact anomaly artifact lineage matches;
3. exact PlayerForm blob SHA matches;
4. positive-participation RB/WR/TE stable-ID coverage >=99%;
5. duplicate positive-participation
   `season/week/team/stable_id` identities = 0;
6. exact team target totals available for >=99% of scoped rows;
7. finite positive offense-snap count on >=99% of scoped rows;
8. finite positive offense_pct on >=99% of scoped rows;
9. finite target count on 100% of scoped rows after deterministic zero fill;
10. target_share_game finite when team_targets > 0;
11. targets_per_offensive_snap finite when offense_snaps > 0;
12. injury-corroborated participation-collapse rows represented > 0;
13. unresolved participation-collapse rows represented > 0;
14. no route metric is inferred from snaps;
15. no target-game/future outcome relation scored;
16. no coefficient fit;
17. no threshold search;
18. no 2026 outcomes;
19. no sportsbook/OddsAPI;
20. production changed = false.

Do not lower gates after execution.

## 11. Required reporting

Report by season and family:
- positive-participation player-games;
- distinct named players;
- team-target coverage;
- snap-count coverage;
- snap-percentage coverage;
- zero-target positive-participation rows;
- injury-corroborated collapse rows;
- unresolved collapse rows;
- median offense_pct;
- median targets per offensive snap;
- median target share.

These summaries are source inventory only.

They are not the modeling objective.

## 12. If source-ready

Freeze one predictive named-player mechanism before scoring.

The next question may compare:

> Does separating the named player's expected participation exposure from his
> own target-earning intensity produce a more faithful next-game target
> projection than treating every historical game as equal full-game exposure?

That candidate must:
- use the player's own strict-prior history;
- predict participation exposure and opportunity earning as separate objects;
- preserve teammate competition / finite team targets;
- use no position coefficient as the player's identity;
- keep snaps explicitly labeled as snaps, not routes;
- evaluate named players directly;
- treat family/position summaries as guardrails only.

No formula is authorized by source readiness alone.

## 13. Integrity at freeze

- predictive outcomes scored: **0**;
- projections generated: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_RECEIVING_OPPORTUNITY_EARNING_PARTICIPATION_EXPOSURE_V1_SOURCE_FROZEN`.
