# PRISM Target-Share Trajectory Observation-State Bridge V1 — Frozen Source Audit

Date: 2026-10-18

Status: **FROZEN SOURCE / LINEAGE AUDIT ONLY / NO TRAJECTORY OUTCOME SCORING**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_TARGET_SHARE_TRAJECTORY_OBSERVATION_STATE_BRIDGE_V1_SOURCE_AUDIT`

## 1. Question

The existing WR/TE Target Share Trajectory V1 is a qualified player-specific
role mechanism.

Its frozen feature is:

`recent2 target share - earlier same-season target share`

using the latest two completed same-team games before the target week.

The newly certified Player Observation Health-State V1 proves that a prior
player-game can be source-labeled before it enters future player history.

This source audit asks:

> For the exact already-frozen WR/TE trajectory rows, can PRISM identify whether
> either game in the frozen RECENT2 window lacked source-certified positive
> offensive participation for the named player?

This is a source/lineage question only.

It does **not** ask whether neutralizing those rows improves a projection.

## 2. Immutable source authorities

### Existing trajectory state

Run:
`37638269235`

Artifact:
`11490707699`

Name:
`player-target-share-trajectory-v1-37638269235`

Digest:
`sha256:2f032cb5e68f5e097e30ca56a2ebe30e505f90233bc60774e4318158fad3fafd`

Use only source fields:
- season;
- week;
- team;
- player_clean_key;
- receiver_id;
- position_group;
- prior_team_games;
- recent2_team_targets;
- recent2_player_targets;
- recent2_share;
- earlier_share;
- trajectory_delta;
- feature_max_week.

The outcome-derived `opportunity_error` field is forbidden in this audit.

Do not rebuild PBP or recompute trajectory.

### Player observation-state source

Run:
`39154145666`

Artifact:
`12146625382`

Name:
`prism-player-observation-health-state-v1-source-39154145666`

Digest:
`sha256:d74f2d3c37a88aa639527aedf719e7e08001d464c722613b46291cda0f74cbeb`

Disposition:
`PRISM_PLAYER_OBSERVATION_HEALTH_STATE_V1_SOURCE_READY`

Use:
`player_observation_health_state_rows.csv`

The health-state artifact is frozen source state only; it contains no predictive
performance score.

## 3. Scope

Primary source bridge:
- WR 2024;
- TE 2024.

Secondary source bridge:
- TE 2025.

This exactly matches the seasons that can feed the existing historical OOS-fold
trajectory integration:
- 2024 joint WR+TE primary;
- 2025 TE-only secondary.

WR 2023 is outside the first bridge because no new integration score is planned
for that season.

No 2026 rows.

## 4. Exact RECENT2 reconstruction

Do **not** recompute target share.

For every eligible trajectory row:

1. derive the team's completed historical game weeks from the immutable
   observation-state artifact's unique `season + team + week` rows;
2. take team weeks strictly before the trajectory target week;
3. select the latest two team weeks;
4. require they agree with the frozen trajectory chronology:
   - both recent weeks < target week;
   - latest recent week = `feature_max_week`;
5. attach the named player by exact `receiver_id -> stable_id`;
6. inspect only the already-frozen observation-state label and
   `positive_participation` flag.

No fuzzy identity.
No current provider fetch.
No target-game data.

## 5. Frozen recent-window state classes

Classify each of the two recent team-games independently:

### `PARTICIPATION_CERTIFIED`
A matching health-state row exists and:
`positive_participation == true`.

This includes both:
- `NO_REPORT_PARTICIPATED`;
- `REPORTED_INJURY_PARTICIPATED`.

V1 does not claim the latter was full-strength.

### `REPORTED_NO_OFFENSIVE_SNAPS`
Matching row state:
`REPORTED_INJURY_NO_OFFENSIVE_SNAPS`.

### `UNRESOLVED_NO_OFFENSIVE_SNAPS`
Matching row state:
`NO_REPORT_NO_OFFENSIVE_SNAPS_UNRESOLVED`.

### `NOT_ON_SCOPED_ROSTER`
The team-week exists in the certified artifact but the target stable player ID
has no scoped weekly-roster row for that team-week.

This is source state, not a performance inference.

## 6. Frozen row-level bridge labels

For each trajectory target row:

`recent2_participation_certified = true`
only if both RECENT2 team-games are `PARTICIPATION_CERTIFIED`.

Otherwise:

`recent2_observation_contaminated = true`.

Also report the contamination topology:
- recent1 only;
- recent2-older only;
- both recent games;
- reported-no-snap present;
- unresolved-no-snap present;
- not-on-roster present.

The word "contaminated" means only:
> the existing trajectory recent window contains at least one observation that
> is not source-certified as positive offensive participation.

It does **not** mean the player was injured, limited, or should be excluded.

## 7. Source gates

Require:

1. both immutable artifact IDs/names/digests verified exactly;
2. trajectory outcome field `opportunity_error` not loaded;
3. only WR 2024 / TE 2024 / TE 2025 bridge rows used;
4. target stable `receiver_id` nonblank = 100%;
5. every target season/team has certified team-week history in the health
   artifact;
6. every row has exactly two prior team-games available;
7. latest reconstructed recent week equals `feature_max_week` = 100%;
8. recent weeks are strictly before target week = 100%;
9. duplicate trajectory target identities = 0;
10. duplicate health stable-ID/team/week identities = 0 after certified source
    deduplication;
11. recent-window classification coverage = 100%;
12. at least one contaminated target row exists in each scored source cell
    (WR 2024, TE 2024, TE 2025);
13. chronology violations = 0;
14. 2026 outcomes used = 0;
15. sportsbook inputs = 0;
16. OddsAPI calls = 0;
17. predictive models fit = 0;
18. thresholds searched = 0;
19. production changed = false.

Do not lower gates after execution.

## 8. Required source outputs

Report for each cell:
- trajectory rows;
- players;
- clean recent2 rows;
- contaminated recent2 rows;
- contaminated share;
- each contamination topology count;
- each recent-game state count;
- identity / chronology audit.

Also report player-level prevalence:
- players with >=1 contaminated target row;
- median contaminated rows per affected player;
- maximum contaminated rows for one player.

Do not report:
- opportunity error;
- target MAE;
- receiving-yard MAE;
- correlation;
- wagering result;
- any target-game performance comparison.

## 9. If source-ready

A pass authorizes one separately frozen integration experiment.

That experiment may compare:

**baseline**
- the already-qualified original Target Share Trajectory V1;

versus

**candidate**
- the exact same frozen trajectory delta;
- set `trajectory_delta = 0` only when
  `recent2_observation_contaminated == true`;
- otherwise preserve the original trajectory delta byte-for-value.

This is a state-quality gate over an existing qualified mechanism.

It may not:
- change recent2 to another window;
- change the four-prior-team-game eligibility;
- fit a coefficient;
- use injury-report status to scale delta;
- create a low-snap threshold;
- remove `REPORTED_INJURY_PARTICIPATED` merely because an injury report exists;
- search position subsets after outcomes.

## 10. If source-blocked

Fail closed.

Do not rebuild trajectory from current PBP.
Do not lower identity coverage.
Do not infer participation from target counts.
Do not treat missing health-state identity as healthy.

## 11. Integrity at freeze

- predictive relationships scored: **0**;
- target/yard outcomes loaded: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- sportsbook/OddsAPI: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_TARGET_SHARE_TRAJECTORY_OBSERVATION_STATE_BRIDGE_V1_SOURCE_AUDIT_FROZEN`.
