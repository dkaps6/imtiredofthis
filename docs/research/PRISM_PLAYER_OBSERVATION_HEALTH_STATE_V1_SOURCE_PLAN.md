# PRISM Player Observation Health-State V1 — Frozen Source / Lineage Audit

Date: 2026-10-18

Status: **FROZEN SOURCE AUDIT ONLY / NO PREDICTIVE SCORING / NO PROJECTION CHANGE**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_PLAYER_OBSERVATION_HEALTH_STATE_V1_SOURCE_AUDIT`

## 1. Why this is the next distinct player-first lane

The WR/TE component-dominance routing family is now CLOSED ALL.

The active queue must not rescue:
- signed-opportunity bridges whose exact source gates are blocked;
- component-instability / dominance / signed-bias summaries;
- generic mechanism-utility on/off rules;
- raw recent-share replacement;
- blocked depth/room/vacancy sources;
- QB passing-TD backlog work unless the user explicitly promotes it.

A separate player-first problem is upstream of all trajectory logic:

> Is a historical player-game observation actually an ordinary role observation, or was it contaminated by injury, non-participation, or return-from-injury state?

A zero, near-zero, or abrupt usage drop must not automatically become evidence that a named player's role declined.

This V1 asks only whether the repository can source-validly label completed historical player observations before those observations are later used as strictly-prior trajectory/role evidence.

No yards, carries, targets, receptions, touchdown outcome relationship is scored here.

## 2. Relationship to existing science

This is distinct from the QUALIFIED Stage-1 active-role participation mechanism.

Stage-1 predicts whether the **target-game player** will have a meaningful role.

This V1 labels the state of an **already-completed prior observation** before that row enters future player history.

It is also distinct from:
- teammate-availability role elasticity;
- Room Competition State;
- RB/TE depth-role transition;
- injury-driven opponent matchup work;
- settlement/DNP grading rules.

No existing projection mechanism is changed.

## 3. Frozen historical sources

Use nflverse / nflreadpy only.

Historical seasons:
- 2024 REG;
- 2025 REG.

Required source families:

### A. Weekly roster identity
`nflreadpy.load_rosters_weekly(season)`

Required semantics:
- season;
- week;
- team;
- stable player ID, preferring GSIS;
- player name;
- position;
- roster status where exposed.

### B. Offensive participation
`nflreadpy.load_snap_counts(seasons=[2024, 2025])`

Required semantics:
- season;
- week;
- team;
- stable player identity where exposed;
- offensive snaps;
- offensive snap percentage.

### C. Weekly injury report
`nflreadpy.load_injuries(seasons=[2024, 2025])`

Required semantics:
- season;
- week / report_week;
- team;
- stable player identity where exposed, otherwise deterministic team + normalized-player identity under uniqueness gates;
- game/report status;
- practice status;
- injury/body-part field where exposed.

### D. Regular-season schedule gate
`nflreadpy.load_schedules(seasons=[2024, 2025])`

Required semantics:
- season;
- week;
- regular-season game type;
- home team;
- away team.

The scoped weekly-roster universe must be restricted to team-weeks with an
actual regular-season game. Bye-week roster rows are not player-game
observations and must never be labeled as DNP/non-participation.

No sportsbook.
No OddsAPI.
No 2026 rows.

## 4. Frozen position scope

Source audit scope:
- QB;
- RB / HB / TB / FB grouped as RB;
- WR;
- TE.

Other positions are excluded from V1.

This is a source-quality audit for PRISM skill-player history, not a league-wide injury model.

## 5. Identity hierarchy

Primary:
- exact GSIS/stable player ID where the source exposes it.

Fallback permitted only for a source that lacks a stable ID:
- same canonical team;
- deterministic normalized player name;
- exactly one matching weekly-roster identity.

Fallback ambiguity is a hard failure for that row.

Do not fuzzy-match names.

Do not infer identity from outcome statistics.

## 6. Frozen completed-observation state vocabulary

The source audit may construct only these descriptive states:

### `NO_REPORT_PARTICIPATED`
- player is in weekly roster universe;
- positive offensive snaps;
- no matching weekly injury-report row.

### `REPORTED_INJURY_PARTICIPATED`
- player is in weekly roster universe;
- positive offensive snaps;
- matching weekly injury-report row exists.

This state does **not** claim the player was limited.
It only says the game cannot automatically be treated as an ordinary healthy observation.

### `REPORTED_INJURY_NO_OFFENSIVE_SNAPS`
- player is in weekly roster universe;
- matching weekly injury-report row exists;
- offensive snaps are zero or no offensive snap row is present.

This is a reported non-participation observation.
It is not allowed to masquerade as a healthy zero-use game.

### `NO_REPORT_NO_OFFENSIVE_SNAPS_UNRESOLVED`
- player is in weekly roster universe;
- no matching weekly injury-report row;
- offensive snaps are zero or no offensive snap row is present.

This state is unresolved.
Do not call it healthy, inactive, DNP, or role decline without a stronger source.

### Return flag
`return_after_reported_nonparticipation = true`
only when:
- the player's immediately prior same-team weekly-roster observation was
  `REPORTED_INJURY_NO_OFFENSIVE_SNAPS`; and
- the current observation has positive offensive snaps.

The return flag is descriptive only.

## 7. Explicit non-labels

V1 does not claim:
- injury severity;
- early exit;
- snap limitation;
- healthy/full-strength status;
- role decline;
- target-game availability;
- future production effect.

Weekly snap percentage may be persisted as descriptive evidence, but no low-snap threshold may be chosen in this source audit.

A later mechanism must freeze any limited-game definition before predictive scoring.

## 8. Frozen source gates

Adjudicate 2024 and 2025 separately.

### Schedule / weekly roster
1. schedule contains nonempty REG Weeks 1-18;
2. all 32 teams represented across the regular season;
3. every scoped roster observation is an actual scheduled team-game; bye-week rows retained = 0;
4. stable player-ID coverage >= 98% in scoped skill-player rows;
5. position-family mapping coverage >= 99%;
6. duplicate season/week/team/player identities = 0 after deterministic deduplication.

### Snap participation
7. source explicitly contains season + week;
8. stable identity coverage >= 98% where stable ID exists;
9. positive-offensive-snap rows join uniquely to scheduled weekly roster >= 98%;
10. offensive snap count/percentage has at least one finite field on >= 99% of positive-participation rows;
11. duplicate season/week/team/player participation identities = 0 after deterministic deduplication.

### Injury report
12. source explicitly contains season + week/report_week;
13. injury rows join uniquely to scheduled weekly roster >= 95%;
14. >=95% of joined injury rows contain at least one nonblank report-status, practice-status, or injury/body-part field;
15. fallback identity ambiguity rows = 0.

### Constructed state
16. every scoped scheduled weekly-roster row receives exactly one V1 observation state;
17. state counts are nonzero for:
   - `NO_REPORT_PARTICIPATED`;
   - `REPORTED_INJURY_PARTICIPATED`;
   - `REPORTED_INJURY_NO_OFFENSIVE_SNAPS`;
18. return-after-reported-nonparticipation rows > 0;
19. chronology violations = 0.

### Integrity
20. outcome production fields used for predictive scoring = 0;
21. 2026 rows used = 0;
22. sportsbook inputs = 0;
23. OddsAPI calls = 0;
24. predictive models fit = 0;
25. thresholds searched = 0;
26. production changed = false.

Do not lower gates after execution.

## 9. Required outputs

Persist:
- source schema/alias report;
- season/position coverage summary;
- scoped player-week observation-state rows;
- injury identity exceptions;
- duplicate/ambiguity audit;
- return-state counts;
- source result JSON.

Report by season and family:
- roster rows / players;
- injury-report rows;
- positive participation rows;
- each observation-state count/share;
- injury->roster identity coverage;
- positive snap->roster identity coverage;
- return rows.

No performance/yards/targets/carries/TD metrics.

## 10. If source-ready

A pass authorizes only a separate frozen mechanism such as:

> Does excluding or downweighting source-labeled contaminated prior observations
> improve this named player's future role/trajectory estimate relative to the
> existing raw-history state?

That later test must:
- use only strictly-prior labels;
- preserve market specificity;
- compare against existing trajectory/carry-snap/entitlement parents without
  silently reopening them;
- freeze any handling of `REPORTED_INJURY_PARTICIPATED` before outcomes;
- keep unresolved rows fail-closed rather than calling them healthy.

No automatic exclusion/downweighting is authorized by source readiness alone.

## 11. If source-blocked

Fail closed.

Do not:
- infer DNP from a zero stat line;
- treat missing snap rows as definitive inactive without roster/injury support;
- lower identity gates;
- fuzzy-match names;
- invent an injury-limited snap threshold;
- use sportsbook availability as health evidence.

## 12. Integrity at freeze

- historical source rows inspected for predictive relationship: **0**;
- predictive candidates scored: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_PLAYER_OBSERVATION_HEALTH_STATE_V1_SOURCE_AUDIT_FROZEN`.

## 13. Pre-implementation freeze amendment

Before any source code was implemented or any source result observed, the V1
contract was amended to add the regular-season schedule gate. This prevents
bye-week weekly-roster rows from being misclassified as no-participation player
games. No source values or candidate outcomes were inspected before this
amendment.
