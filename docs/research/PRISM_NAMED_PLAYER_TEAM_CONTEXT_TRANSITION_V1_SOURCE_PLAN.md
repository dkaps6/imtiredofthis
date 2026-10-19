# PRISM Named-Player Team-Context Transition V1 — Frozen Source / Lineage Audit

Date: 2026-10-18

Status: **FROZEN SOURCE AUDIT ONLY / NO PREDICTIVE SCORING**

Model:
**PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_NAMED_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_AUDIT`

## 1. Player-first question

PRISM is an individual-player projection model.

The Current Role-State Arbitration frontier asks:

> What source-valid football evidence tells PRISM that this named player's
> current role/state is genuinely different from his longer identity now,
> before the target game?

This V1 audits one narrow player-context object:

> Did this exact named RB/WR/TE enter the target team-week after changing NFL
> teams relative to his own immediately prior weekly-roster observation?

Football rationale:

A player's old-team history can still describe the player, but a change of team
also changes:
- teammate competition;
- finite opportunity allocation;
- offensive environment;
- role installation / assimilation context.

Therefore old-team history and current-team history are not automatically the
same current-role state.

This audit only source-certifies that transition.

It does not score whether targets, carries, receptions, yards or touchdowns
change after a move.

## 2. Why this is genuinely distinct

This is not a rescue or relabeling of:

- raw recent-share replacement — CLOSED;
- WR/TE target-share trajectory — protected;
- RB carry/snap player state — protected;
- Stage-1 active-role participation — QUALIFIED;
- Stage-2 active-set conditional allocation — QUALIFIED PARTIAL;
- teammate-availability Role Elasticity V1 — SOURCE BLOCKED;
- RB/TE depth-role transition — SOURCE BLOCKED / occupied;
- WR Anchor / Role-Transmission V1 — CLOSED;
- same-QB allocation/depth/CPOE families — CLOSED;
- playcaller-regime transition V1 — SOURCE BLOCKED;
- injury-report persistence — CLOSED;
- return-from-injury participation restoration — qualified but separate.

Existing WR/TE production features contain separate any-team and same-team
strict-prior participation fields. That does not make this source object
redundant.

Those features summarize history.

This V1 asks whether the **current team itself changed for the named player
before the target game**, creating a current football-state boundary.

No existing production feature is changed.

## 3. Source authority

Use nflverse / nflreadpy weekly roster and schedule source only.

Historical source seasons loaded:
- 2023;
- 2024;
- 2025.

2023 exists only to establish strictly-prior roster/team context for 2024.

Historical adjudication seasons:
- 2024;
- 2025.

Prospective semantic-parity season:
- 2026.

Required source families:

### A. Weekly roster
`nflreadpy.load_rosters_weekly(season)`

Required semantics:
- season;
- week;
- team;
- stable player ID, preferring GSIS;
- player name;
- position.

### B. Schedule
`nflreadpy.load_schedules(seasons=[...])`

Required semantics:
- season;
- week;
- regular-season game type;
- home team;
- away team.

No snaps, targets, carries, receptions, yards, TDs, injury outcomes, fantasy
outcomes or sportsbook fields may define the team-transition state.

## 4. Player scope

Families:
- RB / HB / TB / FB -> RB;
- WR;
- TE.

QB is excluded from this V1 because the first downstream seams are skill-player
opportunity markets:
- RB carries;
- RB targets;
- WR targets;
- TE targets.

Rookies / players with no prior NFL weekly-roster observation are not team
transitions.

They are retained as:

`NO_PRIOR_NFL_ROSTER_OBSERVATION`

and are not used to inflate transition support.

## 5. Identity

Primary and required for transition certification:
- nonblank stable player ID.

No name-only transition may be certified.

Names may be retained for diagnostics only.

No fuzzy matching.

No outcome-stat identity repair.

Canonical team aliases may be normalized only through the repository's existing
team canonicalization.

## 6. Frozen chronology

Define a total football chronology by:

`ordinal = season * 100 + week`

For each stable player ID, sort scheduled weekly-roster observations strictly
by ordinal.

For each target observation in 2024-2026:

`previous_roster_observation`

is the immediately prior scheduled weekly-roster observation for that same
stable player ID across all loaded seasons.

Do not skip an intervening roster observation.

Require:

`previous_ordinal < target_ordinal`

The previous observation may be:
- an earlier week in the same season; or
- a prior-season regular-season observation.

No same/future row may be used.

## 7. Frozen team-context state

Exactly three states:

### `TEAM_CONTEXT_CONTINUED`

- exact prior roster observation exists;
- prior team == current team.

### `TEAM_CONTEXT_TRANSITION`

- exact prior roster observation exists;
- prior team != current team.

Persist:
- prior team;
- current team;
- prior season/week;
- target season/week;
- weeks/observations since first current-team roster appearance;
- whether transition occurred at Week 1;
- whether transition occurred in-season.

The transition is descriptive only.

Do not infer:
- trade vs waiver vs free agency;
- depth-chart promotion;
- role size;
- target/carry gain or loss;
- player quality.

### `NO_PRIOR_NFL_ROSTER_OBSERVATION`

- no strictly-prior loaded NFL weekly-roster observation exists for the stable
  player.

This is not treated as a team change.

## 8. Event de-duplication

A player can remain on his new team for many weeks.

The **transition event** is only the first scheduled roster observation whose
current team differs from the player's immediately prior roster-observation
team.

Subsequent same-team rows are:
`TEAM_CONTEXT_CONTINUED`.

A later move to another team creates a new transition event.

One player can therefore have more than one transition event across seasons,
but never more than one event at the same target team-week.

## 9. Strictly-prior support metadata

For every certified transition event, report without scoring outcomes:

- count of all strictly-prior scheduled roster observations;
- count of strictly-prior observations with the prior team;
- count of current-team scheduled roster observations before the target week;
- latest prior roster season/week/team;
- whether >=3 prior-team roster observations exist;
- whether >=8 prior-team roster observations exist.

These are source-support descriptors only.

They are not participation, opportunity or performance evidence.

A later predictive experiment must separately source-validly attach the exact
market-specific opportunity history it intends to use.

## 10. Historical source gates

Adjudicate 2024 and 2025 separately.

Require all:

1. schedule contains REG Weeks 1-18;
2. all 32 teams represented across regular season;
3. weekly roster scheduled-team-week coverage >= 95%;
4. scoped RB/WR/TE stable-ID coverage >= 99%;
5. duplicate `season + week + team + stable_id` identities = 0 after
   deterministic exact de-duplication;
6. certified transition rows have stable ID coverage = 100%;
7. chronology violations = 0;
8. skipped-prior-observation violations = 0;
9. at least 25 certified team-context transition events;
10. at least 20 distinct named players with a certified transition event;
11. RB transition events > 0;
12. WR transition events > 0;
13. TE transition events > 0;
14. transition rows with >=3 prior-team roster observations >= 75%;
15. no-prior-roster state count > 0 and remains separate from transitions;
16. target-game snaps used = 0;
17. target-game opportunity/performance fields used = 0.

Do not lower these gates after execution.

## 11. 2026 prospective semantic-parity gates

2026 is source/parity only.

No 2026 player outcome may be read or scored.

Require:

18. weekly roster source loads with the same stable-player/team/week semantics;
19. schedule source loads REG team-weeks with the same semantics;
20. at least Weeks 1-6 have scheduled-team roster coverage >= 95%;
21. scoped RB/WR/TE stable-ID coverage >= 99%;
22. duplicate `season + week + team + stable_id` identities = 0;
23. at least one certified 2026 team-context transition event exists;
24. chronology violations = 0;
25. target-game player outcomes used = 0.

This is semantic readiness only.

Production use would still require the target slate's current weekly-roster
snapshot to be refreshed and certified before kickoff.

## 12. Required reporting

Historical, by season:
- scheduled team-weeks;
- roster team-week coverage;
- scoped player-week rows;
- distinct named players;
- stable-ID coverage;
- state counts;
- transition events;
- transition named players;
- Week-1 vs in-season transition events;
- prior-team support >=3 and >=8;
- family transition counts / players / prior-support shares.

2026:
- source max week;
- complete scheduled weeks;
- roster coverage by week;
- transition events / named players;
- schema / stable-ID coverage;
- chronology and duplicate audit.

Persist:
- normalized scheduled weekly-roster rows;
- named-player team-context rows;
- transition-event rows;
- season/family source summary;
- source-result JSON.

Do not report:
- target change;
- carry change;
- yard change;
- projection error;
- MAE/RMSE;
- betting outcome.

## 13. If source-ready

Disposition:

`PRISM_NAMED_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_READY`

Do not change projections.

Freeze a separate market-specific mechanism before scoring.

The first legal predictive question would be:

> For this named player after a source-certified team-context transition, which
> part of his old-team opportunity identity remains predictive while his
> current-team opportunity identity is still sparse?

That later experiment must:
- choose one market seam before scoring;
- use 2024 development;
- use 2025 independent confirmation;
- source-validly attach exact market opportunity history;
- preserve current-team teammate competition / finite opportunity;
- keep old-team player identity separate from current-team allocation;
- fit nothing after outcomes are seen;
- use no 2026 outcomes.

No generic “new-team penalty” is authorized.

## 14. If source-blocked

Disposition:

`PRISM_NAMED_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_BLOCKED`

Fail closed.

Do not:
- treat rookies as team transitions;
- name-match missing IDs;
- lower transition-support gates;
- infer role change from target-game performance;
- infer transaction type;
- fit only the season that passes;
- replace the named player with a position/team average.

Move only to another genuinely distinct named-player current role/state object.

## 15. Integrity at freeze

Before this freeze:
- predictive player outcomes scored: **0**;
- player opportunity relationships inspected under this object: **0**;
- target-game snaps used: **0**;
- target-game targets/carries/receptions/yards/TDs used: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 player outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- paid OddsAPI calls: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_NAMED_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_AUDIT_FROZEN`.
