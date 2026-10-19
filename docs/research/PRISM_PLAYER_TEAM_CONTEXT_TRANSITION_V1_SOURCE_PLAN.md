# PRISM Named-Player Team-Context Transition V1 — Frozen Source / Lineage Audit

Date: 2026-10-18

Status: **FROZEN SOURCE AUDIT ONLY / NO PREDICTIVE SCORING**

Model:
**PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_AUDIT`

## 1. Player-first question

PRISM is an individual-player projection model.

The current frontier asks:

> What source-valid football evidence tells PRISM that this named player's
> current role/state is genuinely different from his longer identity now,
> before kickoff?

This V1 audits one narrow named-player regime object:

> Did this exact named RB/WR/TE change NFL teams before the target game, such
> that his old-team role identity and his current-team competition environment
> are different football states?

A player changing teams is a direct player-state transition.

It is not a position prior.

It is not a teammate absence.

It is not a recent-share transform.

This audit does **not** score whether the player gains or loses carries,
targets, receptions, yards or touchdowns after the move.

## 2. Why this is distinct from protected/closed science

This V1 is not a rescue or relabeling of:

- raw recent-share replacement — CLOSED;
- Opportunity Authority Priority V1 — CLOSED;
- WR/TE target-share trajectory — protected;
- RB carry/snap state — protected;
- active-role participation — QUALIFIED;
- active-set conditional allocation — QUALIFIED PARTIAL;
- teammate-availability Role Elasticity V1 — SOURCE BLOCKED;
- RB/TE depth-role transition — SOURCE BLOCKED;
- same-QB allocation / target-depth / CPOE — CLOSED;
- playcaller regime transition — SOURCE BLOCKED;
- injury-report persistence — CLOSED;
- return-state receiving opportunity bridge — SOURCE BLOCKED.

The existing identity-roster history utilities use team changes for person
resolution. They do not authorize a projection rule for the player's role after
changing teams.

This V1 asks only whether the transition state itself can be reconstructed
cleanly before kickoff.

## 3. Frozen source authority

Primary source family:
- nflverse / nflreadpy weekly rosters.

Historical seasons needed for chronology:
- 2023;
- 2024;
- 2025.

Target historical seasons:
- 2024;
- 2025.

Prospective parity season:
- 2026.

Pregame roster semantics are protected by existing repository authority:
- weekly roster source;
- explicit season + week;
- stable GSIS identity;
- ACT / INA status semantics where available;
- no target-game weekly-stat presence is used to create the roster universe.

The historical input architecture already treats weekly rosters as the
leakage-safe pregame player universe.

## 4. Named-player universe

Allowed target families:
- RB, including HB/TB canonicalized to RB;
- WR;
- TE.

Require:
- stable GSIS player ID;
- canonical team;
- explicit season;
- explicit week;
- one unambiguous team assignment for the named player in the target week.

If the same stable ID appears on more than one team in the same season/week
after exact-row deduplication, that player-week is ambiguous and fails closed.

No name-only player identity is admitted.

## 5. Frozen state construction

Order each named player's weekly roster history chronologically.

For every target player-week:

`current_team = target-week roster team`

`previous_rostered_team = team from the latest strictly-prior weekly roster row`

If no strictly-prior roster row exists:

`TEAM_CONTEXT_NO_PRIOR_ROSTER`

and the row is not called a team transition.

If current team equals previous team:

`TEAM_CONTEXT_CONTINUED`

If current team differs from previous team:

`TEAM_CONTEXT_CHANGED`

Transition kind:

- `OFFSEASON_TEAM_CHANGE` when prior roster row is from an earlier season;
- `INSEASON_TEAM_CHANGE` when prior roster row is from the same season.

The transition is attached to the **first rostered player-week on the new team**
only.

Subsequent weeks on that team are continuation rows.

No target-game snaps, carries, targets, receptions, yards or TDs define the
transition.

## 6. Frozen source metadata

For every transition row persist:

- stable_id;
- family;
- season;
- target week;
- current team;
- previous rostered team;
- previous roster season/week;
- transition kind;
- prior-team roster-week count before transition;
- current-team strictly-prior roster-week count;
- current target-week same-family roster competitor count;
- current target-week same-family competitor stable IDs;
- whether prior-team roster history has >=4 weeks;
- whether prior-team roster history has >=8 weeks.

The competitor set is descriptive source metadata only.

It does not allocate opportunity.

## 7. Current-team competition boundary

A team change necessarily places the named player into a different finite
opportunity environment.

This audit may source-map the current roster competition set.

It may not:
- transfer vacated opportunity;
- boost the new player because a teammate left;
- renormalize target/carry shares;
- reuse blocked teammate-availability elasticity;
- infer depth rank from roster presence.

Any future predictive mechanism must still pass through existing active-role /
finite-opportunity architecture.

## 8. Frozen historical source gates

Adjudicate 2024 and 2025 separately.

Require all:

1. weekly roster source contains explicit season + week;
2. scheduled REG team-week roster coverage >= 95%;
3. stable GSIS ID coverage >= 99% for admitted RB/WR/TE roster rows;
4. ambiguous multi-team stable-ID player-weeks = 0 after exact-row deduplication;
5. previous-roster chronology violations = 0;
6. transition row previous roster week is strictly earlier than target week;
7. at least 40 named-player team-transition events per season;
8. at least 35 distinct named players with a transition per season;
9. at least 10 RB transition players per season;
10. at least 15 WR transition players per season;
11. at least 8 TE transition players per season;
12. >=75% of transition rows have >=4 prior roster weeks with the previous
    team;
13. every transition row has current-team same-family competitor metadata;
14. duplicate transition event identities = 0;
15. target-game player snaps used = 0;
16. target-game opportunity/performance fields used = 0;
17. predictive outcomes scored = 0.

Do not lower gates after execution.

## 9. Frozen 2026 prospective parity gates

Use the same weekly-roster loader and state construction.

Require:

18. 2026 weekly roster source is available;
19. explicit season + week semantics match 2024/2025;
20. stable GSIS ID coverage >= 99%;
21. latest complete observed roster week >= 5;
22. latest complete week covers >= 95% of scheduled REG team-weeks for that
    week;
23. ambiguous multi-team stable-ID player-weeks = 0;
24. at least one source-certified named-player team transition exists in 2026,
    proving the state can be represented prospectively;
25. transition semantics use only roster chronology;
26. 2026 player outcomes used = 0.

This is semantic parity/readiness only.

No 2026 transition outcome may be inspected.

## 10. Integrity gates

27. coefficients fit = 0;
28. thresholds searched = 0;
29. sportsbook inputs = 0;
30. OddsAPI calls = 0;
31. paid OddsAPI calls = 0;
32. production changed = false.

## 11. Required reporting

By season:
- admitted player-weeks;
- stable IDs;
- scheduled team-week roster coverage;
- transition events;
- transition named players;
- offseason vs in-season transitions;
- transition counts by family;
- prior-team >=4-week and >=8-week support;
- competitor metadata coverage;
- ambiguity / chronology audit.

2026:
- latest complete week;
- team-week coverage;
- transition event count only;
- no outcome fields.

Do not report:
- target change;
- carry change;
- yard change;
- projection error;
- MAE/RMSE;
- fantasy outcome;
- betting outcome.

## 12. If source-ready

Disposition:

`PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_READY`

Then freeze one separate market-specific predictive contract before scoring.

The future question must remain player-first:

> For this named player entering a new team context, does his first current-team
> opportunity state behave differently from his own prior-team role identity,
> after accounting for current-team competition and finite opportunity?

A future scorer must:
- use 2024 development;
- use 2025 untouched confirmation;
- keep RB carries separate from RB/WR/TE receiving opportunity;
- preserve teammate competition;
- never use a position coefficient as player identity;
- fit nothing after outcomes are seen.

## 13. If source-blocked

Disposition:

`PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_BLOCKED`

Fail closed.

Do not:
- use name-only identity;
- infer a transition from target-game stats;
- lower event-support gates;
- treat a practice-squad/duplicate-team ambiguity as a clean move;
- use position averages to replace missing player history;
- merge this with teammate vacancy or playcaller state to rescue support.

Move only to another genuinely distinct current role/state source object.

## 14. Integrity at freeze

Before this freeze:
- player transition outcomes scored: **0**;
- player opportunity relationships inspected: **0**;
- 2026 outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_AUDIT_FROZEN`.
