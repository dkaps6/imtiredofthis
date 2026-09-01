# RB-ND2A — Pregame Role-State Source Integrity Audit

## Purpose

Audit the historical football-only data needed to reconstruct RB pregame role/backfield allocation after RB-ND1 showed that player-share/backfield allocation accounts for 60.13% of absolute carry decomposition and the authoritative M94C trace contains no populated historical depth role.

ND2A is source/timing/coverage only. It does not fit or score a football model and does not inspect target outcomes for feature selection.

## Sources to audit

For seasons 2024 and 2025, inspect nflverse/nflreadpy:

1. weekly rosters;
2. participation;
3. injuries/practice/game status;
4. depth charts;
5. player metadata if available.

For each source record:

- exact columns/schema;
- row counts;
- season/week/date fields;
- stable player ID fields;
- team/position fields;
- RB/FB coverage;
- whether the data are pregame-current, postgame-only, or safe only after lagging;
- whether a target-week value can be used without leaking the game;
- Week-1/new-player/team-change support.

## Timing contract

- Weekly roster membership/status may be used for a target week only if the source's week semantics are roster-state rather than target-game participation. ND2A must document fields and make no stronger claim than the provider supports.
- Participation is presumed postgame until proven otherwise. Target-week participation must never be used. Prior-week/rolling lagged participation is eligible if player/team identity coverage is adequate.
- Injury/practice/game-status data may be target-week pregame inputs only when the report itself is explicitly tied to that target week and represents a pregame report. No later-week forward fill.
- Depth data are eligible only if a historical row can be assigned to a target game using an observation timestamp/date before kickoff. A season-level/current snapshot may not be backcast.
- Player metadata may identify rookie/no-NFL-history/team-change state but may not import future team assignment.

## Coverage tests

Build a 2025 M94C RB/FB identity universe from frozen run `33353485070` and report how many player-games can receive, without target-game leakage:

- prior-week participation signal;
- rolling prior participation signal;
- current-week roster status;
- current-week injury/practice/game status;
- provably pregame historical depth role, if available;
- prior-season same-player/team continuity;
- new-team flag derived only from already-known roster history;
- no-prior-NFL-history flag.

Also report Week-1 coverage separately.

## Decision

- If lagged participation + current roster/injury + continuity features cover the majority of M94C RB games, advance to ND2B player-share reconstruction using these blocks.
- If trustworthy historical depth snapshots exist with pregame timestamps, add them as a separately ablated block; otherwise explicitly exclude depth rather than silently backcasting.
- If participation lacks useful player-level role semantics or coverage, do not force it; move to roster/injury/usage-only reconstruction and seek an external archived depth source separately.

Sportsbook data are not used in ND2A.