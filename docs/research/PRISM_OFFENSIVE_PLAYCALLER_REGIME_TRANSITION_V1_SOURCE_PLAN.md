# PRISM Offensive Playcaller Regime Transition V1 — Frozen Source / Lineage Audit

Date: 2026-10-18

Status: **FROZEN SOURCE AUDIT ONLY / NO PREDICTIVE SCORING**

Model:
**PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_OFFENSIVE_PLAYCALLER_REGIME_TRANSITION_V1_SOURCE_AUDIT`

## 1. Player-first question

The current frontier is not asking whether one position's average error improves.

It asks:

> What source-valid football evidence tells PRISM that this named player's
> current role/state is genuinely different from his longer identity now,
> before kickoff?

This V1 audits one deliberately narrow regime object:

> Did this named RB/WR/TE enter the target game under a different primary
> offensive playcaller than his team used in its immediately previous game,
> and can that transition be attached to the named player at pregame
> player + team + week grain with historical/live semantic parity?

The scientific reason this could matter later is player-specific:
a playcaller transition can change how a named player's role is designed,
sequenced and allocated even when the player himself, his position and the
team remain the same.

This audit does **not** score whether any player's carries, targets, receptions,
yards or touchdowns changed.

## 2. Why this is genuinely distinct

This is not a rescue or relabeling of:

- raw recent-share replacement — CLOSED;
- Opportunity Authority Priority V1 — CLOSED;
- WR/TE target-share trajectory — protected;
- RB carry/snap player state — protected;
- Stage-1 active-role participation — QUALIFIED;
- Stage-2 active-set conditional allocation — QUALIFIED PARTIAL;
- teammate-availability role elasticity — SOURCE BLOCKED;
- RB/TE depth-role transition — SOURCE BLOCKED;
- same-QB target allocation — CLOSED;
- same-QB target depth — CLOSED;
- same-QB target CPOE — CLOSED;
- injury-report persistence — CLOSED;
- return-state receiving opportunity bridge — SOURCE BLOCKED.

M68 previously used playcaller information in a **QB residual** research family.
That does not authorize a skill-player rule here.

This V1 uses only the pregame caller-transition state as a source object for
named RB/WR/TE player-weeks. It does not reuse M68's QB outcome models,
opening-script predictors, leverage features or predictive conclusions.

QB is excluded from the target-player scope.

## 3. Historical playcaller authority

Historical 2023-2025 playcaller chronology comes only from the exact frozen
mapping embedded in:

`scripts/backtest/build_qb_playcaller_opening_leverage.py`

Frozen Git blob SHA at this audit boundary:

`6db89b71288c62b3d6a0972a4848dd0449b2524a`

That mapping contains:
- season-opening primary callers for 2023, 2024 and 2025;
- documented midseason effective-week overrides;
- source URLs for those overrides.

The source audit must verify the blob SHA before execution.

Do not rewrite historical caller identity from current web pages.

## 4. 2026 prospective parity authority

Frozen repo snapshot:

`config/prism_playcaller_regime_2026_v1.csv`

Source semantics:
- all-32 season-opening primary offensive playcaller inventory;
- explicit effective week;
- documented midseason overrides only when primary offensive play-calling
  responsibility changed before the target game.

The frozen snapshot includes:
- 32 Week-1 primary caller rows;
- Denver Week-5 override from Davis Webb to Sean Payton.

2026 is parity/readiness only.

No 2026 player outcomes may be read or scored.

The audit may report:
- whether all 32 teams have a Week-1 caller;
- whether effective-week override semantics match the historical contract;
- current maximum effective week represented.

It may not infer that no undocumented change exists merely because no row is
present.

## 5. Named-player universe

Historical player identity authority:

Observation Health-State V1 artifact only:
- run `39154145666`;
- artifact `12146625382`;
- digest
  `sha256:d74f2d3c37a88aa639527aedf719e7e08001d464c722613b46291cda0f74cbeb`;
- file
  `player_observation_health_state_rows.csv`.

Use only identity / chronology fields for the target player universe:
- season;
- week;
- team;
- stable_id;
- family.

Allowed families:
- RB;
- WR;
- TE.

Do not use target-game snaps, targets, carries, receptions, yards, TDs or
observation state to define the current playcaller transition.

Strictly-prior player history support may use prior positive-participation rows
only to count whether the named player has enough same-team history for a later
predictive experiment. Those prior rows are support metadata only.

## 6. Frozen caller state construction

For every scheduled team-week in 2024 and 2025:

`current_playcaller = caller_for(season, week, team)`

`previous_team_game_playcaller =
 caller_for(season, previous_scheduled_team_week, team)`

For Week 1:
- previous team game is the team's final regular-season game from season-1;
- use the prior season's final effective caller from the same historical
  authority.

Frozen state:

`PLAYCALLER_REGIME_CHANGED`
when current caller is nonblank, previous-game caller is nonblank and they
differ.

`PLAYCALLER_REGIME_CONTINUED`
when both are nonblank and equal.

No fuzzy coach identity.

No coordinator title.

Only the primary offensive playcaller matters.

## 7. Frozen player attachment

Attach the team-week caller state to each named RB/WR/TE roster identity using:

`season + week + team`

Persist:
- stable_id;
- family;
- current playcaller;
- previous-game playcaller;
- caller transition state;
- current caller effective start week;
- same-player same-team strictly-prior roster-row count;
- same-player same-team strictly-prior positive-participation row count;
- latest strictly-prior positive-participation week;
- whether the player has at least three strictly-prior same-team
  positive-participation rows.

No player outcome from the target week is permitted.

## 8. Market-specific boundary

This source object may later be tested only at opportunity layers where a
playcaller could reasonably change designed work:

- RB carries;
- RB targets;
- WR targets;
- TE targets.

It is not a universal yards multiplier.

A later receiving-yards or rushing-yards experiment would still require:
- opportunity quantity;
- opportunity type/quality where source-valid;
- player efficiency;
- opponent/environment;
- reliability.

No future predictive test is authorized by this audit alone.

## 9. Frozen historical source gates

Adjudicate 2024 and 2025 separately.

Require all:

1. 32 teams have nonblank current caller identity for every REG team-week;
2. previous-game caller identity coverage >= 99% for Week 2+ team-weeks;
3. Week-1 prior-season final caller coverage = 100% for all 32 teams;
4. duplicate team-week caller identities = 0;
5. caller effective-week chronology violations = 0;
6. at least 10 distinct team-level caller-change events per season when
   offseason Week-1 transitions and documented in-season changes are combined;
7. at least 250 named RB/WR/TE transition player-weeks per season;
8. at least 175 distinct named players exposed to a caller-change row per
   season;
9. stable player ID coverage = 100% in attached transition rows;
10. at least 40% of caller-change player-weeks have >=3 strictly-prior
    same-team positive-participation rows;
11. strictly-prior support chronology violations = 0;
12. target-week offensive snaps used to define caller state = 0;
13. target-week opportunity/performance fields used = 0.

Do not lower gates after execution.

## 10. Frozen 2026 prospective parity gates

Require all:

14. exactly 32 distinct teams have a Week-1 primary caller;
15. duplicate team/effective-week rows = 0;
16. every row has a nonblank source URL and source type;
17. effective week is integer >=1;
18. at least one documented 2026 midseason override exists in the frozen
    snapshot, proving the source representation can express in-season handoff;
19. current 2026 source schema is semantically compatible with historical
    `primary caller + effective week` representation;
20. 2026 player outcomes used = 0.

This is a semantic parity gate, not proof that the snapshot will remain current
forever.

Production use would require event-driven caller-change refresh before each
slate.

## 11. Integrity

21. predictive player outcomes scored = 0;
22. coefficients fit = 0;
23. thresholds searched = 0;
24. sportsbook inputs = 0;
25. OddsAPI calls = 0;
26. paid OddsAPI calls = 0;
27. production changed = false.

## 12. Required reporting

By season:
- REG team-weeks;
- caller identity coverage;
- Week-1 offseason caller-change teams;
- in-season caller-change teams/events;
- total caller-change team-weeks;
- attached RB/WR/TE player-weeks;
- caller-change player-weeks;
- distinct named players on caller-change rows;
- share with >=3 same-team prior positive-participation rows.

By family:
- transition player-weeks;
- distinct named players;
- prior-history support share.

Do not report:
- target change;
- carry change;
- yard change;
- projection error;
- MAE/RMSE;
- betting outcome.

## 13. If source-ready

Disposition:

`PRISM_OFFENSIVE_PLAYCALLER_REGIME_TRANSITION_V1_SOURCE_READY`

Then freeze exactly one predictive mechanism before scoring.

That later question must remain player-first, for example:

> For this named player with a source-certified caller transition and sufficient
> strictly-prior same-team role history, does the first game under the new
> playcaller behave differently from the player's own prior role identity for
> this exact opportunity market?

A future scorer must:
- use 2024 development;
- use 2025 untouched confirmation;
- preserve teammate competition / finite opportunity;
- keep RB carries separate from pass-catcher targets;
- aggregate named player first;
- fit nothing after outcomes are seen.

## 14. If source-blocked

Disposition:

`PRISM_OFFENSIVE_PLAYCALLER_REGIME_TRANSITION_V1_SOURCE_BLOCKED`

Fail closed.

Do not:
- infer caller from coordinator title;
- infer change from offensive statistics;
- lower event-support gates;
- merge caller with QB identity to rescue support;
- use broad coaching-quality grades;
- use fantasy-impact labels from source pages;
- use target-game outcomes to identify the transition.

## 15. Integrity at freeze

Before this freeze:
- skill-player predictive outcomes scored under this mechanism: **0**;
- 2026 outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_OFFENSIVE_PLAYCALLER_REGIME_TRANSITION_V1_SOURCE_AUDIT_FROZEN`.
