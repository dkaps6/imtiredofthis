# Player Role Elasticity from Teammate Availability V1 — Source / Anti-Retest Audit

Date: 2026-10-15

Status: **FROZEN SOURCE AUDIT / NO PREDICTIVE SCORING YET / RESEARCH ONLY**

Version:
`PLAYER_ROLE_ELASTICITY_TEAMMATE_AVAILABILITY_V1_SOURCE_AUDIT`

## Player-first question

This repository is an individual-player projection system.

The question is not:

> Does an inactive teammate improve the position average?

The question is:

> When a named player's competitive room changes before kickoff, does historical
> evidence allow us to characterize how **that player** tends to absorb or lose
> opportunity?

Examples:
- does a specific RB absorb carries when another RB is inactive, or does the room
  remain a committee?
- does a specific WR absorb targets/routes when a teammate is inactive, or do
  those opportunities move elsewhere?
- does a specific TE expand his route/target role when the receiving room changes?

This is **role elasticity / opportunity reallocation**, not generic injury burden.

## Anti-retest boundary

Do NOT reopen:

- QB M78/M79 generic official-inactive identity:
  exact pregame inactive identity qualified as a source but the frozen QB
  predictive test worsened passing yards, attempts and YPA.
- generic injury burden families;
- RB Vacancy Opportunity V1 deterministic
  `vacated rush share -> successor snap-weight transfer`;
- M96 retrospective RB router/threshold/feature search;
- WR simple vacated-target / higher-usage absence entitlement formulas;
- broad situational target residual V1;
- any postgame inference of absence from zero snaps/carries/targets.

The new information sought here is the **named successor's historical response
to room changes**, not another universal vacancy-transfer coefficient.

The M82 integration ledger explicitly left "new injury consequence information
rather than availability identity" as the legal reopening condition after M79.

## Existing RB evidence preserved

RB Vacancy Opportunity V1 remains a separate prospective lane.

Week-3 frozen result:
- disposition `WEEK3_OBSERVATIONAL_MIXED`;
- rush-attempt MAE improved `4.0279 -> 3.8549`;
- rush-yard MAE improved `28.1177 -> 26.6182`;
- DEN worsened in both scored markets;
- PIT improved in both;
- do not rescue with team-specific tuning.

That mixed team behavior is motivation for studying **successor identity /
role elasticity**, not permission to retune the old formula.

## Source authority candidate

Historical pregame absence identity may use only the already-audited NFL.com
official game-day inactive archive semantics from M78.

Important semantics:
- the official inactive designation is a fixed pre-kickoff fact;
- historical article timestamps are NOT predictive features;
- use only player identity/status from the official inactive list;
- no target-game performance is used to create an absence label.

The source collector already supports listed positions and weekly-roster name
bridging across all positions, even though the prior predictive use was QB.

Target historical seasons for this source audit:
- 2024
- 2025

No 2026 outcomes.

## Candidate positions

Audit separately:
- RB/FB
- WR
- TE

Do not pool positions to rescue support.

## Required historical inputs

### Pregame room-change identity
- season
- week
- team
- official inactive player identity
- listed / weekly-roster position
- stable roster identity if available

### Strictly-prior player role history
For each potentially affected active teammate, source only games before target
week:
- team
- player identity
- position
- carries / rush share for RB/FB
- targets / target share for RB/FB/WR/TE
- receptions where needed for auditing
- offensive snap participation / offense_pct where available
- route participation only if a source contract proves exact historical
  semantics; do not fake routes from PBP labels.

### Target-game outcomes
**Forbidden in this source audit.**

They may be attached only in a separately frozen predictive/mechanism contract
after source readiness passes.

## Room definition

A room is team + position-family:

- RB room = RB + FB
- WR room = WR
- TE room = TE

For every official inactive skill player, identify active same-team room
members from weekly roster authority.

This audit does not assume the inactive player's work transfers one-for-one.

## Role-history support state

For each potential successor, audit whether strictly-prior same-team history can
produce:

1. recent opportunity level;
2. recent opportunity share;
3. recent snap participation;
4. room concentration;
5. the successor's prior history in games with comparable room membership, if
   any.

The fifth item is the core role-elasticity source question.

No response coefficient is calculated in this audit.

## Frozen source gates

Each position is adjudicated separately.

For each season (2024 and 2025), require:

1. >= 90% of official inactive skill-player rows bridge to weekly roster
   identity;
2. >= 95% of bridged rows have an unambiguous position-family mapping;
3. >= 90% of candidate successor rows bridge to stable weekly-roster identity;
4. >= 90% of candidate successor rows have strictly-prior same-team opportunity
   history;
5. >= 85% have strictly-prior snap participation;
6. at least 40 target team-games with an official inactive player in that
   position family for WR;
7. at least 20 target team-games for RB/FB;
8. at least 15 target team-games for TE;
9. at least 50 distinct potential successor players with >=1 room-change event
   for WR;
10. at least 30 for RB/FB;
11. at least 20 for TE;
12. at least 20 successor players per position have >=2 historical room states
   (for example, competitor present vs competitor inactive) before any
   predictive elasticity test;
13. chronology violations = 0;
14. target-game outcomes used = 0;
15. sportsbook inputs = 0;
16. OddsAPI calls = 0;
17. predictive models fit = 0;
18. production changes = 0.

The event/player support floors are frozen before source scoring.

## Readiness dispositions

Per position:

`RB_ROLE_ELASTICITY_SOURCE_READY`
`WR_ROLE_ELASTICITY_SOURCE_READY`
`TE_ROLE_ELASTICITY_SOURCE_READY`

or:

`RB_ROLE_ELASTICITY_SOURCE_BLOCKED`
`WR_ROLE_ELASTICITY_SOURCE_BLOCKED`
`TE_ROLE_ELASTICITY_SOURCE_BLOCKED`

Overall:
- all ready:
  `PLAYER_ROLE_ELASTICITY_SOURCE_READY_ALL`
- some ready:
  `PLAYER_ROLE_ELASTICITY_SOURCE_READY_PARTIAL`
- none ready:
  `PLAYER_ROLE_ELASTICITY_SOURCE_BLOCKED`

## If source-ready

Freeze a separate **named-player role-elasticity mechanism experiment** before
attaching outcomes.

That future test must:
- preserve exact player identity;
- compare the named player's own role under different room states;
- distinguish opportunity quantity from efficiency;
- keep RB carries, RB targets, WR targets and TE targets separate;
- compare against the existing canonical player projection, not a position
  average;
- use per-player heterogeneity and player/game clustered inference;
- keep position aggregate MAE/RMSE as guardrails only;
- fit no post-hoc absence coefficient after outcomes.

A positive result would still not automatically authorize production.

## If blocked

Fail closed.

Do not:
- infer inactive status from target-game participation;
- broaden official inactive to questionable/doubtful after seeing support;
- lower event/player floors;
- pool positions;
- use retrospective target-game roster outcomes to manufacture room state.

Then choose another player-identity source family.

## Integrity boundary

This audit:
- uses no sportsbook data;
- makes no OddsAPI call;
- reads no 2026 outcomes;
- fits no predictive model;
- changes no projection;
- changes no production code.

Canonical model status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.
