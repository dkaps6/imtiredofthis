# RB High-Value Opportunity Topology V1 — Source / Anti-Retest Audit

Date: 2026-10-15

Status: **SUPERSEDED BEFORE SOURCE ADJUDICATION / DUPLICATE LANE / DO NOT USE FOR SCIENTIFIC DECISION**

Version:
`RB_HIGH_VALUE_OPPORTUNITY_TOPOLOGY_V1_SOURCE_AUDIT`


## Supersession note

Before any source result from this RB-only contract was used for a scientific decision, a broader frozen player-first contract was reconciled:

`docs/research/PLAYER_HIGH_VALUE_SCORING_OPPORTUNITY_IDENTITY_V1_PLAN.md`

That contract is canonical because it preserves separate channels for:
- RB rushing;
- RB receiving;
- WR receiving;
- TE receiving;
- team scoring-opportunity state;
- named-player high-value entitlement;
- room competition.

This RB-only contract is therefore superseded **before adjudication**. Any workflow artifact produced from it is mechanical/source diagnostic only and may not be used to rescue, contradict, or override the canonical broader V1.

Do not compare the two audits and choose the more favorable result.

## Player-first question

The model already estimates RB rushing opportunity at a broad level.

This V1 asks a different question:

> When a football situation exists, which named RB historically owns that
> specific kind of rushing opportunity?

The conceptual decomposition is:

`TEAM CREATES SITUATION -> RB ROOM OPPORTUNITY EXISTS -> NAMED PLAYER SHARE -> PLAYER OUTCOME`.

This separates:
- **team/environment frequency** of reaching a high-value situation;
from
- **named-player ownership** of the opportunity once the situation exists.

That distinction is central to the player-identity architecture.

## Why this is distinct

Do NOT reopen or relabel:

- M96 retrospective RB routing / threshold research;
- generic RB carry-share modeling;
- RB Week-5 carry/snap allocation shadow;
- RB Vacancy Opportunity V1;
- player role-elasticity teammate-availability V1;
- RB run-location x opponent run-location YPC transmission;
- generic RB efficiency / width rescues;
- WR/TE situational-target residual V1;
- any ATD probability claim.

The closed WR/TE situational-target experiment tested whether third-down,
red-zone and two-minute **target shares** explained future receiving opportunity
residuals. This V1 is RB rushing-only and asks whether a named back has a
historically measurable allocation identity **conditional on the team already
having a specific rushing situation**.

No yardage, touchdown or betting outcome is scored in this source audit.

## Historical source

Use nflverse play-by-play for:
- 2023
- 2024
- 2025

Regular season Weeks 1-18 only.

Required PBP fields:
- `season`
- `week`
- `season_type` or equivalent regular-season marker
- `game_id`
- `play_id`
- `posteam`
- `rush_attempt`
- `rusher_player_id`
- `yardline_100`
- `goal_to_go`
- `down`
- `ydstogo`.

Weekly roster identity:
- season
- week
- team
- GSIS/player ID
- position.

Eligible position family:
- RB
- FB
- HB.

QB scrambles and non-RB designed runs are not part of the RB room denominator.

## Frozen situation definitions

All definitions are frozen before source scoring.

### ANY_RB_RUSH
Eligible RB/FB/HB rush attempt.

### RED_ZONE_RUSH
`yardline_100 <= 20`.

### INSIDE_10_RUSH
`yardline_100 <= 10`.

### INSIDE_5_RUSH
`yardline_100 <= 5`.

### GOAL_TO_GO_RUSH
`goal_to_go == 1`.

### SHORT_YARDAGE_RUSH
`down in {3,4}` and `ydstogo <= 2`.

No alternate yardline threshold may be substituted after seeing coverage.

## Player-level topology state

For each team-game-situation:

`player_situation_share = player RB rushes in situation / team RB-room rushes in situation`.

This is descriptive source state only in V1.

Also materialize:
- player situation attempts;
- team RB-room situation attempts;
- number of active RB/FB/HB rushers in that situation;
- room concentration / HHI for the situation;
- player identity;
- team/game/week identity.

Do not use touchdowns or yards to define the state.

## Why this matters

Two backs can have the same total carry projection but radically different
football identities.

Example:
- Player A receives most between-the-20s work but little goal-line work.
- Player B receives fewer total carries but dominates inside the 5.

Those players should not be treated as equivalent across markets.

This source audit does not claim either player will score a touchdown. It only
tests whether the repository can reconstruct the named player's historical
opportunity topology cleanly enough for a future mechanism study.

## Frozen source gates

Every season must independently pass.

1. Regular-season Weeks 1-18 complete in PBP.
2. >=99% of eligible rush-attempt rows have game/team identity.
3. >=99% of candidate RB rush rows have nonblank rusher player ID.
4. >=95% of candidate rusher IDs map to weekly roster position.
5. >=95% of mapped candidate rush rows classify cleanly as RB/FB/HB.
6. >=100 distinct RB/FB/HB players have >=4 ANY_RB_RUSH games.
7. >=60 distinct players have >=3 RED_ZONE_RUSH games.
8. >=40 distinct players have >=2 INSIDE_10_RUSH games.
9. >=25 distinct players have >=2 INSIDE_5_RUSH games.
10. >=30 distinct players have >=2 GOAL_TO_GO_RUSH games.
11. >=40 distinct players have >=2 SHORT_YARDAGE_RUSH games.
12. >=200 team-games contain at least one RED_ZONE_RUSH.
13. >=150 team-games contain at least one INSIDE_10_RUSH.
14. >=100 team-games contain at least one INSIDE_5_RUSH.
15. >=100 team-games contain at least one GOAL_TO_GO_RUSH.
16. >=100 team-games contain at least one SHORT_YARDAGE_RUSH.
17. situation shares are finite and in [0,1].
18. player situation shares sum to 1 within each nonempty team-game-situation.
19. duplicate player-play identity violations = 0.
20. sportsbook inputs = 0.
21. OddsAPI calls = 0.
22. 2026 outcomes read = 0.
23. predictive models fit = 0.
24. production changed = false.

The support floors are frozen before the audit is run.

## Frozen dispositions

If every season passes every gate:

`RB_HIGH_VALUE_OPPORTUNITY_TOPOLOGY_SOURCE_READY`

Otherwise:

`RB_HIGH_VALUE_OPPORTUNITY_TOPOLOGY_SOURCE_BLOCKED`

Integrity failure:

`RB_HIGH_VALUE_OPPORTUNITY_TOPOLOGY_SOURCE_INTEGRITY_FAILURE`

## If source-ready

Freeze a separate named-player historical mechanism experiment **before**
predictive scoring.

The next test should ask whether strictly-prior same-player situation ownership
predicts his next-game share **conditional on the team actually generating that
same situation**.

That future mechanism must:
- evaluate player-game grain;
- preserve total team situation volume;
- compare against current broad RB allocation rather than a position average;
- keep situation families separate;
- use player- and game-clustered inference;
- require cross-season replication;
- not use TD outcome to choose a winner;
- not convert one positive situation into an automatic ATD model.

Only after a situation-ownership mechanism is qualified could a separate
market-specific integration be considered.

## If blocked

Fail closed.

Do not:
- lower source/support gates;
- merge inside-5 with red zone after seeing counts;
- include QB/WR rushes to inflate support;
- infer missing rusher identity;
- use touchdown outcomes to rescue a sparse cell;
- pool seasons to waive a failed season.

## Scientific boundary

This audit:
- fits no model;
- changes no projection;
- reads no sportsbook;
- makes no OddsAPI call;
- reads no 2026 target outcomes;
- does not create an ATD probability;
- changes no production code.

Canonical model status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.
