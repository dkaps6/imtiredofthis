# Player Active-Role Participation V1 — Source / Anti-Retest Audit

Date: 2026-10-15

Status: **FROZEN SOURCE AUDIT / NO PREDICTIVE SCORING YET / RESEARCH ONLY**

Version:
`PLAYER_ACTIVE_ROLE_PARTICIPATION_V1_SOURCE_AUDIT`

## Player-first question

The full-roster replay exposed a pregame player-state problem:

- 1,837 player-weeks;
- 610 / 1,837 (33.21%) had no weekly-stat row;
- QB 7.81%;
- RB 30.15%;
- WR 31.55%;
- TE 45.31%.

The model nevertheless produced material opportunity for many of those rostered
players.

Do not fix this by deleting players after outcomes.

The player-first question is:

> Among players who are legitimately in the pregame roster universe and are not
> known official game-day inactives, can we reconstruct enough historical
> player-level participation state to test whether **this named player** is
> likely to have an offensive role at all?

This is Stage 1 of opportunity modeling.

Stage 2 remains:
conditional on participating, how much team/room opportunity belongs to the
player?

Do not collapse those questions.

## Anti-retest boundary

This audit does NOT reopen:

- `AVAILABILITY_OPPORTUNITY_RULE_ORDER_GAP_CONFIRMED`:
  definitive unavailable players receiving opportunity because availability
  was applied after opportunity generation. That structural gap remains valid.
- M78/M79 generic official-inactive QB identity as a predictive mean mechanism.
- RB Vacancy Opportunity V1.
- player role-elasticity teammate-availability V1.
- RB carry/snap Week-5 allocation shadow.
- RB receiving-room share.
- WR/TE target-share trajectory.
- WR/TE snap participation already consumed inside WR-R15 / TE-R5P.
- route-participation/YPRR historical work, which remains source-parity blocked.
- postgame DNP/zero settlement rules.

The new scientific object is a separate **probability of offensive
participation / active role**, upstream of opportunity amount.

Existing WR/TE use of prior snap participation is a mean-entitlement input. It
does not constitute a calibrated cross-position zero-state / participation
probability.

## Historical seasons

Source audit:
- 2024
- 2025

No 2026 outcomes.

The two seasons are chosen because the exact frozen official game-day inactive
archive used by M78 is available for both and must be preserved as a pregame
exclusion layer.

## Pregame universe

For each scheduled team-week:

1. begin with weekly roster authority;
2. require stable GSIS/player identity;
3. map to one position family:
   - QB
   - RB = RB/FB/HB
   - WR
   - TE;
4. require weekly roster status `ACT` where status is available;
5. remove players on the exact frozen official game-day inactive list;
6. retain the remaining player even if he ultimately records zero offensive
   snaps or zero box-score opportunity.

Target-game snaps/outcomes may not be used to decide membership in the pregame
universe.

## Pregame inactive authority

Use the frozen corrected M78 official-inactive team-week snapshot:

`data/backtests/qb_m78_official_inactives_corrected/m78_corrected_candidate_teamweek.csv`

Required immutable SHA-256:

`d39aaf0feea101f3e0d2721ebd4118ef33fb1a4d3c76670e2a4f17734e37b609`

Known source shape:
- 1,088 team-weeks total;
- 544 in 2024;
- 544 in 2025.

The archive is used only as a fixed pre-kickoff inactive identity source.

Article timestamps, prose and target-game performance are not predictive
features.

## Participation source

Use historical PFR snap counts through the repository's already-established
snapshot loader.

Required fields / semantics where available:
- season;
- week;
- team;
- player stable identity;
- offensive snaps and/or `offense_pct`.

A team-week with a valid snap source allows a rostered, non-inactive player who
is absent from the player snap table to be represented as zero offensive snaps.

Do NOT interpret player absence as zero if the entire team-week snap source is
missing.

## Source audit outputs

For each position / season report:

- pregame-universe player-games;
- distinct players;
- official-inactive rows removed;
- roster stable-ID coverage;
- inactive-to-roster identity bridge;
- team-week snap-source coverage;
- snap-row stable-ID bridge to roster;
- valid zero-snap player-games;
- positive-snap player-games;
- share with positive offensive participation;
- number of players with >=4 prior same-team eligible games;
- number with >=8 prior same-team eligible games;
- chronology violations.

This is source characterization only.

Do not score a predictor in this audit.

## Frozen source gates

Every position must be adjudicated separately in 2024 and 2025.

Common integrity:

1. exact M78 snapshot SHA matches;
2. 544 official-inactive team-weeks exist in each season;
3. weekly roster team/week coverage >=99%;
4. stable GSIS identity on >=99% of eligible roster rows;
5. official-inactive -> roster identity bridge >=95%;
6. PFR team-week snap-source coverage >=99%;
7. PFR positive-snap player identity -> weekly roster bridge >=98%;
8. chronology violations = 0;
9. no target-game snap/result is used to construct a pregame predictor;
10. sportsbook inputs = 0;
11. OddsAPI calls = 0;
12. predictive models fit = 0;
13. projection values changed = 0;
14. production changed = false.

Per position / season support:

QB:
- >=450 pregame player-games;
- >=45 distinct QBs;
- >=30 QBs with >=4 prior same-team eligible games.

RB:
- >=1,500 pregame player-games;
- >=100 distinct RB/FB/HB;
- >=80 players with >=4 prior same-team eligible games.

WR:
- >=2,000 pregame player-games;
- >=180 distinct WRs;
- >=140 WRs with >=4 prior same-team eligible games.

TE:
- >=1,000 pregame player-games;
- >=100 distinct TEs;
- >=75 TEs with >=4 prior same-team eligible games.

The thresholds are frozen before source scoring.

No position may borrow support from another.

## Frozen dispositions

Per position:

`QB_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY`
`RB_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY`
`WR_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY`
`TE_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY`

or matching `*_SOURCE_BLOCKED`.

Overall:
- all four ready:
  `PLAYER_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY_ALL`
- one or more ready:
  `PLAYER_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY_PARTIAL`
- none:
  `PLAYER_ACTIVE_ROLE_PARTICIPATION_SOURCE_BLOCKED`
- integrity failure:
  `PLAYER_ACTIVE_ROLE_PARTICIPATION_SOURCE_INTEGRITY_FAILURE`.

## Authorized mechanism only if source-ready

For any position that passes, freeze a separate mechanism before scoring.

The intended next scientific question is:

> Does the named player's own strictly-prior participation history forecast his
> target-game offensive participation better than a strictly-prior
> position-level participation baseline?

The mechanism must remain separate from conditional opportunity amount.

Suggested target for that later frozen contract:
- binary target: any offensive snap in a valid target team-week.

Suggested player-specific state:
- same-team last-4 eligible games;
- minimum 3;
- prior participation occurrence only;
- no target-game information.

No exact probability mapping is authorized until that mechanism contract is
frozen.

## If blocked

Fail closed.

Do not:
- infer target-game active role from weekly stats;
- treat missing team-week snap data as player zero;
- lower support/coverage gates;
- pool positions;
- use official inactive identity as a generic mean adjustment;
- rescue with a post-hoc snap-percentage threshold.

Then move to another player-identity information family.

## Scientific boundary

This audit:
- fits no predictive model;
- changes no player projection;
- reads no sportsbook;
- makes no OddsAPI call;
- uses no 2026 outcomes;
- changes no production code.

Canonical status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.
