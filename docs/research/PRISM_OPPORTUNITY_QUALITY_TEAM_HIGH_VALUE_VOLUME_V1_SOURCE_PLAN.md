# PRISM Opportunity Quality — Team High-Value Volume V1 Source Plan

Frozen: 2026-10-16

Status: **FROZEN SOURCE / ANTI-RETEST AUDIT BEFORE ANY NEW OPPORTUNITY-QUALITY SCORING**

Model:
**PRISM — Player Role & Individual Situation Model**

Architecture position:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

## Scientific parents

Player opportunity quantity:
- `PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`;
- `PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL`.

Forward lock:
- `PRISM_WEEK6_SUNMON_STAGE2_SHADOW_LOCKED`;
- run `38990758368`;
- artifact `12085545101`.

High-value topology source:
- `PLAYER_HIGH_VALUE_SCORING_OPPORTUNITY_SOURCE_READY`;
- run `38848080453`;
- artifact `12023852453`;
- digest `sha256:1d8a22bf6240b4d4443ab7432d3224b63eaef97439639e7b47f536aed174adda`.

Terminal anti-retest parent:
- `PLAYER_HIGH_VALUE_OPPORTUNITY_PERSISTENCE_CLOSED`;
- run `38848538570`;
- artifact `12024037594`.

## Exact question

Can historical football data reconstruct a leakage-safe decomposition of:

1. **team high-value opportunity volume** — how many scoring-area opportunities a team creates in a game; and
2. **current named-player entitlement inside that finite opportunity environment**,

without assuming that a player's recent high-value share simply persists?

This source audit asks only whether the required chronology and source lineage exist.

It does **not** score a predictive candidate.

## Why this is not a rescue of the closed V1

The closed V1 asked:

> does the named player's recent high-value share predict his next-game high-value share better than ordinary overall usage share?

That exact idea remains closed.

This source plan does not use:
- last-4 player high-value share as a candidate;
- alternate inside-20/inside-10/inside-5 player-share windows;
- lead-back / WR1 / TE1 subsets;
- player TD outcomes;
- post-result band selection.

Instead, it separates two different objects:

`TEAM HIGH-VALUE VOLUME -> CURRENT PLAYER ENTITLEMENT`.

The player entitlement, if a later mechanism is authorized, must come from already-existing canonical opportunity entitlement / active-role state rather than a new recent high-value-share allocator.

## V1 cells

Only cells with qualified historical Stage-2 opportunity quantity may advance beyond source audit:

1. `RB_RUSH_INSIDE10`
   - finite team pool = team rushing attempts with `yardline_100 <= 10`;
   - named player = RB/FB/HB rusher.

2. `RB_RECEIVING_REDZONE`
   - finite team pool = team targets with `yardline_100 <= 20`;
   - named player = RB/FB/HB receiver.

3. `TE_RECEIVING_REDZONE`
   - finite team pool = team targets with `yardline_100 <= 20`;
   - named player = TE receiver.

WR may be audited descriptively as source context but may not enter a scored V1 mechanism while its 2025 Stage-2 OOS entitlement authority remains blocked.

QB is outside V1 because the exact Stage-2 QB bridge is closed.

## Historical source

Use public football-only sources already accepted by the repository:
- nflverse/nflfastR play-by-play;
- nflverse weekly roster identity;
- canonical schedule/team identity;
- existing Stage-2 source authority only where an exact historical entitlement join is required.

Target source seasons:
- 2022;
- 2023;
- 2024;
- 2025.

Primary as-of target seasons for a later mechanism:
- 2024 development;
- 2025 independent confirmation;
- regular-season Weeks 5-18.

2022-2023 may supply source/history context only. They are not scored as Stage-2-compatible target seasons because the qualified PRISM Stage-2 historical entitlement authority is frozen on 2024-2025.

No earlier season may rescue a failed 2025 confirmation later.

No 2026 outcomes.

## Team high-value topology to reconstruct

For every scheduled team-game:
- total offensive plays where source-supported;
- team rush attempts;
- team targets;
- team inside-10 rushing attempts;
- team red-zone targets;
- team inside-5 rushing attempts as descriptive topology only;
- optional end-zone targets where the already-qualified air-yards source is valid;
- drives/possessions and red-zone-entry counts if a stable drive identity is source-qualified.

The source audit must not silently impute missing high-value opportunities to zero when the underlying PBP/team-game source is incomplete.

## Strictly-prior team state inventory

For each target team-game Weeks 5-18, source audit whether the following can be reconstructed using completed games only:

Team offense:
- prior high-value opportunities per game;
- prior high-value opportunities per relevant parent opportunity
  (inside-10 rushes / team rush attempts; red-zone targets / team targets);
- last up to 4 completed same-season team games;
- minimum 3 prior games.

Opponent allowance:
- same quantities allowed to opponents;
- last up to 4 completed same-season games;
- minimum 3 prior games.

League prior:
- same-season strictly-prior league environment available before the target week.

These are source states only. No combination weight or predictive coefficient is authorized by this plan.

## Player / finite-pool compatibility

For the three eligible cells, prove that historical target player-game identities can be joined to:
- stable player identity;
- the correct team-game high-value pool;
- existing canonical overall opportunity entitlement `q_i` or an exact source-qualified parent;
- Stage-1/Stage-2 chronology where applicable.

Do not construct a replacement opportunity-share mean.

The source audit may report actual player high-value opportunities as a **future scoring label inventory only**.

Target-game high-value counts may not enter any pregame predictor.

## Source gates

Per target season 2024 and 2025:

Core PBP/team topology:
- scheduled team-game coverage >= 99%;
- week/team identity >= 99%;
- yardline coverage >= 99%;
- stable rusher identity >= 99%;
- stable receiver identity >= 99%;
- skill-position roster bridge >= 98%;
- duplicate player-play contradictions = 0.

Finite high-value pools:
- inside-10 rushing pool finite for >= 99% of scheduled team-games;
- red-zone target pool finite for >= 99% of scheduled team-games;
- no player high-value count may exceed its finite team pool;
- team/player high-value count conservation >= 99%.

Strict-prior state:
- >= 95% of target team-games have >=3 strictly-prior same-season team offense games;
- >= 95% have >=3 strictly-prior same-season opponent-allowance games;
- strictly-prior league environment available for 100% of target rows;
- same/future chronology violations = 0.

Player/Stage-2 compatibility, per eligible cell:
- stable player identity >= 99%;
- canonical historical entitlement attach >= 95%;
- finite parent opportunity denominator >= 99%;
- ambiguous identity <= 0.5%.

Support floors per target season:
- `RB_RUSH_INSIDE10`: >= 700 player-games / >= 80 players / >= 300 team-games with at least one inside-10 rushing opportunity;
- `RB_RECEIVING_REDZONE`: >= 700 player-games / >= 80 players / >= 300 team-games with at least one red-zone target;
- `TE_RECEIVING_REDZONE`: >= 700 player-games / >= 75 players / >= 300 team-games with at least one red-zone target.

If these floors prove structurally impossible because zero-event team-games are legitimate football outcomes, report the exact support distribution and fail closed. Do not lower the floors after seeing them.

## Drive / end-zone optional gates

Drive-state and end-zone-target sources are **optional extensions**.

They may be labeled source-ready only if:
- drive identity coverage >= 98% for every target season;
- end-zone target derivation input coverage >= 95% for every target season;
- chronology and identity gates remain zero-violation.

Failure of an optional extension does not block the core inside-10/red-zone topology source if the core gates pass.

It also does not authorize using the optional source.

## Dispositions

Per cell:
- `<CELL>_PRISM_OPPORTUNITY_QUALITY_SOURCE_READY`
- `<CELL>_PRISM_OPPORTUNITY_QUALITY_SOURCE_BLOCKED`
- `<CELL>_PRISM_OPPORTUNITY_QUALITY_SOURCE_INTEGRITY_FAILURE`

Overall:
- `PRISM_OPPORTUNITY_QUALITY_SOURCE_READY_ALL`
- `PRISM_OPPORTUNITY_QUALITY_SOURCE_READY_PARTIAL`
- `PRISM_OPPORTUNITY_QUALITY_SOURCE_BLOCKED`
- `PRISM_OPPORTUNITY_QUALITY_SOURCE_INTEGRITY_FAILURE`

Source readiness is not predictive qualification.

## If source-ready

A separate mechanism freeze must be written before any predictive scoring.

That later freeze must predeclare:
- how team high-value volume is forecast strictly pregame;
- whether opponent allowance contributes and with what frozen weight;
- how the qualified Stage-2 named-player entitlement maps into the high-value pool;
- finite-count distribution;
- all-zero/high-value-empty room behavior;
- development/confirmation seasons;
- individual-player proper score;
- team-volume and room-allocation protection gates;
- player/game dependence-aware inference.

No later mechanism may use recent player high-value share as the direct candidate that V1 already closed.

## Integrity boundary

At this source freeze:
- predictive candidates scored: **0**;
- player high-value-share rescue attempts: **0**;
- TD outcomes used: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- 2026 outcomes: **0**;
- production changed: **false**.
