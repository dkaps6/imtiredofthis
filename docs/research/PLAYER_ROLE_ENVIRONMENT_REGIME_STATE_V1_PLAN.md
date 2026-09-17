# Player Role & Environment Regime State V1 — Frozen Cross-Position Plan

**Status:** PLAN / SOURCE-ENGINEERING ONLY — NO PREDICTIVE RESULT, NO PRODUCTION CHANGE.

**Created:** 2026-09-17

**Branch:** `research-player-role-environment-regime-v1-plan`

## Why this exists

Historical player production is only a good prior when the player's football environment is sufficiently similar to the environment being predicted.

A player can remain the same athlete while his opportunity-generating regime changes materially because of:

- team change;
- quarterback change;
- head coach / offensive coordinator / play-caller change;
- promotion or demotion in positional hierarchy;
- teammate departure/addition;
- injury-created vacancy;
- committee collapse or formation;
- route/slot/outside role change;
- pass-down / two-minute / goal-line role change;
- offensive-line or protection environment change;
- scheme/personnel change;
- rookie/new starter transition;
- return from injury;
- meaningful current-season evidence that supersedes prior-season usage.

This plan creates a structured, time-aware football-context layer for QB, RB, WR and TE.

It is **not** a manual narrative override layer and is **not** a sportsbook-implied projection system.

## Core principle

The system should answer:

> Is the football regime generating this player's opportunities materially different from the historical regime represented by his prior sample?

It should not answer:

> What final stat should we force the player to?

Regime state changes the **prior, weighting, uncertainty, and entitlement context** supplied to downstream models. Final opportunity and efficiency remain learned/conserved football outputs.

## Cross-position architecture

`GAME ENVIRONMENT -> TEAM OPPORTUNITY -> POSITION/ROOM POOL -> PLAYER ROLE/ENVIRONMENT REGIME -> PLAYER ENTITLEMENT -> EFFICIENCY -> JOINT MC`

The regime layer lives between room-level opportunity and player entitlement.

## Global evidence dimensions

Each player-team-week row records independent evidence dimensions rather than one opaque score.

### 1. Team continuity

Fields:

- `team_changed_since_prior_season`
- `team_changed_since_last_game`
- `days_with_current_team`
- `same_team_prior_games`
- `same_system_prior_games`

### 2. Position-room continuity

Fields:

- `credible_competitors_added`
- `credible_competitors_departed`
- `returning_room_opportunity_share`
- `vacated_room_opportunity_share`
- `current_room_competitor_count`
- `room_concentration_prior`
- `room_concentration_current_evidence`

"Credible competitor" requires a frozen role/participation definition; roster presence alone is insufficient.

### 3. Hierarchy / depth evidence

Fields:

- `official_depth_rank`
- `depth_rank_changed`
- `depth_source_authority`
- `depth_freshness_days`

Depth chart is contextual evidence only. No direct rank-to-volume formula is permitted.

### 4. Availability / vacancy

Fields:

- `definitive_unavailable_competitor_count`
- `questionable_competitor_count`
- `injury_created_vacated_share`
- `suspension_or_ir_vacated_share`
- `availability_state_confidence`

### 5. Coaching / scheme continuity

Fields:

- `head_coach_changed`
- `offensive_coordinator_changed`
- `primary_play_caller_changed`
- `position_coach_changed` where source quality permits
- `scheme_continuity_state`
- `play_caller_prior_player_overlap`

No generic "new coach boost" is authorized. These are transition/context variables only.

### 6. Quarterback environment

Applicable especially to WR/TE/RB receiving:

- `primary_qb_changed`
- `current_qb_authority`
- `qb_prior_player_overlap`
- `qb_quality_context_source`
- `qb_style_context` only if separately frozen and leakage-safe

The regime layer may encode that a receiver moved from one QB environment to another. It may not simply award yards because the current QB is "elite" without a frozen football model translating that environment into team pass state / efficiency.

### 7. Offensive-line / protection environment

Applicable especially to QB/RB and indirectly receivers:

- `returning_ol_starter_count`
- `ol_continuity_state`
- `key_ol_absence_count`
- `protection_environment_changed`

Advanced-data protection-history features may later enrich this dimension only after their materializers and temporal contracts are certified.

### 8. Official qualitative role evidence

This is the structured home for football information that matters but is not naturally a box-score statistic.

Allowed source tiers:

**Tier A**
- official team transaction/depth/injury pages;
- official coach/GM/player press conference or transcript;
- league/NFL official transaction and injury information.

**Tier B**
- official team editorial/reporting;
- NFL.com reporting that directly attributes role statements.

**Tier C**
- separately approved high-quality reporting when direct source evidence is unavailable.

Every statement record must preserve:

- source reference / URL;
- publication timestamp;
- available-before-target-kickoff flag;
- speaker;
- speaker authority;
- direct quote vs reporter interpretation;
- normalized role concept;
- player/team;
- confidence;
- expiry / freshness rule.

Normalized role concepts may include:

- `LEAD_ROLE`
- `COMMITTEE_ROLE`
- `WORKLOAD_EXPANSION`
- `WORKLOAD_CONTRACTION`
- `PRIMARY_OUTSIDE_RECEIVER`
- `PRIMARY_SLOT_RECEIVER`
- `PRIMARY_TARGET`
- `DEEP_ROLE`
- `POSSESSION_ROLE`
- `PASS_DOWN_ROLE`
- `TWO_MINUTE_ROLE`
- `GOAL_LINE_ROLE`
- `BLOCKING_HEAVY_ROLE`
- `STARTER_CONFIRMED`
- `STARTER_UNCERTAIN`
- `LIMITED_ROLE`

A qualitative statement is evidence. It never maps directly to a yardage adjustment.

### 9. Strict-prior observed current-season role

After a game is completed, its usage may become pregame evidence for later weeks.

Possible fields:

- snap share;
- route participation;
- target share;
- carry share;
- designed-rush share;
- pass-block snaps;
- two-minute participation;
- red-zone / goal-line opportunity share;
- motion/alignment/route-role summaries where qualified;
- current-season room concentration.

Same-game partial information is forbidden.

## Cross-position state families

Every position gets a shared high-level state plus position-specific substate.

### Shared high-level regime state

- `STABLE_SAME_ROLE`
- `EXPANDED_ROLE`
- `CONTRACTED_ROLE`
- `NEW_TEAM_ROLE_ESTABLISHED`
- `NEW_TEAM_ROLE_UNCERTAIN`
- `NEW_SYSTEM_ROLE_ESTABLISHED`
- `NEW_SYSTEM_ROLE_UNCERTAIN`
- `INJURY_CREATED_EXPANSION`
- `RETURN_FROM_INJURY_TRANSITION`
- `ROOKIE_OR_NEW_STARTER_TRANSITION`
- `CONFLICTING_ROLE_EVIDENCE`
- `INSUFFICIENT_CURRENT_ROLE_EVIDENCE`

These labels describe evidence state. They do not prescribe a projection adjustment.

## Position-specific substates

### QB

Key discontinuities:

- new starter;
- new team;
- new play caller;
- major receiver-room turnover;
- major OL turnover;
- designed-rush role change;
- starter returning from injury.

Possible QB substate:

- `ESTABLISHED_STARTER_STABLE_SYSTEM`
- `ESTABLISHED_STARTER_NEW_SYSTEM`
- `NEW_TEAM_STARTER`
- `NEWLY_PROMOTED_STARTER`
- `STARTER_WITH_REBUILT_RECEIVER_ROOM`
- `STARTER_WITH_MAJOR_PROTECTION_CHANGE`

### RB

Key discontinuities:

- committee -> lead;
- lead -> committee;
- new team;
- competitor departure/addition;
- pass-down role;
- goal-line role;
- injury-created vacancy;
- QB rushing competition.

Possible RB substate:

- `STABLE_LEAD`
- `STABLE_COMMITTEE`
- `VACANCY_EXPANSION`
- `NEW_TEAM_LEAD`
- `NEW_TEAM_COMMITTEE_OR_UNCERTAIN`
- `INJURY_TEMP_EXPANSION`
- `ROLE_CONTRACTION`

This subsumes the separate RB Role Regime State V1 idea; the RB document remains useful as the position-specific detailed specialization.

### WR

Key discontinuities:

- new team;
- WR hierarchy promotion/demotion;
- target competitor departure/addition;
- new QB;
- new play caller;
- slot/outside alignment change;
- route-depth / route-tree role change;
- injury-created target vacancy.

Possible WR substate:

- `STABLE_WR1`
- `STABLE_SECONDARY`
- `NEW_TEAM_PRIMARY`
- `NEW_TEAM_SECONDARY_OR_UNCERTAIN`
- `PROMOTED_PRIMARY_TARGET`
- `VACANCY_TARGET_EXPANSION`
- `QB_ENVIRONMENT_TRANSITION`
- `ALIGNMENT_ROLE_TRANSITION`
- `ROLE_CONTRACTION`

### TE

Key discontinuities:

- new team;
- TE1 promotion;
- blocking-heavy -> route-heavy;
- primary middle-field target role;
- competitor departure/addition;
- new QB;
- scheme change;
- injury-created route/target vacancy.

Possible TE substate:

- `STABLE_TE1`
- `STABLE_COMMITTEE`
- `PROMOTED_ROUTE_TE`
- `NEW_TEAM_PRIMARY_TE`
- `VACANCY_TARGET_EXPANSION`
- `BLOCKING_TO_RECEIVING_ROLE_TRANSITION`
- `ROLE_CONTRACTION`

## Evidence confidence

Do not collapse all context into a single hand-built score.

Store each evidence dimension plus a transparent overall confidence tier.

### HIGH

Multiple authoritative, temporally valid sources agree; or an objective transaction/availability event plus current official role evidence establishes the state.

### MEDIUM

Strong structural transition evidence but limited direct role confirmation; or one authoritative direct role signal with incomplete corroboration.

### LOW

Roster/depth implication only, stale evidence, or meaningful conflict.

### ABSTAIN

Identity, timing, role, or source authority cannot be established.

## Historical-regime weighting concept

A validated future entitlement model may test whether verified regime changes should alter historical weighting.

Permissible candidate mechanisms:

- down-weight old-team opportunity history after a team change;
- down-weight old committee-era shares after a verified competitor departure;
- increase weight on current-room structure;
- increase weight on completed current-season evidence;
- use historical analog priors for similar transition states;
- widen uncertainty when a role change is real but magnitude remains unknown;
- partial pooling between player history and transition-state historical analogs.

Forbidden:

- hand-coded player-specific boosts;
- arbitrary "+20% because WR1";
- automatically trusting a qualitative quote as a carry/target number;
- pulling projection toward sportsbook line;
- target-game result leakage.

## Market discrepancy as diagnostic, not input

Sportsbook lines remain downstream.

Create a `MODEL_MARKET_DISAGREEMENT_AUDIT_V1` record when an eligible model-vs-market gap exceeds a frozen threshold.

For every flagged player, inspect:

1. team continuity;
2. room turnover;
3. current hierarchy;
4. injuries/vacancy;
5. coaching/play-caller change;
6. QB environment for receivers;
7. OL/protection environment for QB/RB;
8. strict-prior current-season role;
9. authoritative qualitative role evidence;
10. whether disagreement comes from opportunity, efficiency, or distribution.

Outcomes:

- `FOOTBALL_INPUT_DEFECT_FOUND`
- `FOOTBALL_ROLE_STATE_STALE`
- `MODEL_MECHANISM_DISAGREEMENT_NO_INPUT_DEFECT`
- `INSUFFICIENT_EVIDENCE`

If football evidence is missing/stale, repair the football data and rerun independently.

If no football defect is found, preserve the disagreement. Vegas is not allowed to become upstream truth.

## Example — DJ Moore, Buffalo

The correct regime interpretation is not:

> Josh Allen is elite, add yards.

It is:

- player changed teams;
- Buffalo acquired him as a major passing-game addition;
- official current depth places him first at one WR position;
- current-team reporting describes him as the intended top outside passing-game playmaker;
- he reunites with a prior offensive coordinator;
- his prior Chicago opportunity environment is therefore not automatically the correct current prior.

Candidate state:

`NEW_TEAM_ROLE_ESTABLISHED + NEW_TEAM_PRIMARY + QB_ENVIRONMENT_TRANSITION + PLAY_CALLER_PRIOR_OVERLAP`

The exact target share / receiving yards must still be determined by the finite pass/WR pool and a validated entitlement/efficiency mechanism.

## Example — Jahmyr Gibbs

Candidate state:

`EXPANDED_ROLE + VACANCY_EXPANSION`

Old split-era share may be stale downward after a meaningful competitor departure and official lead-role evidence.

No fixed carry boost is authorized.

## Example — Kenneth Walker III

Candidate state:

`NEW_TEAM_ROLE_ESTABLISHED + NEW_TEAM_LEAD`

Old-team workload is not assumed transportable. New-room competition and strict-prior current-season evidence govern the transition.

## Historical evaluation program

Build a leakage-safe transition panel across multiple seasons.

Cohorts should include:

- stable same-team same-role;
- same-team promotion;
- same-team contraction;
- new-team established role;
- new-team uncertain role;
- teammate-vacancy expansion;
- injury-created temporary expansion;
- new QB environment for WR/TE;
- new play caller / system;
- rookie/new starter;
- major room turnover.

Evaluate opportunity first.

### QB

- attempts;
- designed rush attempts;
- pass opportunity error;
- distribution/interval calibration.

### RB

- carries;
- targets/receptions;
- rushing-yard MAE holding efficiency constant;
- receiving-yard MAE holding efficiency constant;
- high-workload tail error.

### WR / TE

- routes where historical source allows;
- targets;
- receptions;
- receiving-yard MAE with efficiency separated;
- target-share/room-entitlement error;
- high-volume player cohorts.

Global gates must protect stable-role cohorts from degradation.

## Anti-retest rules

This program is not:

- simple depth-rank remapping;
- a generic "WR1/RB1/TE1 boost";
- a player whitelist;
- market-implied usage;
- another historical efficiency transform;
- an after-the-fact fix for 2026 examples;
- a reason to reopen failed M95/R23-R27D/WR-R11/etc. formulations.

## Engineering sequence

1. Inventory existing current-role/roster/depth/injury/availability assets already in repo.
2. Freeze cross-position `PLAYER_ROLE_ENVIRONMENT_EVIDENCE_V1` schema.
3. Add transaction/team-change and room-turnover evidence.
4. Add authoritative qualitative-role evidence ingestion with timestamps/provenance.
5. Build deterministic state classifier — **no predictive coefficients**.
6. Build leakage-safe historical transition panel.
7. Audit coverage/abstention and historical cohort sizes.
8. Coordinate with active position research lanes.
9. Freeze one position-specific entitlement experiment at a time.
10. Only after qualification, shadow live current season before production promotion.

## Production boundary

This plan does not authorize production changes.

It does not modify:
- M89/M90/C2;
- M38/WR-R15;
- TE-R5P;
- RB P3/R26/R22;
- Issue #535;
- current Full Slate science.

Disposition:

`PLAYER_ROLE_ENVIRONMENT_REGIME_STATE_V1_PLAN_FROZEN_PENDING_SOURCE_ENGINEERING`
