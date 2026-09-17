# NFL Cross-Position Football Context State V1 — Engineering Plan

**Status:** engineering contract / source-and-feature design only.  
**Predictive experiments authorized:** NO.  
**Production changes authorized:** NO.

## Purpose

Create a reusable pregame football-context layer that captures directional changes in a player's environment across positions.

This is broader than the previously tested M76–M77 `exact_personnel_discontinuity` family. M76–M77 is closed and must not be rerun under a new label.

The new family must encode **football consequence**, not merely personnel turnover.

Examples:

- same receiver, better/worse QB environment;
- same RB, more/less backfield competition;
- same QB, stronger/weaker protection and receiver environment;
- same TE, increased/decreased route opportunity;
- player moves teams and changes role, scheme, teammate competition, or supporting cast;
- injury return changes the effective role hierarchy;
- starter promotion creates a new workload state;
- new coordinator or scheme changes the opportunity mechanism.

## Core rule

A context feature must answer:

> **What materially changed about this player's pregame opportunity or efficiency environment relative to the environment that generated his historical production?**

A feature that only answers "some personnel changed" is not sufficient.

## Temporal contract

All context-state values must be known before target kickoff.

Allowed evidence:

- official/current roster;
- declared depth-chart context, treated as contextual evidence rather than workload authority;
- completed prior-game player usage;
- completed prior-game team/QB/offensive-line/receiver-room history;
- official injuries/inactives known before kickoff;
- coaching/scheme identity only when historically timestamped and source-certified;
- schedule/opponent state known pregame.

Forbidden:

- target-game snaps, routes, targets, carries, pressures, outcomes, or tracking geometry;
- sportsbook prices as football-state features;
- retrospective route/coverage labels from the target game;
- same-game partial history.

## Cross-position state dimensions

### 1. Team continuity / team transition

Candidate fields:

- `same_team_as_prior_game`
- `same_team_as_prior_season`
- `team_change_flag`
- `games_since_team_change`
- `new_team_first_game_flag`
- `new_team_first_4_games_flag`

Interpretation:
team change is a trigger, not a directional adjustment by itself.

### 2. Role transition

Candidate fields:

- `role_promotion_score`
- `role_demotion_score`
- `starter_transition_flag`
- `depth_rank_delta`
- `prior_4_game_snap_share_delta`
- `prior_4_game_route_share_delta`
- `prior_4_game_target_share_delta`
- `prior_4_game_carry_share_delta`
- `vacated_opportunity_share`
- `returning_competition_share`

Role must be inferred primarily from usage/participation history; depth chart alone is contextual evidence.

### 3. Teammate competition / room state

Receiver/TE:

- returning target-share concentration;
- vacated target share;
- returning route-share concentration;
- number of established high-share target earners;
- teammate injury-created vacancy;
- new high-priority receiver arrival.

RB:

- returning carry-share concentration;
- returning target-share concentration;
- backfield vacancy;
- QB rushing competition;
- short-yardage competition;
- receiving-back competition.

QB:

- returning WR/TE route-share continuity;
- returning target-share continuity;
- explosive-receiver continuity;
- protection continuity.

### 4. QB environment state for pass catchers

Pregame-safe candidate features:

- `qb_change_flag`
- `qb_starter_continuity_games`
- `qb_prior_efficiency_delta_vs_player_history`
- `qb_prior_attempt_volume_delta_vs_player_history`
- `qb_prior_accuracy_or_completion_environment_delta`
- `qb_prior_explosive_pass_environment_delta`
- `qb_scramble_competition_delta`

The comparison should be between the current QB environment and the QB environment under which the receiver's historical production was generated.

Do not use sportsbook lines.

### 5. Receiver/TE environment state

Candidate features:

- QB-environment delta;
- team pass-opportunity delta;
- room competition delta;
- route/target entitlement delta;
- historical release-space profile from Advanced Feature V1;
- historical route-conditioned spacing profile where pregame route-tendency support exists;
- opponent/team coverage context only when pregame-safe and source-qualified.

### 6. RB environment state

Candidate features:

- team rushing-opportunity delta;
- offensive-line run/protection historical quality;
- QB rushing competition delta;
- backfield competition delta;
- goal-line/short-yardage role continuity;
- receiving ecosystem state;
- injury-created vacancy.

Existing RB science remains authoritative until a separately frozen candidate wins.

### 7. QB environment state

Candidate features:

- protection continuity and strict-prior protection geometry;
- receiver-room quality/continuity;
- team pass-opportunity state;
- opponent pass-rush/coverage context if already source-qualified;
- coordinator/scheme transition;
- center/OL continuity;
- major target-room turnover.

This must not reopen failed M76–M77 generic personnel counts. A new QB candidate must use materially new directional context.

### 8. OL / protection context

OL players may not be direct prop targets, but their state can drive QB/RB/pass-catcher features.

Candidate context:

- projected starter continuity;
- games together;
- starter absences;
- strict-prior blocker interaction geometry;
- strict-prior time-to-min-distance;
- protection-window history;
- replacement-player history.

No target-game blocking geometry is pregame-safe.

### 9. Coaching / scheme state

Candidate fields, source-gated:

- coordinator change flag;
- head coach / play-caller change;
- offensive system continuity;
- early-season games since coordinator change;
- historical pass/rush tendency under current play caller;
- personnel grouping tendency.

These fields remain **SOURCE_PENDING** until a timestamped historical source is certified.

## Environment delta design

The preferred form is not a raw level but a comparison:

`current_environment_state - historical_environment_state_for_player`

Examples:

- current QB prior YPA minus weighted QB-environment YPA from the player's historical receiving sample;
- current room competition minus historical room competition;
- current pass-opportunity rate minus historical team pass-opportunity environment;
- current backfield carry competition minus historical competition;
- current protection history minus QB's historical protection environment.

This directly targets the user's football-base-knowledge concern: a player's historical production should not be treated as context-free.

## Context discontinuity classes

Every player-week may be classified into one or more:

- `STABLE_ENVIRONMENT`
- `NEW_TEAM`
- `NEW_STARTER_ROLE`
- `QB_ENVIRONMENT_CHANGE`
- `ROOM_COMPETITION_INCREASE`
- `ROOM_COMPETITION_DECREASE`
- `INJURY_CREATED_VACANCY`
- `KEY_TEAMMATE_RETURN`
- `OL_ENVIRONMENT_CHANGE`
- `COACHING_SCHEME_CHANGE`
- `MULTI_FACTOR_TRANSITION`

Classification itself is descriptive. Predictive weighting is not authorized by this document.

## Interaction with Advanced Feature V1

Advanced Feature V1 contributes strict-prior historical player/environment evidence:

Receiver:
- `hist_receiver_release_nearest_defender_median_yards`
- `hist_receiver_release_second_defender_median_yards`
- `hist_receiver_release_crowding_2yd_rate`
- `hist_receiver_release_crowding_3yd_rate`
- route-conditioned release geometry where route tendency can be established pregame.

Protection:
- `hist_blocker_snap_distance_median_yards`
- `hist_blocker_min_distance_median_yards`
- `hist_blocker_time_to_min_distance_median_seconds`

Route history:
- player x route throw-space history.

These values describe historical skill/environment interaction. They do not by themselves describe the current target-game environment.

## No-retest boundary

Closed families remain closed:

- M76–M77 generic/exact personnel discontinuity;
- previously rejected generic depth-chart discontinuity;
- previously rejected generic injury burden;
- previously rejected team-level man/zone signal;
- prior NGS/PFR secondary experiments that used materially different coarse feature families.

Reopening requires materially new information.

This context-state lane qualifies as potentially new only when it combines directional environment changes with the newly engineered advanced feature history or other source-qualified context unavailable to the old tests.

## Position coverage

Initial direct target positions:

- QB
- RB
- WR
- TE

Context-provider positions/groups:

- OL
- WR/TE rooms
- RB rooms
- QB
- defensive front
- secondary
- coaching/play caller

The architecture must not assume a single position owns the signal.

## Engineering phases

### Phase A — source inventory and temporal certification

For every proposed context field:
- identify exact source;
- verify historical coverage;
- verify current/live availability where needed;
- define timestamp;
- define identity grain;
- define abstention rules.

### Phase B — context-state materializer

Produce a player-week table keyed by stable GSIS identity with:
- current team;
- prior team;
- role state;
- teammate-room state;
- QB environment;
- OL/protection state where applicable;
- coaching state where source-qualified;
- temporal provenance.

### Phase C — environment-delta panel

Join the player's strict-prior historical production/advanced-feature environment to the current pregame context.

No target-game outcomes included in features.

### Phase D — research coordination

Only after A–C freeze:
- choose one position/market;
- pre-register one candidate hypothesis;
- use frozen train/test cohorts;
- compare to canonical baseline;
- no production promotion unless it wins the existing scientific gates.

## First implementation recommendation

Continue the current WR integration preflight because it is already running and provides the first identity/outcome bridge for Advanced Feature V1.

In parallel, build the generic cross-position context-state source contract so the eventual WR experiment is the first consumer of a reusable system rather than a one-off receiver patch.

## Disposition

**`NFL_CROSS_POSITION_CONTEXT_STATE_V1_ENGINEERING_PLAN_FROZEN`**
