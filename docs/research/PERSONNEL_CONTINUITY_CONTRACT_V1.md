# Personnel Continuity Contract V1

**Status:** ENGINEERING CONTRACT — NO PRODUCTION SCIENCE CHANGE

## Purpose

Define leakage-safe descriptive personnel-continuity and room-turnover state for player-game and team-game context. This layer supplements canonical historical game logs and the Role / Environment Event Ledger; it does not learn outcome-driven regime labels or modify projections.

## Canonical grains

Player state:

`season, week, game_id, player_id`

Team/room state:

`season, week, game_id, team, room`

Every row must include `target_kickoff`, `materialized_at`, `continuity_version`, and source/coverage metadata.

## Strict-prior boundary

Only information knowable before target kickoff may contribute. For target kickoff `T`:

- canonical prior-game usage must come only from games completed before `T`;
- ledger events must satisfy `observed_at < T` and `effective_at <= T`;
- target-game snaps, routes, touches, targets, box score, result, residuals, and postgame role statements are forbidden.

## V1 descriptive fields

### Player continuity

- `same_team_as_prior_game`
- `games_with_current_team_prior`
- `games_since_team_change_prior`
- `games_since_first_active_game_current_team_prior`
- `prior_game_active_flag`
- `prior_game_snap_share` when canonical history supports it
- `prior_game_target_share`
- `prior_game_carry_share`
- `prior_game_route_share` only when the qualified source truly supports routes
- `games_since_depth_role_event_prior`
- `games_since_availability_event_prior`

### Room continuity

For QB/RB/WR/TE and other approved position rooms:

- `room_players_prior_game`
- `room_players_returning_count`
- `room_players_added_count_pregame`
- `room_players_departed_count_pregame`
- `room_returning_prior_target_share`
- `room_departed_prior_target_share`
- `room_returning_prior_carry_share`
- `room_departed_prior_carry_share`
- `room_returning_prior_snap_share` when available
- `room_departed_prior_snap_share` when available
- `room_continuity_coverage_flag`

Shares describe previously observed opportunity only. They are not forecasts of who inherits vacated opportunity.

### Team continuity

- `offensive_players_returning_count`
- `offensive_players_added_count_pregame`
- `offensive_players_departed_count_pregame`
- `defensive_players_returning_count`
- `defensive_players_added_count_pregame`
- `defensive_players_departed_count_pregame`
- `games_since_head_coach_change_prior`
- `games_since_offensive_playcaller_change_prior`
- `games_since_defensive_playcaller_change_prior`
- `team_continuity_coverage_flag`

### OL/DL continuity hooks

When qualified personnel identity is available, materialize descriptive unit continuity without assigning causal blocking responsibility:

- `ol_returning_players_count`
- `ol_prior_start_continuity_count`
- `ol_known_absence_count_pregame`
- `dl_returning_players_count`
- `dl_prior_start_continuity_count`
- `dl_known_absence_count_pregame`

Any later geometry-derived blocker/rusher interaction features must live in their own namespace and preserve interaction-vs-responsibility semantics.

## Identity rules

Stable player IDs are required whenever available. Name-only joins are not canonical. Team aliases must use the repository's canonical normalization. A player changing teams is represented as an explicit continuity transition, not inferred from name matching alone when a stable ID exists.

## Room membership

Room membership is a pregame descriptive roster/depth state, not a target-game usage classification. A player's room may be sourced from canonical position plus qualified pregame roster/depth evidence. Hybrid/ambiguous players must preserve an explicit ambiguity flag rather than being forced into a hindsight category.

## Event-ledger relationship

Transactions, availability, depth movement, coaching changes, play-caller changes, and personnel changes should be sourced through `ROLE_ENVIRONMENT_EVENT_LEDGER_CONTRACT_V1` whenever possible.

The continuity materializer may derive counters and turnover summaries from those events, but every derived value must retain source event IDs or canonical-history lineage.

## Historical reuse rule

Do not independently rebuild player game history. Prior usage/team identity must reuse the canonical historical base under `HISTORICAL_DATA_REUSE_POLICY_V1`. Rehydrate only when an exact retained artifact is unavailable.

## Missingness

Unknown is not zero. Each metric family must expose coverage state. Examples:

- `known`
- `partial`
- `source_unavailable`
- `not_applicable`

A zero departure count is valid only when roster/event coverage is sufficient to distinguish zero from unknown.

## QA invariants

1. No target-game outcome or usage field may influence a target row.
2. No event observed at or after kickoff may influence strict-prior continuity state.
3. Every derived turnover/share field must trace to canonical prior history and/or eligible event IDs.
4. Team changes must be detectable from stable player identity across prior eligible history.
5. Departed opportunity is measured from prior observed opportunity and never automatically reassigned.
6. Room membership ambiguity must remain explicit.
7. Geometry-based proximity/interaction evidence may join later but may not be relabeled as assignment/responsibility here.
8. Missing source coverage must remain distinguishable from a true zero.

## Forbidden fields/labels

This layer may not emit hindsight or betting labels such as:

- `breakout_candidate`
- `role_upgrade_expected`
- `good_matchup`
- `under_candidate`
- `over_candidate`
- `expected_vacated_share_gain`
- any learned multiplier trained on target/future outcomes.

## Downstream authorization boundary

Authorized now:

- source ingestion;
- identity normalization;
- strict-prior joins;
- descriptive continuity materialization;
- QA and coverage audits;
- versioned manifests.

Not authorized by this contract:

- predictive lift testing;
- model weighting/calibration;
- production integration;
- bet-selection changes.

## Disposition

`PERSONNEL_CONTINUITY_CONTRACT_V1_FROZEN`
