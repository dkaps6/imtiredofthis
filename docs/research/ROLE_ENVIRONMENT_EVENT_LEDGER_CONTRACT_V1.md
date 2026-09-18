# Role / Environment Event Ledger Contract V1

**Status:** ENGINEERING CONTRACT — NO PRODUCTION SCIENCE CHANGE

## Purpose

Define a leakage-safe, source-provenanced event ledger for changes in a player's football environment without converting those events into outcome-trained predictive rules.

## Grain

One row per sourced event affecting a player, room, team, or coaching environment.

Required identity fields:

- `event_id`
- `effective_at`
- `observed_at`
- `season`
- `team`
- `player_id` when player-specific
- `room` when position-room-specific
- `event_family`
- `event_type`

## Event families

Allowed V1 families:

- `transaction`
- `availability`
- `depth_role`
- `room_change`
- `team_change`
- `coaching_change`
- `playcaller_change`
- `personnel_change`

No performance outcome, betting result, residual, projection error, or postgame label may be an event family.

## Provenance

Every sourced row must carry:

- `source_name`
- `source_url_or_id`
- `source_published_at` when available
- `observed_at`
- `source_tier`
- `raw_statement_hash`
- `ingest_version`

Derived rows must additionally carry:

- `derived_from_event_ids`
- `derivation_version`
- `derivation_rule`

Sourced statements and derived interpretation must never occupy the same semantic field.

## Strict-prior eligibility

For target game kickoff `T`, an event is pregame-eligible only when:

`observed_at < T` AND `effective_at <= T`

If either timestamp is unknown, the event is ineligible for strict-prior modeling unless an explicit conservative timestamp rule has been approved and versioned.

Later corrections may update the ledger but must not retroactively alter what was knowable at the historical cutoff. Historical materialization must preserve the original `observed_at` boundary.

## Role-state materialization

The ledger may materialize descriptive pregame state such as:

- team tenure games;
- games since team change;
- games since coaching/play-caller change;
- room additions/departures before kickoff;
- active/inactive uncertainty flags;
- prior-game opportunity share deltas;
- room opportunity vacated by known departures;
- depth-chart movement when supported by a timestamped source.

These are descriptive context fields, not automatic projection adjustments.

## Opportunity-change formulas

Any derived opportunity descriptor must be versioned separately from source events. Example allowed descriptors:

- `prior_target_share`
- `prior_carry_share`
- `room_prior_target_share_departed`
- `room_prior_carry_share_departed`
- `games_since_role_event`

Forbidden in this layer:

- learned multipliers from future outcomes;
- labels such as `breakout`, `good_matchup`, `over_candidate`, or `under_candidate`;
- hindsight role classification based on target-game usage.

## Missingness

Unknown is not false. Each family must preserve explicit coverage/missingness indicators. Absence of an event means only `no_event_observed_under_available_sources`, not proof that no real-world change occurred.

## Join surfaces

Player-specific state joins to:

`season, week, game_id, player_id`

Team/room state joins to:

`season, week, game_id, team`

All joins must materialize the target kickoff cutoff and the latest eligible event timestamp used.

## QA invariants

1. Zero events with `observed_at >= target_kickoff` may influence a strict-prior target row.
2. Every derived field must trace to source event IDs or prior-game canonical history.
3. Target-game box score/statistics may not participate in role-state construction.
4. A source correction published after kickoff may not leak into the historical pregame view.
5. Event-family missingness must remain distinguishable from a true zero event count.

## Relationship to historical reuse

The ledger supplements canonical historical player/team game logs. It does not rebuild them. Prior-game usage and team identity should come from the existing canonical historical base whenever available.

## Scientific boundary

This contract authorizes source ingestion, timestamping, QA, and descriptive materialization only. It does not authorize predictive testing, weighting, calibration, bet-selection changes, or production integration.

## Disposition

`ROLE_ENVIRONMENT_EVENT_LEDGER_CONTRACT_V1_FROZEN`
