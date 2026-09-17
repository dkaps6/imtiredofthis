# NFL Data Frontier — Normalized Relational Schema V1

**Status:** frozen Phase-0 engineering contract. Documentation only; no predictive experiment, no production-science change.

**Authority base:** `docs/research/NFL_DATA_INVENTORY_AND_NEW_INFORMATION_FRONTIER.md` and `docs/research/NFL_DATA_INVENTORY_SOURCE_VERIFICATION_2026-09-16.md`.

## Purpose

Define the minimum durable schema for a proprietary NFL relationship-level data platform. The schema is deliberately richer than a final feature table: raw observations, labels, confidence, provenance and QA must remain recoverable so downstream features can be re-derived without re-charting film.

This schema is designed first for the Big Data Bowl 2024 tackling/contact benchmark, but its keys and provenance fields are intentionally reusable for routes, coverage responsibility, blocker-rusher relationships and later video-derived tracking.

## Non-negotiable principles

1. **Store raw relationships, not just aggregates.**
2. **Every derived label carries provenance and version.**
3. **Machine output and human-reviewed truth are distinct fields.**
4. **Abstention is valid.** Unknown/ambiguous labels are not silently guessed.
5. **Temporal integrity is explicit.** Observed target-game data are postgame observations; only strictly-prior summaries may later enter a pregame model.
6. **Source licensing is metadata.** Public competition access is not equivalent to unrestricted production rights.
7. **Identity confidence is first-class.** A geometry row with the wrong player identity is not acceptable ground truth.

---

## Canonical identifiers

All tables should use these identifiers wherever applicable:

- `season` — integer NFL season.
- `week` — integer regular-season week where available.
- `game_id` — source-stable game identifier.
- `play_id` — source-stable play identifier within game.
- `frame_id` — monotonically increasing source frame identifier within play.
- `nfl_id` — source player ID when available.
- `player_key` — internal normalized player identity key.
- `team` / `club` — canonical team abbreviation.
- `source_dataset` — e.g. `BDB_2024`, `BDB_2025`, `ALL22_V1`.
- `source_version` — immutable version string/checksum for the input artifact.
- `label_version` — semantic-label rule/model version.
- `generated_at_utc` — timestamp of derivative generation.

`game_id + play_id` is the minimum play key. `game_id + play_id + frame_id + nfl_id` is the canonical player-frame key.

---

## Table 1 — `games`

One row per game.

Required fields:

- `season`
- `week`
- `game_id`
- `game_date`
- `home_team`
- `away_team`
- `source_dataset`
- `source_version`

Optional metadata:

- `stadium`
- `roof`
- `surface`
- `kickoff_utc`

No betting or sportsbook fields belong in this data-frontier layer.

---

## Table 2 — `plays`

One row per play.

Required fields:

- `game_id`
- `play_id`
- `season`
- `week`
- `possession_team`
- `defensive_team`
- `down`
- `yards_to_go`
- `yardline_side`
- `yardline_number`
- `absolute_yardline_number` where source supports it
- `play_direction`
- `play_description`
- `source_dataset`
- `source_version`

Recommended event-frame fields:

- `snap_frame_id`
- `handoff_frame_id`
- `pass_release_frame_id`
- `pass_arrival_frame_id`
- `first_contact_frame_id`
- `tackle_frame_id`
- `play_end_frame_id`

Each event-frame field may be null when not relevant or not confidently resolved.

---

## Table 3 — `players`

One row per source player identity.

Required fields:

- `nfl_id`
- `player_key`
- `display_name`
- `position`
- `team`
- `season`
- `source_dataset`

Recommended:

- `jersey_number`
- `height`
- `weight`
- `birth_date`

Identity metadata:

- `identity_method` — `source_id`, `roster_join`, `jersey_ocr`, `human_review`, etc.
- `identity_confidence` — 0-1.
- `identity_review_status` — `UNREVIEWED`, `AUTO_ACCEPTED`, `HUMAN_CONFIRMED`, `AMBIGUOUS`.

---

## Table 4 — `player_tracks`

One row per player per frame.

Required fields:

- `game_id`
- `play_id`
- `frame_id`
- `frame_time`
- `nfl_id`
- `player_key`
- `team`
- `x`
- `y`
- `speed`
- `acceleration`
- `orientation`
- `direction`
- `source_dataset`
- `source_version`

Recommended:

- `distance_since_prior_frame`
- `event`
- `is_ball_carrier`
- `tracking_confidence`
- `identity_confidence`

For video-derived tracks also store:

- `pixel_x`
- `pixel_y`
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`
- `homography_version`
- `detector_version`
- `tracker_version`

The video pipeline must never overwrite source-truth NGS coordinates; reconstructed coordinates live as a distinct source/version.

---

## Table 5 — `football_tracks`

One row per football frame when available.

Required:

- `game_id`
- `play_id`
- `frame_id`
- `frame_time`
- `x`
- `y`
- `source_dataset`
- `source_version`

Recommended:

- `event`
- `ball_tracking_confidence`

For video-derived data, ball position may be absent. Event timing can still be retained separately in `plays` / `events`.

---

## Table 6 — `events`

Normalized play events.

Required fields:

- `game_id`
- `play_id`
- `frame_id`
- `event_type`
- `source_dataset`
- `source_event_label`
- `event_confidence`
- `event_method`
- `label_version`

Core event vocabulary for the Phase-0 contact benchmark:

- `BALL_SNAP`
- `HANDOFF`
- `FIRST_CONTACT`
- `TACKLE`
- `MISSED_TACKLE`
- `PLAY_END`

Later extensions:

- `PASS_RELEASE`
- `PASS_ARRIVAL`
- `ROUTE_BREAK`
- `BLOCK_ENGAGEMENT_START`
- `BLOCK_ENGAGEMENT_END`

---

## Table 7 — `contacts`

One row per contact episode between the ball carrier and defender.

Required fields:

- `game_id`
- `play_id`
- `contact_id`
- `ball_carrier_id`
- `defender_id`
- `contact_start_frame_id`
- `contact_end_frame_id`
- `contact_x`
- `contact_y`
- `contact_type`
- `contact_outcome`
- `source_dataset`
- `label_version`

Recommended derived geometry:

- `carrier_speed_at_contact`
- `carrier_accel_at_contact`
- `defender_speed_at_contact`
- `closing_speed`
- `pursuit_angle_deg`
- `relative_heading_deg`
- `distance_to_los_at_contact`
- `yards_from_handoff_to_contact`
- `yards_after_contact_episode`
- `simultaneous_defender_count`
- `is_first_contact`
- `is_tackle_contact`
- `is_missed_tackle`
- `is_broken_contact`

QA fields:

- `machine_confidence`
- `human_review_status`
- `human_reviewer_id` (pseudonymous/internal identifier only)
- `human_label`
- `final_label_source`

---

## Table 8 — `tackles`

One row per defender-play tackle attribution.

Required:

- `game_id`
- `play_id`
- `defender_id`
- `ball_carrier_id`
- `tackle`
- `assist`
- `forced_fumble`
- `missed_tackle`
- `source_dataset`
- `label_version`

This table preserves source PFF/competition labels separately from geometry-derived contact events.

---

## Table 9 — `rb_contact_features`

One row per ball-carrier play. This is a *derived convenience table*, not the authoritative raw record.

Required:

- `game_id`
- `play_id`
- `ball_carrier_id`
- `handoff_frame_id`
- `first_contact_frame_id`
- `tackle_frame_id`
- `first_contact_x`
- `first_contact_y`
- `yards_before_contact`
- `yards_after_first_contact`
- `carrier_speed_at_first_contact`
- `nearest_defender_distance_at_handoff`
- `nearest_defender_distance_at_first_contact`
- `nearest_free_defender_distance_at_handoff`
- `nearest_free_defender_distance_at_first_contact`
- `max_closing_speed_pre_contact`
- `primary_tackler_id`
- `missed_tackler_count`
- `defenders_in_contact_window`
- `feature_version`
- `source_dataset`

Optional later fields:

- `unblocked_defender_flag`
- `blocker_proximity_at_contact`
- `intended_gap`
- `run_concept`
- `point_of_attack_x`

Those optional scheme fields are deliberately excluded from the Phase-0 acceptance gate because they require additional semantic charting.

---

## Future table — `routes`

One row per eligible receiver route.

Planned fields:

- `receiver_id`
- `alignment`
- `route_name`
- `release_frame_id`
- `break_frame_id`
- `target_frame_id`
- `arrival_frame_id`
- path geometry/embedding
- targeted flag
- source truth fields such as `wasRunningRoute`, `routeRan`
- machine confidence / human review

---

## Future table — `coverage_assignments`

One row per receiver-route-defender relationship segment.

Planned fields:

- `receiver_id`
- `primary_defender_id`
- `secondary_defender_id`
- `assignment_start_frame_id`
- `assignment_end_frame_id`
- `coverage_assignment`
- `press_flag`
- `inside_leverage`
- `outside_leverage`
- `top_leverage`
- `help_defender_id`
- `bracket_flag`
- `responsibility_confidence`
- source PFF primary/secondary matchup IDs where available

Nearest-defender geometry is never sufficient by itself to populate semantic responsibility fields.

---

## Future table — `blocking_assignments`

One row per blocker-rusher relationship segment.

Planned fields:

- `blocker_id`
- `rusher_id`
- `assignment_start_frame_id`
- `assignment_end_frame_id`
- `engagement_start_frame_id`
- `engagement_end_frame_id`
- `chip_help_flag`
- `double_team_flag`
- `pressure_allowed`
- `time_to_pressure_allowed`
- `assignment_confidence`

BDB source fields such as `blockedPlayerNFLId1/2/3` remain raw-source columns in a staging table and are normalized into this relationship table.

---

## QA and provenance contract

Every derived table must emit:

- input row counts;
- unique game/play/player counts;
- null rates for required columns;
- duplicate-key count;
- source-version checksum where practical;
- algorithm/label version;
- confidence distribution;
- abstention count/rate;
- human-reviewed count/rate;
- source-vs-derived agreement where source truth exists.

Any run that cannot establish source lineage must fail closed.

## Temporal-use contract

This schema stores postgame observations. It does **not** authorize target-game usage in a pregame model.

If predictive features are ever constructed later, they must be functions only of completed prior games and must be generated under a separately frozen research plan.

## Phase-0 definition of done

Schema V1 is satisfied when the BDB-2024 ingestion/benchmark pipeline can populate at minimum:

- `games`
- `plays`
- `players`
- `player_tracks`
- `football_tracks`
- `events`
- `tackles`
- `rb_contact_features`

with deterministic keys, source provenance and QA output, without accessing sportsbook data or production model code.