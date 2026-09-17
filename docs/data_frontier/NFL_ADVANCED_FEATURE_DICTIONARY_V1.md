# NFL Advanced Feature Dictionary V1 — Frozen Engineering Contract

**Status:** frozen engineering contract; research-only; no predictive experiment or production promotion authorized.

**Parent data-frontier handoff:** `66e9d5d41e7e4e6187ecc2ab317f8a6f2d52c748`

**Branch:** `data-frontier-advanced-feature-contract-v1`

## Purpose

This document freezes the first proprietary advanced-data feature layer derived from the three validated public NFL geometry laboratories already closed in the data-frontier program.

It does **not** restart source discovery, run candidate models, retune production, alter Issue #535, or authorize target-game use of information that is only known after kickoff or after the play.

The authoritative machine-readable contract is:

- `docs/data_frontier/nfl_advanced_feature_dictionary_v1.json`
- schema: `docs/data_frontier/nfl_advanced_feature_dictionary_schema_v1.json`
- validator: `scripts/data_frontier/validate_advanced_feature_dictionary_v1.py`
- focused tests: `tests/test_advanced_feature_dictionary_v1.py`

## Frozen source authorities

| Source key | Official source | Season represented | Frozen source hash | Hash scope | Validated disposition |
|---|---|---:|---|---|---|
| `BDB2021_ROUTE_GEOMETRY` | Kaggle `nfl-big-data-bowl-2021` | 2018 | `55de76561799514779f3fd64b57c02a52430596f4f104c2b6060f3df8fd9e1b4` | source manifest | `BDB2021_ROUTE_NEAREST_DEFENDER_GEOMETRY_LAB_VALIDATED` |
| `BDB2023_PROTECTION_GEOMETRY` | Kaggle `nfl-big-data-bowl-2023` | 2021 | `1c3e1eb6fcd0cf85807c649804fa0c72421c76e973e40a742523f1dd3a929182` | source manifest | `BDB2023_PROTECTION_INTERACTION_GEOMETRY_LAB_VALIDATED` |
| `BDB2026_THROW_WINDOW` | Kaggle `nfl-big-data-bowl-2026-analytics` | 2023 | `228554c6600ac4e73529e5b6309193c8a3371acca7b4bc59792188acd814fb07` | official corpus | `BDB2026_THROW_WINDOW_GEOMETRY_LAB_VALIDATED_WITH_OUTPUT_DEFENDER_SCOPE_LIMIT` |

All three remain **research-access sources**. Authenticated competition access does not imply unrestricted production, live-feed, or commercial deployment rights. Raw competition CSV/video files remain ephemeral in GitHub Actions and are not committed to the repo.

## Temporal firewall

Every V1 field has exactly one temporal class:

| Class | V1 meaning | Count |
|---|---|---:|
| `PREGAME_HISTORICAL_DERIVABLE` | Explicitly named `hist_*` summary computed only from completed observations strictly before target kickoff. Target game is excluded. | 13 |
| `TARGET_GAME_POST_KICKOFF` | Raw geometry that becomes known only after the target game begins. Never usable directly in the target game's pregame model. | 11 |
| `RETROSPECTIVE_VALIDATION_ONLY` | Outcome-window geometry or official retrospective semantic labels reserved for validation, QA and descriptive benchmarks. | 23 |

There is no implicit temporal conversion. A raw post-kickoff field does not become pregame-safe merely because it can be stored. Pregame use requires a separately named `hist_*` field with a strict-prior cutoff.

### Strict-prior cutoff

For every pregame historical field:

1. require `observation_end_time < target_kickoff_utc`;
2. if exact ordering is unavailable, use completed prior NFL weeks only;
3. target-game rows are forbidden;
4. same-game partial history is forbidden;
5. identity must be resolved;
6. source hash must match the frozen registry.

## Semantic firewalls

### Defender proximity is not responsibility

Nearest and second-nearest defender fields are geometric proximity only.

They must never be renamed or interpreted as:

- primary coverage defender;
- matchup assignment;
- shadow assignment;
- banjo/exchange responsibility;
- bracket/help responsibility;
- proof of man/zone responsibility.

This explicitly preserves the M84 lesson.

### BDB 2023 blocked-player semantics

`pff_nflIdBlockedPlayer` is frozen as a **blocked-player / blocking-interaction identity**.

It is not a universal primary blocker-rusher assignment. The source corpus includes meaningful `Pass Route` chip/release interactions and referenced players whose same-play source role can be `Coverage`; V1 preserves those source semantics rather than coercing them.

### BDB 2026 coverage semantics

`team_coverage_man_zone` and `team_coverage_type` are play/team labels.

They do not identify which individual defender was responsible for the targeted receiver. Post-release defender metrics use only defenders present in the official `player_to_predict=true` output scope and abstain when that official scope contains no defender.

## Frozen feature families

### BDB 2021 — route / proximity geometry

Raw target-game geometry:

- `route_label`
- `snap_nearest_defender_distance_yards`
- `snap_second_nearest_defender_distance_yards`
- `throw_nearest_defender_distance_yards`
- `throw_second_nearest_defender_distance_yards`
- `snap_to_throw_nearest_defender_delta_yards`

Retrospective-only arrival geometry:

- `arrival_nearest_defender_distance_yards`
- `arrival_second_nearest_defender_distance_yards`

Pregame strict-prior summaries:

- `hist_player_route_throw_nearest_defender_median_yards`
- `hist_player_route_throw_second_defender_median_yards`
- `hist_player_route_throw_spacing_gap_median_yards`
- `hist_player_route_throw_geometry_sample_count`

Important limitation: BDB 2021 has route labels but no semantic target field. Throw-frame route-runner geometry is therefore not automatically targeted-receiver geometry.

### BDB 2023 — protection / interaction geometry

Raw or retrospective interaction fields:

- `blocker_target_snap_distance_yards`
- `blocker_target_min_distance_yards`
- `blocker_target_terminal_distance_yards`
- `blocker_target_time_to_min_distance_seconds`
- `blocker_target_shared_protection_frames`
- `protection_window_length_seconds`
- `block_interaction_role`
- `blocked_defender_source_role`
- `chip_release_interaction_flag`
- `terminal_frame_fallback_flag`

Pregame strict-prior summaries:

- `hist_blocker_snap_distance_median_yards`
- `hist_blocker_min_distance_median_yards`
- `hist_blocker_time_to_min_distance_median_seconds`

The frozen snap/terminal event-reconciliation logic is inherited exactly from the validated all-weeks geometry checkpoint. Missing snap remains an abstention; terminal last-frame fallback remains explicit lower-confidence provenance.

### BDB 2026 Analytics — throw-window geometry

Target-game release geometry:

- `receiver_release_nearest_defender_distance_yards`
- `receiver_release_second_defender_distance_yards`
- `receiver_release_defenders_within_2yd_count`
- `receiver_release_defenders_within_3yd_count`

Retrospective landing/post-release validation fields:

- `receiver_release_target_to_land_distance_yards`
- `release_nearest_defender_to_landing_zone_yards`
- `terminal_receiver_to_land_distance_yards`
- `terminal_nearest_predicted_defender_to_target_yards`
- `terminal_nearest_predicted_defender_to_land_yards`
- `postrelease_min_nearest_predicted_defender_distance_yards`
- `postrelease_closing_delta_yards`

Official semantic labels, validation-only:

- `source_target_route_label`
- `source_team_coverage_man_zone`
- `source_team_coverage_type`

Pregame strict-prior receiver summaries:

- `hist_receiver_release_nearest_defender_median_yards`
- `hist_receiver_release_second_defender_median_yards`
- `hist_receiver_release_crowding_2yd_rate`
- `hist_receiver_release_crowding_3yd_rate`
- `hist_receiver_release_geometry_sample_count`
- `hist_receiver_route_release_nearest_defender_median_yards`

Retrospective route/coverage benchmarks:

- `route_x_man_zone_release_nearest_defender_median_yards`
- `route_x_coverage_family_release_nearest_defender_median_yards`

The route-conditioned historical lookup does **not** authorize using a realized target-game route label pregame. Any eventual target-game use would require a separately validated, pregame-safe route tendency/scenario mechanism.

## Frozen historical support thresholds

These are engineering abstention thresholds, not predictive tuning parameters:

| Historical summary family | Minimum qualifying strict-prior observations | Confidence bands |
|---|---:|---|
| player × route throw geometry | 5 | LOW 5–9; MEDIUM 10–24; HIGH 25+ |
| blocker interaction history | 10 | LOW 10–19; MEDIUM 20–49; HIGH 50+ |
| targeted receiver release history | 8 | LOW 8–15; MEDIUM 16–39; HIGH 40+ |
| targeted receiver × route release history | 5 | LOW 5–9; MEDIUM 10–24; HIGH 25+ |

These thresholds exist so missing/sparse evidence produces an explicit abstention rather than a fabricated value. They are not evidence that the associated feature is predictive.

## What V1 explicitly forbids

V1 does not authorize:

- predictive experiments;
- model selection;
- production promotion;
- sportsbook/market inputs;
- target-game post-kickoff leakage into pregame features;
- BDB 2026 landing-point or post-release geometry as a pregame predictor;
- exact defender-responsibility claims from proximity;
- universal primary blocker-rusher claims from `pff_nflIdBlockedPlayer`;
- use of BDB 2024/2025 withdrawn source shells;
- committing raw Kaggle CSV/video files.

## Validation requirements

The dedicated validator must pass before this contract is considered structurally valid:

```bash
python scripts/data_frontier/validate_advanced_feature_dictionary_v1.py
PYTHONPATH="${PWD}" pytest -q tests/test_advanced_feature_dictionary_v1.py
```

The validator checks, at minimum:

- the three frozen source hashes;
- unique field names;
- mandatory per-field source/provenance fields;
- the temporal enum;
- explicit target-game exclusion for pregame summaries;
- `hist_*` naming for pregame fields;
- no landing/post-release fields promoted into V1 pregame use;
- explicit abstention logic;
- source-rights/raw-data policy;
- research-only / not-promoted status;
- nearest-defender and blocked-player semantic firewalls.

## Disposition

**`NFL_ADVANCED_FEATURE_DICTIONARY_V1_FROZEN_PENDING_CLEAN_CI`**

Once a clean checkout passes the dedicated validator/tests, V1 is ready for the next **engineering-only** phase: deterministic ephemeral materializers and sanitized QA summaries for these exact fields.

That next phase still does not authorize predictive testing. Any candidate experiment must be separately coordinated with the active research lane after the feature materializers and temporal audits are frozen.
