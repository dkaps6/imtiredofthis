# BDB 2026 Analytics Semantic Checkpoint — 2026-09-17

## Scope

Isolated data-frontier source/semantic validation only. No predictive experiment, production change, sportsbook change, Issue #535 change, or frozen WR/RB/QB/TE science change.

- Branch: `data-frontier-phase0-bdb-contact-v1`
- Workflow: `.github/workflows/data-frontier-bdb2026-analytics-semantic-contract.yml`
- Canonical run: `35266739183`
- Source commit: `64a2fdd53bc3ab145452ee7f39c46aba4f8e8312`
- Run conclusion: **SUCCESS**
- Audit version: `BDB2026_ANALYTICS_SEMANTIC_CONTRACT_V1`

## Core semantic contract

The official BDB 2026 Analytics package forms a clean 18-week pre-throw -> post-throw tracking contract.

### Input/output integrity

- total input rows: **4,880,579**
- total input player-plays: **173,150**
- total output rows: **562,936**
- total output player-plays: **46,045**
- output player-plays missing from input: **0**
- output player-plays with `player_to_predict = false`: **0**
- output frame-count mismatches versus `num_frames_output`: **0**
- plays with multiple ball landing coordinates: **0**
- plays with multiple `num_frames_output` values: **0**
- input plays missing supplementary metadata: **0**

All 18 weekly files pass the same structural contract independently.

## Player-role semantics

Observed pre-throw tracking roles:

- `Defensive Coverage`
- `Other Route Runner`
- `Passer`
- `Targeted Receiver`

`player_side` cleanly separates `Defense` and `Offense` rows. The post-throw output contains only the player-plays flagged `player_to_predict = true` in input, with exact output lengths matching `num_frames_output`.

This establishes `player_to_predict` as a reliable source selector for the players whose post-release trajectories are provided by the official output corpus.

## Supplementary football labels

`supplementary_data.csv` contains **18,009 unique plays** and joins completely to the tracking input play universe.

The file includes both 2023 and 2024 metadata rows; the published weekly input/output tracking family audited here is the 2023 18-week corpus.

### Targeted receiver route

- completeness: **18,005 / 18,009** non-null
- null: **4**
- 12 route classes:
  - ANGLE
  - CORNER
  - CROSS
  - FLAT
  - GO
  - HITCH
  - IN
  - OUT
  - POST
  - SCREEN
  - SLANT
  - WHEEL

### Coverage family

`team_coverage_man_zone`:
- non-null: **18,004 / 18,009**
- `MAN_COVERAGE`: 5,221
- `ZONE_COVERAGE`: 12,783
- null: 5

`team_coverage_type`:
- non-null: **18,004 / 18,009**
- `COVER_0_MAN`: 781
- `COVER_1_MAN`: 4,108
- `COVER_2_MAN`: 332
- `COVER_2_ZONE`: 2,518
- `COVER_3_ZONE`: 5,664
- `COVER_4_ZONE`: 2,860
- `COVER_6_ZONE`: 1,693
- `PREVENT`: 48
- null: 5

### Receiver alignment

Observed alignment families include:
- 1x1
- 2x1
- 2x2
- 3x0
- 3x1
- 3x2
- 3x3
- 4x0
- 4x1

## Interpretation

This source is substantially stronger than a generic tracking-only slice. It provides, on the same play contract:

1. pre-throw offensive and defensive trajectories;
2. explicit `Targeted Receiver`, `Other Route Runner`, `Passer`, and `Defensive Coverage` roles;
3. exact players whose post-throw trajectories are supplied;
4. a single ball landing coordinate and output horizon per play;
5. post-release trajectories with perfect frame-count consistency;
6. targeted-receiver route;
7. receiver alignment;
8. team man/zone and detailed coverage-family labels.

This does **not** by itself identify exact defender responsibility for the targeted receiver. Team coverage labels are team/play semantics, not a defender-assignment label. Do not relabel nearest defender as responsible defender.

## Disposition

**`BDB2026_ANALYTICS_SEMANTIC_CONTRACT_VALIDATED`**

This is a validated ground-truth/geometry laboratory, not a production model promotion.

Recommended next isolated task:

Build a 2023 throw-window geometry benchmark using the validated contract, including targeted-receiver and defender geometry at the final pre-throw frame, ball-landing geometry, and post-release arrival trajectories, while preserving the distinction between proximity and exact coverage responsibility.
