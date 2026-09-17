# Authenticated NFL Kaggle Availability — 2026-09-17

## Scope

This is a read-only source-availability checkpoint for the isolated NFL data-frontier lane.

- Branch: `data-frontier-phase0-bdb-contact-v1`
- Probe workflow: `.github/workflows/data-frontier-kaggle-nfl-availability.yml`
- Corrected workflow run: `35263529056`
- Probe version: `NFL_KAGGLE_AUTHENTICATED_AVAILABILITY_V2`
- Kaggle token: authenticated and active; value never printed or persisted in repo
- Raw competition data downloaded: **NO**
- Raw competition data uploaded: **NO**
- Predictive experiment: **NO**
- Production science changed: **NO**
- Issue #535 changed: **NO**

The probe used the authenticated Kaggle CLI only to list competition-visible files. For paginated listings, the observed count below is only the first visible page; access classification does not depend on complete pagination.

## Availability Matrix

| Competition | Kaggle slug | Status | Authenticated evidence |
|---|---|---|---|
| Big Data Bowl 2020 | `nfl-big-data-bowl-2020` | **ACCESSIBLE_REAL_FILES** | `train.csv` is visible; legacy encrypted competition-test assets also remain listed |
| Big Data Bowl 2021 | `nfl-big-data-bowl-2021` | **ACCESSIBLE_REAL_FILES** | `games.csv`, `players.csv`, `plays.csv`, and `week1.csv` through `week17.csv` are visible |
| Big Data Bowl 2022 | `nfl-big-data-bowl-2022` | **ACCESSIBLE_REAL_FILES** | `PFFScoutingData.csv`, `games.csv`, `players.csv`, `plays.csv`, and tracking for 2018/2019/2020 are visible |
| Big Data Bowl 2023 | `nfl-big-data-bowl-2023` | **ACCESSIBLE_REAL_FILES** | `pffScoutingData.csv`, `games.csv`, `players.csv`, `plays.csv`, and `week1.csv` through `week8.csv` are visible |
| Big Data Bowl 2024 | `nfl-big-data-bowl-2024` | **HOST_WITHDRAWN_OR_README_SHELL** | only `README` is visible; size is exactly 78 bytes |
| Big Data Bowl 2025 | `nfl-big-data-bowl-2025` | **HOST_WITHDRAWN_OR_README_SHELL** | only `README` is visible; size is exactly 78 bytes |
| Big Data Bowl 2026 Analytics | `nfl-big-data-bowl-2026-analytics` | **ACCESSIBLE_REAL_FILES** | paginated listing exposes `supplementary_data.csv`, 2023 weekly input tracking, and output files |
| Big Data Bowl 2026 Prediction | `nfl-big-data-bowl-2026-prediction` | **ACCESSIBLE_REAL_FILES** | paginated listing exposes `test.csv`, `test_input.csv`, and 2023 weekly training inputs |
| NFL Player Contact Detection | `nfl-player-contact-detection` | **ACCESSIBLE_REAL_FILES** | paginated listing exposes tracking/metadata plus Endzone, Sideline, and All29 video files |

## Important source conclusions

### 1. BDB 2024 is genuinely blocked at the official source

The authenticated account can access the competition endpoint, but Kaggle exposes only a single 78-byte `README`. The user independently viewed the README text in the Kaggle UI stating that the dataset was removed at the request of the host. Therefore the Phase-0 BDB-2024 real-corpus benchmark remains preserved but source-blocked.

Recommended repository status for the 2024 tackle corpus:

`SOURCE_BLOCKED_HOST_WITHDRAWN`

Do not use an unofficial mirror as canonical input without separate rights/source verification.

### 2. BDB 2025 is also unavailable from the official competition corpus

The authenticated listing likewise exposes only a single 78-byte `README`. Treat this as a withdrawn/readme-shell source unless a separately authorized source is identified.

Recommended repository status:

`SOURCE_BLOCKED_HOST_WITHDRAWN_OR_SHELL`

### 3. BDB 2023 is a major positive result

The official authenticated corpus is still available, including `pffScoutingData.csv` plus eight weeks of tracking. This restores an executable public ground-truth lane for pass-protection / blocker-rusher research and QA.

This should be the first semantic-responsibility corpus ingested after source packaging/provenance is frozen.

### 4. BDB 2021 is fully useful for receiver/defender geometry

The authenticated listing exposes all 17 weekly tracking files plus games/players/plays. This is a strong ground-truth laboratory for route/coverage geometry and historical receiver-defender interaction work.

### 5. BDB 2022 provides multi-season tracking plus PFF scouting context

The corpus exposes 2018, 2019, and 2020 tracking in addition to PFF scouting data. This is useful for general tracking normalization, special-teams/CV validation, and cross-season identity/geometry infrastructure.

### 6. BDB 2026 and Player Contact Detection are live, executable frontiers

Both 2026 competitions expose real 2023 files. Player Contact Detection exposes real video plus tracking/metadata and is therefore a high-value lawful laboratory for the eventual video -> calibration -> tracking -> contact reconstruction stack.

## Execution consequence

The data-frontier program is **not blocked overall**.

The recommended next source execution order is:

1. **BDB 2023** — download ephemerally, fingerprint, normalize, and audit exact PFF pass-protection responsibility semantics.
2. **BDB 2021** — normalize receiver/defender trajectories and route geometry.
3. **BDB 2026 Analytics / Prediction** — normalize 2023 throw-window / post-release geometry and compare schema overlap with the 2021 tracking stack.
4. **NFL Player Contact Detection** — create the lawful CV/contact benchmark lane using tracking + Endzone/Sideline/All29 video.
5. **BDB 2022** — add cross-season 2018-2020 tracking/PFF validation support.
6. Preserve **BDB 2024** and **BDB 2025** code paths as source-blocked; do not fabricate or substitute unofficial data.

No predictive feature experiment should run until each downloaded corpus passes its own file-contract, provenance, normalization, and fidelity/semantic QA gates.
