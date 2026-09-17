# NFL Data Frontier — Phase-1 Engineering Handoff — 2026-09-17

## Purpose

This closes the isolated NFL advanced-data inventory / source-validation lane and hands the validated data laboratories to the engineering lane.

Repo: `dkaps6/imtiredofthis`

Canonical handoff branch:
`data-frontier-bdb2026-throw-window-benchmark`

This work was deliberately isolated from production science and GitHub Issue #535. It did **not** run predictive candidate experiments, retune models, change production, or redefine frozen WR/RB/QB/TE science.

## What the engineering chat should NOT redo

Do not restart:
- Kaggle source discovery;
- Kaggle authentication debugging;
- competition availability probing;
- BDB 2021 route-label validation;
- BDB 2023 blocker/blocked-player semantic validation;
- BDB 2026 Analytics input/output semantic validation;
- the BDB 2026 throw-window benchmark;
- BDB 2024/2025 access attempts unless an independently authorized source becomes available.

The source/semantic work below is already validated and should be treated as the starting contract.

---

## Canonical validated labs

### 1. BDB 2021 — 2018 route / nearest-defender geometry

Canonical document:
`docs/data_frontier/BDB2021_ROUTE_GEOMETRY_CHECKPOINT_2026-09-17.md`

Official Kaggle source:
`nfl-big-data-bowl-2021`

Source-manifest SHA-256:
`55de76561799514779f3fd64b57c02a52430596f4f104c2b6060f3df8fd9e1b4`

Validated facts:
- all 17 weekly tracking files available;
- 19,239 plays;
- 78,343 route-labeled player-plays;
- zero player-plays with conflicting non-null route labels;
- zero plays missing football tracking;
- 78,312 snap nearest-defender geometry rows;
- 77,833 throw geometry rows;
- 64,879 arrival geometry rows.

Route taxonomy:
`GO, HITCH, FLAT, OUT, CROSS, IN, POST, SLANT, SCREEN, ANGLE, CORNER, WHEEL` plus 107 source `undefined` labels.

Pooled median nearest-defender distance:
- snap: 5.250 yd;
- throw: 2.898 yd;
- arrival: 2.889 yd.

**Mandatory semantic rule:** nearest defender is geometric proximity only. It is NOT true coverage responsibility, matchup assignment, shadow assignment, banjo/exchange responsibility, or proof of man/zone assignment.

Disposition:
`BDB2021_ROUTE_NEAREST_DEFENDER_GEOMETRY_LAB_VALIDATED`

Engineering value:
- route trajectory;
- receiver spacing;
- leverage/proximity;
- congestion;
- route-specific geometric distributions;
- historical route-shape infrastructure.

---

### 2. BDB 2023 — 2021 blocking-interaction / protection geometry

Canonical document:
`docs/data_frontier/BDB2023_BLOCKING_INTERACTION_GEOMETRY_CHECKPOINT_2026-09-17.md`

Official Kaggle source:
`nfl-big-data-bowl-2023`

Source-manifest SHA-256:
`1c3e1eb6fcd0cf85807c649804fa0c72421c76e973e40a742523f1dd3a929182`

Important semantic correction:
`pff_nflIdBlockedPlayer` must be treated as a **blocked-player / blocking-interaction identity**, not as a universal perfect primary blocker-rusher assignment.

Validated interaction counts:
- 46,526 rows contain a blocked-player ID;
- 44,526 source rows are `Pass Block`;
- 2,000 source rows are `Pass Route` chip/release-type interactions;
- 46,095 interactions point to a same-play defender labeled `Pass Rush`;
- 429 point to a defender labeled `Coverage`;
- 2 blocked-player references do not resolve to a same-play PFF row.

For applicable Pass Block rows after explicit NB/null exclusions, blocked-player identity is present on 99.9641%.

All-weeks Weeks 1-8 geometry:
- resolvable interactions: 46,524;
- reconstructed interactions: 46,396;
- pooled coverage: 99.7249%;
- snap geometry among reconstructed: 100%;
- terminal geometry among reconstructed: 100%.

Pooled median geometry:
- blocker-target distance at snap: 2.486 yd;
- minimum distance: 0.767 yd;
- terminal distance: 1.137 yd;
- time from snap to minimum distance: 1.9 sec;
- shared frames: 30.

Event reconciliation is frozen in the canonical checkpoint. Do not casually change snap/pass reconciliation thresholds to force 100% coverage. Missing/untrustworthy events are intentionally abstained.

Disposition:
`BDB2023_PROTECTION_INTERACTION_GEOMETRY_LAB_VALIDATED`

Engineering value:
- blocker/blocked-player trajectory features;
- engagement distance;
- time to engagement / minimum distance;
- chip/release interaction handling;
- protection-geometry QA;
- linking interaction geometry to role-scoped PFF pressure/outcome labels.

---

### 3. BDB 2026 Analytics — 2023 targeted-receiver / throw-window geometry

Semantic-contract document:
`docs/data_frontier/BDB2026_ANALYTICS_SEMANTIC_CHECKPOINT_2026-09-17.md`

Final benchmark document:
`docs/data_frontier/BDB2026_THROW_WINDOW_GEOMETRY_CHECKPOINT_2026-09-17.md`

Official Kaggle source:
`nfl-big-data-bowl-2026-analytics`

Official corpus SHA-256:
`228554c6600ac4e73529e5b6309193c8a3371acca7b4bc59792188acd814fb07`

Canonical benchmark Actions run:
`35268286020`

Validated semantic contract:
- 18 weeks of 2023 pre-throw input;
- 18 weeks of 2023 post-release output;
- supplementary metadata;
- 4,880,579 input rows;
- 173,150 input player-plays;
- 562,936 output rows;
- 46,045 output player-plays;
- zero output player-plays missing from input;
- zero output players with `player_to_predict = false`;
- zero output-frame-count mismatches versus `num_frames_output`;
- zero plays with multiple ball landing coordinates;
- zero input plays missing supplementary metadata.

Observed source roles:
- `Defensive Coverage`;
- `Other Route Runner`;
- `Passer`;
- `Targeted Receiver`.

Supplementary labels include:
- `route_of_targeted_receiver`;
- `receiver_alignment`;
- `team_coverage_man_zone`;
- `team_coverage_type`;
- `pass_length`;
- `pass_location_type`;
- `dropback_distance`;
- `play_action`.

Team coverage labels include:
- COVER_0_MAN;
- COVER_1_MAN;
- COVER_2_MAN;
- COVER_2_ZONE;
- COVER_3_ZONE;
- COVER_4_ZONE;
- COVER_6_ZONE;
- PREVENT.

### Throw-window benchmark

Published 2023 play universe used:
- 14,108 plays;
- 14,107 usable release geometries = 99.9929%;
- 14,107 / 14,107 targeted receivers have terminal post-release geometry;
- targeted-receiver role non-unique: 0;
- targeted receiver not `player_to_predict`: 0;
- targeted receiver missing from output: 0.

Post-release defender geometry:
- 12,966 / 14,107 plays = 91.912%;
- 1,141 plays contain no predicted defender in the official post-release output set.

That 1,141-play limitation is **official output scope**, not an identity/tracking failure.

Pooled release-frame medians:
- nearest defender to targeted receiver: 3.643 yd;
- second-nearest defender: 8.286 yd;
- targeted receiver to ball landing point: 5.294 yd;
- nearest defender to landing point: 7.057 yd.

Terminal post-release:
- targeted receiver to landing point median: 1.186 yd;
- nearest predicted defender to targeted receiver median: 2.394 yd (n=12,966);
- minimum post-release nearest predicted-defender distance median: 2.056 yd (n=12,966).

Descriptive man/zone geometry:
- MAN release nearest defender median: 2.138 yd;
- ZONE release nearest defender median: 4.295 yd.

These are descriptive source findings only, not causal or predictive conclusions.

**Mandatory semantic rule:** team coverage labels are play/team semantics. The corpus does NOT identify exact individual defender responsibility for the targeted receiver. Never relabel nearest defender as responsible defender.

Disposition:
`BDB2026_THROW_WINDOW_GEOMETRY_LAB_VALIDATED_WITH_OUTPUT_DEFENDER_SCOPE_LIMIT`

Engineering value:
- targeted-receiver separation/proximity at release;
- second-nearest-defender spacing;
- target-area crowding;
- landing-zone crowding;
- receiver-to-landing-point displacement;
- post-release closing geometry for official output defenders;
- route x coverage-family geometry.

---

## Source availability status

Canonical document:
`docs/data_frontier/KAGGLE_NFL_AUTHENTICATED_AVAILABILITY_2026-09-17.md`

Authenticated Kaggle status:

- BDB 2020: ACCESSIBLE_REAL_FILES
- BDB 2021: ACCESSIBLE_REAL_FILES
- BDB 2022: ACCESSIBLE_REAL_FILES
- BDB 2023: ACCESSIBLE_REAL_FILES
- BDB 2024: SOURCE_BLOCKED_HOST_WITHDRAWN
- BDB 2025: SOURCE_BLOCKED_HOST_WITHDRAWN_OR_SHELL
- BDB 2026 Analytics: ACCESSIBLE_REAL_FILES
- BDB 2026 Prediction: ACCESSIBLE_REAL_FILES
- NFL Player Contact Detection: ACCESSIBLE_REAL_FILES

BDB 2024 and 2025 each expose only the host-withdrawn 78-byte README shell through the authenticated official competition endpoint. Do not use an unofficial mirror as canonical input without separate rights/source verification.

Player Contact Detection remains a legitimate future lab because the authenticated source exposes tracking/metadata plus Endzone, Sideline, and All29 video. It was **not necessary to mine further to close this discovery phase**.

---

## Data handling / reproducibility contract

The repo intentionally does **not** contain raw Kaggle competition CSVs or videos.

The repository contains:
- workflows;
- schemas/contracts;
- source hashes/manifests;
- validation logic;
- semantic checkpoints;
- aggregate benchmark results;
- sanitized Actions artifacts.

Official competition files are:
1. authenticated with the existing GitHub Actions `KAGGLE_API_TOKEN` secret;
2. downloaded ephemerally to the Actions runner;
3. fingerprinted;
4. processed;
5. discarded after the run.

Do not print, commit, or expose the Kaggle token.

Do not upload raw competition rows simply for convenience. Preserve the existing provenance pattern unless licensing has been explicitly re-reviewed.

---

## Engineering handoff: what to build next

The data-discovery phase is complete. The next lane should be **versioned feature engineering**, not more random source hunting.

Start by creating a feature dictionary / schema that classifies every derived field by:

1. source corpus and source hash;
2. grain (play, player-play, interaction, frame/window);
3. deterministic definition;
4. temporal availability;
5. confidence / abstention rule;
6. whether it is:
   - historical/pregame-derivable under an eventual lawful live source,
   - target-game post-kickoff information,
   - retrospective ground-truth / validation only;
7. semantic limitations;
8. source/license lineage.

Candidate families already justified by validated labs include:

### Receiver / throw-window
- receiver_release_nearest_defender_distance
- receiver_release_second_defender_distance
- receiver_release_defenders_within_2yd
- receiver_release_defenders_within_3yd
- receiver_release_target_to_land_distance
- release_nearest_defender_to_landing_zone
- terminal_receiver_to_land_distance
- postrelease_min_nearest_predicted_defender_distance
- route_x_man_zone_geometry
- route_x_coverage_family_geometry

### Protection
- blocker_target_snap_distance
- blocker_target_min_distance
- blocker_target_terminal_distance
- time_to_min_distance
- shared_protection_frames
- block_interaction_role
- blocked_defender_source_role
- chip/release interaction flags
- terminal-frame-confidence / fallback provenance

### Route geometry
- snap_nearest_defender_distance
- throw_nearest_defender_distance
- arrival_nearest_defender_distance
- second_nearest_defender_distance
- route-specific spacing / congestion summaries

Names above are candidate engineering names, not frozen production names.

---

## Guardrails for engineering

1. **Do not equate nearest defender with coverage responsibility.**
2. **Do not convert `pff_nflIdBlockedPlayer` into an unconditional primary blocker-rusher assignment label.**
3. Preserve source role and chip/release interactions.
4. Preserve abstentions and event-confidence provenance.
5. Keep BDB 2026 post-release defender features explicitly scoped to official `player_to_predict` defenders.
6. Keep retrospective/post-release measurements separate from pregame/live-available features.
7. Do not run predictive tests simply because a feature can be computed. First freeze the feature contract and temporal rules.
8. Do not touch production science or Issue #535 unless the active research coordinator explicitly decides to consume these engineered fields.

---

## Recommended read order for the engineering chat

1. `AGENTS.md`
2. this handoff
3. `docs/data_frontier/KAGGLE_NFL_AUTHENTICATED_AVAILABILITY_2026-09-17.md`
4. `docs/data_frontier/BDB2021_ROUTE_GEOMETRY_CHECKPOINT_2026-09-17.md`
5. `docs/data_frontier/BDB2023_BLOCKING_INTERACTION_GEOMETRY_CHECKPOINT_2026-09-17.md`
6. `docs/data_frontier/BDB2026_ANALYTICS_SEMANTIC_CHECKPOINT_2026-09-17.md`
7. `docs/data_frontier/BDB2026_THROW_WINDOW_GEOMETRY_CHECKPOINT_2026-09-17.md`
8. relevant workflows under `.github/workflows/data-frontier-*`

## Final status

**DATA INVENTORY / NEW-DATA FRONTIER DISCOVERY AND VALIDATION PHASE: COMPLETE**

Validated reusable laboratories now exist for:
- 2018 receiver route / defender-proximity geometry;
- 2021 blocking-interaction / protection geometry;
- 2023 targeted-receiver / coverage-family / throw-window geometry.

The next chat should begin from these validated contracts and engineer a versioned proprietary advanced-data feature layer. It should not restart discovery.
