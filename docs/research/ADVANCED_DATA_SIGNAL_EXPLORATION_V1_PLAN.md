# Advanced Data Signal Exploration V1 — Frozen Reconnaissance Plan

**Status:** ACTIVE RECONNAISSANCE — NO PRODUCTION CHANGE.

**Created:** 2026-09-17

**Branch:** `research-advanced-data-signal-exploration-v1`

**Parent materialization checkpoint:** `47bcd58aecf453f54b3f5db06a9dbdc94b000ad2`

## Purpose

The project now has 47 reproducible advanced-data fields from three validated laboratories. This program determines which of those fields represent stable/persistent football traits, which are mostly play-level state, which are sparse, and which deserve later position-specific predictive testing.

This is intentionally broader than the current WR research lane and does not modify it.

## Source laboratories

### BDB 2021 / 2018 route geometry
- 78,343 route-labeled player-plays
- snap/throw/arrival proximity geometry
- route labels
- no semantic target identity
- no exact defender responsibility

### BDB 2023 / 2021 protection interactions
- 46,396 reconstructed blocking interactions
- blocker-target snap/min/terminal geometry
- time to minimum distance
- PFF role/outcome labels
- interaction identity is not universal assignment responsibility

### BDB 2026 Analytics / 2023 throw-window
- 14,107 targeted-receiver plays with valid release geometry
- explicit target identity
- route
- alignment
- man/zone
- detailed coverage family
- post-release target/output geometry
- no exact defender responsibility

## Research questions

### Q1 — Persistence

Do player-level geometry traits persist across time?

Examples:
- receiver release nearest-defender distance;
- second-defender spacing;
- crowding rate;
- route-conditioned release spacing;
- route-runner throw spacing;
- blocker snap separation;
- blocker minimum interaction distance;
- time to minimum distance.

Primary test:
- early-season vs late-season player/profile correlation;
- player rankings;
- shrinkage with sample size;
- missingness / abstention.

### Q2 — Strict-prior forecastability of geometry

Can a strict-prior player summary forecast the same geometry family in later completed games?

This tests whether a feature is even plausible as a pregame player trait before testing downstream yards/receptions.

Examples:
- historical receiver median release separation -> next-week actual release separation;
- historical crowding rate -> next-week actual crowding;
- historical blocker minimum-distance median -> later interaction geometry.

Metrics:
- MAE;
- median absolute error;
- signed bias;
- Spearman rank correlation;
- sample-size stratification;
- confidence-tier performance.

No sportsbook or production projection is read.

### Q3 — Route / coverage interactions

How much of geometry is explained by:
- route;
- man vs zone;
- detailed coverage family;
- receiver identity;
- route × coverage;
- alignment where available?

This is descriptive/variance-decomposition reconnaissance, not a predictive model.

### Q4 — Protection geometry validation

Do interaction-geometry states line up with source PFF outcomes?

Retrospective-only relationships:
- beaten by defender;
- hit allowed;
- hurry allowed;
- sack allowed;
- defender hit/hurry/sack.

Examples:
- minimum blocker-target distance;
- time to minimum distance;
- terminal separation;
- protection-window length;
- chip/release role.

This validates whether the geometry captures meaningful protection interaction structure.

It is not a pregame prediction result.

### Q5 — Feature redundancy / novelty

Within each lab:
- pairwise correlations;
- rank correlations;
- duplicate-information clusters;
- route/coverage-conditioned redundancy;
- missingness coupling.

Later, after current model feature extraction is safe:
- compare new pregame-safe features against existing production feature families;
- identify genuinely new information.

### Q6 — Sparse / out-of-distribution risk

Quantify:
- percent players meeting V1 sample thresholds by week;
- percentage of future player-weeks with no valid historical feature;
- route-conditioned sparse-history rate;
- blocker-history sparse rate;
- new-player/new-team cases that need role-regime partial pooling.

## Frozen phases

### E0 — Coverage and stability census

No outcomes.

Outputs:
- player counts;
- qualifying history counts;
- sample-size distributions;
- temporal availability curves;
- missingness.

### E1 — Temporal persistence

No box-score outcomes.

BDB2021:
- Weeks 1-8 vs Weeks 9-17 player × route throw geometry.

BDB2023:
- Weeks 1-4 vs Weeks 5-8 blocker interaction geometry.

BDB2026:
- Weeks 1-9 vs Weeks 10-18 targeted-receiver release geometry;
- route-conditioned receiver release geometry.

### E2 — Strict-prior future-geometry validation

No box-score outcomes.

For each target week:
- build historical feature only from prior weeks;
- compare to target week's realized same-family geometry after the game;
- grade by frozen sample-confidence tiers.

Target-game geometry is used only as the validation label, never as a feature.

### E3 — Retrospective semantic validation

BDB2023:
- geometry vs source PFF interaction outcomes.

BDB2026:
- release geometry vs post-release closing / landing geometry;
- route/coverage strata.

These outputs remain retrospective-only.

### E4 — Novelty / redundancy

- feature correlation clusters;
- route/coverage conditional residual structure;
- candidate-family compression.

### E5 — Candidate shortlist

A family may advance only if it satisfies:

1. pregame legality or a clearly defined strict-prior transform;
2. adequate coverage;
3. non-trivial temporal persistence or future-geometry forecastability;
4. non-pathological missingness;
5. clear semantic interpretation;
6. no known duplicate/rejected family under a new name.

This is **not** a predictive win gate.

## Candidate family inventory

### Receiver / route

- `hist_receiver_release_nearest_defender_median_yards`
- `hist_receiver_release_second_defender_median_yards`
- `hist_receiver_release_crowding_2yd_rate`
- `hist_receiver_release_crowding_3yd_rate`
- `hist_receiver_route_release_nearest_defender_median_yards`
- `hist_player_route_throw_nearest_defender_median_yards`
- `hist_player_route_throw_second_defender_median_yards`
- `hist_player_route_throw_spacing_gap_median_yards`

### Protection / blocker

- `hist_blocker_snap_distance_median_yards`
- `hist_blocker_min_distance_median_yards`
- `hist_blocker_time_to_min_distance_median_seconds`

### Retrospective validation families

- arrival proximity;
- landing-zone defender distance;
- terminal target/defender geometry;
- post-release closing delta;
- PFF beaten/hit/hurry/sack labels.

## Position relevance map

### WR
Most relevant:
- release spacing;
- second-defender spacing;
- crowding;
- route-conditioned spacing;
- eventual exposure/matchup features.

Potential mechanisms:
- entitlement;
- catch probability;
- YPT / efficiency;
- distribution width.

### TE
Most relevant:
- release/crowding;
- route-conditioned spacing;
- alignment and coverage family;
- chip/block role context.

### RB
Most relevant:
- route/crowding for receiving role;
- chip/release interaction;
- protection context;
- role-regime state remains primary for opportunity.

### QB
Most relevant:
- receiver aggregate release/crowding state;
- blocker/protection profiles;
- pass-rush/personnel context;
- shared receiver environment.

## Anti-loop rules

Do not:
- rerun M72/M75 aggregate receiver-secondary work and rename it advanced data;
- treat nearest defender as responsibility;
- use BDB2021 route runner as targeted receiver without target truth;
- use post-release/landing geometry pregame;
- use PFF outcome labels pregame;
- infer universal blocker-rusher assignment from interaction identity;
- search many outcome models during reconnaissance;
- use 2026 live examples to tune historical thresholds.

## Artifact policy

Full competition rows and per-row derivatives remain ephemeral.

GitHub may retain:
- code;
- source hashes;
- frozen analysis plan;
- sanitized aggregate tables;
- aggregate correlations/metrics;
- coverage/abstention reports;
- candidate disposition.

## First execution

Run E0 + E1 + E2 + E3 as source-isolated analyses in GitHub Actions.

No production model, sportsbook file, or Issue #535 artifact is read.

Final initial disposition options:

- `ADVANCED_DATA_FAMILY_HIGH_PRIORITY_FOR_FROZEN_PREDICTIVE_TEST`
- `ADVANCED_DATA_FAMILY_RESEARCH_WORTHY_BUT_SPARSE`
- `ADVANCED_DATA_FAMILY_DESCRIPTIVE_ONLY_LOW_PERSISTENCE`
- `ADVANCED_DATA_FAMILY_RETROSPECTIVE_VALIDATION_ONLY`
- `ADVANCED_DATA_FAMILY_SEMANTICALLY_BLOCKED`

## Production boundary

This plan authorizes reconnaissance only.

Any predictive test must be separately preregistered by:
- position;
- target variable;
- baseline;
- feature family;
- folds;
- gates;
- no-retest boundary.

Disposition:

`ADVANCED_DATA_SIGNAL_EXPLORATION_V1_ACTIVE`
