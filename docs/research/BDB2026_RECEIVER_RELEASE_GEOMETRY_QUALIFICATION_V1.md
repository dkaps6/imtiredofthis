# BDB2026 Receiver Release Geometry Qualification V1

**Status:** FROZEN PRE-QUALIFICATION — OUTCOME-FREE  
**Parent context contract:** `FOOTBALL_CONTEXT_SIGNAL_QUALIFICATION_V1_FROZEN`  
**Parent source contract:** `NFL_ADVANCED_FEATURE_DICTIONARY_V1_FROZEN_CI_CERTIFIED`  
**Parent materialization:** `NFL_ADVANCED_FEATURE_MATERIALIZATION_V1_CERTIFIED`  
**Predictive outcomes authorized:** false  
**Sportsbook inputs authorized:** false  
**Production changes authorized:** false

## Purpose

Determine whether the two highest-priority BDB2026 targeted-receiver release-history
descriptors are sufficiently pregame-available, identity-safe, stable, supported and
incremental versus canonical production opportunity state to deserve a separately
frozen predictive experiment.

This qualification does not read receiving yards, receptions, target-game targets,
projection residuals, market lines or bet results.

## Frozen candidates

Exactly two candidate descriptors:

1. `hist_receiver_release_nearest_defender_median_yards`
2. `hist_receiver_release_second_defender_median_yards`

Support field:

`hist_receiver_release_geometry_sample_count`

The certified materializer's minimum support remains **8 prior targeted-receiver
observations**. No alternative support threshold is inspected.

Nearest defender remains geometric proximity only. It is not coverage responsibility.

## Source / materializer

Official BDB2026 Analytics source:
- Kaggle slug: `nfl-big-data-bowl-2026-analytics`
- source season: 2023
- frozen corpus SHA-256:
  `228554c6600ac4e73529e5b6309193c8a3371acca7b4bc59792188acd814fb07`

Reuse the certified materializer exactly from:
`data-frontier-advanced-feature-materializers-v1`

Raw source and full per-row derivatives remain ephemeral.

## Frozen identity bridge

No fuzzy-name join is allowed.

Required path:

`BDB nfl_id -> nflverse players stable-ID crosswalk -> canonical historical player_id / player_identity_key`

Requirements:
- direct stable-ID bridge coverage >= **0.99**
- ambiguous BDB-ID mappings = **0**
- ambiguous canonical stable-ID mappings = **0**
- published duplicate canonical player-week keys = **0**
- fanout = **0**

Names may be retained only as diagnostics and may not rescue failed stable-ID coverage.

## Frozen canonical history

Rehydrate only the ordinary history needed for pregame production state:
- 2022–2023 player-game history
- 2023 is the qualification target season
- 2022 exists only to reconstruct canonical prior-season PlayerForm state

No ordinary historical source is newly invented.

## Frozen eligible universe

Broad target universe:

all canonical **2023 regular-season WR / TE / RB player-game rows**.

The target universe is not restricted to players who happened to be targeted in the
target game and is not restricted to BDB-observed players. This prevents postgame
target participation from inflating pregame coverage.

Week 1 and other cold-start rows remain in the denominator.

Coverage will also be reported descriptively by week and position, but no
late-season or position subset may rescue the broad V1 qualification.

## Pregame legality

For target week W, BDB history must satisfy:

`history_max_source_week < W`

Target-week BDB geometry may never enter its own feature.

Same-game partial history is forbidden.

Landing/post-release BDB fields are forbidden.

Chronology violations must be **0**.

## Stability evidence

Recompute the already-certified outcome-free source persistence from the same frozen
corpus for lineage attachment:

- player-level early period: Weeks 1–9
- player-level late period: Weeks 10–18
- minimum observations in each half: **8**
- statistic: Spearman correlation of player median geometry

This is geometry-to-geometry stability only, not box-score outcome evaluation.

## Outcome-free redundancy audit

Ask whether the candidate history descriptor is reconstructible from canonical
PlayerForm target-opportunity state.

Canonical inputs are exactly:
- `prod_tgt_prior_share`
- `prod_tgt_prior_games`
- `prod_tgt_current_share`
- `prod_tgt_current_games`
- `prod_tgt_playerform_blend`

Temporal redundancy split within source season:
- train: 2023 Weeks 1–9
- holdout: 2023 Weeks 10–18

Missing canonical production shares are filled with deterministic 0.0 exactly as in
the existing analog redundancy audit.

Minimum rows for a scored reconstruction:
- train >= **500**
- holdout >= **200**

Linear reconstruction with intercept only; no model zoo or tuning.

Frozen redundancy interpretation:
- holdout R2 >= **0.90**: `HIGHLY_RECONSTRUCTIBLE_REDUNDANT`
- 0.75 <= R2 < 0.90: `REDUNDANCY_REVIEW`
- R2 < 0.75: `INCREMENTAL_INFORMATION_SURVIVES_REDUNDANCY_GATE`
- insufficient rows: `REDUNDANCY_UNRESOLVED_SOURCE_THIN`

A candidate at R2 >= 0.90 cannot be `READY_FOR_FROZEN_EXPERIMENT`.

## Frozen qualification gates

Inherit `FOOTBALL_CONTEXT_SIGNAL_QUALIFICATION_V1` defaults:

- broad pregame coverage >= **0.80**
- eligible rows >= **500**
- stable-ID coverage >= **0.99**
- duplicate published keys = **0**
- canonical fanout = **0**
- stability evidence present
- clear football mechanism
- source not blocked

Additional V1 requirements:
- chronology violations = **0**
- target-game BDB rows consumed pregame = **0**
- post-release/landing fields consumed pregame = **0**
- redundancy disposition not `HIGHLY_RECONSTRUCTIBLE_REDUNDANT`

Possible dispositions remain:
- `READY_FOR_FROZEN_EXPERIMENT`
- `ENGINEERING_READY_SOURCE_THIN`
- `DESCRIPTIVE_ONLY`
- `SOURCE_BLOCKED`
- `REJECTED_INTEGRITY`

## Mechanism mapping

The candidate may plausibly inform:
- receiver spatial environment / release-space tendency;
- target-quality / catch-conversion / explosive-environment context;
- uncertainty/context around receiving efficiency.

This qualification does not assert any of those outcomes improve.

## No-rescue rules

Do not:
- lower the 8-observation materializer threshold;
- exclude Week 1 from broad coverage;
- use only late-season rows to pass 80%;
- qualify only a favorable position after seeing results;
- substitute route-conditioned target-game route truth;
- add crowding fields after seeing these results;
- use target-game box-score outcomes;
- use sportsbook data;
- post to or modify Issue #535.

If broad V1 is source-thin, preserve that result and move on.

## Required artifacts

Sanitized outputs only:
- source/identity QA
- coverage by week and position
- stability evidence
- redundancy evidence
- qualification inventory
- manifest with source/commit/hashes and explicit no-outcome/no-sportsbook flags

Full per-row BDB/candidate tables remain ephemeral.

## Frozen disposition before execution

`BDB2026_RECEIVER_RELEASE_GEOMETRY_QUALIFICATION_V1_FROZEN_PRE_RESULT`
