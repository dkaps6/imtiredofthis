# BDB2023 Blocker Protection Geometry Qualification V1

**Status:** FROZEN PRE-QUALIFICATION — OUTCOME-FREE  
**Parent contract:** `OL_DL_PROTECTION_CONTEXT_CONTRACT_V1_FROZEN`  
**Source materialization:** `NFL_ADVANCED_FEATURE_MATERIALIZATION_V1_CERTIFIED`  
**Predictive outcomes authorized:** false  
**Production changes authorized:** false

## Purpose

Determine whether the three certified strict-prior blocker-geometry histories contain
enough pregame coverage, stable identity, repeatability and incremental information to
deserve a separately frozen predictive experiment for protection/efficiency context.

This is a distinct mechanism from:
- role/room opportunity;
- event/regime reliability;
- returning-opportunity continuity;
- analog novelty;
- receiver release spacing.

## Frozen candidates

Exactly three:

1. `hist_blocker_snap_distance_median_yards`
2. `hist_blocker_min_distance_median_yards`
3. `hist_blocker_time_to_min_distance_median_seconds`

Certified support threshold remains **10 prior interactions**.

No alternate support threshold may be inspected.

## Source

Official source:
- Kaggle `nfl-big-data-bowl-2023`
- source season: **2021**
- tracking weeks: **1-8**
- frozen source hash:
  `1c3e1eb6fcd0cf85807c649804fa0c72421c76e973e40a742523f1dd3a929182`

Reuse the certified materializer at:

`47bcd58aecf453f54b3f5db06a9dbdc94b000ad2`

Do not redo source discovery.

## Semantic firewall

`pff_nflIdBlockedPlayer` is a blocking-interaction identity.

It is **not** universal primary blocker-rusher responsibility.

This qualification may use the frozen blocker-history geometry but may not:
- claim exact assignment;
- use PFF beaten/hit/hurry/sack labels as pregame predictors;
- use target-week geometry in the target-week feature.

## Stable identity

Required bridge:

`BDB blocker_nfl_id -> nflverse direct nfl_id -> gsis_id -> weekly roster gsis_id`

Names may be retained only as diagnostics and may not rescue mapping.

Frozen integrity:
- direct stable-ID bridge coverage >= **0.99**
- ambiguous BDB nfl_id mappings = **0**
- ambiguous roster GSIS mappings = **0**
- duplicate published player-week keys = **0**
- join fanout = **0**

## Frozen broad pregame universe

Use nflverse **2021 weekly roster** rows for Weeks **1-8**.

Eligible offensive-line rows are those with either:
- `position` in `C,G,T,OT,OG,OL`, or
- `depth_chart_position` in `LT,LG,C,RG,RT,OT,OG,OL,G,T`.

Keep the full broad rostered-OL denominator, including Week 1 and backups.

Do not use target-game snap counts or target-game participation to decide eligibility.

## Pregame legality

For target week W:

`history_max_source_week < W`

Required:
- chronology violations = **0**
- target-game rows used in strict-prior history = **0**
- same-game partial history = **0**

## Stability

Recompute certified source persistence with the frozen reconnaissance split:

- early: Weeks 1-4
- late: Weeks 5-8
- minimum observations in each half: **10**
- statistic: Spearman of blocker median geometry

No pressure/sack outcome label is read for qualification.

## Outcome-free redundancy

Test whether each blocker-history descriptor is largely reconstructible from basic
pregame roster metadata already available without BDB geometry.

Frozen reconstruction inputs:
- offensive-line `position` one-hot
- `depth_chart_position` one-hot when present
- height
- weight
- years experience

No team ID, target-game participation or outcome field is used.

Temporal split:
- train: 2021 Weeks 1-4
- holdout: 2021 Weeks 5-8

Linear least-squares reconstruction with intercept and deterministic one-hot encoding.

Minimum rows:
- train >= **500**
- holdout >= **200**

Frozen interpretation:
- R2 >= **0.90**: `HIGHLY_RECONSTRUCTIBLE_REDUNDANT`
- 0.75 <= R2 < 0.90: `REDUNDANCY_REVIEW`
- R2 < 0.75: `INCREMENTAL_INFORMATION_SURVIVES_REDUNDANCY_GATE`
- insufficient rows: `REDUNDANCY_UNRESOLVED_SOURCE_THIN`

## Frozen qualification gates

Inherit Football Context Signal Qualification V1:

- broad pregame coverage >= **0.80**
- eligible rows >= **500**
- stable-ID coverage >= **0.99**
- duplicate/fanout = **0**
- stability evidence present
- clear protection-context mechanism
- source not blocked

Additional:
- chronology clean
- redundancy not `HIGHLY_RECONSTRUCTIBLE_REDUNDANT`

A source-thin result may not be rescued by restricting to starters, high-snap players,
late weeks, or positions with favorable coverage after seeing the result.

## Mechanism

Potential future component only:
- blocker/protection archetype;
- protection continuity / uncertainty;
- QB efficiency/scramble environment;
- team pass-protection context.

No predictive claim is made here.

## Required outputs

Sanitized aggregate evidence only:
- identity/integrity manifest
- coverage by week and OL position
- stability table
- redundancy table
- qualification inventory

Full per-interaction and per-player history tables remain ephemeral.

## Pre-result disposition

`BDB2023_BLOCKER_PROTECTION_GEOMETRY_QUALIFICATION_V1_FROZEN_PRE_RESULT`
