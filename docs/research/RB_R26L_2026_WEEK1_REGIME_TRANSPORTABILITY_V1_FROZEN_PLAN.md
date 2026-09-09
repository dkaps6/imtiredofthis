# RB R26L 2026 Week-1 Source-Regime Transportability V1 — Frozen Plan

Status: FROZEN BEFORE 2026 AGGREGATE REGIME SCORING
Date: 2026-09-09
Production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Historical source parent: R26J run `34374987828`, artifact `10113466373`, digest `sha256:7d886f228487e052ee1617cc5f42b974cd9da6a9d448f3a86b1de82cb4fc4f46`
Current production evidence parent: canonical post-merge Full Slate run `34317211395`, artifact `10090547415`, digest `sha256:7eab77e41c5879d4f54d87497eee0d1186010784cfc4ef966178930e16fb7c3b`
R26K parent conclusion: `2020_SPECIFIC_MECHANISM_NO_REPLICATED_ROUTER`

## Question

R26K showed that football-coherent states explaining large portions of the harmful 2020 R26 Week-1 allocation error were generally beneficial in 2021-2025 and therefore could not support another historical guard.

R26L asks a prospective/source-only question:

> Does the actual 2026 Week-1 RB vacancy environment resemble the anomalous 2020 source regime or the successful 2021-2025 source regime on source-aligned pregame football state?

No 2026 outcomes are used or available to this study. R26L cannot authorize a projection change by itself.

## Governance

R26L is a transportability/source audit only.

It may NOT:
- exclude 2020 from existing historical results;
- change an R26 prediction;
- refit R9;
- change the R26D meaningful-exit definition;
- change R22 or any receiving-yard mean;
- use sportsbook lines/odds as football inputs;
- use 2026 outcomes;
- use same-week ESPN/nflverse depth charts;
- authorize prospective shadow or production.

A modern-like source result can authorize only a separately frozen R26M-style prospective qualification/synthesis design.

## Current 2026 roster-state source

For source alignment with R26/R26J, 2026 continuity/exits must be built using the same canonical weekly roster machinery already used in R26 research:
- `scripts.backtest.historical_inputs._load_nflreadpy_weekly_sources`
- `normalize_roster`
- canonical allowed roster status `ACT/INA`
- target = 2026 Week 1
- prior snapshot = 2025 regular-season Week 18, determined from the authoritative schedule.

This is NOT the parked 2026 depth-hierarchy study. R26L must not use same-week depth or ESPN depth rank.

If canonical 2026 Week-1 ACT/INA roster rows are unavailable, R26L must fail closed as source-not-ready rather than substitute a different roster population post hoc.

## Current 2026 exited-player receiving state

For each 2026 Week-1 vacancy room, attach strictly-prior receiving identity to departed players using the already-established R26C/R9 as-of mechanics:
- exact target game excluded;
- target-game participation excluded;
- sportsbook excluded;
- `allow_exact_matches=False` semantics inherited from the production-safe identity runtime.

Current source measures needed:
- exit-history coverage;
- summed exited-player last-8 targets/game.

## Primary source-aligned transportability features

Exactly these seven features determine the R26L primary classification:

1. `current_room_n`
2. `continuing_n`
3. `entrants_n`
4. `veteran_entry_n`
5. `veteran_entry_share`
6. `exit_history_coverage`
7. `sum_exit_last8_targets_pg`

They were all part of the R26J pregame source state and can be reconstructed for 2026 with the same roster/strict-prior identity semantics.

No feature may be added to the primary classifier after 2026 aggregate values are visible.

## Secondary production-current diagnostics

The exact post-merge Full Slate artifact may be used only to verify the current production football universe and to report current RB receiving-prior concentration where mechanically possible.

Secondary diagnostics do NOT determine the R26L transportability disposition because current production uses Ourlads as the football-universe authority while R26J's historical room-state population used canonical weekly ACT/INA rosters.

Permitted secondary fields include:
- Ourlads current RB/FB room size;
- normalized current production football target-share HHI;
- normalized current production top-RB target share.

The source difference must be reported explicitly.

## Historical comparison

For each primary feature:
- `v2020` = R26J 2020 Week-1 vacancy-room mean;
- `modern_mean` = unweighted mean of the five R26J season means for 2021-2025;
- `modern_min` / `modern_max` = min/max of those five season means;
- `v2026` = mean across 2026 Week-1 vacancy rooms.

Per-feature source preference:
- `MODERN_CLOSER` if `abs(v2026 - modern_mean) < abs(v2026 - v2020)`;
- `2020_CLOSER` if the reverse;
- `TIE` if equal to numerical tolerance `1e-12`.

## Frozen normalized distance

For each feature, define a historical scale:

`scale = max(max(v2020, modern_max) - min(v2020, modern_min), floor)`

Feature floors:
- count features (`current_room_n`, `continuing_n`, `entrants_n`, `veteran_entry_n`): `0.25`
- rate features (`veteran_entry_share`, `exit_history_coverage`): `0.05`
- target-load feature (`sum_exit_last8_targets_pg`): `0.25`

Then:
- `d2020 = abs(v2026 - v2020) / scale`
- `dmodern = abs(v2026 - modern_mean) / scale`

Aggregate distance is the simple unweighted mean over all seven features.

No feature weights are fitted.

## Frozen transportability dispositions

### `2026_SOURCE_REGIME_MODERN_LIKE_FOR_PROSPECTIVE_QUALIFICATION`
ALL must hold:
1. all source-integrity gates pass;
2. all seven primary features are finite;
3. at least 5 of 7 features are `MODERN_CLOSER`;
4. mean normalized modern distance <= 0.75 * mean normalized 2020 distance;
5. 2026 is not beyond the 2020 value in the 2020-anomalous direction on more than 2 primary features.

Maximum consequence: authorize design of an R26M-style prospective qualification/synthesis study. This does NOT exclude 2020 and does NOT authorize shadow.

### `2026_SOURCE_REGIME_2020_LIKE_NO_MODERN_QUALIFICATION`
ALL must hold:
1. source integrity passes;
2. all seven features finite;
3. at least 5 of 7 features are `2020_CLOSER`;
4. mean normalized 2020 distance <= 0.75 * mean normalized modern distance.

No modern-regime qualification is authorized.

### `2026_SOURCE_REGIME_MIXED_NO_TRANSPORTABILITY_CONCLUSION`
Any valid source result not meeting either directional rule.

No qualification design is authorized.

### `2026_CURRENT_SOURCE_NOT_READY`
Canonical 2026 Week-1 ACT/INA roster source is unavailable or fails minimum coverage.

No scientific transportability conclusion.

## Source integrity gates

Required:
- exact R26J artifact digest verified;
- exact post-merge Full Slate artifact digest verified;
- protected production runtime/model paths clean against production base;
- canonical 2026 Week-1 roster covers at least 30 teams;
- canonical 2025 Week-18 prior roster covers 32 teams;
- allowed roster statuses only `ACT/INA` after normalization;
- current/prior snapshot order exact;
- at least 20 2026 Week-1 vacancy rooms;
- at least 60% departed-player prior-history coverage;
- no target-game outcomes/participation used;
- sportsbook football inputs = 0;
- same-week depth used = false;
- production parameters changed = false.

## Required outputs

- `r26l_2026_week1_room_state.csv`
- `r26l_2026_exited_player_state.csv`
- `r26l_2026_source_summary.csv`
- `r26l_transportability_feature_comparison.csv`
- `r26l_secondary_production_diagnostics.csv`
- `r26l_disposition.json`

## Authority ceiling

R26L can at most authorize **design of a separately frozen prospective qualification/synthesis study**.

It can never directly authorize:
- shadow;
- production;
- removal of 2020;
- a new router;
- an R9 refit.
