# Current-Season State Persistence V1 — Frozen Plan

**Status:** FROZEN BEFORE ANY RESULT OUTPUT  
**Branch:** `research-current-season-state-persistence-v1`  
**Production change authorized:** NO  
**Sportsbook inputs:** PROHIBITED  
**2026 outcome tuning:** PROHIBITED

## Purpose

Quantify how much completed current-season football evidence should change a player's pregame state estimate as the season progresses.

This lane answers a narrow question that the live production architecture currently handles only through fixed blending and selected strict-prior feature paths:

> After 1, 2, 3, 4, ... completed games, does a player's current-season usage/efficiency state predict his next game better than the prior-season baseline, and does the existing PlayerForm four-game pseudo-prior blend improve on either raw source?

This is not a generic personnel-discontinuity retest. It does not reopen M76-M77, M95/M96 retrospective RB router families, or any closed WR/TE efficiency family.

## Existing production state this experiment must respect

Current production is not blind to the active season.

PlayerForm V2 already blends previous-season and completed current-season player metrics using:

`w_current = current_games / (current_games + 4)`

The blended metrics include target share, rush share, route rate when available, YPRR, YPT, YPC, YPA and receptions per target.

Team Context V3 already uses the promoted M89/M90 rolling-eight team/QB context and records how many current-season and prior-season games contribute.

Current-player availability removes definitive unavailable players pregame.

WR-R15 and TE-R5P use strict-prior offensive snap participation, but their shared snap loader is currently hardcoded to source seasons 2020-2025. A 2026 source-readiness audit is part of this lane, but no production source change is authorized by this plan.

## Scientific question

For stable-identity players with both a prior-season baseline and at least one completed game in the target season, compare three strictly pregame estimators of the next-game metric:

1. prior-season baseline only;
2. current-season-to-date baseline only;
3. the exact current PlayerForm four-game pseudo-prior blend.

No estimator may use target-week or future information.

## Historical evaluation frame

Evaluation seasons: 2022, 2023, 2024, 2025.

For each target season S and target week W >= 2:

- prior baseline uses only season S-1 completed games;
- current baseline uses only season S games with week < W;
- actual is the player's metric in season S, week W;
- target-game rows never enter either predictor.

2022-2024 are development/descriptive seasons.

2025 is the replication season. No rule may be changed after inspecting 2025 results.

2026 realized outcomes are not used anywhere in this experiment.

## Direct production-aligned player metrics

QB:
- YPA.

RB:
- team rushing share;
- team target share;
- YPC.

WR:
- team target share;
- YPT;
- catch rate / receptions per target.

TE:
- team target share;
- YPT;
- catch rate / receptions per target.

Route-rate/YPRR are excluded from the primary atlas because the maintained weekly player-stats source does not guarantee live route fields for every season. They remain separate source-qualified lanes.

## Evaluation statistics

For every position x metric x season and for pooled development / 2025 replication:

- n;
- MAE for prior-only, current-only and PlayerForm blend-4;
- RMSE for all three;
- mean error/bias for all three;
- Pearson correlation for all three where defined;
- Spearman correlation for all three where defined;
- `current - prior` state delta versus `next_game - prior` delta correlation;
- sign agreement of those two deltas when both are non-zero.

Also report by completed-current-games bucket:

- 1;
- 2;
- 3;
- 4;
- 5-8;
- 9+.

The key early-season read is therefore explicit for the exact two-game state that exists before 2026 Week 3.

## Interpretation contract

This atlas is diagnostic. It does not itself authorize a new projection model.

A position/metric may be labeled `CURRENT_SEASON_SIGNAL_REPLICATES` only when, in 2025:

- the blend-4 next-game MAE is lower than prior-only MAE; and
- the current-vs-prior state delta has positive next-game delta association.

A label of `BLEND4_NOT_BEST` may be reported when current-only beats blend-4 on 2025, but that does not authorize replacing the blend. Any shrinkage-rule challenger requires a separately frozen experiment using development seasons only for rule selection.

A label of `NO_REPLICATED_CURRENT_SIGNAL` does not mean the position is irreducible; it only closes this simple season-to-date level family for that metric.

## Current-season snap-source audit

In the same run, audit `nflreadpy.load_snap_counts` for 2026 without using those rows as outcomes.

Report:

- 2026 weeks available;
- row counts by available week;
- team coverage;
- non-null offense snap count / offense percentage coverage;
- whether WR-R15 / TE-R5P production source seasons include 2026.

This is a source-readiness result only.

If 2026 prior-week snaps are available but the production adapters omit them, disposition may be `CURRENT_2026_SNAP_SOURCE_AVAILABLE_NOT_CONSUMED`.

That disposition authorizes a follow-up frozen parity/source-continuation test. It does not authorize editing production directly.

## Anti-retest boundaries

Do not use this lane to reopen:

- M76-M77 generic personnel discontinuity;
- M95/M96 exposed-2025 RB routing variants;
- closed WR R17-R20 target-quality families;
- generic injury burden;
- sportsbook game-script features;
- target-game participation.

Current role/depth/availability may be studied only through a separately frozen recipient/state hypothesis, consistent with the existing M95G/M95H and RB-ND2B evidence.

## Artifacts

The canonical workflow must emit:

- row-level persistence panel;
- summary by season / position / metric;
- early-game-count summary;
- 2025 replication summary;
- 2026 snap-source audit;
- machine-readable result JSON.

## Disposition

`CURRENT_SEASON_STATE_PERSISTENCE_V1_PLAN_FROZEN`
