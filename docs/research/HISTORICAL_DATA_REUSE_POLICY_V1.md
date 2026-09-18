# Historical Data Reuse Policy V1

**Status:** FROZEN PROGRAM RULE

**Created:** 2026-09-17

## Purpose

Prevent the Football Context Intelligence Program from repeatedly rediscovering, revalidating, or unnecessarily re-downloading historical NFL base data that the repository has already standardized and used in prior research.

## Core rule

**Reuse first. Rehydrate only when necessary. Never re-research an already-qualified base source merely because a new research lane begins.**

## Existing canonical historical foundation

The repository already contains canonical historical builders and leakage-safe context code for all-position weekly player history and team/game context, including:

- `scripts/backtest/historical_player_logs.py`
- `scripts/backtest/historical_context.py`
- `scripts/backtest/build_historical_inputs.py`
- `scripts/player_form_v2.py`

Canonical weekly player history is normalized from nflverse/nflreadpy at player-game grain and includes, where applicable:

- stable player identity;
- season/week;
- team/opponent/game identity;
- position;
- targets;
- receptions;
- receiving yards;
- rushing attempts;
- rushing yards;
- passing attempts;
- passing yards;
- team opportunity denominators;
- per-game target/rush shares;
- route fields only when the source truly supplies routes.

The historical builder is position-agnostic; QB/RB/WR/TE are not separate raw-history universes.

## Historical season coverage already exercised

The repository has already executed substantial walk-forward/multiseason research across historical seasons.

Examples include:

- M95Q/M91 artifacts retained for 2019, 2020, 2021, 2022, 2023 and 2024;
- RB multiseason reconstruction matrices using 2021-2024 target seasons plus prior seasons;
- 2025 walk-forward infrastructure using canonical 2024-2025 player-game history;
- QB, WR, TE and RB backtests consuming the same canonical historical context framework.

This policy does not claim that every historical study used every season or every feature. It establishes that the base historical player/team pipelines already exist and are authoritative.

## Why the big CSVs are not visible on main

Generated runtime/backtest CSVs are intentionally not committed to the repository.

Examples such as:

- `data/backtests/player_game_logs_history.csv`
- `data/backtests/team_weekly_history.csv`
- `data/backtests/schedule_history.csv`
- `data/player_game_logs.csv`

are generated artifacts, not permanent tracked source files.

Prior workflows often uploaded those files as GitHub Actions artifacts with finite retention.

Therefore:

**absence of a generated CSV from main does not mean the historical data source or pipeline is missing.**

## Required reuse order

Before downloading/rebuilding historical base data, a new research lane must check in this order:

1. **Current preserved canonical artifact**
   - Use an existing non-expired GitHub Actions artifact when it contains the exact required rows/columns and has acceptable lineage.

2. **Existing derived historical asset**
   - Reuse already-persisted component/history outputs when the research question only needs those fields.

3. **Deterministic rehydration from canonical builder**
   - If exact row-level history is needed and no preserved artifact remains, rerun the existing frozen builder against its already-qualified source.
   - This is data materialization, not new source research.

4. **New source acquisition**
   - Only when the requested information is genuinely absent from the historical base.

## What does NOT justify a fresh pull

Do not rebuild historical player logs merely because:

- a new feature family is being considered;
- a new position-specific experiment starts;
- a new branch is created;
- a different target variable is tested;
- historical analog research begins;
- role-regime research begins.

If the needed fields are already in canonical history, reuse them.

## What DOES justify new data

Fresh/new source work is appropriate for information the canonical historical logs do not contain, for example:

- BDB tracking geometry;
- exact/near-exact route geometry;
- protection interaction geometry;
- current role/transaction context not present in box-score history;
- coaching/play-caller history;
- authoritative qualitative role statements;
- true WR-CB responsibility labels if an approved source becomes available;
- current OL/DL personnel/assignment context beyond existing historical stats.

## Historical analog program rule

The future analog engine should consume one canonical historical base table and join additional feature families onto it.

It must not independently rebuild player history separately for QB/RB/WR/TE.

Preferred structure:

`CANONICAL_PLAYER_GAME_HISTORY`
+
`CANONICAL_TEAM_WEEK_HISTORY`
+
optional versioned feature joins:
- role/environment regime;
- personnel continuity;
- advanced geometry history;
- injuries;
- coaching;
- matchup/exposure.

## Durability improvement

Because Actions artifacts expire, the project should create a versioned **historical base manifest** containing:

- seasons;
- row counts;
- schema;
- source versions;
- source hashes where available;
- builder commit;
- deterministic output hashes.

The full row-level data may remain off-Git, but the manifest should make clear when a historical base is byte-identical to a prior build.

If a durable authorized storage surface is later selected, one canonical historical base may be persisted there to avoid repeated network rehydration.

## Scientific boundary

Rehydrating identical historical source rows under the frozen canonical builder does not reopen or rerun prior science.

Existing failed/passed research dispositions remain authoritative.

## Disposition

`HISTORICAL_DATA_REUSE_FIRST_V1_FROZEN`
