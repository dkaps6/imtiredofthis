# Migration 94B — Explicit Football-Only Game State

## Why M94B exists

M94 showed that a direct football-only team-rush regression can modestly improve average team rushing volume, but it still struggles to distinguish true low-rush and high-rush games. The failure is especially important because M92 showed that team-volume error is a major driver of the 20+ and 25+ carry misses.

M94B therefore replaces the one-step question "how many rushes will the team have?" with a more football-mechanical decomposition:

1. How many offensive plays should the team have?
2. What share of those plays should occur while leading, neutral, or trailing?
3. How does this team historically call runs in each of those score states?
4. Convert those pieces into expected team rushing attempts.
5. Keep the existing M91 player-share allocation frozen and measure how the improved team total translates into RB attempts/yards.

## Pregame boundary

No sportsbook spread, total, moneyline, or player prop is used.

Target-game PBP is used only to create labels for training/evaluation. Every feature used for a target game is built from completed games strictly before that target week.

The game-state history uses offensive plays and classifies score differential as:

- lead: greater than +3 points
- neutral: -3 through +3
- trail: less than -3 points

For each completed team-game it records offensive plays, rushing attempts, lead/neutral/trail play shares, state-conditioned rush rates, mean score differential, and a neutral early-down rush tendency.

## Temporal protocol

- 2024 Weeks 1-12: development training
- 2024 Weeks 13-18: choose play-volume model family, state-share model family, and the blend with the frozen M91 baseline
- freeze those architecture choices
- refit on all 2024
- 2025: untouched temporal validation

The 2025 season is never used to select families, thresholds, or blend weight.

## Structured projection

The structured component is:

`predicted offensive plays × sum(predicted state share × pregame state-conditioned rush rate)`

State-conditioned team rush rates use recent prior games with shrinkage toward the league rate so that sparse lead/trail samples do not become brittle.

The 2024 holdout also chooses how much of this structured projection to blend with the frozen M91 football-only baseline. The blend is chosen only from 2024.

## Diagnostic oracles

M94B records three counterfactuals in addition to the candidate:

- actual plays + predicted state mix
- predicted plays + actual state mix
- actual plays + actual state mix

All three continue using the pregame state-conditioned rushing tendencies. These diagnostics are intended to tell us whether the remaining team-volume error is primarily:

- offensive-play count,
- game-state prediction,
- or playcalling tendency within state.

## Promotion gate

M94B is research-only and cannot advance unless the frozen 2025 validation improves all of the following versus M91:

- overall team rushing MAE
- low-rush (20 or fewer) team-game MAE
- high-rush (30+) team-game MAE
- overall RB rushing-attempt MAE
- overall RB rushing-yard MAE
- 20+ carry RB attempt MAE
- legacy all-player rushing-yard guard

A gain limited to workload tails is useful research evidence but is not sufficient for production promotion.

## Data reuse

The run saves the compact 2023-2025 game-state history in the M94B artifact. Downstream RB research can reuse this table instead of re-fetching/reconstructing the same PBP game-state observations on every migration.
