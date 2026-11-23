# Opponent defense by position: does the model miss it, and does a fix help? PREREGISTRATION V1

**Date:** 2026-11-22, frozen before scoring.

**Goal:** a production change, not a study. If the gates below pass, the adjustment is promoted
into the model's final projection for the markets and position groups that pass.

## Why

Code read on 2026-11-22:
- **Rushing projections:** the player's carry share x team rushes x his own yards per carry.
  - The only opponent term is a defensive light-box (+7%) / heavy-box (-6%) switch.
  - `def_rush_epa` is loaded into team context and never used.
- **The ML component** (weight 0.44 on rush_yards) uses only the player's own lagged stats.
- **No component uses yards allowed to the player's position.**
- **Triggering case:** Brock Purdy 10+ rush (SNF week 11). MIN allows the 6th-fewest QB
  rushing yards, 10.4 per game against a league 15.9.

## Feature (strictly prior, same season)

`D = n/(n+4) x (allowed_pg / league_pg - 1)`

- `allowed_pg`: the opponent's mean per-game production allowed to the player's position group
  (QB, RB, WR, TE), in the market's stat, over weeks before the game.
- `league_pg`: the same quantity averaged over all defenses.
- `n`: the opponent's prior games.

## Adjustment

`adjusted = proj x (1 + b x D)`.
- One `b` per market x position group.
- Fit by least squares of `(actual - proj)` on `proj x D`, no intercept.

## Data (the model's real projections, players with posted lines)

- **2024-2025:** full-stack ensemble `proj` (`full_stack_vegas_benchmark_v1/non_qb_detail.csv`)
  for rush_yards, rec_yards and receptions.
- **pass_yards:** base-engine `mc_proj`.
- **2026:** PRISM `model_proj` from the graded boards, weeks 1, 2 and 6-10
  (`data/research/selective_pick/`).

## Folds

- **A:** fit 2024, test 2025.
- **B:** fit 2024+2025, test 2026.
- A cell needs ≥ 150 test rows in fold A and ≥ 60 in fold B. QB rushing exists only in 2026; it
  is fit on 2026 weeks 1-7 and tested on weeks 8-10, and reported separately.

## Promotion gate (per market x position group)

All three must hold:
1. MAE improves in **both** test folds.
2. `b > 0` in both fits: weaker defense means more yards.
3. The fold-B MAE improvement has a game-clustered bootstrap probability of improvement ≥ 0.90.

Cells that pass are promoted with the fold-B `b`. Cells that fail are left unchanged, and the
result says what to test next.

## Prediction

- Receiving cells improve slightly: the 2026 identity study found opponent defense vs
  receptions r 0.26.
- Rushing improves a little.
- `b` lands between 0.2 and 0.5, because the market and the ML layer already absorb part of it.
