# PLAYER OPPORTUNITY SHARE STABILITY V1 — 2023 HOLDOUT FREEZE

Frozen: 2026-10-12

Status: **INDEPENDENT HOLDOUT CONFIRMATION ONLY. FROZEN BEFORE READING 2023 SHARE-SURPRISE RESULTS. NO MODEL CHANGE.**

Parent:
- `docs/research/PLAYER_OPPORTUNITY_SHARE_STABILITY_V1_PLAN.md`
- `docs/research/PLAYER_OPPORTUNITY_SHARE_STABILITY_V1_RESULT.md`

Parent result:
- 2024 and 2025 both replicated in all four frozen cells:
  - RB_CARRY_SHARE
  - RB_TARGET_SHARE
  - WR_TARGET_SHARE
  - TE_TARGET_SHARE

## Purpose

Test whether the exact same player-specific opportunity-share volatility signal
replicates in an untouched historical holdout season.

No transform, threshold, widening coefficient, or production rule is designed
before this holdout is read.

## Holdout authority

Target season:
- **2023 only**

Target weeks:
- regular-season Weeks **5-18**

Public data:
- nflverse / nflreadpy weekly player stats;
- exact weekly roster identity;
- authoritative schedule.

No 2024/2025 target rows enter the holdout scoring population.

## Frozen cells

Identical to V1:

1. `RB_CARRY_SHARE`
   - RB / HB / FB
   - player rushes / team rushes

2. `RB_TARGET_SHARE`
   - RB / HB / FB
   - player targets / team targets

3. `WR_TARGET_SHARE`
   - WR-family
   - player targets / team targets

4. `TE_TARGET_SHARE`
   - TE
   - player targets / team targets

No denominator changes.

## Frozen feature

For each target player-game:

- same stable player identity;
- same team;
- same season;
- games strictly before target week;
- latest up to 8 completed prior games;
- require >=4 prior finite-share games.

Compute:
- `prior_share_mean`
- `prior_share_sd` with population SD, ddof=0
- prior min/max

Primary explanatory feature remains:
- `prior_share_sd`

No prior-season information enters.

## Frozen target

`share_surprise = abs(actual_target_game_share - prior_share_mean)`

No model projection or sportsbook line enters.

## Frozen support floors

Per cell:
- >=300 player-games
- >=100 distinct players
- >=50 games

## Frozen inference

Per cell:
- Spearman rho: prior_share_sd vs share_surprise
- game-cluster bootstrap: 5,000 reps
- player-cluster bootstrap: 5,000 reps
- same deterministic seed family as parent V1, shifted only by holdout season

A cell is:
`PLAYER_SHARE_VOLATILITY_2023_HOLDOUT_CONFIRMED`

only if:
1. support floor passes;
2. raw rho >0;
3. game-cluster 95% CI lower bound >0;
4. player-cluster 95% CI lower bound >0.

Otherwise:
`PLAYER_SHARE_VOLATILITY_2023_HOLDOUT_NOT_CONFIRMED`.

Overall:
- all four confirmed:
  `PLAYER_OPPORTUNITY_SHARE_STABILITY_THREE_SEASON_CONFIRMED`
- one or more fail:
  `PLAYER_OPPORTUNITY_SHARE_STABILITY_HOLDOUT_PARTIAL_OR_FAIL`

No pooled result may rescue a failed cell.

## Integrity

Require:
- exact season = 2023;
- target weeks within 5-18;
- stable identity coverage >=95%;
- non-degenerate actual shares in every cell;
- same/future feature violations = 0;
- same-team feature violations = 0;
- sportsbook inputs = 0;
- OddsAPI calls = 0;
- model residuals = 0;
- fitted predictive models = 0;
- projection changes = 0;
- production changed = false.

## Advancement

Only if all four cells confirm in 2023 may we freeze a separate,
mean-preserving opportunity-uncertainty candidate.

That later candidate must:
- leave expected carry/target share unchanged;
- change uncertainty/confidence only;
- use no threshold selected from V1 or this holdout;
- use no coefficient fit on 2023-2025;
- be prospectively checked on 2026 before production.

If holdout is partial/fail:
- do not rescue with pooling, different windows, alternative denominators, or
  result-derived thresholds.
