# PLAYER OPPORTUNITY SHARE VOLATILITY — MECHANISM DECOMPOSITION V2 PLAN

Frozen: 2026-10-12

Status: **MECHANISM DIAGNOSTIC ONLY. FROZEN BEFORE ADJUSTED RESULTS. NO MODEL CHANGE.**

Parent evidence:
- 2024-2025: all four share-volatility cells replicated;
- untouched 2023 holdout: all four cells confirmed;
- three-season raw signal is therefore established.

## Why this diagnostic is required

The canonical simulator already contains finite-count multinomial allocation
noise around fixed player opportunity probabilities.

Raw realized share SD can therefore contain two components:

1. **ordinary count-sampling variance** that the simulator already represents;
2. **true game-to-game role/share instability** that the simulator currently
   does not represent.

Do not widen allocation from raw SD until these components are separated.

## Target seasons / cells

Seasons:
- 2023
- 2024
- 2025

Weeks:
- 5-18 regular season

Frozen cells:
- RB_CARRY_SHARE
- RB_TARGET_SHARE
- WR_TARGET_SHARE
- TE_TARGET_SHARE

Feature/target history semantics remain identical to V1:
- same player;
- same team;
- same season;
- latest up to 8 strictly-prior games;
- >=4 prior finite-share games.

## Added prior-game denominator state

For every prior history game preserve:
- team rush attempts for RB_CARRY_SHARE;
- team targets for RB/WR/TE target-share cells.

Let:
- `mu = prior_share_mean`
- `s2 = prior_share_sd ** 2`
- `N_j = team opportunity total in prior game j`

Frozen expected ordinary sampling variance:

`sampling_var = mean(mu * (1 - mu) / N_j)`

over qualifying prior games with `N_j > 0`.

Frozen excess role variance:

`excess_share_var = max(0, s2 - sampling_var)`

`excess_share_sd = sqrt(excess_share_var)`

No coefficient is fit.

## Primary mechanism tests

### A. Mean/sampling-adjusted partial Spearman

Question:
does prior share SD still predict target-game absolute share surprise after
controlling for share scale and expected finite-count sampling noise?

Variables:
- X = prior_share_sd
- Y = share_surprise
- controls:
  - prior_share_mean
  - expected_sampling_sd = sqrt(sampling_var)

Method:
1. rank-transform X, Y and both controls within the season/cell;
2. regress ranked X on intercept + ranked controls by ordinary least squares;
3. regress ranked Y on intercept + ranked controls;
4. Pearson correlation of the two residual vectors = partial Spearman rho.

This regression is diagnostic residualization only. It is not a predictive model
and no fitted coefficient enters a candidate.

### B. Excess-SD Spearman

Spearman:
`excess_share_sd` vs `share_surprise`.

This asks whether the estimated volatility beyond ordinary multinomial sampling
tracks next-game instability.

## Frozen inference

For each season/cell and for BOTH primary tests:
- raw statistic must be >0;
- game-cluster bootstrap 5,000 reps;
- player-cluster bootstrap 5,000 reps;
- both 95% CI lower bounds must be >0.

Bootstrap samples recompute ranks / partial residualization from scratch.

Support floor:
- >=300 player-games
- >=100 players
- >=50 games

## Per-cell disposition

A cell is:
`PLAYER_SHARE_EXCESS_VOLATILITY_MECHANISM_CONFIRMED`

only if BOTH primary tests pass in ALL THREE seasons 2023, 2024, 2025.

Otherwise:
`PLAYER_SHARE_EXCESS_VOLATILITY_MECHANISM_NOT_CONFIRMED`.

No pooling across positions/cells may rescue a failure.

Overall:
- >=1 confirmed cell:
  `PLAYER_OPPORTUNITY_EXCESS_VOLATILITY_EXISTS`
- none:
  `PLAYER_OPPORTUNITY_VOLATILITY_EXPLAINED_BY_MEAN_AND_COUNT_NOISE`

## Descriptive fields

May report:
- fraction of rows with positive excess variance;
- raw SD vs expected sampling SD;
- prior-share mean;
- excess-SD quartiles.

Descriptive fields cannot rescue a primary failure.

## Integrity

Require:
- exact seasons 2023-2025;
- target weeks 5-18;
- same/future feature violations = 0;
- same-team history violations = 0;
- all denominator source weeks strictly prior;
- stable identity coverage >=95%;
- sportsbook inputs = 0;
- OddsAPI calls = 0;
- model residuals = 0;
- predictive candidate fits = 0;
- projection changes = 0;
- production changed = false.

## Advancement

Only cells with confirmed excess-volatility mechanism may advance to a
mean-neutral allocation-uncertainty candidate.

For a confirmed cell, the future candidate should use **excess** role variance,
not raw share SD, so ordinary multinomial count noise is not double-counted.

If a cell fails:
- do not widen it from raw SD;
- do not rescue with mean-share buckets, different windows, alternate
  denominators, or result-derived thresholds.
