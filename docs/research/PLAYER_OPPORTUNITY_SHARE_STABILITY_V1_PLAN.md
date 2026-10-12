# PLAYER OPPORTUNITY SHARE STABILITY V1 — DIAGNOSTIC PLAN

Frozen: 2026-10-12

Status: **DIAGNOSTIC ONLY — FROZEN BEFORE TARGET-GAME SHARE-SURPRISE SCORING — NO PROJECTION OR DISTRIBUTION CHANGE.**

Parent roadmap:
- `docs/research/INDIVIDUAL_OPPORTUNITY_ALLOCATION_ROADMAP_V1.md`

## Question

Does a player's own strictly-prior week-to-week opportunity-share volatility
identify a player whose next-game opportunity allocation is inherently less
stable?

This is a player-specific uncertainty question, not a new mean allocator.

The primary hypothesis is:

> larger strictly-prior share SD -> larger absolute next-game share surprise.

A positive result could justify a separately frozen uncertainty/confidence
candidate later. It does not authorize mean changes or production integration.

## Anti-retest boundary

This is distinct from:

### Opportunity State Conflict Uncertainty V1
Closed/null:
- used disagreement between PlayerForm and Bayes opportunity states;
- did not test realized strictly-prior share volatility.

Do not reopen disagreement-based widening.

### WR room regime / offensive-regime history
Closed:
- Receiver Room Targets-Per-Play rescue;
- Offensive Regime Boundary hard reset;
- WR Anchor / Role-Transmission.

V1 does not alter prior-season room history or reset any mean.

### Target-depth distribution shadow
Closed:
- used target-depth dispersion as a symmetric rec-yard width transform.

V1 uses opportunity-share volatility and does not alter receiving-efficiency
shape.

### RB retrospective mean program
M96 remains closed. V1 does not alter RB mean, routing or allocation.

## Data authority

Public nflverse / nflreadpy weekly player stats normalized through the existing
PlayerForm V2 semantics.

Target seasons:
- 2024
- 2025

Target weeks:
- regular-season Weeks 5-18

Weeks 1-4 are intentionally excluded from scoring because the frozen feature
requires at least four same-season, same-team prior games.

No sportsbook data.

## Frozen cells

Evaluate independently:

1. `RB_CARRY_SHARE`
   - target position: RB / HB / FB
   - realized share: `rush_share_game = player rushes / team rushes`

2. `RB_TARGET_SHARE`
   - target position: RB / HB / FB
   - realized share: `tgt_share_game = player targets / team targets`

3. `WR_TARGET_SHARE`
   - target position: WR-family
   - realized share: `tgt_share_game`

4. `TE_TARGET_SHARE`
   - target position: TE
   - realized share: `tgt_share_game`

These denominators match the existing PlayerForm / opportunity-decomposition
semantics. Do not switch to within-room shares after results.

## Frozen feature

For each target player-game and cell:

- same player stable identity;
- same current team;
- same current season;
- games strictly before target week;
- latest up to 8 completed prior games;
- require >=4 prior games with finite share.

Compute:

- `prior_share_mean`
- `prior_share_sd` using population SD (`ddof=0`)
- `prior_share_min`
- `prior_share_max`

Only `prior_share_sd` is a primary explanatory feature.

No prior-season share enters this V1.

## Frozen target

`share_surprise = abs(actual_target_game_share - prior_share_mean)`

This is an intrinsic opportunity-allocation stability target.

It is not:
- projection error;
- betting error;
- final-model residual;
- sportsbook-relative error.

The target-game share is read only after the feature is frozen for that row.

## Frozen support

A row is scoreable only if:
- stable player identity;
- exact target position/cell;
- >=4 prior same-season same-team finite-share games;
- finite target-game share;
- target game is Weeks 5-18.

A season/cell is supported only if:
- >=300 player-games;
- >=100 distinct players;
- >=50 games.

## Frozen primary inference

For every season/cell:

1. Spearman rho:
   `prior_share_sd` vs `share_surprise`

2. game-cluster bootstrap:
   - 5,000 reps
   - seed `20261012`
   - resample authoritative target game clusters
   - recompute Spearman rho

3. player-cluster bootstrap:
   - 5,000 reps
   - seed `20271012`
   - resample stable player identities
   - recompute Spearman rho

Expected sign: **positive**.

A season/cell is `VOLATILITY_SIGNAL_SUPPORTED` only if:
- support floor passes;
- raw rho > 0;
- game-cluster 95% CI lower bound > 0;
- player-cluster 95% CI lower bound > 0.

## Frozen per-cell disposition

A cell is:

`PLAYER_SHARE_VOLATILITY_REPLICATED`

only if `VOLATILITY_SIGNAL_SUPPORTED` in BOTH 2024 and 2025.

Otherwise:

`PLAYER_SHARE_VOLATILITY_NOT_REPLICATED`.

No pooled cross-position result may rescue a failed cell.

Overall:
- >=1 replicated cell:
  `PLAYER_OPPORTUNITY_SHARE_STABILITY_SIGNAL_CONFIRMED`
- none:
  `PLAYER_OPPORTUNITY_SHARE_STABILITY_NULL`

## Descriptive-only secondary fields

May report, but may not rescue the primary gate:
- share-surprise means by prior-SD quartile;
- prior share mean;
- position/cell row counts;
- player counts;
- week counts.

No quartile threshold may become a rule from this same result.

## Integrity gates

Must report and require:
- target seasons exact 2024/2025;
- target weeks exact 5-18;
- same/future feature violations = 0;
- same-team feature violations = 0;
- stable identity coverage among scoreable candidates >=95%;
- sportsbook inputs used = 0;
- OddsAPI calls = 0;
- model residuals used = 0;
- fitted predictive models = 0;
- candidate projection changes = 0;
- production changed = false.

## Advancement

If a cell replicates:
- freeze a separate uncertainty/confidence candidate BEFORE any transformation;
- preserve mean projection exactly;
- test distribution/probability calibration or prospective error stratification;
- no threshold chosen from this diagnostic;
- require independent holdout/prospective confirmation before production.

If no cell replicates:
- close share-volatility uncertainty V1;
- do not rescue with entropy, alternative windows, within-room denominators,
  position pooling, or result-derived cutoffs.
