# PLAYER OPPORTUNITY SHARE STABILITY — 2023 HOLDOUT RESULT

Date: 2026-10-12

Status: **ALL FOUR CELLS CONFIRMED IN UNTOUCHED 2023 HOLDOUT. THREE-SEASON SIGNAL CONFIRMED. NO MODEL CHANGE.**

Frozen holdout:
- `docs/research/PLAYER_OPPORTUNITY_SHARE_STABILITY_2023_HOLDOUT_FREEZE.md`

Canonical authority:
- workflow run: `38294106319`
- exact head: `cad154e0f2fd0bdd2d1d5f502eda77c13ae2f636`
- artifact: `11767104207`
- artifact digest: `sha256:e5a042d84ba575fedd31148f9c5e9a5bebc9bd8f1ae36e2decd9df173280e8ce`
- workflow conclusion: `SUCCESS`

Disposition:
`PLAYER_OPPORTUNITY_SHARE_STABILITY_THREE_SEASON_CONFIRMED`

## Holdout panel

- season: **2023 only**
- weeks: 5-18
- scoreable player-games: **6,559**
- distinct players: **464**
- stable identity coverage: **100%**
- same/future feature violations: **0**
- same-team feature violations: **0**
- all four actual-share cells non-degenerate
- sportsbook inputs: 0
- OddsAPI calls: 0
- model residuals: 0
- predictive models fit: 0
- projection changes: 0
- production changed: false

## RB carry share

- rows: **1,464**
- players: 137
- games: 208
- Spearman rho: **+0.6409**
- game-cluster 95% CI: **[+0.5985, +0.6793]**
- player-cluster 95% CI: **[+0.5512, +0.7120]**

Quartile shape:
- Q1 prior SD ~0.0123 -> next-game share surprise **0.0245**
- Q4 prior SD ~0.2039 -> next-game share surprise **0.1979**

Disposition:
`PLAYER_SHARE_VOLATILITY_2023_HOLDOUT_CONFIRMED`

## RB target share

- rows: **1,464**
- Spearman rho: **+0.6116**
- game-cluster 95% CI: **[+0.5757, +0.6470]**
- player-cluster 95% CI: **[+0.5333, +0.6799]**

Quartile shape:
- Q1 prior SD ~0.0070 -> surprise **0.0114**
- Q4 prior SD ~0.0688 -> surprise **0.0592**

Disposition:
`PLAYER_SHARE_VOLATILITY_2023_HOLDOUT_CONFIRMED`

## WR target share

- rows: **2,283**
- players: 208
- Spearman rho: **+0.4686**
- game-cluster 95% CI: **[+0.4318, +0.5055]**
- player-cluster 95% CI: **[+0.4049, +0.5274]**

Quartile shape:
- Q1 prior SD ~0.0140 -> surprise **0.0255**
- Q4 prior SD ~0.0941 -> surprise **0.0814**

Disposition:
`PLAYER_SHARE_VOLATILITY_2023_HOLDOUT_CONFIRMED`

## TE target share

- rows: **1,348**
- players: 119
- Spearman rho: **+0.5699**
- game-cluster 95% CI: **[+0.5263, +0.6102]**
- player-cluster 95% CI: **[+0.4815, +0.6451]**

Quartile shape:
- Q1 prior SD ~0.0065 -> surprise **0.0120**
- Q4 prior SD ~0.0723 -> surprise **0.0652**

Disposition:
`PLAYER_SHARE_VOLATILITY_2023_HOLDOUT_CONFIRMED`

## Three-season interpretation

The parent 2024-2025 result and independent 2023 holdout all agree:

- RB carry-share volatility predicts next-game carry-share instability;
- RB target-share volatility predicts next-game target-share instability;
- WR target-share volatility predicts next-game target-share instability;
- TE target-share volatility predicts next-game target-share instability.

This is a replicated individual-player opportunity uncertainty feature.

It is materially different from the closed Opportunity State Conflict
Uncertainty V1:
- that lane used disagreement between model states and was null;
- this lane uses realized strictly-prior player share volatility and replicates.

## Current simulator gap

The canonical simulator currently allocates target/carry opportunities from
fixed player probabilities plus multinomial count noise.

It does not use player-specific prior opportunity-share volatility to vary those
probabilities across simulated game states.

The generic `rules_volatility_mult` acts on yardage/efficiency noise and does
not represent role/share uncertainty.

## Next legal moves

1. Freeze and preserve the raw share-volatility feature prospectively for Week 6
   before target-game outcomes.
2. Freeze a separate mean-preserving allocation-uncertainty candidate:
   - expected target/carry share unchanged;
   - finite team opportunity conserved;
   - player/room opportunity probabilities receive overdispersion based only on
     strictly-prior share volatility;
   - no threshold selected from 2023-2025;
   - no fitted widening coefficient from 2023-2025;
   - no production promotion without prospective 2026 evidence.
