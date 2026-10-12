# PLAYER OPPORTUNITY SHARE VOLATILITY — MECHANISM DECOMPOSITION V2 RESULT

Date: 2026-10-12

Status: **EXCESS ROLE VOLATILITY CONFIRMED FOR RB CARRY SHARE, RB TARGET SHARE, AND WR TARGET SHARE. TE TARGET SHARE FAILS THE FROZEN THREE-SEASON GATE. NO MODEL CHANGE.**

Frozen plan:
- `docs/research/PLAYER_OPPORTUNITY_SHARE_VOLATILITY_MECHANISM_V2_PLAN.md`

Canonical authority:
- workflow run: `38295515693`
- exact head: `81cd2beeb6f8da9fd5e485d70daee546285e6905`
- artifact: `11768432238`
- artifact digest: `sha256:7b31dd29e4b94674d6948cf84af24fae7181d1a4f502a07beaa43919d4d7f3a6`
- workflow conclusion: `SUCCESS`

Overall disposition:
`PLAYER_OPPORTUNITY_EXCESS_VOLATILITY_EXISTS`

Panel:
- seasons: 2023, 2024, 2025
- weeks: 5-18
- rows: **19,623**
- distinct players: **694**
- stable identity coverage: **100%**
- feature chronology violations: **0**
- denominator chronology violations: **0**
- same-team violations: **0**
- sportsbook inputs: **0**
- OddsAPI calls: **0**
- model residuals: **0**
- predictive candidate fits: **0**
- projection changes: **0**
- production changed: **false**

## What V2 removed

V1 established that raw strictly-prior player share SD predicts next-game share
surprise in all four cells across three seasons.

V2 asked whether that signal is merely ordinary finite-count noise already
represented by the simulator.

For every row:

`sampling_var = mean(mu * (1 - mu) / N_j)`

`excess_share_var = max(0, prior_share_sd^2 - sampling_var)`

Two frozen tests were required:
1. partial Spearman of raw SD vs next-game surprise after controlling ranked
   prior mean share and ranked expected sampling SD;
2. Spearman of `excess_share_sd` vs next-game surprise.

Both game-cluster and player-cluster 95% CI lower bounds had to be >0 in every
season for a cell to confirm.

## RB carry share — CONFIRMED

2023:
- adjusted partial rho: **+0.3614**
- partial game CI low: **+0.3053**
- partial player CI low: **+0.2764**
- excess-SD rho: **+0.4989**
- excess game CI low: **+0.4452**
- excess player CI low: **+0.4149**
- positive excess-variance rows: **64.89%**

2024:
- adjusted partial rho: **+0.1870**
- partial game CI low: **+0.1304**
- partial player CI low: **+0.1064**
- excess-SD rho: **+0.4609**
- excess game CI low: **+0.4141**
- excess player CI low: **+0.3710**
- positive excess-variance rows: **70.50%**

2025:
- adjusted partial rho: **+0.3149**
- partial game CI low: **+0.2611**
- partial player CI low: **+0.2349**
- excess-SD rho: **+0.5024**
- excess game CI low: **+0.4606**
- excess player CI low: **+0.4183**
- positive excess-variance rows: **59.61%**

Disposition:
`PLAYER_SHARE_EXCESS_VOLATILITY_MECHANISM_CONFIRMED`

## RB target share — CONFIRMED

2023:
- adjusted partial rho: **+0.1498**
- partial game CI low: **+0.0984**
- partial player CI low: **+0.0712**
- excess-SD rho: **+0.3293**
- excess game CI low: **+0.2868**
- excess player CI low: **+0.2494**
- positive excess-variance rows: **44.33%**

2024:
- adjusted partial rho: **+0.0933**
- partial game CI low: **+0.0445**
- partial player CI low: **+0.0344**
- excess-SD rho: **+0.3053**
- excess game CI low: **+0.2598**
- excess player CI low: **+0.2253**
- positive excess-variance rows: **43.53%**

2025:
- adjusted partial rho: **+0.0846**
- partial game CI low: **+0.0307**
- partial player CI low: **+0.0196**
- excess-SD rho: **+0.3049**
- excess game CI low: **+0.2626**
- excess player CI low: **+0.2292**
- positive excess-variance rows: **32.62%**

Disposition:
`PLAYER_SHARE_EXCESS_VOLATILITY_MECHANISM_CONFIRMED`

## WR target share — CONFIRMED

2023:
- adjusted partial rho: **+0.1459**
- partial game CI low: **+0.1014**
- partial player CI low: **+0.0944**
- excess-SD rho: **+0.2502**
- excess game CI low: **+0.2106**
- excess player CI low: **+0.1920**
- positive excess-variance rows: **51.25%**

2024:
- adjusted partial rho: **+0.1952**
- partial game CI low: **+0.1470**
- partial player CI low: **+0.1335**
- excess-SD rho: **+0.2927**
- excess game CI low: **+0.2498**
- excess player CI low: **+0.2212**
- positive excess-variance rows: **53.38%**

2025:
- adjusted partial rho: **+0.1826**
- partial game CI low: **+0.1426**
- partial player CI low: **+0.1216**
- excess-SD rho: **+0.2730**
- excess game CI low: **+0.2363**
- excess player CI low: **+0.2044**
- positive excess-variance rows: **53.58%**

Disposition:
`PLAYER_SHARE_EXCESS_VOLATILITY_MECHANISM_CONFIRMED`

## TE target share — NOT CONFIRMED

2023:
- adjusted partial rho: **+0.0525**
- partial game CI low: **+0.0025**
- partial player CI low: **-0.0134**
- excess-SD rho: **+0.2554**
- excess game CI low: **+0.2056**
- excess player CI low: **+0.1731**
- mechanism gate: **FAIL**

2024 and 2025 both pass the adjusted mechanism tests, but the frozen contract
requires all three seasons.

Disposition:
`PLAYER_SHARE_EXCESS_VOLATILITY_MECHANISM_NOT_CONFIRMED`

Do not use raw TE target-share SD to widen target-allocation uncertainty under
this family.

## Interpretation

For RB carry share, RB target share and WR target share, player-specific
game-to-game role instability remains after removing:

- player mean-share scale;
- ordinary finite-count sampling variance already represented by fixed-share
  multinomial allocation.

Therefore the canonical simulator is missing a real player-specific opportunity
uncertainty component in those three cells.

TE remains protected from this candidate family.

## Next legal move

Freeze a mean-identical distribution diagnostic for the three confirmed cells.

The first test should isolate allocation uncertainty before modifying the joint
simulator:

- condition on target-game team opportunity count strictly as a component
  diagnostic;
- baseline marginal law: Binomial(N, prior mean share);
- candidate marginal law: Beta-Binomial with the same mean and latent share
  variance derived directly from the frozen `excess_share_var`;
- no fitted widening coefficient;
- no threshold;
- unsupported/invalid rows remain baseline;
- primary evaluation = proper distribution score with three-season replication;
- no production change.

Only after a mean-neutral marginal distribution candidate succeeds should a
joint team-conserving implementation be designed.
