# TE Player Allocation Joint Dirichlet V1 — Freeze

Date: 2026-10-14

Status: **FROZEN BEFORE JOINT SCORE / RESEARCH ONLY**

Parents:
- `TE_PLAYER_ALLOCATION_DIFFICULTY_CONFIRMED`
- `PLAYER_OPPORTUNITY_DIFFICULTY_TARGET_ALLOCATION_SOURCE_READY`
- TE-R5P promoted mean authority

Canonical parent:
- allocation-only run `38656442803`
- exact head `e936e3eb03bf085c7f2bce1249cfa97435eb4b37`
- artifact `11936407975`
- digest `sha256:7ded306cd0a049902130ccc5cab995231b192120c02679c3852757bc8df29cc0`

## Question

Can the confirmed named-TE allocation-difficulty state be represented as a **finite-team, team-conserving target-allocation distribution** that improves joint and individual probabilistic forecasts while leaving TE-R5P means unchanged?

This is TE only. WR does not advance from the allocation-only bridge.

## Frozen universe

- regular-season 2024 development;
- regular-season 2025 independent confirmation;
- only TE rows already scored by `PLAYER_ALLOCATION_ONLY_DIFFICULTY_V1`;
- same stable historical player identity;
- no new source or refit.

## Pregame mean authority

For each TE row:
- `T_hat` = recovered pregame team target total;
- `q_i` = promoted TE-R5P full-team target share;
- promoted TE target mean remains exactly `T_hat * q_i`.

No target mean changes.

## Convert named-player allocation width to overdispersion

Parent candidate count-scale width:
`sigma_i = player allocation MAE * sqrt(pi/2)`.

Desired count variance:
`V_i = sigma_i^2`.

Ordinary multinomial variance under the pregame mean:
`B_i = T_hat * q_i * (1-q_i)`.

For rows with:
- `T_hat > 1`;
- `0 < q_i < 1`;
- `V_i > B_i`;

derive exactly:

`rho_i = (V_i / B_i - 1) / (T_hat - 1)`.

A TE is candidate-eligible only when `0 < rho_i < 1`.

No clipping, cap, floor, coefficient, or fitted transform.

Rows with underdispersion, invalid rho, or insufficient player history remain canonical multinomial and are absorbed into the complement.

## Actual team target count for conditional distribution scoring

The target-game **realized team target count N** is used only as the conditioning total for distribution scoring, exactly as in the previously frozen joint-allocation historical diagnostics.

Infer N only from target-game outcome identity:
- for any TE with `actual_targets > 0` and `actual_share > 0`,
  `N = actual_targets / actual_share`;
- require N to be integer within `1e-8`;
- require all positive-target TEs in the same team-game to imply the same N within `1e-8`;
- rooms with no valid N inference are not scoreable and cannot be silently imputed.

N never constructs `q_i`, `rho_i`, or player eligibility.

## Room law

For each scoreable team-game with >=1 eligible TE:

Categories:
1. each eligible TE separately;
2. one complement containing every ineligible TE plus all non-TE target mass.

Baseline probabilities:
- eligible `q_i`;
- complement `1 - sum(q_i)`.

Require complement probability strictly in (0,1).

Actual counts:
- each eligible TE's integer actual targets;
- complement = `N - sum(eligible TE targets)`.

Require exact nonnegative conservation.

### Baseline

`Y ~ Multinomial(N, p)`.

### Candidate

Room overdispersion:

`rho_room = SUM[q_i(1-q_i)rho_i] / SUM[q_i(1-q_i)]`

`kappa = 1/rho_room - 1`

`alpha_j = p_j * kappa`

`Y ~ DirichletMultinomial(N, alpha)`.

No coefficient is fit.

## Primary joint proper score

Per room:
`gain_joint = NLL_multinomial - NLL_dirichlet_multinomial`.

Positive favors the candidate.

Support per season:
- >=200 candidate-applied team-game rooms;
- >=28 teams;
- >=100 distinct games.

For **2024 and 2025 separately**:
- mean joint gain >0;
- game-cluster bootstrap 5,000 reps, 95% CI lower >0;
- team-cluster bootstrap 5,000 reps, 95% CI lower >0.

Seeds:
- game `2026101421`;
- team `2026101422`.

2024 cannot rescue 2025.

## Marginal protection

For every eligible TE in a candidate-applied room:

Baseline marginal:
`Binomial(N, q_i)`.

Candidate marginal:
`BetaBinomial(N, alpha_i, alpha_0-alpha_i)`.

Use exact discrete CRPS.

For each season:
- mean marginal candidate CRPS <= baseline CRPS;
- expected-count gap <= `1e-12`.

Marginal CRPS cannot rescue failed room-level joint score.

## Named-player confirmation

On 2025 eligible TE marginals:
- player-cluster bootstrap 5,000 reps for marginal CRPS gain, 95% CI lower >0;
- among TEs with >=6 candidate-applied games:
  - median per-player marginal CRPS gain >0;
  - fraction with positive mean marginal CRPS gain >0.50.

Position aggregate alone cannot qualify the candidate.

## Integrity

Require:
- exact 2024/2025 seasons only;
- no target mean changes;
- exact room count conservation;
- finite probabilities summing to 1 within `1e-12`;
- no sportsbook;
- no OddsAPI;
- no 2026 outcomes;
- no production change.

## Disposition

PASS only if every support, 2024 joint, 2025 joint, marginal-protection, named-player, and integrity gate passes:

`TE_PLAYER_ALLOCATION_JOINT_DIRICHLET_CONFIRMED`.

Otherwise:

`NO_ACTIONABLE_TE_PLAYER_ALLOCATION_JOINT_DIRICHLET`.

If support is insufficient because too few rooms have inferable N or valid overdispersion:

`TE_PLAYER_ALLOCATION_JOINT_DIRICHLET_SOURCE_LIMITED`.

## No rescue

After scoring, do not:
- change prior window/min games;
- clip or rescale rho;
- select TE1/TE2 subsets;
- pool seasons to rescue a failed 2025 confirmation;
- alter TE-R5P means;
- add WR/RB;
- use sportsbook information.

A pass authorizes only a separately frozen TE-R5P Monte Carlo shadow/integration candidate. It does not change production.
