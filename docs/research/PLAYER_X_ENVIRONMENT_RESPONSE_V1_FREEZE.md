# PLAYER X ENVIRONMENT RESPONSE V1 — FROZEN HISTORICAL DIAGNOSTIC

Date frozen: 2026-10-13

Status: **FROZEN BEFORE SCORING / DIAGNOSTIC ONLY / NO PRODUCTION CHANGE**

Parent:
- `docs/research/PLAYER_FIRST_INDIVIDUALIZATION_FRONTIER_V1.md`

Production reconstruction parent:
- `main@812ac2cde79fe3846174613a2f2777e57b34b739`

## Question

Does a named player's own strictly-prior residual response to the model's existing shared football environment adjustment predict his next residual?

This is distinct from a generic defense-v-position multiplier, the failed direct target-depth/run-location matchup formulas, and unconditional same-player bias correction. The existing environment multiplier is not replaced or refit.

## Frozen cells

- `WR_REC_YARDS`: WR/LWR/RWR/SWR, market `rec_yards`, environment `rules_pass_eff_mult`.
- `TE_REC_YARDS`: TE, market `rec_yards`, environment `rules_pass_eff_mult`.
- `RB_RUSH_YARDS`: RB/HB/FB, market `rush_yards`, environment `rules_rush_eff_mult`.

QB is excluded; M89/M90 remains protected.

## Historical authority and chronology

Target seasons: 2024 and 2025.
Target weeks: 7-18.
Each season is evaluated independently.

For each target row:
- baseline = exact reconstructed full-stack football mean from the frozen production parent;
- residual = `actual - baseline_projection` (positive means underprojection);
- environment = `log(shared_environment_multiplier)`;
- history = last up to 8 eligible earlier same-season games for the same stable player;
- minimum history = 6 games;
- no cross-season player residual history.

Target-game outcome never enters its feature history.

## Frozen player-response signal

For each target row, on the prior same-player history:

`x_j = log(environment_multiplier_j)`

`e_j = actual_j - baseline_projection_j`

Require at least six finite prior rows and at least three distinct x values after rounding to 1e-8.

Compute:
- `response_rho = Spearman(x_prior, residual_prior)`;
- `env_percentile = (count(x_prior <= x_current) + 0.5) / (n_prior + 1)`;
- `relative_environment = 2 * env_percentile - 1`;
- `player_environment_response_signal = response_rho * relative_environment`.

No coefficient is fit and no projection is changed.

## Controls

The partial-rank diagnostic controls:
- same-player `prior_mean_residual`;
- target `baseline_projection`.

Signal and target residual are rank transformed and separately residualized against an intercept plus ranked controls.

## Primary statistics

For each cell and each season independently report:
1. raw Spearman(signal, target residual);
2. partial Spearman after controls;
3. Q4 minus Q1 mean target residual across signal quartiles;
4. game-cluster bootstrap 95% CIs for raw and partial rho;
5. player-cluster bootstrap 95% CIs for raw and partial rho;
6. player-level descriptive distributions without post-hoc subgroup selection.

Frozen support floors:
- WR: >=250 rows, >=50 players, >=50 games;
- TE: >=125 rows, >=25 players, >=50 games;
- RB: >=125 rows, >=25 players, >=50 games.

Bootstrap:
- 2,000 replicates;
- deterministic seeds fixed in code;
- whole-game or whole-player cluster resampling.

A cell-season is supported only if:
- support floor passes;
- raw rho > 0;
- raw game-cluster CI lower > 0;
- raw player-cluster CI lower > 0;
- partial rho > 0;
- partial game-cluster CI lower > 0;
- partial player-cluster CI lower > 0;
- Q4-Q1 residual gap > 0.

A cell earns `PLAYER_X_ENVIRONMENT_RESPONSE_REPLICATED` only if the same frozen rule passes independently in BOTH 2024 and 2025.

No pooled-season, majority-season, role-subgroup, or threshold rescue.

## Integrity gates

Must all pass:
- exact target seasons only;
- target weeks 7-18 only;
- stable nonblank player identity;
- zero same/future history rows;
- all used environment multipliers finite and >0;
- exact full-stack baseline authority reproduced;
- sportsbook inputs = 0;
- OddsAPI calls = 0;
- parameters fit = 0;
- projection changes = 0;
- production changes = 0.

## Dispositions

At least one replicated cell:
`PLAYER_X_ENVIRONMENT_RESPONSE_HAS_REPLICATED_CELLS`

No replicated cell:
`PLAYER_X_ENVIRONMENT_RESPONSE_NULL`

Integrity failure:
`PLAYER_X_ENVIRONMENT_RESPONSE_INTEGRITY_FAILURE`

A replicated cell may only advance to a separately frozen integration candidate. A null closes this exact response-state construction.

After results are visible do not retune the history window, minimum games, percentile convention, cells, control set, or season gates.

Sportsbook inputs: **0**
OddsAPI calls: **0**
2026 outcomes: **0**
Projection changes: **0**
Production changes: **0**
