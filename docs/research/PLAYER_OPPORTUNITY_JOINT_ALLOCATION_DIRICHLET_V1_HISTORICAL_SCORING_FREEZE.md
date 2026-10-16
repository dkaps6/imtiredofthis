# PLAYER OPPORTUNITY JOINT ALLOCATION — DIRICHLET V1 HISTORICAL SCORING FREEZE

Frozen: 2026-10-13

Status: **FROZEN BEFORE JOINT HISTORICAL SCORE RESULTS. NO PRODUCTION CHANGE.**

Parents:
- `docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_PLAN.md`
- `docs/research/PLAYER_OPPORTUNITY_ALLOCATION_UNCERTAINTY_BETABINOMIAL_V1_RESULT.md`
- mechanics run `38400116643` = SUCCESS

Authorized cells only:
- `RB_CARRY_SHARE`
- `WR_TARGET_SHARE`

Not authorized:
- RB target share
- TE target share

## Historical universe

Seasons:
- 2023
- 2024
- 2025

Weeks:
- regular-season 5-18

Pregame context:
- exact historical model context rebuilt at the target-week cutoff;
- canonical current mean rules only;
- no target-game outcomes used in feature construction.

Target-game realized team opportunity count is read only for conditional
distribution scoring.

## Canonical probability authority

### RB carry share

For every target team-week:
1. rebuild canonical historical metrics;
2. take `rules_rush_share`;
3. apply the canonical stable top-five rushing pool;
4. apply the canonical 0.95 modeled-player mass cap;
5. preserve the residual rushing bucket.

The resulting final player probabilities are the baseline `q_i`.

### WR target share

For every target team-week:
1. rebuild canonical historical metrics;
2. take `rules_tgt_share`;
3. apply the canonical M38 WR hierarchy sharpening;
4. apply the canonical 0.95 modeled-player mass cap;
5. preserve the residual receiver bucket.

The resulting final player probabilities are the baseline `q_i`.

No raw historical share mean replaces canonical `q_i`.

## Candidate eligibility

A player may enter the eligible Dirichlet group only when:
- the V2 excess-variance feature exists strictly pregame;
- the cell is authorized;
- `rho_i = excess_share_var / (prior_share_mean*(1-prior_share_mean))`
  is finite and strictly between 0 and 1;
- canonical `q_i` is finite and strictly between 0 and 1.

All other players plus the canonical residual bucket form the complement.

Require:
- >=1 eligible player;
- complement probability strictly between 0 and 1;
- room rho strictly between 0 and 1.

Otherwise the room remains exact baseline.

## Joint baseline law

For exact target-game team opportunity count N:

`Y ~ Multinomial(N, q_eligible_1, ..., q_eligible_m, complement)`

## Joint candidate law

Use the already-frozen group/complement mechanics:

`rho_room = SUM[q_i(1-q_i)rho_i] / SUM[q_i(1-q_i)]`

`kappa = 1/rho_room - 1`

`alpha_j = p_j * kappa`

where `p` contains eligible canonical probabilities plus complement.

Then:

`Y ~ DirichletMultinomial(N, alpha)`

No coefficient is fit.

## Realized room counts

For the same target-game team opportunity total N:
- eligible player count = actual player rushes or targets;
- complement count = N - SUM(eligible player counts).

Require exact nonnegative conservation.

## Primary proper score

Room-level negative log score.

Per-room improvement:

`joint_logscore_improvement = NLL_baseline - NLL_candidate`

Positive favors candidate.

Each cell/season is supported only if:
- >=200 candidate-applied team-game rooms;
- >=28 distinct teams;
- >=100 authoritative games.

## Frozen inference

For each cell and season independently:
- mean joint log-score improvement >0;
- game-cluster bootstrap 5,000 reps;
- team-cluster bootstrap 5,000 reps;
- both 95% CI lower bounds >0.

Seed family:
`20261013`.

All three seasons must pass.

## Marginal protection

For each eligible player in candidate-applied rooms:

Baseline marginal:
`Binomial(N, q_i)`

Candidate marginal:
`BetaBinomial(N, alpha_i, alpha_0-alpha_i)`

Compute discrete CRPS.

For each cell/season:
- mean marginal CRPS candidate <= baseline;
- max expected-count gap <=1e-12.

Marginal CRPS cannot rescue a failed joint score.

## Integrity

Require:
- exact seasons 2023-2025;
- weeks 5-18;
- canonical probability sums + residual = 1 within 1e-12;
- realized counts conserve N exactly;
- source feature max week < target week;
- no sportsbook inputs;
- no OddsAPI;
- no fitted coefficient;
- no threshold search;
- production unchanged.

## Per-cell disposition

A cell receives:

`JOINT_DIRICHLET_ALLOCATION_HISTORICAL_PASS_FREEZE_PROSPECTIVE_SHADOW`

only if every primary and protection gate passes in 2023, 2024 and 2025.

Otherwise:

`JOINT_DIRICHLET_ALLOCATION_HISTORICAL_FAIL_CLOSED`.

No pooled RB/WR result may rescue a failed cell.

## No-rescue boundary

After results:
- do not tune rho;
- do not cap/floor rho;
- do not change eligible-player minimums;
- do not change the top-five rushing pool;
- do not change M38;
- do not change the 0.95 cap;
- do not select volatility buckets or player roles;
- do not add RB target or TE target;
- do not change the history window.

Historical PASS authorizes only a frozen 2026 prospective shadow.
