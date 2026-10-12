# PLAYER OPPORTUNITY ALLOCATION UNCERTAINTY — BETA-BINOMIAL V1 FREEZE

Frozen: 2026-10-12

Status: **MEAN-IDENTICAL COMPONENT DISTRIBUTION DIAGNOSTIC. FROZEN BEFORE DISTRIBUTION SCORING. NO JOINT SIMULATOR OR PRODUCTION CHANGE.**

Parent authority:
- `docs/research/PLAYER_OPPORTUNITY_SHARE_STABILITY_V1_RESULT.md`
- `docs/research/PLAYER_OPPORTUNITY_SHARE_STABILITY_2023_HOLDOUT_RESULT.md`
- `docs/research/PLAYER_OPPORTUNITY_SHARE_VOLATILITY_MECHANISM_V2_PLAN.md`
- `docs/research/PLAYER_OPPORTUNITY_SHARE_VOLATILITY_MECHANISM_V2_RESULT.md`

V2 canonical authority:
- run `38295515693`
- exact head `81cd2beeb6f8da9fd5e485d70daee546285e6905`
- artifact `11768432238`
- digest `sha256:7b31dd29e4b94674d6948cf84af24fae7181d1a4f502a07beaa43919d4d7f3a6`

Confirmed V2 cells:
- `RB_CARRY_SHARE`
- `RB_TARGET_SHARE`
- `WR_TARGET_SHARE`

Not authorized:
- `TE_TARGET_SHARE`

## Question

Does adding the measured **excess player role/share variance** improve the
probabilistic distribution of next-game player opportunities while keeping the
expected player share exactly unchanged?

This is a component diagnostic. It deliberately conditions on the realized
target-game team opportunity count to isolate individual allocation uncertainty
from team-volume forecasting error.

It is not a deployable pregame projection by itself.

## Evaluation universe

Seasons:
- 2023
- 2024
- 2025

Weeks:
- regular-season 5-18

Rows:
- exact V2 rows for the three confirmed cells;
- stable player identity;
- finite `prior_share_mean`;
- finite `excess_share_var`;
- target-game team opportunity count `N > 0`;
- integer target-game player opportunity count `0 <= y <= N`.

Target-game N/y are read only after the strictly-prior V2 features are frozen.

## Target-game opportunity counts

- `RB_CARRY_SHARE`
  - N = actual team rushing attempts under the same PlayerForm denominator used
    in V1/V2;
  - y = actual player rushing attempts.

- `RB_TARGET_SHARE`
  - N = actual team targets;
  - y = actual RB targets.

- `WR_TARGET_SHARE`
  - N = actual team targets;
  - y = actual WR targets.

No denominator changes.

## Baseline marginal law

The current fixed-share multinomial allocation implies, marginally for one
player:

`Y ~ Binomial(N, mu)`

where:
- `mu = prior_share_mean`.

This is the frozen baseline for this component diagnostic.

## Candidate marginal law

For:
- `v = excess_share_var`.

If:
- `v <= 0`: candidate is exactly the baseline Binomial law.

If:
- `0 < v < mu * (1 - mu)` and `0 < mu < 1`:

derive the Beta concentration by method of moments:

`kappa = mu * (1 - mu) / v - 1`

`alpha = mu * kappa`

`beta = (1 - mu) * kappa`

and score:

`P ~ Beta(alpha, beta)`

`Y | P ~ Binomial(N, P)`

therefore:

`Y ~ BetaBinomial(N, alpha, beta)`.

If the method-of-moments parameters are mathematically invalid or non-finite,
the row remains baseline and is reported as candidate-ineligible.

No epsilon-based scientific shrinkage, cap, floor, fitted coefficient or
threshold is allowed. Pure numerical handling must not change an otherwise
valid distribution.

## Exact mean identity

For every row:

baseline expected count:
`E[Y]_base = N * mu`

candidate expected count:
`E[Y]_cand = N * mu`

Required max absolute expected-count gap:
- <= `1e-12`.

The candidate is distribution-only.

## Primary proper score

Discrete CRPS for count outcome y:

`CRPS = SUM_{k=0}^{N} (F(k) - 1[y <= k])^2`

using:
- Binomial CDF for baseline;
- Beta-Binomial CDF for candidate.

Per-row improvement:

`crps_improvement = CRPS_base - CRPS_candidate`

Positive favors candidate.

## Frozen primary inference

For each cell and season independently:

Support:
- >=300 rows
- >=100 players
- >=50 games

Inference:
- mean CRPS improvement >0;
- game-cluster bootstrap 5,000 reps;
- player-cluster bootstrap 5,000 reps;
- seed family `20261012`;
- both 95% CI lower bounds >0.

A cell passes primary only if all three seasons pass.

## Frozen log-score protection

For every supported season/cell:

- compute Binomial log PMF of realized y;
- compute candidate log PMF;
- use negative log score `NLL = -log(P(y))`.

Candidate mean NLL must be <= baseline mean NLL in every season.

Log score cannot rescue a CRPS failure.

## Descriptive-only diagnostics

May report:
- changed-row fraction (`v > 0` and valid Beta parameters);
- baseline/candidate variance ratio;
- 80% and 90% central count-interval coverage;
- candidate-ineligible reasons;
- CRPS on changed rows only.

These cannot rescue the all-row primary gate.

## Per-cell disposition

A confirmed V2 cell earns:

`BETA_BINOMIAL_ALLOCATION_UNCERTAINTY_HISTORICAL_PASS`

only if ALL pass:
1. support floor in 2023, 2024, 2025;
2. mean CRPS improvement >0 in all three seasons;
3. game-cluster 95% CRPS-improvement lower bound >0 in all three;
4. player-cluster 95% CRPS-improvement lower bound >0 in all three;
5. mean NLL non-worse in all three;
6. max expected-count mean gap <=1e-12;
7. no source chronology violation;
8. no sportsbook inputs;
9. no OddsAPI calls;
10. no fitted coefficient;
11. no threshold search;
12. no production change.

Otherwise:
`BETA_BINOMIAL_ALLOCATION_UNCERTAINTY_HISTORICAL_FAIL_CLOSED`.

No pooled result may rescue a failed cell.

## Advancement

A historical PASS does **not** authorize replacing the joint simulator.

For a passing cell:
1. lock the exact excess-variance feature prospectively in 2026;
2. design a separate team-conserving joint allocation implementation;
3. preserve player means and finite team volume;
4. use semantic RNG isolation;
5. require prospective calibration / distribution improvement before production.

If a cell fails:
- do not tune excess variance;
- do not add shrinkage;
- do not choose volatility buckets;
- do not alter the 8-game history;
- do not pool positions;
- do not convert raw V1 SD into a widening rule.

TE remains outside this V1 candidate family regardless of other results.
