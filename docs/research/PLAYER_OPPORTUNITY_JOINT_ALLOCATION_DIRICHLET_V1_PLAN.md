# PLAYER OPPORTUNITY JOINT ALLOCATION — DIRICHLET GROUP/COMPLEMENT V1 PLAN

Frozen: 2026-10-13

Status: **MECHANICS / HISTORICAL-INTEGRATION CANDIDATE ONLY. NO PRODUCTION CHANGE. FROZEN BEFORE JOINT SCORE RESULTS.**

Scientific parents:
- `PLAYER_OPPORTUNITY_SHARE_STABILITY_THREE_SEASON_CONFIRMED`
- `PLAYER_OPPORTUNITY_EXCESS_VOLATILITY_EXISTS`
- `BETA_BINOMIAL_ALLOCATION_UNCERTAINTY_HAS_PASSING_CELLS`

Historically supported cells:
- RB carry share
- WR target share

Explicitly unsupported in this family:
- RB target share
- TE target share

## Problem

The canonical simulator currently uses fixed player allocation probabilities and
multinomial draws.

That conserves finite team volume, but it assumes the player's underlying role
share is fixed within a simulated game distribution.

The three-season evidence shows that RB carry share and WR target share have
player-specific latent role variance beyond ordinary multinomial/binomial count
noise.

The marginal Beta-Binomial V1 improved proper allocation scores, but independent
player Beta-Binomial draws cannot be inserted directly into production because
they do not jointly conserve one finite team opportunity total.

## Candidate objective

Create one joint allocation law that:

1. preserves the canonical player mean probabilities exactly;
2. preserves the finite team target/carry total exactly in every simulation;
3. introduces latent role-state overdispersion only from pregame
   `excess_share_var`;
4. uses no fitted widening coefficient;
5. does not use Week-6 or later outcomes;
6. does not alter team play volume, pass rate, efficiency, or mean entitlement.

## Transport the historical volatility scale to canonical model probabilities

For one historically supported player feature:

- raw prior share mean: `mu_i`
- excess latent share variance: `v_i`

Define a dimensionless latent-role correlation:

`rho_i = v_i / (mu_i * (1 - mu_i))`

when:
- `0 < mu_i < 1`
- `0 < v_i < mu_i * (1-mu_i)`.

Otherwise the player is candidate-ineligible.

At simulation time let `q_i` be the player's **canonical final allocation
probability after all existing mean rules**:
- for RB carry: after current rush-share filtering/capping;
- for WR targets: after M38 WR hierarchy sharpening and current team-share
  capping.

The candidate never replaces `q_i` with the raw historical mean.

The player-specific desired latent variance on the canonical probability scale
is descriptive only:

`v_i_star = rho_i * q_i * (1 - q_i)`.

## Joint group/complement law

For one supported cell within one team:

Eligible set `E`:
- RB carry: Beta-Binomial-eligible RB players only;
- WR target: Beta-Binomial-eligible WR players only.

All other modeled players plus the canonical residual bucket form one
**complement** category for the first-stage latent draw.

Let:
- `q_i` = canonical probability for eligible player i;
- `c = 1 - SUM_E(q_i)`.

Require:
- at least one eligible player;
- `0 < c < 1`.

### Room-level latent correlation

Use the probability-variance-weighted average of the player-specific
correlations:

`rho_room = SUM_E[q_i(1-q_i) * rho_i] / SUM_E[q_i(1-q_i)]`.

Equivalently:

`rho_room = SUM_E[v_i_star] / SUM_E[q_i(1-q_i)]`.

This is algebraic and outcome-free. No coefficient is fit.

Require:
- `0 < rho_room < 1`.

Then:

`kappa = 1/rho_room - 1`.

Dirichlet parameters:
- `alpha_i = q_i * kappa` for each eligible player;
- `alpha_c = c * kappa` for the complement.

Draw latent first-stage probabilities:

`P ~ Dirichlet(alpha_1, ..., alpha_m, alpha_c)`.

Given exact team opportunity total N:

`Y_first ~ Multinomial(N, P)`.

This conserves N exactly.

### Complement allocation

Let canonical baseline probabilities for all non-eligible modeled players and
the residual bucket be `r_j`.

Conditional on complement count `Y_c`, allocate that count using their
unchanged relative baseline probabilities:

`Y_noneligible | Y_c ~ Multinomial(Y_c, r_j / c)`.

Therefore:
- every simulation conserves exact team N;
- every category's expected count remains `N*q_j`;
- unsupported players receive no player-specific volatility input;
- their aggregate complement count varies only as the mathematically necessary
  counterweight to eligible-player role uncertainty.

## Exact mean identities

Must hold analytically and in mechanics tests:

For every player/category:
`E[Y_i]_candidate = N * q_i`.

Team:
`SUM_i Y_i = N` in every simulation.

No mean allocator changes.

## Semantic RNG isolation

The research implementation must use a dedicated allocation RNG stream derived
deterministically from the parent seed.

Changing allocation uncertainty must not perturb:
- team plays;
- pass/rush totals;
- efficiency shocks;
- catch-rate RNG;
- receiving/rushing efficiency RNG;
- QB efficiency RNG;
- touchdown RNG.

No production simulator edit is authorized in the mechanics stage.

## Mechanics gates

Before historical scoring:

1. exact finite team count conserved in every draw;
2. empirical mean allocation gap <=0.01 opportunities in a >=100,000-draw
   deterministic unit simulation;
3. no eligible players -> exact baseline allocation path;
4. invalid rho -> exact baseline allocation path;
5. one eligible player reduces to the matching Beta-Binomial first-stage
   marginal within numeric tolerance;
6. complement internal expected shares remain baseline;
7. only RB carry / WR target cells accepted;
8. no fitted coefficient;
9. no sportsbook / OddsAPI;
10. production unchanged.

## Historical integration test

Only after mechanics pass, freeze a separate scoring run using 2023-2025
pregame historical model contexts.

Historical candidate must use:
- canonical final allocation probabilities `q_i`, not hindsight shares;
- strictly-prior excess-variance fields;
- exact target-game team opportunity count only for conditional distribution
  scoring, not feature construction.

Primary joint score:
- room/team-game multinomial log score versus Dirichlet-Multinomial log score
  on eligible players + complement;
- RB carry and WR target scored separately;
- 2023, 2024, 2025 independently;
- game-cluster and team-cluster bootstrap;
- all three seasons must improve.

Marginal protection:
- eligible-player count CRPS must be non-worse in every season;
- expected-count gap <=1e-12 analytically.

No pooled result may rescue a failed cell.

## No-rescue boundary

After scoring, do not:
- tune rho;
- cap/floor rho from outcomes;
- choose volatility buckets;
- change the 8-game feature window;
- pool RB carry and WR target;
- add RB target or TE target;
- choose only certain player roles;
- alter canonical mean shares.

Failure closes this exact V1 law.

## Advancement

A historical PASS authorizes only:
1. a separately frozen 2026 prospective joint-allocation shadow;
2. comparison against the already-locked Week-6+ excess-variance fields;
3. no production promotion without prospective calibration evidence.
