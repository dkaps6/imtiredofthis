# PLAYER OPPORTUNITY JOINT ALLOCATION — DIRICHLET V1 HISTORICAL SCORE FREEZE

Frozen: 2026-10-13

Status: **FROZEN BEFORE HISTORICAL JOINT-SCORE RESULTS. NO PRODUCTION CHANGE.**

Parents:
- `docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_PLAN.md`
- `docs/research/PLAYER_OPPORTUNITY_JOINT_ALLOCATION_DIRICHLET_V1_MECHANICS_RESULT.md`
- `docs/research/PLAYER_OPPORTUNITY_ALLOCATION_UNCERTAINTY_BETABINOMIAL_V1_RESULT.md`
- `docs/research/PLAYER_OPPORTUNITY_SHARE_VOLATILITY_MECHANISM_V2_RESULT.md`

Candidate cells, adjudicated independently:
1. `RB_CARRY_SHARE`
2. `WR_TARGET_SHARE`

RB target share and TE target share remain excluded.

## Historical universe

- seasons: 2023, 2024, 2025
- regular-season weeks: 5-18
- leakage-safe historical model context for each target week
- strictly-prior V2 excess-share variance
- target-game team opportunity count used only as the conditioned scoring total N
- no sportsbook / OddsAPI input

## Canonical mean authority

For each target team-week reconstruct the canonical model allocation probabilities
from the exact pregame model inputs.

RB carry:
- use canonical `rules_rush_share`;
- apply canonical top-5 rushing-share filter;
- apply canonical 0.95 modeled-share cap and residual semantics.

WR target:
- use canonical `rules_tgt_share`;
- apply canonical M38 WR hierarchy sharpening;
- apply canonical 0.95 modeled-share cap and residual semantics.

The candidate must use these final probabilities q_i.
Do not substitute historical realized mean share for q_i.

## Volatility transport

For an eligible player with frozen V2:
- raw prior mean mu_i
- excess share variance v_i

`rho_i = v_i / (mu_i * (1-mu_i))`

only when mathematically valid.

Use the exact joint group/complement law frozen in V1 mechanics.
No coefficient, shrinkage, floor, cap, or threshold may be fit from outcomes.

## Primary joint score

For each team-game/cell with >=1 eligible player and a valid complement:
collapse the outcome vector to:
- each eligible player separately;
- one complement category containing all other players + residual opportunities.

Baseline:
`Multinomial(N, q)`

Candidate:
the frozen Dirichlet-Multinomial group/complement law.

Primary per-room improvement:
`joint_log_score_improvement = log P_candidate(y) - log P_baseline(y)`

Positive favors candidate.

A season/cell is supported only if:
- >=100 team-games;
- >=50 distinct eligible players.

Inference:
- mean joint log-score improvement >0;
- game-cluster bootstrap 5,000 reps;
- team-cluster bootstrap 5,000 reps;
- both 95% CI lower bounds >0.

## Marginal protection

For every eligible player-row:
- compare canonical-mean Binomial count CRPS to the implied candidate marginal;
- candidate mean remains exactly N*q_i;
- mean candidate CRPS must be <= baseline in every season;
- no season may have statistically supported marginal CRPS harm.

## Exact invariants

Require:
- team opportunity total conserved exactly;
- candidate expected count for every category = N*q_i analytically;
- max expected-count gap <=1e-12;
- source max week < target week;
- canonical q derived without target-game outcomes;
- no identity remapping chosen from results;
- sportsbook inputs =0;
- OddsAPI calls =0;
- fitted coefficients =0;
- threshold searches =0;
- 2026 outcomes =0;
- production changed =false.

## PASS gate

A cell receives:
`JOINT_ALLOCATION_DIRICHLET_HISTORICAL_PASS_FREEZE_FORWARD_SHADOW`

only if ALL pass in 2023, 2024, and 2025:
1. support floor;
2. mean joint log-score improvement >0;
3. game-cluster 95% lower bound >0;
4. team-cluster 95% lower bound >0;
5. marginal CRPS non-worse;
6. no supported marginal harm;
7. mean identity/invariants pass.

Otherwise:
`JOINT_ALLOCATION_DIRICHLET_HISTORICAL_FAIL_CLOSED`.

No pooled three-season result may rescue a failed season.
No RB result may rescue WR or vice versa.

## No-rescue rule

After scoring do not:
- tune rho;
- tune room aggregation;
- alter complement construction;
- choose volatility buckets;
- change the history window;
- select player subgroups;
- add RB-target or TE-target cells;
- replace canonical q with realized prior mean.

A historical PASS authorizes only a separately frozen 2026 prospective shadow.
