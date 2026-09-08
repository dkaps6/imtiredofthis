# RB R18 Canonical MC Tail Adapter Parity V1 — Frozen Plan

Date frozen: 2026-09-08
Parents:
- `RB_R16_UPSIDE_TAIL_SIGNAL_SUPPORTED_DIAGNOSTIC_ONLY`
- `RB_R17_DISTRIBUTION_SIGNAL_SUPPORTED_RESEARCH_ONLY`
Parent runs:
- R16 `34286363931`
- R17 `34286785433`
Status: PLAN FROZEN BEFORE EXECUTION

## Purpose

R16 showed strict-prior upside-tail discrimination. R17 showed that conditioning an empirical residual distribution on those tail probabilities can modestly improve probabilistic receiving-yard forecasts while preserving the mean exactly.

R18 is an integration/parity experiment, not a new football hypothesis and not a production promotion.

Research question:

> Can the R17 tail-shape mechanism be layered onto the canonical `scripts/simulation_v2.py` Monte Carlo output without changing the canonical RB receiving mean, finite opportunity allocation, non-RB outcomes, or other RB component markets, while retaining the canonical simulation dependence structure as much as possible?

## Frozen architecture

Do NOT modify `scripts/simulation_v2.py` in R18.

Implement a separate research adapter that consumes a completed canonical `SimulationResult` and returns a shadow `SimulationResult`.

The adapter may change only:
- RB `rec_yards`
- RB `rush_rec_yards`, and only by exactly the same draw-level delta applied to RB `rec_yards`

The adapter must leave exactly unchanged:
- every non-RB simulation array for every market
- RB `receptions`
- RB `rush_att`
- RB `rush_yards`
- RB `anytime_td`
- QB passing outcomes
- all target/carry allocation logic and any allocation trace produced before the adapter

## Rank-preserving tail reshape

For each eligible RB:

1. Read the canonical RB receiving-yard draw vector `x`.
2. Compute its canonical sample mean `mu`.
3. Generate an R17 nested empirical tail-mixture target draw vector using supplied strict-prior `p30` and `p50` and historical residual pools `<30`, `30-49`, `>=50`.
4. Floor target draws at zero and rescale them so their sample mean equals `mu` exactly.
5. Sort the target draws.
6. Assign the sorted target draws to the rank order of the canonical draw vector.

This is empirical-copula / rank-preserving quantile mapping. The intent is to change the RB receiving-yard marginal tail shape while preserving the canonical ordering induced by target counts, shared pass-efficiency shocks, pace/game-state shocks, and other simulator dependence.

No p30/p50 calibration or residual-pool boundary may be tuned in R18. R18 consumes R16/R17 information as frozen parents.

## Frozen test surfaces

### A. Canonical simulation parity fixture

Run `scripts/simulation_v2.py` on a deterministic fixture containing at minimum:
- 2 teams in one game
- at least 2 RBs
- at least 2 WRs
- at least 1 TE
- at least 1 QB
- multiple markets per player where supported

Use a fixed simulation seed and at least 10,000 iterations so parity and distribution behavior are measurable.

Run canonical simulation once. Apply R18 shadow adapter to a deep copy of its output.

### B. Determinism replay

Run the adapter twice with identical inputs, risk probabilities, residual pools, and adapter seed. Outputs must be bitwise/array-equal.

### C. R17 historical distribution replay

Using the immutable R17 fold inputs and OOS p30/p50 probabilities, run the same R18 tail-draw generator (before rank assignment) over the 2024/2025 historical player-games and confirm it remains statistically consistent with the frozen R17 candidate. This is a code-path parity check, not a new tuning surface.

## Frozen gates

R18 passes only if ALL gates are true.

### Mechanical / conservation gates

1. `canonical_mean_parity`: for every adapted RB, absolute difference between candidate and canonical `rec_yards` sample mean <= 1e-8 yards.
2. `non_rb_exact`: all non-RB arrays across all markets are exactly equal before vs after adapter.
3. `rb_component_exact`: for RBs, `receptions`, `rush_att`, `rush_yards`, `anytime_td`, and any other non-receiving component arrays are exactly equal.
4. `rush_rec_identity`: for every adapted RB and every draw, candidate `rush_rec_yards == canonical_rush_yards + candidate_rec_yards` within 1e-10.
5. `allocation_trace_exact`: any canonical target/carry allocation trace supplied to the adapter is unchanged exactly.
6. `nonnegative_rec_yards`: every adapted receiving-yard draw >= 0.
7. `finite_draws`: every adapted draw is finite.

### Dependence / determinism gates

8. `rank_preservation`: Spearman rank correlation between canonical and adapted RB `rec_yards` draws >= 0.9999 for every adapted RB with nonconstant canonical draws.
9. `deterministic_replay`: identical adapter inputs/seed produce exactly equal arrays.

### R17 code-path replay guards

10. `r17_combined_q90_guard`: historical R18-generator q90 pinball <= frozen R17 comparator q90 pinball 3.665298977473567.
11. `r17_combined_q95_guard`: historical R18-generator q95 pinball <= frozen R17 comparator q95 pinball 2.4849645782257346.
12. `r17_combined_crps_guard`: historical R18-generator CRPS <= frozen R17 comparator CRPS * 1.005. This is a <=0.5% degradation guard because the purpose of R18 is integration parity, not retuning R17.
13. `r17_cat30_brier_guard`: historical R18-generator combined Brier30 <= frozen R17 comparator 0.05748601013634733.
14. `r17_cat50_fragility_guard`: historical R18-generator Brier50 must be <= frozen comparator in each fold plus 0.00010. This explicit fold guard addresses R17's fragile 50+ result and was frozen before R18 execution.

### Governance gates

15. `canonical_simulation_unmodified`: R18 must not modify `scripts/simulation_v2.py`.
16. `sportsbook_zero`: sportsbook inputs added = 0.
17. `production_parameters_zero`: production parameters changed = 0.

## Interpretation rules

PASS means the supported R17 distribution mechanism has a mechanically safe shadow adapter compatible with the canonical Monte Carlo output. PASS does NOT mean the adapter is live or promoted.

After PASS, the next separately frozen step should build/refit the strict-prior tail scorer and residual-pool artifact needed to score 2026 player-games prospectively, then run a full-slate shadow/parity gate before any promotion.

FAIL means the R17 research distribution cannot yet be safely composed with the canonical simulator. Diagnose the failed mechanical or distribution gate before changing the adapter architecture.
