# Rush-Attempt Zero-MC Allocation Lineage Audit V1 — Frozen Plan

Date: 2026-09-26
Status: FROZEN DIAGNOSTIC PLAN — NO REPAIR AUTHORIZED

## Question

A historical systems-integrity diagnostic found 7,137 `rush_att` rows where the calibrated ensemble mean is nonzero while canonical MC is exactly zero. The effect is especially harmful for QB and directionally harmful for RB, while blanket ensemble injection worsens WR/TE. Therefore no position carveout or generic repair is authorized.

The next legitimate question is purely architectural:

**Where, exactly, does a positive rushing-opportunity signal first become zero on the canonical Monte Carlo path?**

## Frozen lineage

For each historical player-week, trace these stages without changing production logic:

1. `rush_att` output-row `rules_rush_share`;
2. the exact player row retained by `simulation_v2.simulate` after its one-row-per-player collapse;
3. that retained row's pre-top-five rushing share and stable team rank;
4. literal top-five membership and post-selector rushing share;
5. final multinomial player probability after the 0.95 player-mass cap/residual bucket;
6. expected carries from final probability;
7. realized multinomial mean carries;
8. keyed `SimulationResult` lookup mean;
9. canonical `mc_proj`.

The audit must identify the **first zero stage** for every zero-MC row.

## Cohort

Leakage-safe historical pregame reconstruction:
- seasons 2024 and 2025;
- prior seasons 2023 and 2024 respectively;
- all available regular-season weeks;
- same historical context/build path used by the current component backtest;
- no sportsbook data;
- no Week-3 target-game outcomes.

Primary diagnostic subsets:
- all `rush_att` rows with positive reported `rules_rush_share` and `mc_proj == 0`;
- QB and RB/FB subsets;
- all positions as a control.

## Frozen classifications

Each zero-MC row is classified once, in this order:

- `OUTPUT_SHARE_NONPOSITIVE`
- `SELECTED_ROW_MISSING`
- `SELECTED_ROW_SHARE_NONPOSITIVE`
- `TOP5_EXCLUDED`
- `FINAL_PROBABILITY_ZERO`
- `REALIZED_MC_ZERO_WITH_POSITIVE_PROBABILITY`
- `LOOKUP_ZERO_AFTER_POSITIVE_REALIZED`
- `CANONICAL_ZERO_AFTER_POSITIVE_LOOKUP`
- `UNEXPLAINED`

No thresholds are fit. Top-five is the production constant, not a candidate.

## Required checks

- canonical `mc_proj` must equal same-seed keyed lookup mean within 1e-9;
- keyed lookup must equal realized carry-array mean within 1e-9;
- top-five membership must reproduce `simulation_v2._top_n_shares(..., 5)` exactly;
- final probabilities must reproduce `_allocate_counts` probability normalization exactly;
- report any difference between the `rush_att` output row's share and the simulator-selected row's share;
- report simulator-selected market distribution;
- report zero-MC first-stage counts by season and position;
- specifically materialize the Kirk Cousins 2024 Week 1 ATL example if present.

## Governance

Diagnostic only:
- candidate variants = 0;
- fitted parameters = 0;
- production mutations = 0;
- no sportsbook upstream;
- no QB/RB carveout;
- no alternate top-N;
- no threshold search;
- no repair design until the lineage result is known.

A mechanical failure may be repaired only to execute this exact frozen audit. The science/query itself may not be changed after results are observed.
