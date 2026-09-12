# Historical Market Vegas Benchmark V1 — result record

**Disposition:** `BASE_MC_ENGINE_HISTORICAL_MARKET_BENCHMARK_NO_EDGE_NAIVE_SIDE_SELECTION`

## What this measures

Whether the current codebase's shared Monte Carlo opportunity/simulation
engine — the common foundation every position's projection is built on —
beats real 2024-2025 sportsbook lines, using the free `gcampb41/nfl_data-`
Action Network-derived archive (repaired reconciliation; see
`scripts/backtest/prepare_free_qb_prop_archive.py` and the generalized
`scripts/backtest/prepare_free_market_prop_archive_v1.py`).

Betting rule: naive, no edge threshold. Every row where the model's mean
disagrees with the book line is treated as a bet on that side. This is
deliberately the simplest possible rule; it is not the master workbook's
PLAY/LEAN/PASS edge-threshold logic.

## What this does NOT measure

This grades the base MC component only. It does **not** include:
- the ML/State ensemble blend (`data/model_ensemble_weights.csv`);
- the QB `M89`/`M90` `QB_PASS_SYNTHESIS_V1` residual correction, which
  internal confirmation work showed meaningfully reduces QB pass-yards MAE
  (60.6 -> 56.6 in the M90 sealed 2023 confirmation);
- RB `P3`/`R26`, which are qualified for the 2026 Week-1 route only and are
  out of scope for a 2024-2025 historical grade;
- any edge-threshold or EV-based bet selection.

This result is therefore a **lower bound**, not a verdict on the full
production stack.

## Result (see `historical_market_vegas_benchmark_summary.csv`)

Across every market (pass_yards, rush_yards, rec_yards, receptions,
rush_rec_yards) and both seasons, the base engine underperformed the
market: Vegas MAE was lower than model MAE in every market, the model was
closer than Vegas only 40-46% of games, and naive per-disagreement betting
returned roughly -1% to -6% ROI per unit (all-markets: 2024 -4.9%, 2025
-1.0%). This is consistent with a model near breakeven-to-slightly-behind
against `-110`-style vig with no edge filter, not a model with a
demonstrated market edge.

## Reproduction

```bash
python scripts/backtest/build_historical_inputs.py --season 2025 --prior-season 2024 --weeks 1-18 --out-dir <dir>
python scripts/backtest/build_historical_inputs.py --season 2024 --prior-season 2023 --weeks 1-18 --out-dir <dir>
python scripts/backtest/historical_player_logs.py --seasons 2023,2024,2025 --schedule <combined schedule> --out <player_logs.csv>
python scripts/backtest/build_market_vegas_benchmark_projections_v1.py --season <season> --prior-season <prior> --weeks 1-18 --player-logs <player_logs.csv> --team-weekly <...> --schedule <...> --universe-dir <...> --out <proj.csv>
python scripts/backtest/prepare_free_market_prop_archive_v1.py --projection-file <proj_2024.csv> --projection-file <proj_2025.csv> --out-dir <dir>
python scripts/backtest/grade_historical_market_vegas_benchmark_v1.py --projection-file <proj_2024.csv> --projection-file <proj_2025.csv> --props <historical_market_props.csv> --out-dir <dir>
```

## Anti-reinvention rule

Do not rerun this exact naive-side, base-engine-only comparison and treat a
different result as evidence either way without changing what's actually
being tested (full stack vs. base engine, edge-threshold vs. naive
selection). The next legitimate step is grading the **full** production
stack (ensemble + QB synthesis) with an **edge-threshold** selection rule,
not repeating this exact configuration.
