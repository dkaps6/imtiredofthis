# WR Phase 4C Stage 1 — Pass-Volume Source Lineage Audit

**STATUS: SOURCE-ONLY DUE DILIGENCE. NO STAGE-1 FIT/OUTCOME. NO PRODUCTION CHANGE.**

This record closes Claude's conditional Stage-1 plan-review ask in Issue #535 comment `5687745803`: confirm that the historical `plays_est` / `dropback_rate` source used by the football-only script submodel has zero Vegas/market lineage.

## Exact source chain

The PR #558 workflow builds `team_weekly_history.csv` through:

1. `.github/workflows/research-market-implied-game-script-v1.yml`
2. `scripts/backtest/build_historical_inputs.py`
3. `scripts/backtest/historical_inputs.py::build_all_historical_inputs()`
4. `scripts/backtest/historical_inputs.py::build_team_weekly_from_pbp()`
5. `scripts/utils/pbp.py::get_pbp()` -> `nflreadpy.load_pbp()` with `nfl_data_py` fallback.

Inside `build_team_weekly_from_pbp()`:

- `off_play = (qb_dropback == 1 OR rush_attempt == 1)`
- `plays_est = len(g)` where `g` is the team's completed-game offensive-play frame
- `dropback_rate = mean(qb_dropback)` over those offensive plays

The calculation uses completed nflverse PBP event fields. It does not read schedule `spread_line`, `total_line`, moneylines, implied totals, player-prop markets, or any sportsbook-derived field.

The schedule builder is a separate path. In the PR #558 workflow, market schedule columns are loaded later by the diagnostic script and joined as candidate regressors *against* the already-built PBP history; they are not inputs to `plays_est` or `dropback_rate`.

## Stage-1 implication

The Stage-1 script target

`realized_pass_volume = plays_est * dropback_rate`

and its strictly-prior rolling football inputs derived from those two fields have **zero market/Vegas lineage**.

Therefore Arm C's `pred_realized_pass_volume` is market-free upstream. Arm B owns the frozen market information (`market_total`, `<38` indicator), so `C > B` retains the intended incremental interpretation: football-only predicted pass-volume state adds information beyond the market environment.

## Boundaries

- This audit does not score Stage 1.
- It does not alter the frozen target, features, model, gates, or thresholds.
- The optional WR-room magnitude-floor suggestion from Claude is not adopted: conditions 3/4 already impose the materiality floor at the team-pool level, while condition 6 remains a transport/coherence gate.
- The eventual Stage-1 result record must repeat this lineage statement and contextualize the frozen `0.10` target/team-game floor against the Phase-4B Layer-2 reported baseline MAE reference (~`4.520381`, about `2.21%`).
