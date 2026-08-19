# Walk-forward backtest architecture

## Non-negotiable time boundary

For target season `S`, week `W`, model features may use only rows where:

- `season < S`, or
- `season == S and week < W`.

Target-week outcomes and future weeks are comparison labels only. They must never be used to build PlayerForm, TeamForm, ML features, state transitions, rules context, or the projected player universe.

## Pregame player universe

The historical player universe must be supplied from a pregame source such as a historical roster/depth snapshot. It must not be inferred from target-week box-score participation. Using target-week participants would leak who actually played.

`build_historical_player_inputs()` therefore accepts an explicit `pregame_universe` and ignores target-week result rows when constructing features.

## Week 1

For Week 1, current-season evidence is empty by design. Player and team baselines fall back to the prior season. This mirrors the production preseason architecture.

## Optional enrichments

Historical injuries, weather, coverage and WR/CB assignments require timestamp-safe snapshots. A weekly result artifact labeled as the target week is not automatically safe. Backtest Migration 1 hard-fails when week-tagged optional data is at or after the target cutoff.

## Model reuse

Backtesting must use the same canonical Bayesian, rules, simulation, ML, state and ensemble modules as production. Backtest-specific code is responsible for reconstructing historical inputs, not for inventing a parallel prediction model.

Backtest Migration 1 establishes the historical context factory and cutoff guards. Migration 2 will connect these explicit historical contexts to the production component runners and generate out-of-sample component predictions.
