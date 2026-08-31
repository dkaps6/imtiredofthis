# Migration 91 — RB Temporal Baseline

## Purpose

M91 expands the current canonical rushing backtest from one scored target season to a two-season prospective baseline before any new RB tuning.

The existing 2025 walk-forward already uses 2024 as prior history. M91 adds 2024 as an independently scored out-of-sample target season using only 2023-and-earlier information at each prediction cutoff, then reproduces 2025 using 2024-and-earlier information. This is the same temporal principle used in the QB research: prior-season data may inform a future game, but target-game results never enter the pregame projection.

M91 is diagnostic only. It does not change production projections, ensemble weights, simulation rules, sportsbook inputs, or the promoted top-5 rushing allocation architecture.

## Target markets

Primary RB markets:

- rushing attempts
- rushing yards
- rushing + receiving yards

The baseline also records the workload context needed to isolate the specific failure mode already visible in 2025: true bell-cow / workload-spike games being projected too conservatively.

## Temporal design

### 2024 target season

- prior season: 2023
- target weeks: 1-18
- historical schedule/team/player/injury/weather context is rebuilt at the explicit target-season cutoff
- each week is scored prospectively

### 2025 target season

- prior season: 2024
- target weeks: 1-18
- same canonical code path and feature semantics
- each week is scored prospectively

The 2024 and 2025 prediction files are then combined only for evaluation. They are never pooled in a way that lets later-season results alter earlier predictions.

## RB-specific diagnostics

M91 reports each component (MC, ML, State) by season and combined for:

- all RB rows
- actual 10+ carry games
- actual 15+ carry games
- actual 20+ carry games
- actual 25+ carry games
- actual bell-cow games with 15+ carries and at least 55% of team rushes
- actual bell-cow games with 15+ carries and at least 60% of team rushes
- actual RB1 rows
- projected RB1 rows

Metrics include MAE, median absolute error, 90th-percentile absolute error, RMSE, bias, correlation, underprojection rate, and large-under/large-over miss rates.

M91 also measures:

- projected vs actual share of total team rushing attempts
- projected vs actual share of the RB-only rushing pool
- RB1 identification accuracy by team-week
- projected vs actual leader carry volume

These diagnostics separate three different questions that were previously blended together:

1. Did the model identify the correct lead back?
2. Did it allocate the backfield share correctly?
3. Did it project enough total carries when that back became a true workload spike / bell-cow?

## Anti-leakage / anti-overfit rules

- No sportsbook or player-prop line enters the football projection.
- No target-week result enters the pregame context.
- No new coefficients or thresholds are tuned in M91.
- M30 top-5 rushing allocation remains frozen for this baseline.
- 2024 and 2025 must be reported separately as well as combined; a pooled improvement may not hide a seasonal regression.

## Next research phase

After M91 establishes the stable baseline, the RB work should split into three explicit problem families rather than changing everything at once:

1. **Workload concentration / bell-cow attempts** — depth-chart ownership, competing-back injuries, snap/carry concentration, team committee behavior, and total rush-volume suppression.
2. **Rushing yards conditional on opportunity** — efficiency, defensive front/box context, explosive-rush potential, offensive-line context, and workload-dependent YPC behavior.
3. **Rushing + receiving yards** — receiving role, route/target share, checkdown environment, and combined touch/opportunity projection.

Any candidate architecture should be frozen before its final temporal confirmation. Because 2025 has already been inspected extensively during earlier rushing work, a later promotion candidate should also receive an earlier temporal rotation (for example 2023 scored from 2022 prior history) rather than treating 2025 as pristine unseen confirmation data.

## Future Vegas benchmark

Sportsbook lines remain downstream only. Once a trustworthy historical pregame prop-line source is available, the correct benchmark is a paired same-player/game/market comparison:

- sportsbook line vs actual
- model pregame projection vs actual
- absolute-error difference on the exact common cohort
- model win / tie / loss rate against the line
- bias and RMSE for both
- paired bootstrap probability that the model is closer to the actual result

This benchmark measures whether the football projection is genuinely closer to what happened than the market line without contaminating the football model with sportsbook information.
