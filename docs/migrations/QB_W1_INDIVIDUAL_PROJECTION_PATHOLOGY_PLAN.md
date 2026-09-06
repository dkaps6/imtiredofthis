# QB Week 1 Individual Projection Pathology Audit — Frozen Plan

## Purpose

Audit the promoted M89/M90 Week-1 passing-yards projections at the **individual QB level**, not merely by slate-wide MAE or average market gap.

This is a production-diagnostic audit. It does not reopen broad QB mean feature hunting and it does not tune projections toward sportsbook lines.

## Frozen parents and evidence

- Production/audit parent: `94752b4e946ae9325237f8702604d95b46d84fe9`
- Week-1 no-odds audit run: `34059395746`
- Week-1 artifact: `9997012665` (`qb-2026-w1-no-odds-market-audit`)
- Historical M89 validation run: `33331073376`
- Historical M89 artifact: `9737913528` (`m89-qb-data-integrity-casebook-synthesis`)
- Promoted architecture remains `QB_PASS_SYNTHESIS_V1` / M89 confirmed by M90.

## Individual-history scorecard

Using the untouched M89 2024-2025 validation trace, calculate for each QB:

- games
- base MAE
- synthesis MAE
- synthesis signed bias
- median absolute error
- 90th percentile absolute error
- 30+ / 50+ / 75+ yard miss rates
- synthesis MAE change versus the corrected base

These are diagnostic only. A player's full-sample 2024-2025 scorecard is **not** allowed as a feature in the 2026 projection.

Any later player-specific calibration test must rebuild these quantities walk-forward using only games strictly before each target game.

## Week-1 internal projection audit

For every posted Week-1 QB row, retain:

- ensemble projection
- MC / ML / State projections
- promoted synthesis projection
- synthesis correction
- component range
- predicted attempts
- predicted YPA
- model SD
- sportsbook line and model-minus-line, downstream only

Predeclared internal diagnostic flags:

- `SYNTHESIS_CAP`: absolute synthesis correction >= 44.999 yards
- `LARGE_SYNTHESIS_MOVE`: absolute synthesis correction >= 30 yards
- `LARGE_COMPONENT_DISAGREEMENT`: max(MC, ML, State) - min(MC, ML, State) >= 40 yards
- `PLAYER_HISTORY_HIGH_MAE`: historical synthesis MAE >= 50 yards with at least 8 historical games
- `PLAYER_HISTORY_DIRECTIONAL_BIAS`: absolute historical synthesis bias >= 15 yards with at least 8 historical games
- `PLAYER_HISTORY_SYNTHESIS_WORSENED`: historical synthesis MAE > historical base MAE with at least 8 historical games

Sportsbook discrepancy thresholds are reported descriptively but are **not** scientific gates and cannot cause a projection change.

## Frozen interpretation

The audit will classify each QB by the internal mechanisms above and quantify how much of the current market disagreement is associated with:

1. large residual-synthesis movement;
2. disagreement among the production components;
3. historically high player-specific error/volatility;
4. persistent historical directional bias.

No production change is authorized by this audit alone.

A later correction test is authorized only if this audit identifies a coherent, pregame-available, historically testable mechanism. Any such test must be separately frozen and walk-forward. No sportsbook line may enter that correction.
