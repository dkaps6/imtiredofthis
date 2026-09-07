# QB-PD3 Internal Disagreement Reliability — Frozen Plan

## Status
Frozen before results. Diagnostic only. Production unchanged.

## Lineage
- Production QB mean: `QB_PASS_SYNTHESIS_V1` (M89/M90).
- Week-1 pathology audit: `QB_W1_INDIVIDUAL_PROJECTION_PATHOLOGY`, canonical run `34064097525`.
- Player-error persistence: `QB_PD2_PLAYER_ERROR_PERSISTENCE`, canonical run `34064528914`, disposition `NO_ACTIONABLE_QB_PLAYER_ERROR_PERSISTENCE`.
- Historical evidence: exact M89 2024-2025 validation trace, 884 rows.
- Sportsbook fields in that historical trace are explicitly ignored; only football-side components may enter this audit.

## Hypothesis
Large Week-1 discrepancies often coincide with disagreement among the football model's own MC/ML/state components and/or very large synthesis corrections. Those are observable pregame model-state properties. They may identify historical games in which the promoted synthesis layer is less reliable even though recent player-specific error history itself is not persistent.

This is a reliability diagnostic, not a market correction and not a new mean model.

## Frozen states
Using only columns available before the game:
1. `COMPONENT_RANGE_40`: `component_range >= 40` yards.
2. `SYNTH_MOVE_30`: `abs(football_residual_correction) >= 30` yards.
3. `SYNTH_CAP_45`: `abs(football_residual_correction) >= 44.999` yards.
4. `MULTI_FLAG_2`: at least two of the three states above are true.

These thresholds are inherited from the already-frozen Week-1 pathology audit; they are not chosen from this result.

## Outcomes
For each state vs its complement, pooled and separately for 2024 and 2025, report synthesis/base MAE, RMSE and bias; synthesis-minus-base absolute-error delta; median/p75/p90 synthesis absolute error; 30+/50+/75+/100+ synthesis miss rates; and sample counts. Also report continuous Spearman relationships for component range and absolute synthesis correction against absolute synthesis error and synthesis-minus-base absolute error.

## Frozen actionable-state gate
A predeclared state is actionable only if all are true: pooled N >= 60; N >= 20 in both seasons; pooled synthesis MAE is >=5.0 yards worse in-state than out-of-state; that penalty is positive in both seasons; pooled synthesis-minus-base absolute-error delta is >=+2.0 yards in-state; that delta is nonnegative in both seasons; in-state 75+ miss rate is >=3 percentage points higher than out-of-state; no sportsbook field is used and no leakage is present. `SYNTH_CAP_45` can only be descriptive if it misses N.

## Disposition
If any state passes: `QB_INTERNAL_RELIABILITY_STATE_SUPPORTED`, authorizing a separate frozen synthesis-trust candidate only. If none pass: `NO_ACTIONABLE_QB_INTERNAL_RELIABILITY_STATE`. No nearby threshold search.

## Production
No production change is authorized by this diagnostic.
