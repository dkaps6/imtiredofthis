# RB-PD2 Yard-Difficulty MC-Width V1 — Pre-Result Amendment 3

**FROZEN BEFORE ANY CANDIDATE RESULT. RESEARCH ONLY.**

This supplements the original frozen plan (`222c78cb0314e79e0f9ce6e147ab127a30f68f10`) and prior prospective amendments. No candidate output was used to choose these rules.

## A. Dependence-aware CRPS bootstrap

Keep the existing player-clustered Gate-C bootstrap unchanged and add a second required crossed player × game bootstrap.

`game_key = (season, week, min(team, opponent), max(team, opponent))`.

For each replicate, independently sample unique players with replacement and unique games with replacement. Convert both samples to multiplicities. Each observed row gets weight `player_multiplicity * game_multiplicity`. Compute the weighted paired mean of `candidate_crps - baseline_crps`. Redraw zero-weight replicates.

Freeze 10,000 valid replicates, seed `42027`, and require `P(weighted mean delta < 0) >= 0.95`.

Final Gate C requires BOTH the original player-clustered probability `>=0.95` and this crossed player × game probability `>=0.95`. This is additive and cannot rescue failure of the original gate.

## B. Interval and tail boundaries

Freeze the evaluator's existing conventions:

- quantiles: `np.quantile(..., method="linear")`;
- 80% interval: q10 to q90, endpoints inclusive;
- 90% interval: q05 to q95, endpoints inclusive;
- tail probability at 50/75/100 yards: `mean(draw >= threshold)`;
- realized tail label: `actual >= threshold`.

## C. Exact high-difficulty quartile

Freeze the evaluator's existing global-Q75 rule on the primary eligible population:

- threshold = linear-interpolated 75th percentile of primary-row `difficulty_score`;
- include all rows with `difficulty_score >= threshold`;
- ties at the threshold are all included, so the slice may exceed exactly 25%;
- use the same pooled threshold for the overall high-difficulty slice and each season's Gate-G subset.

Do not replace this with exact-top-N selection or a fixed `difficulty_score >= 0.75` rule after output.

## D. Provenance clarifications

- `EXPECTED_PARENT_ROWS=5607` and `EXPECTED_PARENT_SCOREABLE=4652` are fail-closed integrity expectations from the canonical PR #556 result, not tuning targets.
- The `0.30` width coefficient and `0.50` onset come from the prior WR-R3 Lane-B uncertainty precedent in `docs/migrations/WR_R3_COMBINED_CALIBRATION_PLAN.md` / `scripts/backtest/evaluate_wr_r3_combined_calibration.py`; they were not fit to this RB candidate.
- Structural assertions that no sportsbook/production/carry/YPC/allocation change occurs remain code-contract checks, not result-driven claims.

## Disposition

A positive qualification now requires every original hard gate plus the new crossed player × game bootstrap gate. Otherwise the original failure dispositions remain unchanged. Qualification is still research-only and requires separate forward/shadow confirmation before any production change.
