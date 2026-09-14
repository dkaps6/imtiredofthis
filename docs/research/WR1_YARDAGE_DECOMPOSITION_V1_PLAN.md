# WR1 YARDAGE DECOMPOSITION V1 — FROZEN DIAGNOSTIC PLAN

Status: **FROZEN BEFORE ANY RESEARCH CANDIDATE OR PRODUCTION CHANGE**

Branch: `research-wr1-yardage-decomposition-v1`
Parent: PR #600 authority-exact head `1167f9fdadde452deb84d3891097865da2f163d5`

## Why this study exists

PR #600 repaired the historical Vegas benchmark so football authority must reproduce before sportsbook lines are joined. The corrected benchmark exposed a specific WR problem: receiving-yard directional performance is much weaker than WR receptions, especially among top WRs.

This study is diagnostic only. It may identify the failed football layer; it cannot promote or tune a fix.

## Frozen cohort

1. Historical season: **2024 only** for the main Vegas-linked WR analysis, because WR-R15's scientific confirmation contract authorizes 2023-2024 and the real historical prop archive used here covers 2024-2025. We do not invent 2025 WR-R15 scientific authority.
2. WR universe: exact 2024 identities from run `34238301577`, artifact `10061328722`.
3. Model WR1 is **`wr_rank == 1` from the WR-R15/M38 authority artifact**, exactly one WR1 anchor per team-game. Ourlads/lane `role == WR1` is reported only as a secondary label audit and never defines the scientific WR1 cohort.
4. WR2+ (`wr_rank > 1`) is the control population.
5. Sportsbook rows come only from PR #600 authority-exact run `34843204550`, artifact `10346639168`, after football parity PASS.

## Frozen benchmark arms

- `AUTHORITY_OOS`: exact promoting OOS projection stored by the WR authority artifact and carried into #600.
- `CURRENT_PRODUCTION_ORDER`: audited #549 production-order replay with current repository receiving ensemble weights, as emitted by #600 on the same exact identities.

Both must be reported. No arm may replace or silently broaden the other.

## Frozen decomposition

For every line-matched model-WR1 receiving-yard row:

- projected targets: exact WR-R15/M38 OOS `pred_targets` (WR1 is the immutable M38 anchor)
- actual targets: authority `actual_targets`
- projected receptions: current production-order receptions projection on the same identity
- actual receptions: actual receptions from the exact #600 projection/grade row
- projected receiving yards: current production-order receiving-yard projection
- actual receiving yards: exact outcome from #600

Define effective factors:

- projected catch rate = projected receptions / projected targets
- actual catch rate = actual receptions / actual targets
- projected YPR = projected yards / projected receptions
- actual YPR = actual yards / actual receptions
- projected YPT = projected yards / projected targets
- actual YPT = actual yards / actual targets

The effective-factor identities are diagnostic bookkeeping, not a claim that the final ensemble is literally parameterized as those three independent components.

### Error attribution

1. **3-factor Shapley decomposition** of signed receiving-yard error across TARGETS, CATCH RATE, and YPR. All six replacement orders receive equal weight. Contributions must sum to projected yards minus actual yards within floating-point tolerance.
2. **2-factor Shapley decomposition** across TARGETS and YPT for all rows where YPT is defined; this is the robust fallback for zero-reception rows.
3. Report absolute-contribution share, signed contribution, and dominant mechanism for the full cohort, Vegas wins, Vegas losses, OVER calls, UNDER calls, UNDER losses, and OVER losses.

## Frozen tail / failure slices

Report without tuning:

- actual receiving yards >= 100
- absolute model receiving-yard error >= 30
- actual YPT quartiles, using cohort-wide quartile cutpoints
- projected YPT quartiles
- target-error quartiles
- production projection < Vegas line (UNDER) vs > line (OVER)
- authority OOS side vs current-production side flips

## WR1-vs-WR2+ control

Report raw WR1 vs WR2+ results and a matched control comparison.

Matching rule is frozen as one-nearest-neighbor **with replacement**, separately for each WR1 row, among WR2+ rows in the same week bucket (`W1-4`, `W5-9`, `W10-13`, `W14-18`). Distance uses z-scored:

- projected targets
- Vegas receiving-yard line

No outcome field, actual statistic, directional result, sportsbook price, or model error may enter matching.

## Production-order drift audit

On identical WR1 rows, compare `AUTHORITY_OOS` vs `CURRENT_PRODUCTION_ORDER`:

- count directional side flips
- OVER->UNDER and UNDER->OVER counts
- net wins gained/lost on flipped rows
- projection delta on flipped and non-flipped rows
- football MAE / bias / p90 AE by arm

This is necessary because a better football MAE can still change line direction unfavorably.

## Context fields

Strictly pregame context from the frozen component trace may be used **descriptively only** in V1, including pass volume, pressure/context-availability flags, and existing coverage/matchup availability flags. V1 may not fit a corrective model from those fields.

## Stop rules

- No sportsbook line/odds as upstream football features.
- No M38 multiplier retuning.
- No WR-R15 refit or WR1-anchor mutation.
- Do not recycle failed NGS R11, generic vacancy mass, or snap-depth-only candidates.
- Do not infer WR-CB assignments that are not present in frozen source data.
- No candidate coefficients, thresholds, gates, or subgroup routers may be chosen from V1 output.
- A follow-up candidate requires a new frozen plan before evaluation.

## Required outputs

- `wr1_cohort_audit.csv`
- `wr1_vs_wr2plus_scoreboard.csv`
- `wr1_production_order_drift.csv`
- `wr1_error_decomposition_rows.csv`
- `wr1_error_decomposition_summary.csv`
- `wr1_tail_failure_summary.csv`
- `wr1_matched_control_summary.csv`
- `WR1_YARDAGE_DECOMPOSITION_V1_RESULT.md`

## Disposition vocabulary

V1 may conclude only one of:

- `TARGET_VOLUME_DOMINANT`
- `CATCH_TRANSLATION_DOMINANT`
- `YPR_EFFICIENCY_DOMINANT`
- `MIXED_TARGET_AND_EFFICIENCY`
- `PRODUCTION_ORDER_DRIFT_DOMINANT`
- `NO_SINGLE_MECHANISM_DOMINANT`

These are diagnostic labels, **not promotion decisions**.
