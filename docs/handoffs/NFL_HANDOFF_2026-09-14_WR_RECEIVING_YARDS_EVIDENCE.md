# WR RECEIVING-YARDS EVIDENCE LEDGER — 2026-09-14

Companion to `docs/handoffs/NFL_HANDOFF_2026-09-14_WR_RECEIVING_YARDS_CURRENT.md`.

## Exact authority artifacts
- WR-R15: run `34238301577`, artifact `10061328722`, digest `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`.
- WR1 decomposition: run `34858963515`, artifact `10353787250`, digest `sha256:3e6fb21956e0f2681a732379412b6eff0c3e17dd561d7358e8995e279fb42f5d`.
- Authority-exact benchmark PR #600: run `34843204550`, artifact `10346639168`, digest `sha256:e0041386c7f6600c6e8a8781d0d91d7603c4d1fde75d9aa3d0f0055aaa746225`.
- Claude shared-tail trace: run `34865304105`, artifact `10356647408`, digest `sha256:17aca6a561543ad3240cacf4bd7639d277e77cd138196d456585ef131365ffd1`.
- Claude shared-tail evaluation: run `34873351292`, artifact `10358739434`, digest `sha256:c366c7d419b1d3eb266538776b1c5ac39d9e0357d9e244885a7899d2bee5a158`.

## WR1 decomposition summary
On the exact 494 matched true-WR1 rows:
- current production order: 237 correct / 257 incorrect, `47.9757%`
- mean bias `-9.2562 yd`
- MAE `29.1775`
- target/catch/YPR absolute-error shares `40.61% / 27.34% / 32.04%`
- collapsed target-vs-YPT `50.25% / 49.75%`.

On incorrect-direction rows:
- target/catch/YPR `42.60% / 25.26% / 32.14%`
- target-vs-YPT `51.50% / 48.50%`.

Actual 100+ WR1 games:
- n=78
- only `25.64%` correct direction
- bias and MAE both `67.43 yd` low.

Actual YPT top quartile:
- n=124
- `31.45%` correct direction
- bias `-39.17 yd`
- MAE `42.70`.

WR1 UNDER misses:
- n=170
- average underprojection `40.98 yd`
- 34.12% became 100+ receiving-yard games.

## Identity lesson
Do not use literal depth-chart `role == WR1` as the scientific WR1 identity. The decomposition artifact found 1073 literal-WR1 rows vs 516 true model WR1 rows, with 403 team-games having at least two literal WR1 labels and a maximum of three. Use the M38/R15 hierarchy (`wr_rank == 1`).

## Production-order drift diagnostic
On the same 494 true-WR1 rows:
- authority OOS: 253-241, MAE `29.123760`
- current production order: 237-257, MAE `29.177472`
- 98 side flips
- authority won 57 flipped rows vs 41 current-production wins.
This is not authorization to revert #599. #599 was promoted on football accuracy, not line direction.

## Closed shared-tail experiment
Frozen canonical cohort = exact 884 C2 team-games (444/440).
2025 holdout results:
- Spearman `0.0887`
- residual gap `+3.12 yd`
- bootstrap probability `75.58%`
- 100+ rate ratio `1.237x`
- catastrophic 40+ miss ratio `1.013x`
- opportunity conditional `2/4`
- 2024 coherence negative.
Disposition: no actionable QB-C2 -> WR1 shared-tail signal. No percentile/threshold rescue.

Claude's 849-row actual-usage sensitivity independently reached the same substantive failure. The canonical authority remains the preregistered 884-row result.

## Anti-retest closures
Do not repeat generic versions of:
- M72 explosive weapon x defense
- M75 separation/cushion/aDOT/YACOE/secondary quality
- R7 persistent explosive/YAC/air-yard traits
- R9-R11 NGS
- R3 residual persistence combined calibration
- C1/C3 joint receiver/QB adjustments
- ND3 vacancy/dynamic entitlement
- fabricated player-level WR-CB assignments.

## Open scientific target
The unresolved target is football-only WR receiving-yard efficiency/translation conditional on already-projected opportunity, especially high-efficiency and high-ceiling games. Any next candidate must be demonstrably new relative to the closures above, temporally valid, leakage-safe, preregistered, and independently challenged by GPT-5.6 and Claude before result exposure.
