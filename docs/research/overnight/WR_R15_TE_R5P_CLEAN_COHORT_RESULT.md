STATUS: RESEARCH ONLY — NOT PROMOTED. WR-R15/TE-R5P re-applied to the
identity-clean cohort (PR #541/#542). No production/model/threshold change.

# WR-R15/TE-R5P applied to the identity-clean benchmark

Direct answer to: "you tested Vegas against my base ensemble, not against my
full researched/layered model." Correct — `CLEAN_BENCHMARK_INDEPENDENT_REGRADE_V1_RESULT.md`
tested only `mc_proj`/`ml_proj`/`state_proj` + frozen ensemble weights, the
same base stack that existed before WR-R15/TE-R5P were promoted. This
re-applies those two promoted models to the same clean cohort, using the
identical method already used once on the corrupted cohort (PR #534) —
`scripts/research/apply_wr_r15_te_r5p_to_clean_benchmark_v2.py`, same join
keys, same `grade()` function, same PLAY/LEAN/STRONG gate. Nothing re-fit.

WR-R15 applied to 3,079 rows (2023-2024 WR receptions/rec_yards only, per
its own frozen validation contract — no 2025 extrapolation). TE-R5P applied
to 2,662 rows (2023-2025 TE receptions/rec_yards, full overlap).

## Before (base ensemble) vs after (WR-R15/TE-R5P applied), STRONG tier

| market | model_mae before | model_mae after | vegas_mae before | vegas_mae after | ROI before | ROI after | still loses to Vegas |
|---|---:|---:|---:|---:|---:|---:|:---|
| rec_yards | 21.603 | 21.025 | 19.999 | 19.834 | -3.36% | -2.89% | YES |
| receptions | 1.707 | 1.642 | 1.546 | 1.542 | -0.87% | -0.49% | YES |

## Honest read

Your researched layers **measurably help** — this is real, not nothing.
Model MAE drops on both touched markets, ROI improves on both (receptions
nearly halves its loss, -0.87% to -0.49%), and this happens on the clean,
correctly-matched cohort, not a corrupted one. That's genuine evidence
WR-R15 and TE-R5P are doing what they were built to do: making the
projection more accurate than the base ensemble underneath them.

**They do not yet flip the verdict.** The model still loses to Vegas on raw
MAE in both markets after the improvement, and ROI is still negative in
every tier. WR-R15/TE-R5P narrow the gap to Vegas; they do not close it,
at least not on this benchmark's fidelity-limited probability layer (same
disclosed caveat as before: `Normal(mean=proj, sd=component_sd)`, not a
full replay of the promoted stack's actual simulation distribution — this
adjustment still only swaps the point mean, not the underlying MC output
that would also affect `component_sd`/STRONG-tier assignment).

## What's still not tested against Vegas at all yet

QB M89/90 pass_yards synthesis reconstruction was **not** re-run on the
clean cohort in this pass. Unlike WR-R15/TE-R5P (a single join), M89/M90 is
a 6-stage pipeline (`build_m89_2023_training_trace.py`,
`build_m90_rotated_training_trace.py`, `run_m89_pregame_synthesis.py`,
`correct_m89_team_semantics.py`, `run_m90_qb_synthesis_confirmation.py`,
plus an integrity audit) — re-running it correctly against the clean
cohort is comparable in scope to the identity rebuild itself, not a quick
follow-up. Flagging as the next explicit research task rather than
improvising a shortcut. Its last (pre-rebuild, corrupted-cohort) result was
STRONG-tier ROI -1.40% — closer to breakeven than the base QB ensemble's
-5.3%, but still negative, and that number needs the same re-verification
treatment as everything else once it's re-run clean.

RB P3/R26/R22 remain correctly out of scope for any 2024/2025 historical
backtest regardless of cohort cleanliness — their promotion contract scopes
them to Week 1 of the 2026 season specifically; there is no historical
in-season version of what they predict to test against.

## Files

`scripts/research/apply_wr_r15_te_r5p_to_clean_benchmark_v2.py`,
`clean_v2_non_qb_detail_wr_r15_te_r5p_applied.csv`,
`clean_v2_non_qb_summary_wr_r15_te_r5p_applied.csv`.
