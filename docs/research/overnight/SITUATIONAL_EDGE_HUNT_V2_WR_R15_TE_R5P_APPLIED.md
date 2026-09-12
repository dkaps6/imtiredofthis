STATUS: RESEARCH ONLY — NOT PROMOTED — AWAITING USER APPROVAL.

CORRECTION (Issue #535 checkpoint 28): this doc's title, file name, and
prose originally said "TE-R5P" throughout. The artifact actually applied for
TE (`te_r5_oos_player_casebook.csv`, run `34132127351`/artifact
`10022512461`) is **TE-R5**, an earlier scientific-decomposition model, not
the later production-certified **TE-R5P** (authorized_by_run `34152797603`,
artifact `10029942404` — see
`data/models/te_r5p_production_model_v1/te_r5p_production_model_v1.json`).
The numbers below are unchanged and are a real, correctly-computed result
for TE-R5 as actually applied; they were never a valid test of TE-R5P. The
text below is corrected to say TE-R5. A genuine TE-R5P historical replay is
separate, not-yet-done follow-up work (see PR #549).

# Situational Edge Hunt V2 — WR-R15/TE-R5 Applied

Direct follow-up to `SITUATIONAL_EDGE_HUNT_V1_RESULT.md`. That result found a
candidate edge (receptions, model's top confidence quartile, UNDER side,
+2.66% ROI both seasons independently) on a cohort where receptions/rec_yards
were graded on the base MC+ML+State+ensemble engine only — the promoted
`WR_R15_WR1_ANCHORED_PARTICIPATION` model and the TE-R5
scientific-decomposition model were not yet applied (disclosed gap in
`data/backtests/full_stack_vegas_benchmark_v1/README.md`). This closes that
gap using the actual canonical validation artifacts for those two models
(not a re-fit, not new modeling) and re-checks whether the edge holds.

## What was applied, and to what scope

- **WR-R15**: joined `wr_r15_confirmation_predictions.csv` (canonical OOS
  artifact, run `34238301577`, artifact `10061328722` — the exact IDs
  WR-R15's own production certification is pinned to), `WR_R15_WR1_ANCHORED_PARTICIPATION`
  variant only, to WR position rows. **2023-2024 only** — WR-R15 was never
  validated for 2025 (`scientific_confirmation_seasons: [2023, 2024]` is the
  frozen production contract, and `scientific_confirmation_2025_used` must be
  `False`), so 2025 WR receiving rows are correctly left on the base
  ensemble, not silently "confirmed" with an unvalidated projection.
- **TE-R5** (not TE-R5P — see correction note above): joined
  `te_r5_oos_player_casebook.csv` (canonical artifact, run `34132127351`,
  artifact `10022512461`), `candidate_receptions_r5`/`candidate_rec_yards_r5`,
  to TE position rows. Full 2023-2025 coverage.
- Re-graded with the exact same arithmetic as the original benchmark
  (`scripts/backtest/grade_full_stack_vegas_benchmark_v1.py::grade()`,
  unmodified, same PLAY/LEAN/STRONG thresholds) — nothing about the grading
  rule changed, only the input projection for the rows these two models
  actually cover.
- 3,082 rows got a WR-R15 projection (2023-2024 WR receptions/rec_yards);
  2,669 rows got a TE-R5 projection (2023-2025 TE receptions/rec_yards).
  Everything else (RB, QB, 2025 WR) is byte-identical to the original
  benchmark.

Reproduce: `python3 scripts/research/apply_wr_r15_te_r5p_to_vegas_benchmark_v1.py`
(reads cached artifacts from `/tmp/` — see script docstring for exact sources;
not committed here since they're large and the artifacts themselves are the
citable source of truth).

## Aggregate effect (STRONG_ONLY_PLAY_TIER)

| Market | Before (base engine only) | After (WR-R15/TE-R5 applied) |
|---|---|---|
| receptions | win rate 54.1%, ROI -0.6% | win rate 54.3%, ROI **-0.12%** |
| rec_yards | win rate 51.6%, ROI -2.9% | win rate 51.8%, ROI **-2.56%** |

Both markets move closer to breakeven, neither crosses it in aggregate. Small
but real, directionally consistent improvement from applying WR-R15 (the
real production model) plus TE-R5 (an earlier scientific-decomposition
model, not the promoted TE-R5P) instead of the generic ensemble — as
expected, not a surprise, but not yet confirmation that the actual promoted
TE-R5P is pulling its weight, since it wasn't the model tested here.

## Does the V1 candidate edge survive?

**Yes, with the edge holding but softening.** Re-running the same slice
(receptions, `signal == STRONG_EDGE`, top quartile of `prob_edge`, UNDER
side):

| | V1 (base engine) | V2 (WR-R15/TE-R5 applied) |
|---|---:|---:|
| n | 1,258 | 1,105 |
| Pooled ROI | +2.66% | +1.50% |
| 2024 ROI | +2.43% | +2.07% |
| 2025 ROI | +2.90% | +0.92% |
| Win rate | 52.5% | 52.9% |

Still positive independently in both seasons — the candidate survives the
harder test. n shrank because the confidence quartile boundary shifts once
better projections change which bets qualify as "top quartile."

## The more interesting split: rows the new models actually touched

Splitting the V2 top-quartile/UNDER population by whether WR-R15/TE-R5
actually changed that row's projection:

| | n | Pooled ROI | Win rate |
|---|---:|---:|---:|
| Touched by WR-R15/TE-R5 | 396 | **+2.94%** | 54.0% |
| Untouched (base engine) | 709 | +0.70% | 52.3% |

The rows these two models touched show a *stronger* edge than the rows
still on the base engine — encouraging, and the direction you'd want, but
remember only WR-R15 here is the actual promoted production model; the TE
side is TE-R5, not TE-R5P. **Do not over-read this**: broken out by season,
the touched group is 2024 n=375 (+4.46% ROI, strong) vs. **2025 n=21
(-24.35% ROI)** — and that tiny 2025 slice is almost entirely TE-R5 (since
WR-R15 has zero 2025 coverage by design). n=21 is nowhere near enough to
draw a conclusion either way; it's a couple of losses away from flipping
sign. Treat the "touched rows are stronger" read as suggestive, not
confirmed, until there's a larger 2025 TE-only sample (from the actual
TE-R5P model, per the follow-up in PR #549) or (if you ever want it) a
genuine WR-R15 2025 confirmation pass.

## Bottom line

The V1 candidate is not an artifact of grading receptions on the wrong
model — it survives contact with a real entitlement layer (WR-R15, the
actual promoted model; TE-R5, an earlier decomposition model, not TE-R5P),
softer but still real and still consistent both seasons. These models
appear to be concentrating value in the direction you'd hope, but the
evidence for that specific claim is thin (n=21 in the one season/position
cell that would prove it). This remains a lead worth a confirmatory look
with more data, not something to size a bet off yet.
