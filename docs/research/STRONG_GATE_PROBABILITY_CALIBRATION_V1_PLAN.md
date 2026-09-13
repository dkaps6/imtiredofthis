# STRONG/LEAN Gate Probability Calibration V1 — Frozen Plan

**STATUS: FROZEN BEFORE ANY CANDIDATE OUTPUT. RESEARCH ONLY. NO PRODUCTION CHANGE.**

## Motivation

Two independently confirmed defects (Issue #535 checkpoints 30/45):

1. The empirical-MC fair probability is severely overconfident. Calibration
   bins (`docs/research/overnight/empirical_model_p_calibration_bins_v1.csv`)
   show realized over-rate stuck at roughly 51-55% across almost the entire
   claimed-probability range 0.55-0.95 — a model claiming "97.6% confident"
   realizes only a 54.9% win rate. Only the top bin (>=0.95 claimed) shows a
   realized rate meaningfully above 50%, and even that is modest (54.9%).
2. The STRONG gate's second condition (`prob_edge >= 3pp`) is algebraically
   redundant with the first (`EV >= 5%`) under realistic sportsbook vig —
   proven directly from the odds/EV formulas, not just observed on one
   dataset (checkpoint 45). It contributes essentially nothing as an
   independent safeguard.

Together these explain why STRONG fires on ~87% of graded rows even when fed
the correct simulated-outcome distribution: the gate's *threshold logic* is
fine in isolation, but the *probability it's applied to* doesn't mean what it
claims to mean, and the intended second safeguard never actually screens
anything out.

**This plan fixes the probability, not the threshold.** `EV >= 0.05` and
`prob_edge >= 0.03` stay exactly as they are in
`scripts/master_betting_workbook_core_v2.py` and
`scripts/backtest/grade_full_stack_vegas_benchmark_v1.py`. Confirmed via
direct code reading that this repo's live pricing pipeline
(`run_pricing_v2.py`) already computes `fair_prob` from the real simulated
Monte Carlo distribution, not a `component_sd` Normal approximation — that
specific defect (checkpoint 41) is a backtest-only fidelity gap, not a live
one. The redundant-threshold and overconfidence findings, by contrast, apply
to the same EV/no-vig/edge arithmetic used identically in the live gate
(`signal()` in `scripts/master_betting_workbook_core_v2.py`), so this
candidate is aimed at a real, live-relevant defect.

## Design

**Input**: `empirical_fair_prob_detail.csv`, the per-row output of
`scripts/research/grade_empirical_fair_prob_v1.py`'s existing, already-CI-run
2024/2025 historical reconstruction
(`.github/workflows/research-historical-fair-probability-reconstruction-v1.yml`).
Carries `season`, `market`, `p_over`, `over_odds`, `under_odds`,
`over_novig`, `under_novig`, `actual`, `line`, `actual_side` per row. No new
historical rebuild; reuses the exact same already-validated reconstruction.

**Calibration target**: `p_over` itself (the probability the empirical MC
distribution assigns to the OVER outcome), not `best_model_p` (which already
reflects a side choice) — the overconfidence is a property of the raw
probability estimate, not of side selection.

**Method**: per-market isotonic regression (`sklearn.isotonic.IsotonicRegression`,
monotonic, non-parametric, appropriate given the calibration bins show a
non-linear, largely-flat-then-slightly-rising realized-rate shape that a
simple Platt/logistic rescaling would not capture), fit on
`(raw p_over, realized over/under outcome)` pairs for decided (non-push) rows
of ONE season only, per market. Five markets (`pass_yards`, `rec_yards`,
`receptions`, `rush_rec_yards`, `rush_yards`), matching how every other
diagnostic tonight has been sliced.

**Genuine two-directional holdout**: fit calibrators on 2024, freeze, apply
blind to 2025; separately fit on 2025, freeze, apply blind to 2024. Neither
direction touches the test season during fitting. Minimum 100 decided rows
required per market/season to fit a calibrator; below that, fail closed
(report as `INSUFFICIENT_ROWS`, do not extrapolate).

**Recomputation**: after calibrating `p_over` -> `p_under = 1 - p_over`,
recompute `ev_over`/`ev_under`/`best_ev`/`best_model_p`/`prob_edge`/`signal`
from the calibrated probabilities using the exact same, unmodified
`implied_prob`/`no_vig`/`ev_roi`/`signal` functions already used everywhere
else in this repo's grading — no new threshold logic invented. `over_novig`/
`under_novig` (the market's own no-vig probability) are untouched; only the
model's own probability estimate changes.

**Grading, avoiding the selection-effect trap from earlier tonight**: report
both (a) `ALL_NO_FILTER` — the same fixed row set, before vs. after
calibration, isolating the calibration effect from any tier-membership
shift, and (b) the resulting `STRONG_EDGE` tier under calibration (new
membership, new coverage rate, new realized win rate/ROI on whatever rows
survive). Both matter: (a) proves whether calibration changes anything on
identical rows; (b) is what a live gate change would actually produce.

## Expected result, stated before running (so it can't be moved after)

Given the calibration bins already show realized rates hovering near 50-55%
across almost the whole 0.55-0.95 claimed range, the honest expectation is
that calibrated EV rarely clears +5% except at the extreme tail — meaning
STRONG coverage should collapse to a small fraction of its current ~87%,
not stay similar. **A large coverage drop is the expected, correct outcome
of a working fix, not a failure of this experiment.** The candidate is
judged on whether the resulting (much smaller) STRONG tier shows a
realized win rate/ROI that is non-negative and better than the current
STRONG tier's, on the same held-out season — not on how many rows it keeps.

## What this is not

Not a production change. Not a retuning of the 0.05/0.03 threshold
constants. Not a claim that live `run_pricing_v2.py` probabilities are
themselves miscalibrated in the same way measured here — this uses the
historical reconstruction's empirical MC arrays, the best available
same-mechanism proxy for live probabilities, but a live-data confirmation
is a distinct follow-up this plan does not claim to satisfy. Any decision
to actually change `scripts/master_betting_workbook_core_v2.py`'s live gate
requires a separate, explicit approval after this result is reviewed.
