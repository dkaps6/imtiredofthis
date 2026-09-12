STATUS: RESEARCH ONLY — NOT PROMOTED. No production/model/weight/threshold
change. Design frozen and posted to Issue #535 checkpoint 24 before any run
against real data. This experiment is led by Claude, per explicit user
instruction ("propose that but I want you to lead it") — GPT-5.6 was invited
to attack the design before execution, not asked to co-own it.

# Distribution Widening V1 — Design

Checkpoints 12/20 (Issue #535) found the empirical MC distribution
reconstructed in PR #546 (`persist_historical_simulated_outcomes_v1.py`) is
under-dispersed relative to realized forecast residuals in every one of 5
markets individually — 45.4%-69.4% of realized `|actual - proj|` SD depending
on market, confirmed per-market, not an aggregate blending artifact. This
tests the direct next question: does correcting that specific gap — and
nothing else — move calibration/ROI further?

## Method, single variable only

1. **FIT phase** (one season, no sportsbook data touched at all): for each
   row, rescale its raw historical MC array to its own frozen projection
   mean (exactly production semantics, reused unmodified from
   `grade_empirical_fair_prob_v1.py::rescale_outcomes`), compute that row's
   within-row simulated SD, and compare the market-level mean of those SDs
   against the market-level realized residual SD
   (`std(actual - proj)` across rows). The ratio is the widening factor `k`,
   frozen per market.
2. **APPLY phase** (the OTHER season only, `k` frozen from step 1, blind):
   widen each row's mean-aligned rescaled array by `k` around its own
   (unchanged) mean — this cannot alter `model_mae`, since the point
   projection is never touched, only the spread around it. Recompute
   `p_over` empirically on the widened array. Grade against real Vegas lines
   with the exact same frozen PLAY/LEAN/STRONG gate used everywhere else in
   this research thread (`grade_full_stack_vegas_benchmark_v1.py`'s
   `ev_roi`/`signal`/`no_vig` logic, unmodified).
3. Both directions run (fit 2024 → test 2025, and fit 2025 → test 2024) so
   every reported number is a genuine out-of-sample result, never
   fit-on-what-you-graded.

A hard runtime assertion (`max_abs_mean_shift <= 1e-8`) enforces that
widening never moves the row's mean — the only thing this experiment is
allowed to change is spread, so a bug that leaked into the mean fails loud
rather than silently confounding the result with a second variable.

## Explicitly not done here

- No mean/weight/threshold change.
- Not combined with PR #545's held-out ensemble weights (one variable at a
  time, per the standing agreement in Issue #535).
- No post-hoc factor search — if widening doesn't help, that is the result,
  not a starting point for tuning `k` until it does.
- Not a distribution *shape* change (still Gaussian-shaped resampling around
  the same mean-aligned empirical array) — this tests dispersion width only.

## Files

`scripts/research/fit_and_apply_distribution_widening_v1.py`,
`tests/test_fit_and_apply_distribution_widening_v1.py`,
`.github/workflows/research-distribution-widening-2024-2025-holdout-v1.yml`.

Validated locally against a synthetic fixture (deliberately narrow simulated
SD vs. wide realized residual SD, mimicking the exact real-world pattern)
before spending any CI compute: the fit phase recovers the true dispersion
ratio, `model_mae`/`vegas_mae` are provably bit-identical between the
unwidened and widened variants (confirming the mean-invariance contract),
and a widening factor of 1.0 is confirmed to be an exact no-op.
