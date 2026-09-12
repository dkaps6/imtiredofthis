# NFL HANDOFF — 2026-09-12 — HISTORICAL MARKET FALSIFICATION CURRENT

## Status

This is the active research checkpoint on branch `research-historical-market-falsification-v1`.

Branch base at creation: `22057ba9571d7976fdc4f7e6c5ddc9d6d6f1bd7e`.

Latest `main` reconciled during this checkpoint: `f4f0422b89a0a9f9d71dee0b195e619fa8557013` (PR #536, Claude holdout downgrade merged).

GitHub is canonical; chat memory is secondary.

## Concurrent research rule

The user currently has Claude working in the repository at the same time. This branch is an isolated GPT-5.6 research lane:

- do not write directly to `main`;
- do not force-update shared branches;
- re-fetch `main` before every material checkpoint, merge, or canonical-state claim;
- reconcile rather than duplicate concurrent work;
- preserve positive and negative evidence;
- use GitHub Issue #535 as the shared cross-audit control room.

## Priority override

Historical Vegas falsification is the highest-priority issue. Do not present the player-prop stack as historically profitable overall, do not treat large live `edge_pct` values as validated wagering edges, and do not change upstream football science until mean-quality failure is separated from betting-translation failure.

Season-long Weeks 1-18 continuity remains real but secondary until this separation is closed.

## Canonical baseline evidence

The repository contains the 2024-2025 full-stack Vegas benchmark under `data/backtests/full_stack_vegas_benchmark_v1/`.

Baseline disposition: `FULL_STACK_EDGE_THRESHOLD_HISTORICAL_BENCHMARK_QB_SYNTHESIS_GATES_CONFIRMED_NO_FOOTBALL_ONLY_EDGE`.

QB M89/M90 football synthesis improves its own football baseline materially but remains negative against the historical market at the PLAY/STRONG tier. Non-QB rush yards, receiving yards, rush+receiving yards and receptions also remain negative in aggregate.

Claude subsequently:

- applied contract-valid WR-R15 / TE-R5P point projections in PR #534;
- found the prior receptions-UNDER candidate softened;
- ran a genuine cross-year threshold holdout in PR #536;
- obtained fit-2024 -> test-2025 ROI +0.61% (n=577) and fit-2025 -> test-2024 ROI +1.59% (n=543);
- correctly downgraded that candidate to directionally plausible but unconfirmed / not promotable.

## GPT-5.6 independent pass 1

Full record: `docs/research/HISTORICAL_MARKET_FALSIFICATION_GPT56_PASS1.md` at commit `e393ef46222e81077258065221911dddac422a96`.

### Finding 1 — a real football-mean gap exists

After WR-R15 / TE-R5P point means are applied where their historical contracts permit, Vegas line MAE remains lower than model MAE in every non-QB market:

- rush_yards: model 20.6553 vs Vegas 19.0783;
- rec_yards: model 20.8604 vs Vegas 19.8043;
- rush_rec_yards: model 32.5282 vs Vegas 27.1973;
- receptions: model 1.6190 vs Vegas 1.5326.

All-market non-QB: model 16.1567 vs Vegas 14.6805.

QB M89/M90 remains valuable relative to the project's prior QB base (combined MAE 61.1505 -> 59.7739, bias -10.0387 -> +2.5483, correlation 0.1974 -> 0.2604), but the Vegas-line benchmark is still more accurate on MAE. Do not call M89/M90 bad football science; call it improved football science that has not demonstrated superiority to Vegas.

`rush_rec_yards` is the largest current mean-quality gap. Receptions is the closest lane to Vegas.

### Finding 2 — the historical probability translator is not production-equivalent

Historical grader `scripts/backtest/grade_full_stack_vegas_benchmark_v1.py` defines outcome sigma as:

`std(mc_proj, ml_proj, state_proj)`

and converts `(projection - line) / component_sd` through a Normal CDF.

That is inter-model point-estimate disagreement, not player-outcome variance.

Production `scripts/run_pricing_v2.py` instead obtains `fair_prob` directly from the final simulated outcome array after applying the final football mean:

`p_over = mean(adjusted_outcomes > line)`.

Therefore the historical `prob_edge` / EV / STRONG results are directionally useful but are not a literal historical replay of production fair probability.

### Finding 3 — STRONG_EDGE is almost non-selective, a major calibration warning

Post WR-R15 / TE-R5P point-mean patch:

- non-QB STRONG = 15,428 / 16,973 = **90.90%** of all rows;
- rush_yards = 88.78% STRONG;
- rec_yards = 89.24% STRONG;
- rush_rec_yards = 94.48% STRONG;
- receptions = 91.98% STRONG.

QB synthesis = 709 / 812 = **87.32%** STRONG; 93.23% LEAN_OR_STRONG.

A gate requiring estimated EV >= 5% and probability edge >= 3 percentage points should not classify almost the entire sportsbook board as a strong edge if its fair probabilities are calibrated. Treat this as a high-priority downstream-calibration/fidelity failure.

### Finding 4 — WR-R15 / TE-R5P V2 changes the mean but not the historical uncertainty model

`scripts/research/apply_wr_r15_te_r5p_to_vegas_benchmark_v1.py` replaces `adjusted_proj` with the promoted specialist point mean but leaves `mc_proj`, `ml_proj`, and `state_proj` unchanged. The unchanged historical grader then derives `component_sd` from those old components.

Production applies WR-R15 / TE-R5P upstream of joint simulation, so the actual production distribution changes with the entitlement specialist.

Thus V2's MAE effect is informative, while its `prob_edge` ranking / ROI are still fidelity-limited and are not a complete grade of the promoted WR/TE stack.

### Finding 5 — the free market archive drops line-moving player/book groups

`scripts/backtest/prepare_free_market_prop_archive_v1.py` groups captured source rows by season/week/market/player/book, collects unique line values, and drops a group entirely whenever it contains more than one line.

This means:

- historical props whose lines moved across captured snapshots are excluded;
- the benchmark is biased toward static-line groups;
- it is not a full representative closing-board sample;
- for constant-line groups, odds use `iloc[-1]`, but this function itself does not enforce timestamp sorting before selecting the final price.

The source policy must be quantified and frozen before treating ROI as a clean close/T-minus historical test.

### Finding 6 — two-sided price completeness is not guaranteed before no-vig

Book selection ranks DraftKings before FanDuel before considering price completeness. A one-sided DK row can therefore beat a complete FD row.

The full-stack `no_vig()` fallback returns the one available implied probability unchanged when the opposite price is missing. That is not a true two-way de-vig probability.

Quantify prevalence before assigning effect size; future fidelity gate should require complete two-sided prices or fail closed.

### Finding 7 — side asymmetry exists, but not a universal UNDER edge

Historical STRONG-tier examples:

- rush_yards UNDER: +0.57% ROI pooled; OVER: -9.23% ROI;
- rec_yards: OVER -2.65%, UNDER -3.02%;
- baseline receptions: OVER +2.23%, UNDER -0.92%;
- rush_rec_yards: both sides materially negative.

Rush-yards OVER is a conspicuous failure cohort, but even rush UNDER changes sign by year. Current 2026 UNDER concentration is not validated by a broad historical universal-UNDER rule.

## Current interpretation

The project has **two problems at once**:

1. historical football point means are generally not as accurate as Vegas lines;
2. the historical fair-probability / edge translator is also materially non-production-equivalent and appears severely overconfident.

Therefore negative ROI is a legitimate falsification signal, but it does not yet cleanly prove that production's literal Monte Carlo `fair_prob` engine itself loses historically. Conversely, benchmark-fidelity limitations do not erase the real point-mean gap versus Vegas.

Do not collapse these into a single diagnosis.

## Frozen next work

Do NOT tune edge thresholds to these results yet.

1. Reconstruct actual historical production-style outcome distributions where feasible.
2. Apply WR-R15 / TE-R5P upstream of historical simulation, not only as point-mean replacements.
3. If literal simulation cannot be reconstructed, calibrate outcome variance prospectively on training folds and test on untouched folds; do not treat raw component disagreement as outcome sigma without validation.
4. Freeze a line/price timestamp policy and preserve line-moving groups rather than dropping them.
5. Require complete two-sided prices for no-vig comparison.
6. Add calibration bins, Brier/log loss, STRONG coverage, year/side breakdowns, non-QB bias/correlation and model-closer-than-Vegas metrics.
7. Only after benchmark fidelity is repaired should the project decide whether upstream football science must be reopened and in which specific lanes.

## Cross-audit

Post this pass to GitHub Issue #535 and ask Claude to independently falsify:

- the ~90.90% non-QB / ~87.32% QB STRONG coverage;
- one-sided DK / complete FD prevalence;
- line-movement group-drop counts and source row chronology;
- availability of historical simulation arrays / inputs for literal empirical fair-prob reconstruction;
- the two-problem interpretation (mean gap + separate translation/fidelity gap).

No production promotion is authorized from this checkpoint.
