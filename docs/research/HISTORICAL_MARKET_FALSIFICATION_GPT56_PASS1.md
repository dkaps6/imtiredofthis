# GPT-5.6 Independent Historical Market Falsification — Pass 1

Status: RESEARCH ONLY. No production-science change authorized.

Canonical `main` reconciled through `f4f0422b89a0a9f9d71dee0b195e619fa8557013` (PR #536 merged). This pass was performed independently of Claude's candidate-hunt conclusion, while using the same repository artifacts as evidence.

## Executive verdict

The present evidence supports **two distinct failure classes at once**:

1. **Football mean gap:** on the current historical benchmark cohorts, the football-only point projections are generally less accurate than the Vegas line on MAE. This is real and should not be explained away as only a pricing problem.
2. **Betting-translation fidelity gap:** the historical full-stack grader does not reconstruct production fair probability faithfully. It uses disagreement among three point-estimate components as if it were outcome volatility, which makes the historical `prob_edge` / EV / STRONG classifications materially suspect. The fact that about 91% of non-QB props and 87% of QB props qualify as `STRONG_EDGE` is a major calibration warning.

Therefore the negative historical ROI is a legitimate falsification signal, but it is **not yet a clean historical falsification of production's actual Monte Carlo fair-probability engine**. The point-mean evidence and the probability/ROI evidence must be separated.

## 1. Football projection quality vs Vegas

### Non-QB, after WR-R15 / TE-R5P point means are applied where contract-valid

Source: `docs/research/overnight/non_qb_summary_wr_r15_te_r5p_applied.csv`.

ALL_NO_FILTER:

| Market | Model MAE | Vegas-line MAE | ROI/unit |
|---|---:|---:|---:|
| rush_yards | 20.6553 | 19.0783 | -3.60% |
| rec_yards | 20.8604 | 19.8043 | -3.46% |
| rush_rec_yards | 32.5282 | 27.1973 | -5.21% |
| receptions | 1.6190 | 1.5326 | -0.78% |
| all markets | 16.1567 | 14.6805 | -2.85% |

Vegas is closer on MAE in all four non-QB markets. `rush_rec_yards` has the largest mean-quality gap and is currently the weakest football-mean lane in this comparison. `receptions` is the closest to Vegas and also the closest to breakeven.

STRONG_ONLY_PLAY_TIER still shows Vegas closer in every market:

- rush_yards: model 21.3924 vs Vegas 19.6361
- rec_yards: model 21.0193 vs Vegas 19.8404
- rush_rec_yards: model 33.1540 vs Vegas 27.4923
- receptions: model 1.6367 vs Vegas 1.5409

### QB pass yards

Source: `data/backtests/full_stack_vegas_benchmark_v1/qb_synthesis_summary.csv` and `qb_synthesis_scoreboard_reconstruction_v1.csv`.

QB M89/M90 football synthesis is a real improvement over its base football model:

- combined base MAE 61.1505 -> football synthesis 59.7739
- combined bias -10.0387 -> +2.5483
- combined correlation 0.1974 -> 0.2604

The improvement is present in both seasons, especially 2025. That supports the synthesis as useful football science relative to its own base. But in the Vegas benchmark's 812 matched rows, synthesis MAE is still 58.5264 vs Vegas-line MAE 56.7980, and STRONG-tier ROI is -1.40%.

Conclusion: do not label M89/M90 a failed model. Label it **better than our prior football baseline but not demonstrated to beat the market line** on this benchmark.

## 2. Historical probability translator is not production-equivalent

Source: `scripts/backtest/grade_full_stack_vegas_benchmark_v1.py`.

The historical full-stack grader computes:

- `component_sd = std(mc_proj, ml_proj, state_proj)`
- `p_over = NormalCDF((proj - line) / component_sd)`

This `component_sd` is **inter-model point-estimate disagreement**, not observed player-performance variance and not Monte Carlo outcome variance. It is an epistemic disagreement proxy. Correlated component models can agree closely while football outcomes remain highly volatile; treating this spread as outcome sigma can make probabilities extreme.

Production does something different. Source: `scripts/run_pricing_v2.py`.

Production:

- generates the actual Monte Carlo outcome array;
- applies the final football target mean by rescaling that outcome distribution;
- computes `p_over = mean(adjusted_outcomes > line)`;
- records `model_sd = std(adjusted_outcomes)`.

So the historical benchmark's high-confidence probability results are not a literal replay of production `fair_prob`.

## 3. STRONG edge classification is almost non-selective

After WR-R15 / TE-R5P point means are applied:

- non-QB ALL rows = 16,973
- non-QB STRONG rows = 15,428
- **90.90% of all non-QB rows are labeled STRONG_EDGE**

By market:

- rush_yards: 2,461 / 2,772 = **88.78%** STRONG
- rec_yards: 5,232 / 5,863 = **89.24%** STRONG
- rush_rec_yards: 2,481 / 2,626 = **94.48%** STRONG
- receptions: 5,254 / 5,712 = **91.98%** STRONG

QB synthesis:

- 709 / 812 = **87.32%** STRONG
- 757 / 812 = **93.23%** LEAN_OR_STRONG

A gate requiring estimated EV >= 5% and probability edge >= 3 percentage points should not classify nearly the entire market as a strong edge if fair probabilities are well calibrated. This is the clearest current evidence that the historical probability translator is materially overconfident or otherwise mis-specified.

Claude's holdout cutoffs reinforce the same concern: the top-quartile receptions `prob_edge` cutoffs are about 0.422 and 0.427, yet realized holdout win rates are only about 52.7%-52.9%. The exact model probability depends on each row's market probability, so do not infer a universal fixed fair probability from the cutoff alone, but the combination is consistent with severe overconfidence.

## 4. WR-R15 / TE-R5P historical application still has a distribution mismatch

Source: `scripts/research/apply_wr_r15_te_r5p_to_vegas_benchmark_v1.py`.

The V2 historical patch:

- starts from original `mc_proj`, `ml_proj`, `state_proj`, `ensemble_proj`;
- creates `adjusted_proj`;
- overwrites only `adjusted_proj` with WR-R15 / TE-R5P specialist values where valid;
- then calls the unchanged historical `grade()`.

Therefore the upgraded specialist mean is evaluated with a `component_sd` still derived from the **old base MC/ML/State components**.

Production WR-R15 / TE-R5P is different: source `scripts/run_pricing_with_full_roster_universe_v3_core.py` shows these specialists are applied at the target-entitlement seam **before joint simulation**, so they alter simulated player distributions as well as means.

Implication:

- V2's MAE change is informative about point-mean quality.
- V2's `prob_edge`, STRONG ranking, and ROI are **not a full historical replay of the promoted WR/TE production stack**.

This is a substantive fidelity gap, not a cosmetic one.

## 5. Market archive construction creates a selection/timestamp problem

Source: `scripts/backtest/prepare_free_market_prop_archive_v1.py`.

The archive normalizer groups by season/week/market/player/book and collects unique line values. If a group contains more than one line value, it executes `continue` and drops the entire group.

Consequences:

1. Props whose lines moved across captured snapshots are systematically excluded.
2. The benchmark is therefore disproportionately a **static-line cohort**, not necessarily a representative historical betting board.
3. For a group whose line stayed constant, the script uses the last non-null over/under price via `iloc[-1]`, but this function does not itself sort by a timestamp before doing so. Unless upstream source row order is guaranteed chronological, `archived_latest_per_book` is not independently enforced here.
4. The data still does not represent a frozen T-minus-X-minutes or official close snapshot.

This does not automatically make the benchmark optimistic or pessimistic, but it prevents interpreting it as a clean closing-line historical test and may create selection bias.

## 6. One-book selection and de-vig fidelity gap

Source: `scripts/backtest/grade_historical_market_vegas_benchmark_v1.py` and `grade_full_stack_vegas_benchmark_v1.py`.

Book selection sorts by `book_rank` before `price_count`:

- DraftKings rank 0
- FanDuel rank 1

So DraftKings is selected whenever present even if its row has only one side priced and FanDuel has both sides.

The full-stack `no_vig()` fallback returns the single available implied probability unchanged if the opposite side is missing. That is not a true two-way de-vig probability.

Need to quantify how often this actually occurs before assigning effect size, but the correct fidelity gate should either:

- require complete two-sided prices for the chosen book, or
- prioritize a complete two-sided row before book preference,
- and otherwise fail closed for no-vig probability comparisons.

Flat 1-unit stake arithmetic and push accounting appear standard.

## 7. OVER / UNDER asymmetry is real but not universal

The V1 situational detail shows notable side-specific failures:

### Rush yards, STRONG tier
- UNDER: n=1,698, win 53.42%, ROI **+0.57%**
- OVER: n=763, win 47.18%, ROI **-9.23%**

The OVER lane is a major historical failure cohort. UNDER is only slightly positive pooled and changes sign by year (2024 negative, 2025 positive), so this is not yet a season-invariant edge.

### Receiving yards, STRONG tier
- OVER: ROI -2.65%
- UNDER: ROI -3.02%

Both fail.

### Receptions, baseline STRONG tier before specialist patch
- OVER: n=635, ROI **+2.23%**
- UNDER: n=4,732, ROI **-0.92%**

Again there is no universal historical rule that UNDER is inherently better. Claude's narrower high-confidence receptions-UNDER candidate is a cohort-specific hypothesis and its genuine cross-year holdout has already been downgraded to unconfirmed.

### Rush+rec
Both sides are materially negative.

## 8. What the negative ROI currently proves — and what it does not

It DOES prove:

- our current historical football means, as reconstructed, generally do not beat Vegas line MAE;
- the current historical betting translator does not demonstrate a broad profitable edge;
- the live board's very large `edge_pct` values should not be assumed calibrated from this history;
- market/side-specific failure modes exist and deserve explicit diagnostics.

It does NOT yet prove:

- that production's literal Monte Carlo `fair_prob` path is historically negative, because that exact distribution path is not what the historical full-stack grader used;
- that WR-R15 / TE-R5P's promoted distributional effect is historically captured by V2, because V2 replaced their means without reconstructing their simulation distributions;
- that threshold tuning alone can fix the problem.

## 9. Frozen next-step recommendation

Do NOT tune thresholds against these results yet. Repair benchmark fidelity first.

### A. Distribution / probability fidelity
1. Reconstruct historical outcome distributions using the actual production simulation path where feasible.
2. Apply WR-R15 / TE-R5P upstream of historical simulation exactly as production does, within their frozen historical scientific scope.
3. For lanes where literal historical simulation cannot be reconstructed, fit/calibrate an outcome-variance model only on training folds and test it on untouched seasons; do not use MC/ML/State point disagreement as outcome sigma without empirical calibration.
4. Report probability calibration bins, Brier score, log loss, STRONG-tier coverage, and realized win rate by predicted-probability bin.

### B. Market-source fidelity
1. Freeze a line/price timestamp policy prospectively.
2. If the archive contains multiple snapshots, preserve line-moving groups rather than dropping them and select a defined snapshot (preferably close or a fixed pre-kickoff horizon).
3. Require complete two-sided prices for no-vig calculations.
4. Report book/source and timestamp coverage explicitly.

### C. Mean-model diagnostics
1. Add non-QB bias, correlation, and `model_closer_than_vegas_rate` to the full-stack benchmark.
2. Break these out by year, market, position, role and side.
3. Prioritize `rush_rec_yards` mean quality and rush-yards OVER failure as diagnostic lanes, not immediate post-hoc fixes.

### D. Confirmation discipline
After fidelity repairs, rerun a genuinely held-out market test. No upstream model science change should be promoted until that separation tells us whether the remaining failure is mean quality, distribution calibration, or both.

## Cross-audit requests for Claude

Please independently try to falsify these specific findings:

1. Reproduce the ~90.90% non-QB / ~87.32% QB STRONG coverage and determine whether there is any legitimate reason production-equivalent fair probabilities should be this non-selective.
2. Quantify the frequency of incomplete DK two-sided prices and whether FanDuel often has a complete alternative.
3. Quantify how many source groups are dropped because multiple line values occur and verify whether source row ordering truly makes `iloc[-1]` chronological.
4. Check whether historical simulation arrays or sufficient historical production inputs already exist to reproduce actual empirical `fair_prob` rather than the component-SD Normal proxy.
5. Challenge the conclusion that the current benchmark simultaneously shows a real mean gap vs Vegas and a separate downstream probability-calibration/fidelity gap.
