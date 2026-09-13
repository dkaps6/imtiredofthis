# VEGAS_CONFIRMATION_LIKELIHOOD_V1 — Frozen Plan

**STATUS: FROZEN BEFORE ANY 2025 CANDIDATE OUTPUT. RESEARCH ONLY. NO PRODUCTION CHANGE.**

## Objective

PR #560 showed that when the pregame Vegas game script is later confirmed by the realized game, the margin→RB and total→WR/TE usage relationships move materially toward the ground-truth ceiling. This experiment asks the only actionable follow-up: **can we identify those more-trustworthy games before kickoff using the market's own pregame geometry, and does doing so sharpen player-usage effects on a blind season?**

This does not reopen PR #558, does not alter `project_game_script()`, and does not use player props or target-game outcomes as features.

## Frozen split

- Train: 2023 + 2024 regular season.
- Test: 2025 regular season only.
- One game per classification row.
- No reverse rotation, no threshold/model-family search, no 2025 outcome used in fitting or threshold selection.

## Frozen labels

Reuse PR #560's corrected definitions.

Primary (`T=7`):
- `margin_confirmed_7`: predicted and actual winner side agree **and** `abs(actual_margin_home - predicted_margin_home) <= 7`.
- `total_confirmed_7`: predicted and actual total lie on the same side of PR #560's frozen total split **and** `abs(actual_total - predicted_total) <= 7`.

Sensitivity (`T=3`), identical definitions with tolerance 3. T=3 is reported only; it cannot select a different model or threshold.

The total split is the same pooled 2023-2025 pregame `predicted_total` median used by PR #560. It is an already-frozen label convention from the parent study, not a fitted feature or test-outcome-derived tuning step.

## Frozen pregame features

Base features for both labels:
1. signed home spread (`predicted_margin_home`)
2. absolute spread
3. posted total (`predicted_total`)
4. distance from `abs(spread)` to nearest canonical NFL key margin in `{3, 7, 10, 14}`

Optional moneyline block, all-or-nothing:
5. favorite no-vig moneyline probability
6. spread-vs-moneyline consistency residual

Moneyline block eligibility is frozen as **>=80% complete two-sided home+away moneylines in both the 2023-24 train set and the 2025 test set**. If either side misses that bar, both moneyline-derived features are omitted everywhere.

American moneylines are converted to implied probabilities and normalized two-way to remove vig. `favorite_ml_prob = max(p_home_novig, p_away_novig)`.

If eligible, the spread-vs-moneyline residual is leakage-safe: on 2023-24 only, fit `StandardScaler -> LogisticRegression` from `abs(spread)` to the pregame favorite's no-vig moneyline probability converted to a binary favorite-win proxy is **not allowed** because that would require outcomes. Instead fit a train-only deterministic linear mapping from `abs(spread)` to `favorite_ml_prob`; residual = observed pregame favorite no-vig probability minus train-fitted expected probability. Apply the frozen mapping to 2025. This is cross-market consistency, not cross-book consensus.

No weather, injuries, team-strength fields, player data, our model projections, prop lines, or outcome-derived features are allowed upstream.

## Frozen model

One model per label and tolerance:

`StandardScaler -> LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, class_weight=None, random_state=42)`

No hyperparameter tuning, no alternate model family, no feature selection after output.

## Frozen classification gates — blind 2025

For each primary T=7 label independently:
1. ROC AUC > 0.55.
2. Brier score strictly better than constant 2023-24 train-prevalence probability.
3. High-likelihood subset has >= +10 percentage-point confirmation-rate lift versus all eligible 2025 games.
4. Adequate class support: both positive and negative 2025 labels >=30 rows.

Selector threshold is frozen before 2025 scoring as the **75th percentile of 2023-24 fitted predicted confirmation probability** for that label. No 2025 quantile or alternate cut may be used.

T=3 sensitivity must be directionally supportive for the same feature/model specification: AUC > 0.50 and selected confirmation-rate lift > 0.00. It need not satisfy the T=7 numeric gates.

## Frozen downstream utility gates — blind 2025

Build the PR #560 team-game player-usage frame for 2025, then select only games above the frozen train-derived T=7 confirmation-probability threshold for the matching hypothesis.

Primary volume metrics:
- margin model -> `rb_rush_att`
- total model -> `wrte_targets`

For each:
1. Reproduce the 2025 PR #560 unconditional Vegas arm on the same eligible games.
2. Selected-arm Cohen's d must be **strictly greater** than the unconditional Vegas-arm Cohen's d, moving toward the ground-truth actual-outcome arm.
3. Selected high/low split must retain >=30 team-game rows on each side.

`rb_rush_yards` and `wrte_rec_yards` are reported as secondary diagnostics only and cannot rescue a failed primary-volume gate.

## Disposition

- If either T=7 classification model fails its classification gates: that hypothesis gets `NO_ACTIONABLE_PREGAME_CONFIRMATION_STATE` and no downstream rescue is allowed.
- If classification passes but its primary downstream usage gate fails: confirmation may be statistically predictable but is not useful for this player-volume objective; stop that hypothesis.
- Only a hypothesis passing **both** classification and downstream gates becomes `QUALIFIED_PREGAME_CONFIRMATION_CANDIDATE`, still research-only and not production-authorized.
- No combined overall PASS is claimed unless both margin/RB and total/WRTE hypotheses pass independently.

## Source / integrity constraints

- Use merged PR #559/#560 loaders and player-log builder; do not rederive line sign conventions.
- No paid historical source and no claim of historical line movement/cross-book consensus.
- Preserve exact row counts, class prevalence, feature coverage, coefficients, thresholds, AUC/Brier/lift, effect sizes, and season/test identities in artifacts.
- Any mechanical failure before scoring may be repaired without changing this plan. Any scientific change requires a new frozen plan before seeing replacement output.
