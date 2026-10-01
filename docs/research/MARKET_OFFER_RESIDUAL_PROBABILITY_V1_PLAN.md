# Market Offer Residual Probability V1 — Frozen Plan

**STATUS: FROZEN BEFORE ANY REAL RESULT. RESEARCH ONLY. NO PRODUCTION CHANGE.**

## Motivation

Market-Relative Bet Selector V1 Stage 1 closed the scalar fair-line blend:
model-minus-market level disagreement did not add stable two-directional held-out
information for QB pass yards, TE receiving yards, or TE receptions.

This V1 is intentionally a different architecture. It does not estimate a fair
stat line and does not recalibrate the raw production Monte Carlo probability.
It directly asks whether a small fixed downstream discriminative model can
predict the realized OVER/UNDER outcome of an **actual sportsbook offer** better
than that book's own no-vig probability.

Sportsbook information remains strictly downstream of football generation.

## Anti-retest boundary

This is not:
- isotonic calibration of raw model probabilities;
- scalar shrinkage of model projection toward the market;
- analog / k-NN reliability;
- edge-bin / side / home-away / week-bucket slice mining;
- threshold optimization;
- a football-model feature or mean change.

No hyperparameter, feature, subgroup, or threshold search is allowed after the
first result.

## Source scope

First execution is limited to exact historical authorities with symmetric
2024/2025 support:
- QB pass yards: exact M89 OOS authority;
- TE receiving yards: exact TE-R5P OOS authority;
- TE receptions: exact TE-R5P OOS authority.

WR is source-blocked because WR-R15 forbids 2025 confirmation.
RB is source-blocked because no 2024-2025 retrospective promoted authority
exists.

Historical sportsbook source remains the existing free Action Network-derived
DK/FD archive. No paid OddsAPI call.

## Offer-level population

One row per exact:
`game_id + player_clean_key + market + book + line`.

Require:
- exact football authority row;
- valid offer line;
- both OVER and UNDER prices for the same book+line;
- finite final outcome;
- non-push result.

For each player-market-game:
- `crossbook_median_line` = median eligible book line;
- `line_range` = max line - min line;
- `offer_count` = eligible book offers.

Each offer receives sample weight:
`1 / offer_count`
so one player-market-game contributes total weight 1 regardless of book count.

## Frozen features

Fit a separate model per market using exactly five features:

1. `model_gap = model_projection - offer_line`
2. `abs_model_gap = abs(model_gap)`
3. `line_vs_crossbook_median = offer_line - crossbook_median_line`
4. `line_range`
5. `novig_over_centered = book_novig_over_probability - 0.5`

No interaction terms, polynomials, role/position subgroups, player priors,
weather, home/away, week buckets, or feature selection.

## Target

`y_over = 1(actual > offer_line)`.

Pushes are excluded before fitting/evaluation.

## Frozen model

Per market and training season:

`StandardScaler -> LogisticRegression`

Exact settings:
- L2 penalty;
- `C=1.0`;
- `solver='lbfgs'`;
- `max_iter=1000`;
- no class weighting;
- no hyperparameter search;
- training sample weights as defined above.

## Genuine two-direction holdout

Run both:
- fit 2024 -> test 2025;
- fit 2025 -> test 2024.

Support floor per direction:
- >=100 training player-market identities;
- >=100 test player-market identities;
- >=20 training game clusters;
- >=20 test game clusters.

## Baseline

For each offer the frozen baseline probability is that same book/line's
two-sided no-vig OVER probability.

The candidate is judged against the sportsbook probability, not against 0.5.

## Metrics

Primary:
- weighted log loss.

Secondary guards:
- weighted Brier score;
- weighted ROC AUC.

Define per-offer log-loss improvement:
`baseline_logloss - candidate_logloss`.

## Game-cluster bootstrap

- test-season game_id cluster;
- 10,000 resamples;
- seed 20261001;
- candidate model remains frozen from training season;
- preserve offer sample weights;
- report percentile 95% CI for weighted mean log-loss improvement.

## Direction pass gate

A direction passes only if:
1. candidate weighted log loss < baseline weighted log loss;
2. candidate weighted Brier < baseline weighted Brier;
3. candidate weighted AUC >= baseline weighted AUC;
4. mean log-loss improvement > 0;
5. bootstrap 95% CI lower bound for log-loss improvement > 0.

## Market pass gate

A market is `MARKET_OFFER_RESIDUAL_PROBABILITY_V1_PASS` only if BOTH holdout
directions pass every direction gate.

Otherwise:
`NO_VERIFIED_MARKET_OFFER_RESIDUAL_PROBABILITY_SIGNAL_V1`.

No pooled rescue.

## What a pass would authorize

A pass does **not** authorize betting or production.

It would authorize one separately frozen sparse-actionability study translating
the held-out probability into actual offer EV with an abstention rule, followed
by prospective 2026 pre-kickoff shadow confirmation.

Because 2024/2025 outcomes have been repeatedly exposed elsewhere in the
project, this study is historical discovery/replication evidence, not pristine
prospective confirmation.

A failure closes this exact five-feature logistic architecture. No C tuning,
feature addition/removal, interactions, market subgroup, side subgroup, or
threshold rescue.
