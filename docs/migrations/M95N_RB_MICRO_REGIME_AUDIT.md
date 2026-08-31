# M95N — Conditional Player-Game Micro-Regime Audit

Research-only diagnostic. M95M showed that M95K's stable-workhorse reranking is cross-season nonstationary: 20+ improved strongly in 2025 W13-18 but reversed in 2023 W13-18, while 25+ was unstable in both same-window checks.

## Question

Are we forcing one shared stable-workhorse response function across player-games that should be treated differently? Specifically, does agreement between current pregame context and historical feed/ceiling information identify a stable micro-regime, while disagreement requires a conditional response rather than a universal rerank?

## Fixed audit design

No model is fit. No candidate probabilities are changed. No feature or coefficient search is allowed.

For stable workhorses only, evaluate three scopes:

- 2023 Weeks 13-18 opened M95L confirmation;
- 2025 Weeks 13-18 same-window comparison;
- 2025 full reused research population for context.

For each target (20+ and 25+):

1. `baseline_context_rank` = within-scope percentile rank of the frozen M95F baseline tail probability. This represents the existing pregame football/role context model.
2. `feed_score` = fixed average percentile rank of four pregame historical feed/ceiling fields.
3. Split both at the median without using outcomes.
4. Assign four interpretable micro-regimes:
   - `aligned_high`: current context high + feed history high;
   - `context_only`: current context high + feed history low;
   - `history_only`: current context low + feed history high;
   - `aligned_low`: both low.

Secondary descriptive axes are fixed before the run: projected team volume, backfield concentration, matchup/run-defense weakness, and recent role momentum. They are diagnostics only and cannot select a model in M95N.

## Integrity rules

- pregame inputs only;
- outcome labels only for evaluation after regime assignment;
- no sportsbook inputs;
- no new model fit;
- no feature search;
- no coefficient search;
- no production change;
- M95K remains failed as a universal frozen architecture after M95L;
- opened 2023 labels may explain failure but may not be used to retune M95K and call it confirmation.

## Interpretation rule

Evidence for micro-regime dependence exists if the aligned-high vs aligned-low ordering is directionally stable across 2023 and 2025 same-window populations while the preferred side of the discordant `context_only` vs `history_only` pair flips by season. Such a result would support testing a separately precommitted agreement-gated architecture in the next migration, not selecting whichever expert wins on opened 2023 labels.
