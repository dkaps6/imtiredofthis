# Market-Relative Bet Selector V1 — Pre-Result Implementation Amendment

**STATUS: FROZEN BEFORE ANY STAGE-1 RESULT.**

This amendment resolves implementation details that were implicit in
`MARKET_RELATIVE_BET_SELECTOR_V1_PLAN.md`. It does not change the research
question, eligible markets, source hierarchy, or stopping rules.

## Stage-1 source scope for the first execution

Only markets with symmetric 2024/2025 exact historical authority are eligible
for the first V1 Stage-1 execution:

- QB `pass_yards` — `QB_PASS_SYNTHESIS_V1` exact M89 OOS authority;
- TE `rec_yards` — `TE_R5P_PRODUCTION_MODEL_V1` exact OOS authority;
- TE `receptions` — `TE_R5P_PRODUCTION_MODEL_V1` exact OOS authority.

WR is not eligible for the symmetric V1 holdout because WR-R15's frozen
authority forbids 2025 confirmation. RB is not eligible because the promoted RB
stack has no 2024-2025 retrospective authority. Neither may be rescued with a
generic historical model.

Sportsbook history is rebuilt from the repo's existing free Action
Network-derived archive path. The archived latest eligible DK/FD line per book
is treated as the historical market snapshot for this retrospective
incremental-information study. It is not claimed to be a live-open or
timestamp-specific intraday state.

## Consensus market anchor

For each exact `game_id + player_clean_key + market`:
- use all identity-clean eligible historical book rows;
- consensus line = median of the available book lines;
- one-book groups are retained and explicitly marked `consensus_book_count=1`;
- no outcome information enters consensus construction.

The Stage-1 level-information test uses only the consensus line. Book-specific
prices are reserved for later stages.

## Beta fit

For each market and training season, define:

- `x = model_projection - consensus_line`
- `y = actual_stat - consensus_line`

Fit the no-intercept least-squares coefficient:

`beta_raw = sum(x*y) / sum(x*x)`

Then apply the frozen constraint:

`beta = clip(beta_raw, 0, 1)`

If `sum(x*x) <= 1e-12`, the direction fails closed as
`DEGENERATE_MODEL_GAP`.

No alternate loss, intercept, transform, sign inversion, or subgroup fit is
allowed after results are seen.

## Stage-1 support floor

A fit/test direction is scoreable only if:
- training rows >= 100;
- test rows >= 100;
- training game clusters >= 20;
- test game clusters >= 20.

Otherwise the direction is `INSUFFICIENT_SUPPORT`.

## Primary held-out metric

For each test row:

- baseline absolute error = `abs(consensus_line - actual)`
- candidate absolute error = `abs(market_relative_fair_line - actual)`
- paired improvement = baseline absolute error - candidate absolute error

Positive improvement favors the candidate.

A direction passes the point criterion only if mean paired improvement > 0.

## Game-cluster bootstrap

- cluster unit: `game_id`;
- resamples: 10,000;
- deterministic seed: 20260930;
- sample test-season game clusters with replacement;
- include every row belonging to each sampled cluster, preserving repeated
  clusters when selected multiple times;
- beta remains frozen from the training season; it is not refit in bootstrap;
- report percentile 95% CI of the mean paired improvement.

A direction passes the inference criterion only if the 2.5th percentile is
strictly > 0.

## Market Stage-1 gate

A market is `STAGE1_INCREMENTAL_LEVEL_SIGNAL_PASS` only if BOTH directions:
- are scoreable;
- have constrained beta > 0;
- have held-out candidate MAE < market-consensus MAE;
- have mean paired improvement > 0;
- have bootstrap 95% CI lower bound > 0.

Otherwise the market is
`NO_VERIFIED_INCREMENTAL_MODEL_LEVEL_SIGNAL_V1` or
`SOURCE_REPLAY_BLOCKED` / `INSUFFICIENT_SUPPORT` as applicable.

No pooled rescue and no Stage-2 probability work for a failed market.
