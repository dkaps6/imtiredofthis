# Market-Relative Bet Selector V1 — Frozen Plan

**STATUS: FROZEN BEFORE ANY NEW RESULT. RESEARCH ONLY. NO PRODUCTION CHANGE.**

## Why this exists

The current football model can disagree strongly with sportsbook lines, but that disagreement is not itself proof of betting edge.

Already-established evidence:
- 2026 Weeks 1-3 production postmortem: raw stated probabilities are severely overconfident and raw edge does not rank outcomes monotonically.
- historical empirical-MC diagnosis: the old STRONG gate fired on the large majority of rows while realized performance stayed near market break-even;
- pure isotonic probability recalibration sharply reduced coverage but failed its held-out ROI bar in both directions;
- broad situational/edge slices have not produced a stable, validated selector.

Therefore this study does **not** retune the old edge threshold and does **not** fit another scalar probability calibration. It asks a more basic downstream question:

> After the sportsbook market has set a line, does the model's independent football projection add incremental information beyond that market line, and if so, how much of the model-market disagreement is actually trustworthy?

The football model remains upstream and sportsbook-independent. Sportsbook information enters only in this downstream betting-selection study.

## Non-goals

This plan does not:
- change any football feature, coefficient, ensemble weight, simulation, or projection;
- feed sportsbook information upstream into football generation;
- optimize a threshold against realized ROI;
- use 2026 Weeks 1-3 outcomes to fit or tune the selector;
- reopen the failed isotonic-calibration lane;
- reopen broad situational slice mining;
- use CLV that was not legitimately captured pregame;
- spend OddsAPI credits or acquire paid data.

## Data readiness gate

Before any scientific score, build an authority manifest by market.

A market is scoreable only if historical rows provide, for the exact same pregame player-market-offer identity:
1. leakage-safe football projection from a historically legitimate reconstruction;
2. pregame sportsbook line and side prices;
3. exact game/player/market identity;
4. final outcome;
5. no known publication/availability/quarantine failure;
6. a documented statement of which current production authorities are and are not historically reproduced.

Markets that cannot meet this gate are reported as **SOURCE/REPLAY BLOCKED** rather than silently approximated.

2026 Weeks 1-3 may be used only as already-exposed descriptive context. They may not fit coefficients, choose features, choose thresholds, or alter gates.

## Market anchor

For each player-market-game, construct a **consensus market anchor** from the eligible pregame books available at the frozen snapshot.

Primary consensus line:
- median eligible book line for that exact player-market-game.

If multiple books share the same line, prices may be summarized for execution diagnostics, but the football model is never modified by sportsbook data.

A single-book offer is then evaluated relative to:
- the consensus line;
- the model projection;
- that book's actual line and price.

This separates two distinct sources of potential value:
1. **information value** — whether the model adds anything beyond consensus;
2. **execution value** — whether a specific book offers a better line/price than consensus.

## Stage 1 — incremental model information beyond market

For each scoreable market, define:

- `market_line` = frozen consensus line;
- `model_gap = model_projection - market_line`;
- `actual_residual = actual_stat - market_line`.

Fit exactly one market-level coefficient on the training season:

`actual_residual = beta * model_gap`

Frozen constraints:
- no intercept in V1;
- `beta` constrained to `[0, 1]`;
- no sign inversion;
- no position subgroup, role subgroup, side subgroup, week subgroup, or top-N rescue;
- no alternate nonlinear transform after outcomes are seen.

Interpretation:
- `beta = 0`: the model adds no usable level information beyond the market;
- `beta = 1`: trust the full model-market disagreement;
- `0 < beta < 1`: shrink the model disagreement toward the market.

Candidate downstream fair-stat center:

`market_relative_fair_line = market_line + beta * model_gap`

This is a **betting-layer fair line**, not a replacement football projection.

## Genuine holdout

Run both directions where authority permits:
- fit 2024, test 2025;
- fit 2025, test 2024.

A market passes the Stage-1 incremental-information gate only if, in both directions:
1. held-out candidate MAE is lower than the market-line baseline MAE;
2. the candidate improvement is positive under a game-cluster bootstrap;
3. the 95% bootstrap CI for paired MAE improvement excludes zero in the favorable direction;
4. beta is > 0 without violating the frozen [0,1] constraint.

If either direction fails, that market is **NO VERIFIED INCREMENTAL MODEL LEVEL SIGNAL** for V1 and no bet-selector logic is built from it.

No pooled rescue is allowed.

## Stage 2 — honest uncertainty around the market-relative residual

Only markets that pass Stage 1 may continue.

On the training season only, estimate the empirical distribution of:

`candidate_residual = actual_stat - market_relative_fair_line`

V1 uses a market-level empirical residual distribution only.
No player-specific, role-specific, side-specific, or edge-bin-specific variance fitting.

On held-out rows, translate the candidate fair-stat center and frozen residual distribution into OVER/UNDER probabilities at each real book line.

This creates a downstream probability tied to **historically realized error around the market-relative fair line**, rather than treating raw Monte Carlo disagreement as automatically calibrated betting confidence.

## Stage 3 — sparse actionability test

For every eligible book offer in the held-out season:
- compute raw break-even probability from actual price;
- compute candidate side probability from Stage 2;
- compute candidate EV.

The primary selector is intentionally conservative:

**ACTIONABLE only when the lower bound of a game-cluster bootstrap confidence interval for candidate EV is > 0.**

Otherwise: **PASS**.

No fixed +3%, +5%, +10%, top-K, confidence-tier, or unit-sizing threshold is fit in V1.

The study must report:
- percentage of offers passed;
- percentage actionable;
- number of unique player-markets;
- number of game clusters;
- win rate;
- units and ROI;
- calibration/Brier score;
- actionability by market;
- comparison with raw production EV ranking;
- whether larger candidate EV ranks realized results monotonically.

A useful selector is expected to be sparse. High abstention is not a failure.

## Execution-quality diagnostic

Separately from the primary science, measure whether actionability is concentrated in cases where the chosen book is favorable versus consensus:
- better line than consensus;
- same line but better price;
- market-wide disagreement/dispersion.

These are diagnostics only in V1. They cannot rescue a failed Stage-1 or Stage-3 primary result.

## Primary V1 disposition

### `MARKET_RELATIVE_SELECTOR_V1_PASS`
Only if at least one market:
1. passes Stage 1 in both holdout directions;
2. produces a non-empty actionable set in both holdout directions;
3. has positive held-out ROI in both directions;
4. has game-cluster bootstrap evidence consistent with positive EV in both directions;
5. does not depend on a single week/game cluster for >20% of actionable rows.

### `MARKET_RELATIVE_SELECTOR_V1_NULL`
If source-ready markets fail to show incremental model information or no market produces a robust sparse actionable set.

### `MARKET_RELATIVE_SELECTOR_V1_SOURCE_BLOCKED`
If historically legitimate reconstruction is insufficient to answer the question.

No rescue variants after terminal disposition.

## Prospective 2026 shadow

Regardless of historical disposition, any future live candidate must remain **shadow-only** first.

For Week 4+:
- freeze the current football projection;
- freeze the consensus line and each eligible book offer at run time;
- freeze candidate fair line/probability/actionability before kickoff;
- never rewrite the lock after outcomes;
- grade only after final participation/outcome evidence is available.

2026 Weeks 1-3 are never retroactively used to choose the Week-4+ rule.

A production betting gate would require a separate promotion decision after sufficient prospective support.

## Relationship to prior failed lanes

This study is intentionally not:
- isotonic recalibration of raw model probability;
- generic distribution widening;
- edge-threshold tuning;
- STRONG/LEAN revival;
- situational subgroup mining;
- an upstream Vegas blend inside the football model.

It is a downstream test of whether independent football disagreement has **incremental value over the market**, followed by an honest abstaining execution layer.

## Cost boundary

Use existing historical sportsbook artifacts and existing live Full Slate odds snapshots only.
No extra OddsAPI pull is authorized solely for this research.
