# 2026 Weeks 1–2 production-decision backtest — findings

Source boards:
- `data/market_track_record/boards/2026_wk01.csv`
- `data/market_track_record/boards/2026_wk02.csv`

Canonical selected settlement rows:
- `data/market_track_record/graded/2026_wk01_wk02_graded.csv`

Reproduce:
`PYTHONPATH=. python scripts/operations/backtest_full_report_v1.py --season 2026 --weeks 1,2`

## What counts as a bet

This scorecard now reproduces the deployed downstream **Best Snapshot** decision,
not mean-vs-line direction.

For every concrete captured book+line offer:

1. compute OVER and UNDER expected ROI from the archived side-specific
   `fair_prob` and captured American price;
2. choose the higher-EV side, matching production;
3. across real offers for the same player-market, keep the highest-EV offer;
4. if the best EV is nonpositive, production says PASS and no bet is graded;
5. an exact best-EV tie that represents different wagers fails closed rather
   than using row order;
6. an identical-wager tie uses a deterministic normalized provider key;
7. `consensus_line` is diagnostic only and does not choose the wager.

Sportsbook information remains downstream only. None of these rules alter the
football projection.

## Settlement integrity

Roster identity alone does not prove sportsbook action.

For selected player props absent from the weekly stats table:

- positive PFR snap participation -> verified zero outcome;
- explicit inactive/DNP with no participation on the captured DraftKings or
  FanDuel offer -> VOID;
- missing/ambiguous participation evidence -> fail closed as unresolved.

Canonical W1/W2 settlement inventory:

- selected settlement rows: **810**
- decided bets: **802**
- void DNP rows: **8**
- stats-table outcomes: **791**
- snap-confirmed verified-zero outcomes: **11**
- additional positive-EV rows still unresolved: **2** (Joshua Palmer receiving
  yards and receptions, Week 2); they are excluded from W/L and units.

`anytime_td` remains ungraded by standing project policy.

## Headline

**802 decided bets: 397-405 (49.5%), -44.09 units.**

- Week 1: **204-205**, -20.92u
- Week 2: **193-200**, -23.17u
- model MAE: **18.42**
- selected sportsbook-line MAE: **17.38**
- model closer than selected line: **44.9%**

The old 438-428, 439-427 and 440-426 records are superseded. They graded a
different wager-selection or settlement convention and are not production
track records.

## By market

| market | W-L | win% | units | model MAE | line MAE | model bias | line bias | closer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pass_yards | 36-17 | 67.9% | +14.99 | 56.50 | 60.88 | -1.21 | -2.65 | .623 |
| rush_yards | 71-66 | 51.8% | -1.55 | 21.17 | 18.84 | -7.44 | -1.55 | .438 |
| rush_rec_yards | 33-33 | 50.0% | -3.81 | 32.15 | 26.55 | -20.67 | -1.15 | .439 |
| rec_yards | 132-145 | 47.7% | -27.65 | 22.72 | 21.51 | -7.95 | -4.30 | .419 |
| receptions | 125-144 | 46.5% | -26.07 | 1.72 | 1.57 | -0.74 | -0.20 | .454 |

QB pass yards remains the standout early market after production-decision
alignment: **36-17, +14.99u**. Removing pass yards leaves the other markets
**361-388 (48.2%)**.

That is encouraging, not a promotion/tuning instruction. It is still only
53 bets across 30 game clusters.

## Finding 1 — probabilities remain materially overconfident

| selected-side stated probability | n | mean stated | actual hit | record | units |
|---|---:|---:|---:|---:|---:|
| <=50% | 18 | 45.7% | 16.7% | 3-15 | -11.40 |
| 50-55% | 65 | 53.6% | 40.0% | 26-39 | -12.64 |
| 55-60% | 130 | 57.4% | 50.0% | 65-65 | -3.44 |
| 60-65% | 129 | 62.4% | 51.2% | 66-63 | -3.28 |
| 65-70% | 89 | 67.5% | 57.3% | 51-38 | +7.47 |
| >70% | 371 | 81.3% | 50.1% | 186-185 | -20.81 |

The >70% group is the clearest problem: stated probability averages 81.3% but
realizes essentially 50/50.

## Finding 2 — simulated distributions still look too narrow

Realized projection-error SD versus the model's stated `model_sd`:

| market | stated SD | realized error SD | ratio |
|---|---:|---:|---:|
| pass_yards | 54.62 | 78.83 | 1.44x |
| rush_yards | 15.65 | 29.23 | 1.87x |
| rec_yards | 17.46 | 31.16 | 1.78x |
| receptions | 1.62 | 2.14 | 1.32x |

This remains a strong cross-position diagnostic, but two live weeks do **not**
license a global SD rescale. Existing position-specific distribution
authorities must be evaluated independently.

## Finding 3 — declared probability edge still does not rank outcomes monotonically

Quintiles of absolute fractional `edge_pct` after fixing its units:

| quintile | n | W-L | win% | units | model bias |
|---|---:|---:|---:|---:|---:|
| Q1 smallest | 161 | 68-93 | 42.2% | -33.03 | +2.30 |
| Q2 | 160 | 87-73 | 54.4% | +5.20 | -3.74 |
| Q3 | 160 | 78-82 | 48.8% | -10.67 | -4.83 |
| Q4 | 160 | 78-82 | 48.8% | -11.59 | -5.59 |
| Q5 largest | 161 | 86-75 | 53.4% | +6.01 | -18.35 |

The ordering is non-monotonic. The largest-edge quintile also has by far the
largest negative projection bias. Raw `edge_pct` is therefore not supported
as a confidence-ranking or staking variable from this sample.

## Finding 4 — cluster-aware inference finds no certified slice

The board contains mechanically related props within the same NFL games.
Inference therefore centers each decided bet by its own price-implied
break-even probability, sums residuals within game, tests across game clusters,
then applies BH/FDR.

With corrected fractional edge bins:

- **50** slices meet the sample/cluster gates;
- **0 of 50 survive BH-FDR at q=0.10**;
- QB pass yards raw game-cluster one-sided p = **0.0119**, but it does not
  survive the multiple-comparisons gate;
- TE overall raw cluster p = **0.9449**;
- TE receiving yards raw cluster p = **0.9727**.

So QB pass yards remains a prospective lead, not a certified betting edge.

## Finding 5 — rush + receiving yards still has a construction/bias concern

For the 66 production-selected decided `rush_rec_yards` bets:

- mean model projection: **53.79**
- mean selected line: **73.30**
- mean actual: **74.45**
- model bias: **-20.67**
- selected-line bias: **-1.15**
- side mix: **59 UNDER / 7 OVER**

The selected market line is close to unbiased while the model is about 21
yards low. This remains a seam/construction audit target, not proof of one
specific cause.

## Finding 6 — TE remains the clearest positional weakness

TE production-selected decided bets:

- overall: **58-75 (43.6%), -21.45u**
- receiving yards: **28-41, -16.09u**
- receptions: **30-34, -5.37u**
- Week 1: **32-37**
- Week 2: **26-38**

Week-2 TE remains close to the market number on average:

- mean projection minus selected line: **-0.21**
- median absolute projection-line gap: **1.26**
- model bias: **-0.53**
- OVER: **6-17**
- UNDER: **20-21**

This is not evidence of a simple sign inversion. Receiving yards is materially
worse than receptions, and the model often sits near the posted number while
failing to separate the better side.

That matters because independent historical evidence says TE target-share
state is highly persistent, while TE-R1 attributed about 45% of receiving-yard
error mass to targets/entitlement and about 55% to catch-rate + YPR efficiency.
PR #627 also begins feeding 2026 strict-prior snap participation into TE-R5P
starting Week 3 without refitting coefficients.

The next TE diagnostic should therefore separate **entitlement/opportunity
error from downstream efficiency/distribution error**, rather than label the
entire TE authority "bad."

## Finding 7 — the board still carries a systemic low projection bias

Across decided bets:

- overall model bias: **-6.05**
- selected-line bias: **-2.09**
- OVER bets: 178, model bias **+4.81**
- UNDER bets: 624, model bias **-9.14**

The production Best Snapshot still produces a heavily UNDER-skewed board, and
large negative projection bias is concentrated there. Treat this as a
cross-position diagnostic, not as evidence that UNDER itself is predictive.

## What this does and does not license

Nothing here authorizes a model retune.

The repaired production scoreboard says:

- keep QB passing-yard parameters frozen and score prospectively;
- investigate TE entitlement-vs-efficiency first;
- audit rush+receiving construction separately;
- evaluate probability/distribution calibration authority-by-authority;
- do not use sportsbook lines upstream;
- do not globally rescale distributions from two live weeks;
- do not interpret raw `edge_pct` as trustworthy ranking signal yet.
