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

Before EV selection, the historical grader also replays the original downstream
publication blockers from evidence pinned in immutable commit
`8133975f505365234dbdb75ff0fad0c715f68e31`:

- Week 1: KICKED_OFF_LOCKED teams LAR/NE/SEA/SF are ineligible; the recovered
  priced board already contains zero rows for those teams. The source run also
  recorded zero core unresolved sportsbook-player rows and zero current-roster
  mismatches after identity repair, with definitive-unavailable players excluded
  before opportunity construction.
- Week 2: the exact source workbook blocks Amon-Ra St. Brown receiving yards
  and receptions as `UNMATCHED_CURRENT_ROSTER`, and Deebo Samuel Sr.
  rush+receiving yards as `LINEAGE_NOT_RESOLVED` / `RESEARCH ONLY`.
- The separate final-board quarantine remains independently enforced (including
  the Week-2 Carson Wentz quarantine).

These gates affect only whether a priced offer was publishable as a production
bet. They do not alter the football projection or use outcomes.

## Settlement integrity

Roster identity alone does not prove sportsbook action.

For selected player props absent from the weekly stats table:

- positive PFR snap participation -> verified zero outcome;
- explicit inactive/DNP with no participation on the captured DraftKings or
  FanDuel offer -> VOID;
- missing/ambiguous participation evidence -> fail closed as unresolved.

Canonical W1/W2 settlement inventory:

- selected settlement rows: **805**
- decided bets: **797**
- void DNP rows: **8**
- stats-table outcomes: **786**
- snap-confirmed verified-zero outcomes: **11**
- additional positive-EV rows still unresolved: **2** (Joshua Palmer receiving
  yards and receptions, Week 2); they are excluded from W/L and units.

`anytime_td` remains ungraded by standing project policy.

## Headline

**797 decided bets: 395-402 (49.6%), -42.86 units.**

- Week 1: **204-205**, -20.92u
- Week 2: **191-197**, -21.94u
- model MAE: **18.33**
- selected sportsbook-line MAE: **17.30**
- model closer than selected line: **44.9%**

The old 438-428, 439-427, 440-426, 397-405 and intermediate 396-404 records are superseded. They graded a different wager-selection, settlement, quarantine, or publication-gate convention and are not the final production track record.

## By market

| market | W-L | win% | units | model MAE | line MAE | model bias | line bias | closer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pass_yards | 36-16 | 69.2% | +15.99 | 56.03 | 60.65 | -2.79 | -4.10 | .635 |
| rush_yards | 70-66 | 51.5% | -2.45 | 21.32 | 18.96 | -7.49 | -1.59 | .434 |
| rush_rec_yards | 32-33 | 49.2% | -4.69 | 32.54 | 26.79 | -21.09 | -1.33 | .431 |
| rec_yards | 132-144 | 47.8% | -26.65 | 22.53 | 21.37 | -7.71 | -4.09 | .420 |
| receptions | 125-143 | 46.6% | -25.07 | 1.71 | 1.57 | -0.73 | -0.19 | .455 |

QB pass yards remains the standout early market after production-decision
alignment: **36-16, +15.99u**. Removing pass yards leaves the other markets
**359-386 (48.2%)**.

That is encouraging, not a promotion/tuning instruction. It is still only
52 bets across 30 game clusters.

## Finding 1 — probabilities remain materially overconfident

| selected-side stated probability | n | mean stated | actual hit | record | units |
|---|---:|---:|---:|---:|---:|
| <=50% | 18 | 45.7% | 16.7% | 3-15 | -11.40 |
| 50-55% | 65 | 53.6% | 40.0% | 26-39 | -12.64 |
| 55-60% | 129 | 57.5% | 50.4% | 65-64 | -2.44 |
| 60-65% | 128 | 62.4% | 50.8% | 65-63 | -4.15 |
| 65-70% | 89 | 67.5% | 57.3% | 51-38 | +7.47 |
| >70% | 368 | 81.4% | 50.3% | 185-183 | -19.70 |

The >70% group is the clearest problem: stated probability averages 81.3% but
realizes only about 50/50 in this two-week sample.

## Finding 2 — simulated distributions still look too narrow

Realized projection-error SD versus the model's stated `model_sd`:

| market | stated SD | realized error SD | ratio |
|---|---:|---:|---:|
| pass_yards | 54.85 | 78.75 | 1.44x |
| rush_yards | 15.71 | 29.33 | 1.87x |
| rec_yards | 17.43 | 30.95 | 1.78x |
| receptions | 1.61 | 2.14 | 1.33x |

This remains a strong cross-position diagnostic, but two live weeks do **not**
license a global SD rescale. Existing position-specific distribution
authorities must be evaluated independently.

## Finding 3 — declared probability edge still does not rank outcomes monotonically

Quintiles of absolute fractional `edge_pct` after fixing its units:

| quintile | n | W-L | win% | units | model bias |
|---|---:|---:|---:|---:|---:|
| Q1 smallest | 160 | 68-92 | 42.5% | -32.03 | +1.80 |
| Q2 | 160 | 86-74 | 53.8% | +3.32 | -3.78 |
| Q3 | 158 | 78-80 | 49.4% | -8.67 | -4.41 |
| Q4 | 159 | 78-81 | 49.1% | -10.81 | -5.58 |
| Q5 largest | 160 | 85-75 | 53.1% | +5.33 | -18.48 |

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
- QB pass yards raw game-cluster one-sided p = **0.0081**, but it does not
  survive the multiple-comparisons gate;
- TE overall raw cluster p = **0.9449**;
- TE receiving yards raw cluster p = **0.9727**.

So QB pass yards remains a prospective lead, not a certified betting edge.

## Finding 5 — rush + receiving yards still has a construction/bias concern

For the 65 production-selected decided `rush_rec_yards` bets:

- mean model projection: **53.97**
- mean selected line: **73.73**
- mean actual: **75.06**
- model bias: **-21.09**
- selected-line bias: **-1.33**
- side mix: **58 UNDER / 7 OVER**

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

- overall model bias: **-6.10**
- selected-line bias: **-2.13**
- OVER bets: 177, model bias **+4.38**
- UNDER bets: 620, model bias **-9.09**

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
