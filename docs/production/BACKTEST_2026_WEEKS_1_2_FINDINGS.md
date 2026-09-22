# 2026 Weeks 1–2 graded backtest — findings

Source: `data/market_track_record/boards/2026_wk01.csv`, `2026_wk02.csv`
Full graded rows: `data/market_track_record/graded/2026_wk01_wk02_graded.csv` (866 rows)
Reproduce: `PYTHONPATH=. python scripts/operations/backtest_full_report_v1.py --season 2026 --weeks 1,2`

One bet per player-market. The median consensus line selects the intended
model side; W/L, Vegas error and units are then attached to the nearest
captured real-book quote whose own line agrees with that side. Equidistant
cross-book ties use the lexicographically smallest normalized provider
`book` key (`book_title` fallback), never line direction or price. Duplicate
rows at one canonical book+line use the least
favorable captured American price; an unresolved same-book/equidistant-line
ambiguity fails closed. The rule is outcome-independent and row-order
invariant. On Weeks 1–2 it yields 866 identity-resolved bets and no
abstentions. `anytime_td` is not graded (standing policy), which leaves
2,908 archived rows unmeasured. Zero pushes.

## Headline

866 bets, **440-426 (50.8%)**, −41.94 units. Model MAE 18.08 vs line 17.13;
the model is closer than the line on 45.0% of bets. Week 1 is 226-212; Week 2
is 214-214.

## The board is one market

| market | W-L | win% | units | model MAE | line MAE | model bias | line bias | closer |
|---|---|---|---|---|---|---|---|---|
| pass_yards | 38-20 | 65.5% | +13.95 | 57.53 | 61.10 | −1.92 | −3.55 | .603 |
| rush_yards | 78-73 | 51.7% | −2.65 | 20.00 | 17.92 | −6.60 | −1.41 | .430 |
| receptions | 148-145 | 50.5% | −22.18 | 1.72 | 1.59 | −0.63 | −0.10 | .454 |
| rush_rec_yards | 33-34 | 49.3% | −4.84 | 31.84 | 26.34 | −20.20 | −1.07 | .433 |
| rec_yards | 143-154 | 48.1% | −26.21 | 22.45 | 21.40 | −7.27 | −4.12 | .431 |

`pass_yards` is the only market where the model is more accurate than the
line. Remove it and the remaining 808 bets go **402-406 (49.8%)**.

Within QB, `pass_yards` is 38-20 and `rush_yards` is exactly 26-26. Whatever
is working is the passing-yards path specifically — the one carrying the
promoted M89/M90 synthesis — not "the QB model".

## Finding 1 — the stated probability is not a probability

| stated band | bets | mean stated | actual | gap |
|---|---|---|---|---|
| ≤50% | 103 | 45.2% | 49.5% | +4.3pp |
| 50–55% | 62 | 52.8% | 50.0% | −2.8pp |
| 55–60% | 126 | 57.5% | 45.2% | −12.3pp |
| 60–65% | 112 | 62.4% | 53.6% | −8.8pp |
| 65–70% | 91 | 67.4% | 57.1% | −10.3pp |
| **>70%** | **372** | **81.4%** | **50.8%** | **−30.6pp** |

Above roughly 55% stated, realized hit rates remain far below the stated
probabilities and are not monotonic with confidence. The largest band is also
the most confident and the most populous — 43% of the board sits at a mean
stated 81.4% and returns 189-183, −18.72 units.

Overconfidence is not concentrated anywhere. In the >70% band: QB 19-10,
WR 79-68, RB 72-76, TE 19-29.

## Finding 2 — the mechanism is a distribution that is too narrow

Realized projection-error spread against the model's own stated `model_sd`:

| market | stated sd | realized error sd | ratio |
|---|---|---|---|
| pass_yards | 54.44 | 78.67 | 1.45× |
| rush_yards | 15.23 | 28.13 | 1.85× |
| rec_yards | 17.40 | 31.18 | 1.79× |
| receptions | 1.62 | 2.17 | 1.33× |

The simulated outcome distribution is 1.3–1.9× too tight. A distribution that
narrow pushes P(over) and P(under) toward the extremes, which is exactly the
overconfidence in Finding 1. This is one defect, not two.

## Finding 3 — the model's own declared edge does not rank its bets

Quintiles of |`edge_pct`|:

| quintile | W-L | win% | units | model bias |
|---|---|---|---|---|
| Q1 smallest | 82-92 | 47.1% | −23.50 | −0.21 |
| Q2 | 97-76 | 56.1% | +7.24 | −0.71 |
| Q3 | 83-90 | 48.0% | −18.39 | −3.69 |
| Q4 | 86-87 | 49.7% | −11.42 | −5.40 |
| Q5 largest | 92-81 | 53.2% | +4.13 | −17.76 |

Not monotonic. The largest-edge quintile also carries the largest projection
bias (−17.76): a big declared "edge" is mostly the model being wrong about the
number, not disagreeing usefully with the market. Ranking or sizing by
`edge_pct` has no support here.

### Cluster-aware inference check

The board is not 866 independent trials: many bets share the same NFL game and
are mechanically related. The significance layer therefore centers each bet
by its own captured-price break-even probability, sums those residuals inside
each game, and tests across independent game clusters before BH/FDR correction.

All 38 slices meeting the sample gate were cluster-testable; **0 of 38 survive
BH-FDR at q=0.10**. QB pass yards has a raw game-cluster-aware one-sided
p-value of **0.0307**, but it does **not** survive the multiple-comparisons
gate. That makes it an encouraging frozen prospective lead, not a certified
bet-selection edge.

## Finding 4 — `rush_rec_yards` looks like a construction defect

| | mean |
|---|---|
| model projection | 53.7 |
| sportsbook line | 72.9 |
| actual | 73.9 |

The line is essentially unbiased (−1.07). The model is **19 yards low on a
73-yard market**, roughly 26%, and consequently picks UNDER on 59 of 67 bets.
A systematic one-directional shortfall of that size on a market that is the
sum of two components the model already projects separately is not variance.
`rush_yards` bias is −6.60 and `rec_yards` is −7.27; summed that is about
−13.9, so roughly −6 yards of the gap is unexplained by the parts.

## Finding 5 — TE is a no-edge market, not an inverted one

TE overall 60-82 (42.3%). Week 1 36-36, Week 2 24-46.

Week 2 TE: mean projection minus line **−0.07**, median |projection − line|
**1.16 yards** on a ~16-yard line. The model is agreeing with the number and
then taking whichever side the rounding lands on. TE bias is only −0.29 in
Week 2, so this is not the under-projection problem — it is an absence of
signal. Week 2 OVERs went 8-25 while UNDERs went 16-21.

The 60-82 aggregate is poor descriptively, but the rows are clustered by
game and should not be treated as 142 independent trials. The cluster-aware
slice test does not produce a multiple-comparisons-surviving TE signal. The
mechanism still does not require a sign error: the model is essentially on the
market number, leaving no demonstrated edge to harvest at that line proximity.

## Finding 6 — a systemic low bias drives a 2:1 UNDER book

Overall model bias −5.55 vs line bias −2.01. By side: UNDER bets carry a
−9.82 bias, OVER bets +2.59. The board is 568 UNDER to 298 OVER because the
projections run low, not because the model found 568 unders.

This replicates the under-projection finding from the 2024/2025 clean-cohort
re-grade, now on live 2026 slates.

## What this does and does not license

Nothing here is a tuning instruction. The calibration and `model_sd` findings
are defects with identified mechanisms and should be evaluated authority by
authority rather than globally rescaled from two weeks. The `pass_yards`
result is one market over two weeks at n=58 across 30 game clusters; its raw
cluster-aware p-value is encouraging but does not survive FDR. It needs Weeks
3–6 under frozen parameters before it can support a stronger claim.
