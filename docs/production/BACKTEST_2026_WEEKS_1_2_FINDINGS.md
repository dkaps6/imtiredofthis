# 2026 Weeks 1–2 graded backtest — findings

Source: `data/market_track_record/boards/2026_wk01.csv`, `2026_wk02.csv`
Full graded rows: `data/market_track_record/graded/2026_wk01_wk02_graded.csv` (866 rows)
Reproduce: `PYTHONPATH=. python scripts/operations/backtest_full_report_v1.py --season 2026 --weeks 1,2`

One bet per player-market. The median consensus line selects the intended
model side; W/L, Vegas error and units are then attached to the nearest
captured real-book quote whose own line agrees with that side. If no
compatible quote exists the player-market abstains. The rule is
outcome-independent and row-order invariant. On Weeks 1–2 it repaired 12
straddling rows and created no abstentions among the 866 identity-resolved
bets. `anytime_td` is not graded (standing policy), which leaves 2,908
archived rows unmeasured. Zero pushes.

## Headline

866 bets, **439-427 (50.7%)**, −47.96 units. Model MAE 18.08 vs line 17.11;
the model is closer than the line on 44.8% of bets.

## The board is one market

| market | W-L | win% | units | model MAE | line MAE | model bias | line bias | closer |
|---|---|---|---|---|---|---|---|---|
| pass_yards | 38-20 | 65.5% | +13.60 | 57.53 | 61.21 | −1.92 | −4.48 | .603 |
| rush_yards | 78-73 | 51.7% | −3.58 | 20.00 | 17.78 | −6.60 | −2.12 | .430 |
| receptions | 147-146 | 50.2% | −25.05 | 1.72 | 1.59 | −0.63 | −0.14 | .454 |
| rush_rec_yards | 33-34 | 49.3% | −4.87 | 31.84 | 26.34 | −20.20 | −1.22 | .418 |
| rec_yards | 143-154 | 48.1% | −28.06 | 22.45 | 21.40 | −7.27 | −4.52 | .428 |

`pass_yards` is the only market where the model is more accurate than the
line. Remove it and the remaining 808 bets go **401-407 (49.6%)**.

Within QB, `pass_yards` is 38-20 and `rush_yards` is exactly 26-26. Whatever
is working is the passing-yards path specifically — the one carrying the
promoted M89/M90 synthesis — not "the QB model".

## Finding 1 — the stated probability is not a probability

| stated band | bets | mean stated | actual | gap |
|---|---|---|---|---|
| ≤50% | 96 | 45.3% | 50.0% | +4.7pp |
| 50–55% | 69 | 52.8% | 53.6% | +0.8pp |
| 55–60% | 140 | 57.5% | 47.1% | −10.4pp |
| 60–65% | 113 | 62.3% | 51.3% | −11.0pp |
| 65–70% | 91 | 67.6% | 54.9% | −12.6pp |
| **>70%** | **357** | **81.3%** | **50.4%** | **−30.9pp** |

Above roughly 55% stated, the number carries no information: every band lands
near 50% regardless of what the model claimed. The largest band is also the
most confident and the most populous — 41% of the board sits at a mean stated
81.3% and returns 180-177, −22.43 units.

Overconfidence is not concentrated anywhere. In the >70% band: QB 19-9,
WR 73-65, RB 70-75, TE 18-28.

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
| Q1 smallest | 83-91 | 47.7% | −22.99 | +0.64 |
| Q2 | 97-76 | 56.1% | +5.32 | −2.30 |
| Q3 | 85-88 | 49.1% | −14.29 | −3.25 |
| Q4 | 84-89 | 48.6% | −15.49 | −5.65 |
| Q5 largest | 90-83 | 52.0% | −0.50 | −17.21 |

Not monotonic. The largest-edge quintile also carries the largest projection
bias (−17.21): a big declared "edge" is mostly the model being wrong about the
number, not disagreeing usefully with the market. Ranking or sizing by
`edge_pct` has no support here.

## Finding 4 — `rush_rec_yards` looks like a construction defect

| | mean |
|---|---|
| model projection | 53.7 |
| sportsbook line | 72.7 |
| actual | 73.9 |

The line is essentially unbiased (−1.22). The model is **19 yards low on a
73-yard market**, roughly 26%, and consequently picks UNDER on 59 of 67 bets.
A systematic one-directional shortfall of that size on a market that is the
sum of two components the model already projects separately is not variance.
`rush_yards` bias is −6.60 and `rec_yards` is −7.27; summed that is about
−13.9, so roughly −6 yards of the gap is unexplained by the parts.

## Finding 5 — TE is a no-edge market, not an inverted one

TE overall 60-82 (42.3%). Week 1 36-36, Week 2 24-46.

Week 2 TE: mean projection minus line **+0.08**, median |projection − line|
**1.05 yards** on a ~16-yard line. The model is agreeing with the number and
then taking whichever side the rounding lands on. TE bias is only −0.29 in
Week 2, so this is not the under-projection problem — it is an absence of
signal. Week 2 OVERs went 8-25 while UNDERs went 16-21.

At n=142 a 60-82 record is about 2.1 SD from a coin, so the record is
suggestive but the *mechanism* is clear and does not require a sign error:
there is no edge to harvest at that line proximity, and juice does the rest.

## Finding 6 — a systemic low bias drives a 2:1 UNDER book

Overall model bias −5.55 vs line bias −2.36. By side: UNDER bets carry a
−9.82 bias, OVER bets +2.59. The board is 568 UNDER to 298 OVER because the
projections run low, not because the model found 568 unders.

This replicates the under-projection finding from the 2024/2025 clean-cohort
re-grade, now on live 2026 slates.

## What this does and does not license

Nothing here is a tuning instruction. The calibration and `model_sd` findings
are defects with identified mechanisms and should be fixed as such. The
`pass_yards` result is one market over two weeks at n=58 with game-clustered
bets; it needs Weeks 3–6 under frozen parameters before it means anything.
