# How far up his own price ladder did each week-9 player actually get?

**Date:** 2026-11-11 · **Script:** `scripts/research/analyze_alt_ladder_depth_v1.py`
**Board:** `data/market_track_record/graded/w9_2026_alt_board_graded_complete_v1.csv`
**Rows:** `data/research/alt_ladder_depth_2026_wk09_{allbooks,fanduel}_v1.csv` (per-ladder),
`data/research/alt_ladder_survival_2026_wk09_{allbooks,fanduel}_v1.csv` (survival curves)

Covers **every week-9 player on the board**: 966 (player, market, book) ladders over
5,115 rungs, 13 games, 185 players. FanDuel subset: 359 ladders, 1,924 rungs.

---

## 0. The premise is exactly right, and it is provable

An alt ladder is **monotone by construction** — `hit` is derived from one `actual` per
player, so clearing the rung priced -200 forces the -400 and -750 rungs to clear too.

**Monotonicity violations: 0 / 966.** Checked by the script and asserted on every run;
a non-zero count would invalidate everything below.

This is why the rung is the wrong unit. Week 9, FanDuel:

- raw rungs priced >= -200: **1,321 rungs, 33.7% hit**
- distinct **ladders** reaching -200 or better: **198 / 359 = 55.2%**

Both are correct and they answer different questions. The rung count is dominated by
the deep plus-money rungs sitting inside otherwise-winning ladders. **"How often does a
player get there" is the ladder number.** Everything here is per ladder.

Two columns carry the result, and together they bracket each player in price space:

- **`hwm_price`** — the price of the deepest rung he *cleared* (his high-water mark).
  Blank means he missed even the cheapest rung offered.
- **`died_price`** — the price of the cheapest rung he *missed*.

---

## 1. The answer: how many cleared -200 or better

| | all books | FanDuel |
|---|---|---|
| ladders | 966 | 359 |
| **cleared -200 or better** | **503 (52.1%)** | **198 (55.2%)** |
| cleared nothing at all | 237 (24.5%) | 75 (20.9%) |
| median price of the first MISSED rung | **-116** | -118 |

**About half of all ladders reach -200, and a quarter clear nothing whatsoever** — not
even the cheapest rung the book offered, which is routinely -1,400.

## 2. Working up the ladder — all books, every player

`offered` = ladders holding a rung at this price or better. `cleared` = those where such
a rung hit. `pick_roi` = ROI of betting **one** rung per ladder: the cheapest priced at
or above the level, which is what "give me about -200 on this guy" actually buys.

| level | offered | cleared | rate | pick price | pick hit | breakeven | **edge** | **ROI** |
|---|---|---|---|---|---|---|---|---|
| -1000 | 966 | 703 | 72.8% | -400 | 72.8% | 75.0% | -2.2 | **-2.8%** |
| -600 | 966 | 670 | 69.4% | -300 | 69.4% | 71.9% | -2.5 | -3.6% |
| -400 | 966 | 626 | 64.8% | -220 | 64.8% | 67.6% | -2.8 | -4.4% |
| -300 | 966 | 582 | 60.2% | -167 | 60.2% | 63.5% | -3.2 | -5.2% |
| **-200** | **961** | **503** | **52.3%** | -92 | 52.3% | 56.3% | -3.9 | **-7.0%** |
| -150 | 947 | 444 | 46.9% | -33 | 46.9% | 50.8% | -3.9 | -7.0% |
| -110 | 896 | 374 | 41.7% | +88 | 41.7% | 44.4% | -2.7 | **-6.7%** |
| **+100** | **887** | **351** | **39.6%** | +143 | 39.6% | 42.0% | **-2.4** | **-6.5%** |
| +120 | 822 | 287 | 34.9% | +170 | 34.9% | 37.9% | -3.0 | -9.4% |
| **+150** | **683** | **180** | **26.4%** | +211 | 26.4% | 33.1% | **-6.8** | **-21.7%** |
| +200 | 520 | 102 | 19.6% | +284 | 19.6% | 26.9% | -7.3 | -25.6% |
| +300 | 367 | 55 | 15.0% | +420 | 15.0% | 20.1% | -5.1 | -21.8% |
| +500 | 313 | 23 | 7.3% | +718 | 7.3% | 12.6% | -5.3 | -43.7% |
| +800 | 254 | 10 | 3.9% | +1065 | 3.9% | 8.9% | -5.0 | -50.8% |
| +1000 | 215 | 7 | 3.3% | +1365 | 3.3% | 7.0% | -3.8 | -54.5% |

**The curve is flat, then it falls off a table at +150.** Everything from -1000 up to
+120 loses 3-9%, which is roughly the vig. From +150 up it loses 22-55%.

FanDuel is the same shape and milder on the favorite side: **-0.7% to -5.3% from -600
through +120**, then **-16.3% at +150** and -81.9% at +1000.

## 3. Why +150 is the cliff — the amplification law

This is the part that generalizes beyond week 9. Look at `edge_pts`, the gap between the
pick's hit rate and its breakeven, in percentage points:

| level | breakeven | edge (pts) | ROI | **ROI cost per point of edge** |
|---|---|---|---|---|
| -1000 | 75.0% | -2.2 | -2.8% | **1.27x** |
| -400 | 67.6% | -2.8 | -4.4% | 1.57x |
| -200 | 56.3% | -3.9 | -7.0% | 1.79x |
| -110 | 44.4% | -2.7 | -6.7% | 2.48x |
| +100 | 42.0% | -2.4 | -6.5% | 2.71x |
| +150 | 33.1% | -6.8 | -21.7% | 3.19x |
| +300 | 20.1% | -5.1 | -21.8% | 4.27x |
| +500 | 12.6% | -5.3 | -43.7% | 8.25x |
| +1000 | 7.0% | -3.8 | -54.5% | **14.34x** |

**The error barely moves. The price does all the damage.** We are off by two to seven
points everywhere on the ladder. At -1000 that costs 1.3x its size in ROI; at +1000 it
costs **14x**. A fixed calibration error is a rounding error on a favorite and a
catastrophe on a longshot, because it is a far larger *fraction* of a small win
probability.

Independently confirmed on the prediction side (FanDuel, all markets, pick at each level,
positive = PRISM overstates):

| level | n | predicted | actual | error |
|---|---|---|---|---|
| -400 | 359 | 67.5% | 68.2% | **-0.7** |
| -200 | 359 | 55.3% | 55.2% | **+0.2** |
| -110 | 349 | 42.7% | 43.6% | **-0.8** |
| +100 | 349 | 40.6% | 41.5% | **-1.0** |
| +150 | 275 | 31.2% | 28.4% | **+2.8** |
| +300 | 155 | 18.7% | 15.5% | **+3.2** |

PRISM is **essentially unbiased from -400 through +100** and starts overstating at +150 —
**the calibration cliff and the ROI cliff are at the same price.** That rules out bad
luck as the explanation and matches the decile table in `PRISM_W9_COMPLETE_GRADING_V1` §4,
which was measured a different way on the same board.

**This is the mechanism behind every alt failure recorded in this repository**: the
LOTTERY tier at -46.5%, the >1.6x depth cliff at -53.5%, and the Egbuka rung. None of
them were unlucky. They were a 3-point error sold at 10x amplification.

## 4. Per market, all books

| market | ladders | cleared nothing | cleared -200+ | ROI @-200 | ROI @-110 | ROI @+100 | ROI @+150 |
|---|---|---|---|---|---|---|---|
| `rec_yards` | 398 | 25% | **58%** | +0.2% | +1.9% | **+4.0%** | -18.0% |
| `pass_yards` | 72 | 15% | 56% | -5.4% | -5.5% | -2.2% | -3.8% |
| `receptions` | 244 | 23% | 50% | -6.0% | -6.0% | -4.8% | -14.5% |
| `rush_yards` | 185 | 30% | 45% | -19.7% | -17.5% | -21.2% | -42.2% |
| `rush_rec_yards` | 67 | 21% | 43% | -20.2% | -29.8% | -37.8% | -44.0% |

The ordering is **the same ordering** the sign rule produced independently
(`PRISM_ALT_TABLE_SUPPORT_AUDIT_V1`, r = +0.860) and the same ordering the complete
grading produced. Three different methods, one ranking.

`rec_yards` near even money is the only non-negative cell. FanDuel, `rec_yards`:
**+5.4% at -110 and +8.2% at +100 (n=140)**, replicated on DraftKings at +3.8% and +4.9%
(n=141/138) — but **negative on BetMGM (-9.4%, -6.6%, n=68)**.

**Cells to distrust outright:** `rec_yards` at +250/+300/+400 reads +41.5%/+89.6%/+429%
on **n=33/9/1**. Those are one or two lucky ladders, not a signal. Any level with
pick_n < 50 in a single market is noise.

## 5. What is established and what is not

**ESTABLISHED, and it is the firmest result in this project.** Rungs priced **+150 and
up lose heavily.** FanDuel at +150: **-16.3%, 95% CI [-33.9%, +0.4%], P(>0) = 2.8%**
(cluster bootstrap by game, 13 games, 20k draws — the game is the resampling unit
because legs inside a game are correlated). Replicated on all three books and in every
market. This is a *negative* finding and the inference supports it.

**DESCRIPTIVE, no inference needed.** 52.1% of ladders reach -200 or better; 24.5% clear
nothing; the median ladder dies at **-116**.

**MECHANISM, large n and consistent.** The amplification law in §3, confirmed on both the
ROI side and the prediction side at n=349.

**SUGGESTIVE ONLY — DO NOT BET THIS YET.** `rec_yards` near even money.
FanDuel +100: **+8.2% but 95% CI [-13.2%, +29.2%], P(>0) = 77.0%.** Pooled FD+DK over
the -110..+120 band: **+4.3%, CI [-16.2%, +23.1%], P(>0) = 67.4%.** One week, 13 game
clusters, and I searched a grid of 5 markets x 9 levels before this cell stood out —
so treat it as a hypothesis with a mechanism, not a measurement. What makes it worth
carrying forward rather than discarding: it sits at the intersection of two findings
established *before* this analysis (the sign rule's market ranking, and the decile
calibration), and it replicates on the two books with real coverage.

**NOT CLAIMABLE ANYWHERE.** A positive edge. Nothing in this analysis establishes one.

## 6. Where the ladders die, named

The `died_price` column is the useful one for reading a near-miss. The -200 failures that
still cleared something came up **exactly one rung short**:

| player | market | actual | cleared | at | died at | price |
|---|---|---|---|---|---|---|
| Matthew Stafford | pass_yards | 228 | 224.5 | -205 | 249.5 | -106 |
| Tetairoa McMillan | rec_yards | 56 | 49.5 | -210 | 59.5 | -128 |
| Tyler Allgeier | rush_yards | 46 | 39.5 | -210 | 49.5 | -114 |
| Caleb Williams | rush_yards | 16 | 14.5 | -210 | 19.5 | -114 |
| T.J. Hockenson | receptions | 4 | 3.5 | -250 | 4.5 | -118 |
| Alvin Kamara | receptions | 3 | 2.5 | -245 | 3.5 | -102 |

**The modal death is in the -150 to -100 band** (133 of 721), with a second mode at
<= -300 (148). The ladder's own boundary and the best-performing price band are the same
place, which is not a coincidence: that is where outcomes are genuinely near 50/50, where
the book's price is fairest, and where a small edge is worth the most per point.

The 75 FanDuel shutouts are the opposite failure — the cheapest rung on the board missed:
Carson Wentz (81 passing yards against a 124.5 rung at **-1400**), Josh Jacobs (5 rushing
yards against 19.5 at -1200), Chris Bell (0 catches against 0.5 at -1400), Tucker Kraft
(1 receiving yard against 19.5 at -800), Ashton Jeanty (**-5** receiving yards against
14.5 at -700). A -1400 rung implies 93.3%. Five of them lost in one week.

44 FanDuel ladders ran to +200 or better, and they are overwhelmingly `receptions`:
C.J. Stroud pass_yards to **+1260** (384 yards), Dawson Knox receptions to **+920**
(4 catches off a 1.5 main), Tank Dell **+880**, Kayshon Boutte **+750**. Note Knox:
**I recommended him at -280 and he ran to +920.** That is the mirror image of the Egbuka
error — too shallow rather than too deep. Hindsight-perfect rung selection returns
+132.5%; betting the cheapest rung on every ladder returns -2.4%. The gap between those
two numbers is the entire problem, and §5 says we cannot yet close it.

## 7. Consequences

1. **A hard price ceiling is now justified by inference, not taste.** No rung priced
   above about **+120**. The current builder's `PRICE_MAX = 130` already sits there, and
   this is independent confirmation it is in the right place.
2. **This formally closes the HIGH/LOTTERY tier question** from
   `PRISM_W9_COMPLETE_GRADING_V1` §4 on the measurement side. Those tiers live entirely
   above +150, where the loss is the one statistically supported result on the board.
   Keeping them is a choice to spend money for entertainment, and it should be labeled
   that way rather than modeled.
3. **The near-even-money band is where the work should go**, not the deep end. It is
   where PRISM is unbiased, where amplification is lowest, and where `rec_yards` showed
   the only non-negative cell.
4. **Carry `rec_yards` at -110..+120 into week 10 as a preregistered hypothesis** with
   the target stated in advance, rather than re-searching the grid next week.
5. **`rush_yards` and `rush_rec_yards` are unbettable at every price tested.** -17% to
   -44%, no band exempt.
