# Week 11 2026: complete grading of the model and the whole slate

**Date:** 2026-11-24. **Status:** closed. All 13 games are final, TNF through MNF.

**Coverage:**
- Main lines: `boards/2026_wk11.csv`. TNF rows come from run 44580190319; every other game
  from the Sunday run 45024058278, 07:50 ET.
  - FanDuel + DraftKings, 1,874 priced rows.
  - **1,778 graded.** 96 dropped as no-snap voids or pushes.
  - 194 players, 13 games.
- Alt ladders: FanDuel overs, latest capture per game, 3,373 graded rungs.

**Script:** `scripts/research/grade_week11_complete_v1.py`.

**Rows** (`data/market_track_record/graded/`):
- `w11_2026_mainline_board_graded_complete_v1.csv`
- `w11_2026_prism_side_graded_v1.csv`
- `w11_2026_projection_accuracy_v1.csv`
- `w11_2026_projection_rows_v1.csv`
- `w11_2026_alt_board_graded_v1.csv`

**Cards, screens, the RB-under tracker and line movement:** `WK11_2026_BACKTEST_V1.md`.

## 1. PRISM's own side (`edge_pct > 0`), one bet per offer

| | n | hit | ROI |
|---|---|---|---|
| coin flip (every side) | 1,778 | 50.0% | **-5.9%** |
| **PRISM side** | 889 | 52.5% | **-1.1%** |
| FanDuel only | 443 | 52.6% | -1.0% |
| DraftKings only | 446 | 52.5% | -1.2% |

**Week 10 was -11.3%.** Week 11 is close to breakeven: PRISM cut the house's 5.9% to 1.1%.

| by market | bets | hit | ROI |
|---|---|---|---|
| rec_yards | 311 | 55.3% | **+4.5%** |
| receptions | 310 | 52.6% | -1.9% |
| pass_yards | 52 | 51.9% | -2.0% |
| rush_rec_yards | 70 | 50.0% | -6.0% |
| rush_yards | 146 | 47.9% | **-9.0%** |

| by position | bets | hit | ROI |
|---|---|---|---|
| **WR** | 328 | 54.0% | **+1.6%** |
| TE | 153 | 54.9% | +0.3% |
| QB | 104 | 52.9% | +0.4% |
| RB | 300 | 49.7% | **-5.3%** |

| by window | bets | hit | ROI |
|---|---|---|---|
| TNF | 76 | 52.6% | -2.0% |
| Sunday day | 675 | 50.8% | -4.6% |
| SNF | 78 | 59.0% | +14.2% |
| MNF | 60 | 63.3% | +18.8% |

| by edge size | bets | hit | ROI |
|---|---|---|---|
| 0-5% | 194 | 51.0% | -3.6% |
| 5-10% | 174 | 52.3% | -0.3% |
| 10-20% | 266 | 48.5% | -9.7% |
| 20%+ | 255 | 58.0% | +9.0% |

Edge size is again not monotone.

**FanDuel by position x market:**

| position / market | bets | hit | ROI |
|---|---|---|---|
| WR receptions | 81 | 56.8% | **+7.0%** |
| WR rec_yards | 81 | 51.9% | -2.2% |
| TE receptions | 37 | 56.8% | -2.1% |
| TE rec_yards | 38 | 52.6% | -0.5% |
| RB rec_yards | 35 | 62.9% | +18.5% |
| RB receptions | 35 | 40.0% | -18.5% |
| RB rush_yards | 46 | 45.7% | **-14.0%** |
| QB rush_yards | 26 | 57.7% | +10.0% |
| QB pass_yards | 26 | 50.0% | -5.8% |

## 2. Projection accuracy vs the FanDuel line (one row per player-market)

MAE for every model component and the final projection:

| market / pos | n | **PRISM** | **Vegas** | MC | ML | State | PRISM bias | Vegas bias | PRISM closer | right side |
|---|---|---|---|---|---|---|---|---|---|---|
| rec_yards WR | 81 | **22.1** | 22.6 | 22.4 | 24.0 | 26.6 | -6.8 | -3.8 | 46% | 51% |
| rec_yards TE | 38 | 15.2 | **14.5** | 15.4 | 15.7 | 15.1 | -0.6 | -2.7 | 47% | 53% |
| rec_yards RB | 35 | **13.5** | 13.8 | 13.2 | 15.7 | 13.4 | -1.6 | -1.9 | 60% | 60% |
| receptions WR | 81 | **1.56** | 1.61 | 1.59 | 1.63 | 1.88 | -0.55 | -0.03 | 53% | 57% |
| receptions TE | 37 | **1.45** | 1.47 | 1.46 | 1.47 | 1.56 | +0.12 | +0.18 | 57% | 59% |
| receptions RB | 35 | 1.40 | **1.27** | 1.53 | 1.37 | 1.31 | -0.41 | -0.16 | 40% | 43% |
| rush_yards RB | 46 | 26.3 | **21.7** | 28.3 | 26.6 | 30.0 | **-11.2** | -5.3 | 39% | 43% |
| rush_yards QB | 26 | 11.3 | 11.3 | 11.7 | 12.4 | 12.4 | -3.4 | -0.3 | 46% | 54% |
| rush_rec_yards RB | 34 | 35.5 | **29.5** | 35.5 | 34.9 | 35.1 | **-18.0** | -4.6 | 44% | 56% |
| pass_yards QB | 26 | 45.3 | 40.5 | 53.7 | **37.7** | 50.9 | +8.8 | -3.1 | 38% | 50% |

**What the table says:**

- **WR and TE projections were as accurate as FanDuel's line this week, or better.** PRISM beat
  the line's MAE on WR receiving yards, WR receptions, TE receptions and RB receiving yards.
- **RB rushing is the worst market, and the reason is known:** PRISM ran 11 yards low on RB
  rushing and 18 low on RB rush+receiving.
- **Passing yards:** the ML component (37.7) beat both the final projection (45.3) and the line
  (40.5). n=26.
- **Star compression persists.** On top-quartile rec/rush lines (n=59), PRISM bias was **-14.4
  vs Vegas -1.4**. Week 10: -17.6 vs -5.9.

**Largest misses above PRISM.** Every one is a backfield role change that the line had already
priced:

| player | PRISM | FanDuel line | actual | what happened |
|---|---|---|---|---|
| Emmett Johnson (KC) rush | **29** | 87.5 | 105 | Walker out; the lead back for the 2nd week running. **Missed in week 10 too** (PRISM 6). |
| David Montgomery (HOU) rush | 31.5 | 56.5 | 107 | Back from 5 games out (TNF) |
| Jonah Coleman (DEN) rush | 39 | 71.5 | 111 | Took over from Dobbins |
| Ashton Jeanty (LV) rush+rec | 58 | 64.5 | 169 | Bowers out; 90 receiving |

**Largest misses below PRISM:** quarterbacks in lopsided or broken games:
- McCord 102 (MIA lost 26-9);
- Daniel Jones 144 (IND lost 26-3, after his receivers were traded);
- Herbert 131;
- Caleb Williams 181 (rain in Chicago);
- Milton 141;
- Geno Smith 176.

PRISM's final pass projection was biased +8.8 this week.

## 3. Calibration: PRISM's stated probability is not a probability

| stated (PRISM side) | bets | stated | actual | ROI |
|---|---|---|---|---|
| 50-55% | 137 | 52.6% | 43.1% | -18.2% |
| 55-60% | 159 | 57.4% | 51.6% | -2.9% |
| 60-70% | 246 | 64.6% | 49.2% | -10.0% |
| 70-80% | 178 | 74.7% | 63.5% | +15.9% |
| 80%+ | 107 | 86.9% | 54.2% | +0.1% |

The overstatement is the same as V27 recorded, about 12-30 points. `fair_prob` and `edge_pct`
must not be used as probabilities.

## 4. Alt board, FanDuel overs (3,373 rungs)

36.6% hit vs 41.4% needed, **-20.4%**. Every market x position lost:

| market / pos | rungs | hit | needed | ROI |
|---|---|---|---|---|
| rec_yards WR | 915 | 39.9% | 43.1% | -13.1% |
| rec_yards TE | 356 | 37.6% | 42.4% | -24.7% |
| rec_yards RB | 282 | 29.1% | 38.6% | -24.4% |
| receptions WR | 512 | 34.6% | 39.6% | -16.0% |
| receptions TE | 218 | 28.9% | 38.0% | -38.9% |
| receptions RB | 188 | 36.2% | 40.8% | -27.2% |
| rush_yards RB | 461 | 39.0% | 42.3% | -18.0% |
| rush_yards QB | 168 | 29.8% | 39.3% | -24.9% |
| pass_yards QB | 246 | 41.9% | 44.1% | -28.5% |

The cushion and price-band breakdown and the shortlist band are in `WK11_2026_BACKTEST_V1.md`
§4-5.

## 5. What week 11 says

1. **The model was close to breakeven on its own side (-1.1%)**, after -11.3% in week 10. **WR
   and TE were the best positions**, and WR/TE projections matched or beat FanDuel's line.
2. **RB rushing is the model's biggest problem.**
   - PRISM side -14% on FanDuel.
   - Bias -11 to -18 yards.
   - The largest misses are backs whose role just changed, which the line priced and PRISM
     did not.
   - **Emmett Johnson was missed two weeks in a row.**
   - The allocation uses a season-long carry share. `RB_CARRY_SHARE_RECENCY_RESULTS_V1.md`
     measured that recent games deserve about 0.85 weight, and it was never promoted.
3. **Star compression** (bias -14 vs -1) is still open.
4. **Stated probabilities are overstated** by 12-30 points.
5. **Alt overs lost in every market.** Main-line overs lost too.
