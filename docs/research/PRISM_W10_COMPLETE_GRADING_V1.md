# Week 10 2026 — complete grading of the whole slate

**Date:** 2026-11-17 · **Status:** closed. Every priced week-10 game is final.

**Coverage.** Three paid runs: TNF (43327080728), Sunday (43730255663), MNF (43891730307) —
13 of the slate's 14 games. **NE@DET (09:30 Munich) was never priced** (the Sunday run fired
after kickoff). Main lines from `data/market_track_record/boards/2026_wk10.csv` (1,956 rows,
FanDuel + DraftKings); alt ladders from `alt_props/2026_wk10_{thu6,sun,mon}.csv`, keeping only
the **latest capture per game** (the Thursday capture priced the whole week; counting it with
the Sunday capture would double-count Sunday games).

**Rows (durable, committed):**
- `data/market_track_record/graded/w10_2026_mainline_board_graded_v1.csv` — 1,916 graded main-line rows
- `data/market_track_record/graded/w10_2026_alt_board_graded_complete_v1.csv` — 3,520 FanDuel alt
  over rungs, latest capture per game, 13 games
- card grading: `WK10_2026_BACKTEST_V1.md`

Graded with `scripts/research/grade_mainline_board_v1.py` (snap-count zeros, unpriced rows
dropped). `anytime_td` is not on the week-10 board.

## 1. PRISM's own side, one bet per offer — worse than a coin flip this week

The two-sided board grades at exactly 50% and its ROI is the hold (-5.5%), so every PRISM claim
is on **one bet per offer: the side PRISM rates above the price** (`edge_pct > 0`).

| | n | ROI |
|---|---|---|
| coin flip (both sides) | 1916 | **-5.5%** |
| **PRISM-picked side, both books** | 958 | **-11.3%** |
| PRISM edge over a coin flip | | **-5.7 pts**, 95% CI [-12.8, +1.5] by game, P(>0) = 6% |

Weeks 6-8 were **+3.5 pts** over a coin flip (`PRISM_PRIOR_WEEKS_PRICE_BAND_V1.md` §3). Week 10
reverses it. One week; the CI includes zero.

FanDuel only (478 bets): **47.3% hit vs 53.5% needed, -11.6%.**

| | bets | hit | ROI |
|---|---|---|---|
| pass_yards | 26 | 50.0% | -5.8% |
| rec_yards | 167 | 43.7% | **-17.6%** |
| receptions | 167 | 51.5% | -4.8% |
| rush_rec_yards | 42 | 45.2% | -14.8% |
| rush_yards | 76 | 46.1% | -13.5% |
| OVER side | 168 | 41.7% | **-20.8%** |
| UNDER side | 310 | 50.3% | -6.6% |
| TNF / SUN / MNF | 38 / 394 / 46 | 44.7 / 49.2 / 32.6% | -14.9 / -8.1 / -38.7% |

Bigger PRISM edges did not hit more: 0-5% edge -15.8%, 5-10% -15.9%, 10-20% -6.6%, 20%+ -10.9%.

## 2. Projection accuracy vs the market (one row per player-market, FanDuel line)

| market | n | PRISM MAE | Vegas MAE | PRISM bias | Vegas bias | projection on right side of line |
|---|---|---|---|---|---|---|
| pass_yards | 26 | **67.1** | 69.3 | -10.6 | -17.7 | 50.0% |
| rec_yards | 167 | 23.5 | **22.0** | -7.5 | -6.6 | 46.1% |
| receptions | 167 | 1.56 | **1.52** | -0.3 | 0.0 | 50.3% |
| rush_rec_yards | 42 | 34.4 | **28.4** | -17.6 | -7.6 | 45.2% |
| rush_yards | 76 | 20.0 | **16.3** | -7.0 | -2.9 | 46.1% |

**Passing yards is again the only market where PRISM beats the market's line.** Everywhere else
the line is closer and PRISM is biased low.

**Star compression is the largest measurable defect this week.** On the top quartile of book
lines (rec + rush, n=61): **PRISM bias -17.6 vs Vegas -5.9.** The week's biggest misses were
stars far above PRISM: Ja'Marr Chase 231 rec yds (PRISM 62, book 89.5), JSN 171 (77 / 89.5),
McCaffrey 161 rec (23 / 32.5), Pickens 145 (45 / 57.5), Burrow 409 (276 / 289.5), Purdy 389,
Mahomes 348, James Cook 157 rush (73 / 91.5).

**Role changes PRISM missed and the book did not:** Emmett Johnson (KC) 101 rush yds on 88%
snaps against a PRISM 6.3 and a book 73.5 — a 27%-snap backup who became the lead back;
Charbonnet 94 (PRISM 38 / book 65.5); Corum 100 (39 / 57.5); Hampton 105 (43 / 34.5).

Misses explained by state, already in the card backtest: Daniel Jones 108 (receivers traded;
PRISM 246), Rodgers 143 (pulled in a blowout), Henry 10 (BAL QB change flipped the run/pass mix).

## 3. The alt board — the week-9 cushion edge did not replicate

FanDuel alt over rungs, players who played, latest capture per game: **3,520 rungs, 37.7% hit vs
41.4% needed, -10.6%.**

| market | rungs | hit | ROI |
|---|---|---|---|
| pass_yards | 244 | 46.7% | **+11.7%** |
| rec_yards | 1634 | 39.4% | -3.9% |
| receptions | 993 | 33.5% | -20.3% |
| rush_yards | 649 | 36.4% | -21.3% |

| cushion (PRISM / rung) | rungs | hit | needed | ROI |
|---|---|---|---|---|
| < 1.0 | 2364 | 24.6% | 25.8% | -9.1% |
| 1.0-1.25 | 311 | 52.4% | 61.3% | -17.2% |
| 1.25-1.5 | 222 | 60.8% | 71.0% | -15.5% |
| **1.5-2.0** | 240 | 70.8% | 77.1% | **-9.6%** |
| 2.0-3.0 | 197 | 66.5% | 81.0% | -18.0% |
| 3.0+ | 186 | 78.5% | 84.2% | -6.8% |

At -150 to -400 — the week-9 "sweet spot" that returned **+13.5%** — the 1.5-2.0 band returned
**-15.1%** (n=90). **Every cushion band lost money in week 10.** The week-9 result was one slate
and it did not survive the second. Passing-yards alt rungs (+11.7%) are the one place the board
made money, consistent with passing yards being PRISM's best main-line market.

## 4. What week 10 says, plainly

1. **PRISM's side lost to a coin flip** (-5.7 pts) after three weeks of beating one (+3.5). Two
   samples, opposite signs; neither is established.
2. **The cushion strategy that built the week-9 and week-10 cards did not replicate.** It cannot
   be called an edge on two weeks that disagree.
3. **The defects that cost the most are known and open:** star compression (bias -17.6 vs -5.9)
   and role changes PRISM's allocation does not see (Emmett Johnson). Both are V27 §6 items.
4. **Passing yards is the market where PRISM has held up** on both main lines and alt rungs.
