# Can we tell, before the game, which model picks will win? RESULTS V1

**Date:** 2026-11-20.

**Preregistration:**
- `SELECTIVE_PICK_IDENTIFICATION_PREREGISTRATION_V1.md`;
- frozen in commit 6515f57 before scoring;
- sha256 `1d5ebc367c9dcc38afe2ba9352fc7657721f8cb9f6b6b39908578420e80fdb35`.

**Script:** `scripts/research/evaluate_selective_pick_identification_v1.py`.

**Rows:** `data/research/selective_pick/`.

**Verdict: all six selectors FAIL.** None was profitable in both seasons. **Every one of them was
positive in 2025 and negative in 2024:** that is a season effect, not a selector.

## 1. The answer in one paragraph

**We can find picks that win more often. The book already charges for exactly that.**
- The best selector (S6, a classifier on everything we know pregame) hit **58.5%** on seasons it
  never saw.
- It still lost money (-1.4%), because what it learned was to pick the bets the book prices at
  about -150.
- Across 17,000 lines, the win rate in every price band matches the win rate the price implies,
  to within about 1-2 points (§3).
- So "find the 49% we get right" turns into "find the bets that win more often **than the
  price says**". Nothing pregame that we have found does that in both seasons:
  - not the model's edge;
  - not the side;
  - not the price;
  - not the spread or total;
  - not the model's record on that player;
  - not all of them combined.

## 2. The selectors (fit on one season, scored blind on the other, both directions)

Full-stack production ensemble, 2024-2025, rush_yards, rec_yards, receptions and rush_rec_yards.
16,969 lines.

| selector | bets | hit | ROI | 2024 | 2025 | verdict |
|---|---|---|---|---|---|---|
| C0 every model pick | 16,969 | 52.2% | -2.9% | -5.2% | -0.1% | comparator |
| C1 every UNDER, model ignored | 16,969 | 51.6% | -3.6% | -5.2% | -1.6% | comparator |
| C2 the model's OVER picks | 3,911 | 51.3% | -4.9% | -6.8% | -2.7% | comparator |
| **S1 the model's UNDER picks** | 13,058 | 52.4% | **-2.3%** | -4.7% | +0.6% | FAIL |
| S2 model UNDER, biggest third of edges | 4,345 | 53.2% | -2.2% | -4.2% | +0.1% | FAIL |
| S3 market x side cells that won last season | 7,510 | 51.5% | -4.5% | -4.5% | (none chosen) | FAIL |
| S4 price better than -110 | 3,097 | 47.5% | -2.0% | -5.0% | +0.7% | FAIL |
| S5 the model's record on this player ≥60% | 3,554 | 52.1% | -3.6% | -6.5% | -0.7% | FAIL |
| **S6 classifier, top 20%** | 3,394 | **58.5%** | **-1.4%** | -3.8% | +1.5% | FAIL |

CIs are clustered by game and by player.
- S1: [-4.3%, -0.2%] by game, [-4.2%, -0.4%] by player.
- S6: [-4.4%, +1.5%] by both.
- Full table: `selective_pick_selectors_v1.csv`.

**Notes:**
- **S3, fit on 2025, picked nothing.** That is not a missing result: on 2024, which it was fit
  on, no market x side cell had positive ROI over 150+ bets. The 2025 → 2024 direction picked
  five cells, and they lost 4.5% on 2024.
- **The 2024/2025 split is the main pattern.** The model was close to breakeven in 2025 (-0.1%)
  and lost 5.2% in 2024. Any filter looks good on 2025 and bad on 2024. A filter that only
  works in the model's good year is not something to bet on.

## 3. Why the high-hit-rate selector still loses: the price already knows

Hit rate by the win chance implied by the price (vig included), across all 16,969 model picks:

| price-implied win | bets | implied | actual hit | ROI |
|---|---|---|---|---|
| under 45% (about +130) | 525 | 41.3% | 40.4% | -4.6% |
| 45-50% | 1,285 | 47.9% | 46.7% | -2.4% |
| 50-53% | 4,460 | 52.2% | 51.5% | -1.4% |
| 53-56% (about -120) | 7,904 | 53.8% | 51.5% | -4.2% |
| 56-60% | 1,377 | 58.1% | 57.9% | -0.4% |
| over 60% (about -150+) | 1,418 | 63.3% | 61.7% | -2.6% |

- **The actual hit rate follows the price in every band.**
- **S6's picks:** price-implied 59.9% vs actual 60.7% (fit on 2024), and 58.9% vs 56.7% (fit on
  2025).
- It found the bets the book already marked as likely winners, and paid full price for them.
- **What hit rate a "good pick" needs depends on its price.** A 58% pick at -150 is a losing
  bet.

## 4. Does the model add anything? Yes, a little, and less than the vig

| on the same rows | ROI |
|---|---|
| UNDER where the model also says UNDER (S1) | -2.3% |
| UNDER on every row (C1) | -3.6% |
| **UNDER where the model says OVER (C3)** | **-7.9%** |

When the model says OVER, the under does clearly worse. **So the model's direction is real
information.** It moves the result by about 1-5 points, and the vig is about 4.5. That is the
same conclusion as the 2026-11-04 passing-yards study: real direction, not enough to beat the
price.

## 5. 2026 check (PRISM boards, weeks 1, 2, 6-10, FanDuel + DraftKings)

No selector passed, so this is descriptive only (`selective_pick_2026_check_v1.csv`).

| market | the model's UNDER picks | blind UNDER | the model's OVER picks |
|---|---|---|---|
| rush_yards | **57.5%, +8.3%** (n=569, 5/7 weeks; CI [-1%, +18%]) | +5.0% | -11.7% |
| rec_yards | -2.2% | -3.3% | -6.9% |
| receptions | -1.2% | -1.5% | -9.4% |
| rush_rec_yards | -10.6% | -8.5% | -9.8% |
| pass_yards | +2.5% | -9.9% | +7.8% (the retracted edge) |
| all four non-QB | -0.6% | -1.5% | |

The 2026 rushing-under result matches the RB-under tracker's 2026 backtest. On the 2024-25
lines, the same "model says UNDER on rush_yards" bets went -1.0% and then +0.9%. **2026 rushing
unders look like a season regime.** The RB-under rule that grew out of them already failed on
2024-25.

**Passing yards** (secondary; base engine on 2024-25, not the QB synthesis):
- the model's picks: -6.0%;
- its OVER picks: -14.6%.
- In 2026 weeks 9-10 the passing-yards picks went **49.1%, -7.2%** (n=112), after the 57.4%,
  +8.2% that was retracted on 2026-11-04.

## 6. Prediction scored

- S1: unders beat overs, the model adds 1-2 points, TRACK or FAIL. **Right:** it adds 1.3
  points over blind unders and fails.
- S2 no better than S1: **right** (-2.2% vs -2.3%).
- S3 picks receptions or under cells and holds weakly: **wrong**. It picked those cells and lost
  4.5%.
- S4 noise: **right**.
- S5 FAIL: **right**.
- S6 at 53-54%, TRACK at best: **direction right, number wrong**. It hit 58.5% by choosing
  expensive bets, and it FAILs.
- "No selector PASSES": **right**.

## 7. A data defect found and handled

In `data/backtests/full_stack_vegas_benchmark_v1/non_qb_detail.csv` (and the base archive),
`game_id` is the **previous season's** game for that team and week. For example, 2024 week 1 NYJ
(vs SF) carries `2023_01_BUF_NYJ`.
- Team, opponent, line and actual are correct. Actuals were checked against nflverse: 100% of
  rush_yards, rec_yards and receptions rows match.
- This study rebuilt the game key from the nflverse schedule; it matched the file's opponent on
  100% of rows.
- **Anything else that clusters or joins on that `game_id` is wrong.**
  - The 2026-11-20 RB-under historical test used schedule-derived game ids, so it is unaffected.
  - The defect is in the projection builder's game-id attachment
    (`build_market_vegas_benchmark_projections_v1.py`). It is noted here and not yet fixed.

## 8. What this means for betting

1. **There is no hidden 49%.** On 2024-25 lines, the pregame information we have does not pick
   out a group of model picks that beats the price in both seasons.
2. **The price is the book's own pick-quality score, and it is accurate.** Any rule that finds
   "picks that hit 58%" has to be checked at the price paid, not by hit rate.
3. **The model has real direction, about 1-5 points.** It is worth using to **avoid** bets: do
   not take an under the model says is an over (-7.9%), and do not take overs the model does not
   back.
4. **What the model cannot supply has to come from outside it.** Information the line does not
   have yet, or a better number than the line, is something the model cannot produce from box
   scores. The candidates:
   - news before the line moves;
   - the better of two books' lines on the same player.
