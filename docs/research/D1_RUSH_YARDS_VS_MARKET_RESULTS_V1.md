# D1 carry shares → RB rushing yards vs the FanDuel line: RESULTS V1

**Date:** 2026-11-19.

**Preregistration:** `D1_RUSH_YARDS_VS_MARKET_PREREGISTRATION_V1.md`, frozen at d081097.

**Script:** `scripts/research/evaluate_d1_rush_yards_vs_market_v1.py`.

**Rows:** `data/research/d1_rush_yards_vs_market_v1{,_rows}.csv`.

**Verdict: PRIMARY NULL. The named risk fired.**
- The carries gain does not survive into rushing-yard accuracy at this sample size.
- Two betting-relevant findings came out of it: one about **mean vs median**, and one about
  **who is right when model and book disagree.**

## Results: 221 FanDuel RB rush_yards offers, 2026 wk 6-10, 69 games, players who played

| projection | MAE | bias (actual - proj) | median residual | right side vs line | ROI on its side |
|---|---|---|---|---|---|
| R0, production share | 24.89 | +10.6 | +3.2 | 47.5% | -10.7% |
| **R1, D1 share** | 24.71 | **+2.5** | **-4.1** | **40.3%** | -24.3% |
| PRISM `model_proj` | 23.28 | +7.5 | +0.5 | 48.4% | -9.0% |
| FanDuel line | **21.27** | +3.0 | -2.5 | | |

- **Primary (R0 → R1 MAE):** +0.18 yards, 95% CI [-1.10, +1.50]. **Null.** The CI includes 0.
- **Side accuracy:** R1 trails R0 by 7.2pp. The minimum detectable effect at n = 221 is 9.4pp,
  so this is **not significant, but adverse.**
- The reduced-form projector (share x rolling team carries x shrunk YPC) is cruder than
  production. PRISM's own `model_proj` beats both R0 and R1 on MAE. **Only R0 vs R1 is a clean
  comparison.**

## Finding 1: a mean-correct projection is the wrong tool for an over/under decision

- D1 fixes the **level**: bias falls from +10.6 to +2.5, about where the book's line sits.
- RB rushing yards are right-skewed: the median result is below the mean.
  - R1's **median** residual is -4.1, and the actual lands below R1's number in **55%** of
    games.
  - Comparing a mean projection with the line therefore says OVER too often. R1 said OVER on
    54% of offers, while the actual went over only 46% of the time.
- The book's line sits **between** the mean and the median: actual under the line 54% of the
  time, mean actual 3 yards above it.
- **The side decision has to come from the distribution (P(actual > line)), not from mean vs
  line.**
  - Production's `fair_prob` does come from the simulated distribution. This test's naive
    mean-vs-line rule does not.
  - Prior distribution work: `PRISM_PLAYER_DISPERSION_RESULTS_V1.md` and the RB PD2
    width/difficulty studies.

## Finding 2: when the carries model sits above the line, the book is usually right

| R1 vs line | offers | actual went over |
|---|---|---|
| R1 above the line ("over") | 120 | **38%** |
| R1 below the line ("under") | 101 | 56% (44% under) |
| all offers | 221 | 46% |

R1 above the line is a **negative** signal. Where the volume model is higher than the book, the
book usually holds information the model lacks: late news, committee changes, game script.
This is the same mechanism as the "book has already cut him" flag in
`rb_rush_situation_flags_v1.py` (UNDER 66.7% when the volume projection is more than 1.2x the
line). **It is a second, independent sighting of that pattern**, from a different projector, on
a different row set.

## What this means

1. **For the model:**
   - D1 is a better **carries** model: -19 to -21% carries MAE, prospective test pending.
   - At this n it does not yet make a better **yards** model.
   - It must never drive a side decision through mean vs line.
2. **For betting:**
   - Do not bet RB rushing **overs** because a volume model is above the line. In this
     sample, those went over 38% of the time.
   - The consistent RB edge stays where the reverse-engineering found it: the **under** when
     the RB is in a bad situation, including when the book has cut him.
3. **Prediction scored:**
   - "R1 beats R0 on MAE": right in direction only. It is not significant.
   - "Side accuracy not resolvable": right. The difference is adverse and inside the MDE.
   - **Named risk fired.**

No sportsbook variable entered any projection; lines were used for grading only. Nothing
promoted.
