# TD1 target shares → receiving yards and receptions vs the FanDuel line: RESULTS V1

**Date:** 2026-11-19.

**Preregistration:** `TD1_REC_YARDS_VS_MARKET_PREREGISTRATION_V1.md`, frozen at d18ceab.

**Script:** `scripts/research/evaluate_td1_rec_yards_vs_market_v1.py`.

**Rows:** `data/research/td1_rec_yards_vs_market_v1{,_rows}.csv`.

**Verdict: PRIMARY NULL** (rec_yards).
- **Receptions: the level defect is removed.** TD1 eliminates the "every receiver half a
  reception low" bias, but that is not a significant MAE gain.
- The lesson from rushing repeats: opportunity fixes do not become yards gains at this n, and
  mean projections on skewed yardage markets lean OVER.

## Results: FanDuel offers, 2026 wk 6-10, WR/TE/RB who played

| | C0 (production share) | **TD1** | PRISM `model_proj` | FanDuel line |
|---|---|---|---|---|
| **rec_yards** (n=715) MAE | 21.83 | 22.02 | 21.75 | **21.07** |
| bias / median residual | +7.9 / +0.8 | **+2.6** / -3.5 | +5.4 / -2.0 | +5.2 / -1.5 |
| right side / ROI / % OVER | 51.3% / -3.4% / 45% | 50.1% / -5.7% / 68% | 51.9% / -2.3% / 58% | |
| **receptions** (n=713) MAE | 1.605 | 1.597 | 1.557 | **1.531** |
| bias / median residual | +0.56 / +0.43 | **+0.06 / -0.06** | +0.28 / +0.05 | +0.06 / -0.5 |
| right side / ROI | 53.3% / -5.5% | 54.3% / -2.9% | 54.8% / -1.6% | |

**WR1** (pregame rank 1 WR, n=124):

| | C0 | TD1 | PRISM | line |
|---|---|---|---|---|
| rec_yards bias | +13.7 | **+1.2** | +13.1 | +6.1 |
| receptions bias | +1.03 | **+0.05** | +0.75 | +0.13 |

- **rec_yards: MAE gain -0.19, 95% CI [-0.76, +0.40].** Null, slightly the wrong way.
- **receptions: +0.008 [-0.043, +0.057].** Null. WR1 receptions: +0.16 [-0.004, +0.33], just
  short.
- Side MDE: 5.2pp overall, 12.6pp for WR1. **No side difference is resolvable.**

## What it shows

1. **TD1 fixes levels.**
   - Receiving-yards bias falls from +7.9 to +2.6, and for WR1s from +13.7 to +1.2, which is
     the alpha compression.
   - **Receptions bias falls from +0.56 to +0.06**, matching the line's +0.06. This is the
     additive "half a reception low" defect in `PRISM_RECEIVING_MASS_DECOMPOSITION_V1.md` §2,
     removed at the share/budget stage of the reduced-form chain.
2. **Level fixes do not cut yardage error.** YPT noise dominates: yards per target repeats at
   r 0.25-0.31. The book's line still has the lowest MAE in both markets.
3. **The skew lesson repeats.** TD1's mean is right, but its rec_yards median residual is -3.5,
   and comparing it with the line says OVER 68% of the time. For receptions, a count with little
   skew, mean and median agree and nothing is distorted.
4. **Across rushing and receiving the pattern is now consistent:**
   - the allocative + budget fixes are real **opportunity and level** fixes: carries -19/-21%,
     targets -7/-8%, level biases removed;
   - they do **not** yet make a better **yardage** projection;
   - and the side decision must use the distribution, never the mean vs the line.

## Prediction scored

- "Small improvement, probably not significant": **wrong in sign**, but not significant.
- "Alphas rise toward the line": **right.** WR1 bias +13.7 → +1.2, past the line's +6.1.
- "Sides not resolvable": **right.**
- **Named risk** (the D2 pattern, yards worse): **its direction appeared**, -0.19, but not
  significantly.

No sportsbook variable entered any projection. Lines were used for grading only. Nothing
promoted.
