# First real test of alt-line profitability — 2026 week 9

**Date:** 2026-11-08 · **Rows:** `data/research/alt_graded_2026_wk09_v1.csv` (4,306 graded rungs)
**Status:** first out-of-sample grading of alt rungs against **real offered prices** in this
repository. 13 of 15 week 9 games final; TB@CHI and BUF@MIN excluded as unsettled.

`PRISM_ALT_PRICING_VALIDATION_2026_W9.md` established that no such test existed because
`backtest_alt_clearance_table_v1.py` never referenced `offered_american`. This is that test.

## Headline: market selection dominated model selection

Flat 1u on every available alt OVER rung:

| market | n | hit% | breakeven | P&L | ROI |
|---|---|---|---|---|---|
| **rec_yards** | 1401 | 59.3% | 58.2% | **+40.5u** | **+2.9%** |
| pass_yards | 651 | 46.9% | 47.2% | -4.4u | -0.7% |
| rush_yards | 673 | 50.2% | 58.9% | -124.2u | -18.5% |
| rush_rec_yards | 321 | 42.7% | 56.5% | -108.9u | -33.9% |
| **receptions** | 1260 | 31.0% | 35.4% | **-377.9u** | **-30.0%** |
| ALL | 4306 | 46.5% | 49.9% | -574.9u | -13.4% |

Receiving yards was the only profitable market. Receptions was the worst by a wide margin.

## Rung depth is the second axis

| frac_of_line | n | hit% | breakeven | ROI |
|---|---|---|---|---|
| 0.0-0.5 | 322 | 82.6% | 86.8% | -4.8% |
| 0.5-0.7 | 598 | 76.8% | 78.9% | -2.5% |
| 0.7-0.85 | 487 | 67.4% | 71.3% | -5.1% |
| 0.85-1.0 | 719 | 52.9% | 56.2% | -5.4% |
| 1.0-1.2 | 616 | 45.0% | 46.4% | -2.4% |
| 1.2-1.5 | 936 | 26.0% | 29.9% | -14.9% |
| 1.5-2.0 | 272 | 12.5% | 18.0% | **-42.2%** |
| >2.0 | 356 | 4.2% | 8.4% | **-59.4%** |

Deep OVER rungs are heavily taxed. Any tier built from frac > 1.5 is structurally
losing regardless of model. The shipped HIGH and LOTTERY tiers were built there.

## The shipped filter worked overall, and inverted by market

| selection | n | hit% | breakeven | P&L | ROI |
|---|---|---|---|---|---|
| bet everything | 4306 | 46.5% | 49.9% | -574.9u | -13.4% |
| triple-intersection | 215 | 51.2% | 44.0% | +45.4u | +21.1% |
| **triple-intersection + >=6 games (shipped)** | 155 | 51.6% | 43.9% | **+34.1u** | **+22.0%** |
| ...receptions subset | 85 | 50.6% | 35.6% | +37.1u | **+43.7%** |
| ...rec_yards subset | 59 | 52.5% | 55.7% | -4.3u | **-7.3%** |

The filter turned the market's worst category (-30.0%) into its best selection
(+43.7%), and turned the market's only profitable category (+2.9%) into a loss
(-7.3%). All of the week's profit came from receptions picks.

## Requiring all three estimates was the error on yards markets

Within `rec_yards`:

| filter | n | P&L | ROI |
|---|---|---|---|
| PRISM beats price alone | 461 | **+46.1u** | **+10.0%** |
| own record beats price alone | 580 | +50.3u | +8.7% |
| clearance table beats price alone | 237 | +17.1u | +7.2% |
| all three (shipped) | 84 | +1.4u | +1.6% |

Every single criterion outperformed the intersection. The intersection cut 461
profitable legs to 84 and removed the edge with them. This is the predicted
consequence of the measured per-market bias: PRISM runs 5.71 yards low on
`rec_yards` and 4.13 low on `rush_yards`, so `p_prism` was the binding constraint
on 35.8% of rec_yards and 54.8% of rush_yards rungs, systematically deleting
yards OVERs. The diagnosis was made before the slate and not acted on.

Profitable rec_yards depth bands: 0.7-0.85x (+10.6%), 1.0-1.2x (+8.3%),
1.2-1.5x (+7.1%). Unprofitable: <=0.7x and 0.85-1.0x.

## Shipped ticket results

SAFE won, MODERATE won, MEDIUM won (Foster Moreau inactive, leg voids),
HIGH RISK lost (2/3), LOTTERY lost (0/4). Flat 1u on the 12 settled legs:
**+7.04u**. Dawson Knox (BUF, Monday) and Roschon Johnson (CHI, Sunday night)
unsettled at time of writing.

## Changes this evidence supports

1. **Weight by market.** `rec_yards` and `pass_yards` are playable; `receptions`,
   `rush_yards` and `rush_rec_yards` are not, blind. Receptions is playable only
   with a selection filter, and only that filter made money this week.
2. **Do not require all three estimates on yards markets.** Use PRISM-beats-price
   as the primary screen there; keep the intersection for receptions, where it
   demonstrably added 73pp of ROI over the blind market.
3. **Cap rung depth at frac <= 1.5**, and prefer 0.7-0.85 and 1.0-1.5.
4. **Bias-correct `model_proj` per market** before computing any probability, using
   the measured residuals. This is the same defect as the `model_sd` finding and
   has the same fix: calibrate the distribution to observed outcomes.
5. One week, 13 games. Do not promote any of this on a single slate; re-test on
   week 10 before changing production.
