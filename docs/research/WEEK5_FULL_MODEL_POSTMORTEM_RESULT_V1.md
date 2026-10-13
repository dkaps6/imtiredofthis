# WEEK 5 FULL MODEL POSTMORTEM V1 — RESULT

Date: 2026-10-13

Status: **COMPLETE SCORECARD. PLAYER-LEVEL FAILURE DECOMPOSITION STILL ACTIVE. NO WEEK-5 OUTCOME TUNING.**

Frozen protocol:
- `docs/research/WEEK5_FULL_MODEL_POSTMORTEM_PROTOCOL_V1.md`

Canonical pregame authority:
- full-slate run `37860592615`
- artifact `11585973062`
- digest `sha256:bf52aa4bc30bf6e8cc6029a0a86c0f33cdfaa09f587c38a138f23cef6b7436f4`
- production behavior authority `24704e5f1ef872ec86c36446d2c73f6d377c824c`

Canonical postmortem authority:
- run `38395384228` SUCCESS
- exact head `89505891d8ead7addd746507d0aa81cabe6c04ef`
- artifact `11812200405`
- digest `sha256:5888cab39506ae0bf70034dfb60f297defd28dd3219ea8118870738706f55ac3`

No sportsbook refetch occurred.

## Full underlying football board

- unique player-market rows: **396**
- settled rows: **392**
- DNP/inactive rows: **4**
- unresolved rows: **0**

Overall:
- model MAE: **18.9288**
- model RMSE: **32.3325**
- model bias (model-actual): **-3.9636**
- median preserved-line MAE: **17.8431**
- model closer than median line: **50.26%**
- model direction hit rate vs median line: **55.32%**

The full model was directionally useful, but the market remained slightly better
on raw mean accuracy in Week 5.

## Full-board by position

QB:
- rows 58
- MAE **39.0588**
- median-line MAE **36.0172**
- model closer **43.10%**
- direction hit **50.00%**

RB:
- rows 134
- MAE **18.3801**
- median-line MAE **16.7537**
- model closer **50.75%**
- direction hit **57.36%**

WR:
- rows 142
- MAE **13.8202**
- median-line MAE **13.9965**
- model closer **55.63%**
- direction hit **60.00%**

TE:
- rows 58
- MAE **12.5735**
- median-line MAE **11.6034**
- model closer **43.10%**
- direction hit **44.83%**

WR is the only primary position with better raw Week-5 mean accuracy than the
preserved median book line.

## Full-board by market

pass_yards:
- rows 29
- MAE **60.8366**
- median-line MAE **56.2586**
- direction hit **51.72%**

rush_yards:
- rows 78
- MAE **22.4157**
- median-line MAE **20.1346**
- direction hit **54.55%**

rec_yards:
- rows 129
- MAE **22.5079**
- median-line MAE **22.1124**
- direction hit **53.13%**

receptions:
- rows 128
- MAE **1.5979**
- median-line MAE **1.6250**
- direction hit **55.65%**

rush_rec_yards:
- rows 28
- MAE **28.5483**
- median-line MAE **26.1429**
- direction hit **70.37%**

## Model-vs-line gap diagnostic

Pregame disagreement size was not a clean confidence score.

Gap quintiles:
- Q1 mean gap 0.20 -> MAE 3.59 -> direction 50.0%
- Q2 mean gap 0.77 -> MAE 5.39 -> direction 67.5%
- Q3 mean gap 2.99 -> MAE 21.60 -> direction 48.7%
- Q4 mean gap 7.10 -> MAE 24.48 -> direction 53.9%
- Q5 mean gap 21.26 -> MAE 39.52 -> direction 56.4%

Large disagreement clearly correlated with larger realized error. It did not
produce monotonic hit-rate improvement. This reinforces the earlier rule:
**raw biggest-edge sort is not a valid confidence selector.**

## Largest football misses

Among the largest absolute errors:
- Matthew Stafford pass yards: model 242.3, actual 383
- Drake Maye pass yards: 238.5 -> 374
- Jordan Love pass yards: 232.1 -> 367
- Jahmyr Gibbs rush+rec: 78.1 -> 207
- Tyler Shough pass yards: 223.0 -> 96
- Tyson Bagent pass yards: 208.9 -> 87
- Bo Nix pass yards: 238.5 -> 123
- Bucky Irving rush yards: 50.3 -> 165
- Josh Allen rush yards: 21.4 -> 121
- Deshaun Watson pass yards: 238.2 -> 155
- CeeDee Lamb rec yards: 81.0 -> 9

These are not yet assigned causal labels in this result. The protocol requires a
separate evidence-based player failure casebook.

## Production-selected settlement

Canonical selected-settlement result:
- archived rows: 354
- decided bets: **350**
- wins-losses: **199-151**
- hit rate: **56.86%**
- units: **+26.17**
- ROI: **+7.48%**
- selected model MAE: **19.31**
- selected Vegas-line MAE: **18.17**
- model closer rate: **52.0%**
- void DNP rows: 4
- unresolved rows: 0

By selected market:
- rush+rec: 20-6, **+11.64u**, +44.8% ROI
- receptions: 66-45, **+14.11u**, +12.7%
- rec yards: 61-51, **+3.05u**, +2.7%
- pass yards: 14-12, **+0.47u**, +1.8%
- rush yards: 38-37, **-3.10u**, -4.1%

By selected position:
- QB: 25-28, **-5.73u**
- RB: 75-49, **+17.27u**
- WR: 73-50, **+14.51u**
- TE: 26-24, **+0.12u**

Important interpretation:
the downstream selection/price layer added value in Week 5 even though the raw
football means were not globally superior to the market. Do not conflate the
two.

## Scientific boundary

No new threshold, input, coefficient, role rule, or model change may be fitted
from Week 5 alone.

Next required work:
1. player-level failure decomposition;
2. Week-5 frozen opportunity-lock grading;
3. ticket/concentration appendix;
4. continuity comparison with Weeks 1-4.
