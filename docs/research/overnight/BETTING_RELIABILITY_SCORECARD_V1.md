# Betting Reliability Scorecard V1

Source: `data/backtests/full_stack_vegas_benchmark_v1/non_qb_detail.csv` (16973 real graded bets, 2024-2025, non-QB markets only; QB pass_yards has no comparable row-level detail file -- see `data/backtests/full_stack_vegas_benchmark_v1/qb_*_summary.csv` separately).
Bar for VALIDATED: positive ROI independently in each of 2024 and 2025, with >= 25 bets in each season. Same bar `situational_edge_hunt_v1.py` already uses; that script tested 146 finer situational slices and found zero pass. This scorecard stays at a coarser grain (market x signal tier x edge quartile x side) so a pass here is less likely to be a multiple-comparisons artifact.

## Validated cuts: 1

| cut                    | market     |    n |   win_rate |   roi_per_unit |   n_2024 |   roi_2024 |   n_2025 |   roi_2025 | validated_both_seasons_positive   |   signal | prob_edge_q   |   side |
|:-----------------------|:-----------|-----:|-----------:|---------------:|---------:|-----------:|---------:|-----------:|:----------------------------------|---------:|:--------------|-------:|
| market_x_edge_quartile | receptions | 1231 |    0.52234 |      0.0284951 |      599 |  0.0294594 |      632 |  0.0275812 | True                              |      nan | EDGE_Q4_HIGH  |    nan |

## Market-level summary (all bets, for context)

| market         |    n |   win_rate |   roi_per_unit |   n_2024 |   roi_2024 |   n_2025 |    roi_2025 | validated_both_seasons_positive   |   signal |   prob_edge_q |   side |
|:---------------|-----:|-----------:|---------------:|---------:|-----------:|---------:|------------:|:----------------------------------|---------:|--------------:|-------:|
| receptions     | 5712 |   0.53659  |     -0.0134989 |     2868 | -0.0291136 |     2844 |  0.00224747 | False                             |      nan |           nan |    nan |
| rec_yards      | 5863 |   0.512536 |     -0.0349829 |     3025 | -0.0600512 |     2838 | -0.00826276 | False                             |      nan |           nan |    nan |
| rush_yards     | 2772 |   0.50938  |     -0.0359705 |     1433 | -0.0617497 |     1339 | -0.00838153 | False                             |      nan |           nan |    nan |
| rush_rec_yards | 2626 |   0.505712 |     -0.0521272 |     1938 | -0.0669339 |      688 | -0.0104188  | False                             |      nan |           nan |    nan |

## Market x signal tier (STRONG_EDGE / LEAN_EDGE / NO_EDGE)

| market         |    n |   win_rate |   roi_per_unit |   n_2024 |    roi_2024 |   n_2025 |     roi_2025 | validated_both_seasons_positive   | signal      |   prob_edge_q |   side |
|:---------------|-----:|-----------:|---------------:|---------:|------------:|---------:|-------------:|:----------------------------------|:------------|--------------:|-------:|
| receptions     | 5367 |   0.540712 |    -0.00550771 |     2688 | -0.0238346  |     2679 |  0.0128808   | False                             | STRONG_EDGE |           nan |    nan |
| rush_yards     | 2461 |   0.514831 |    -0.0246944  |     1273 | -0.0580479  |     1188 |  0.0110454   | False                             | STRONG_EDGE |           nan |    nan |
| rec_yards      | 5361 |   0.515575 |    -0.0291931  |     2759 | -0.0577425  |     2602 |  0.00107882  | False                             | STRONG_EDGE |           nan |    nan |
| rush_rec_yards | 2481 |   0.507457 |    -0.0487509  |     1808 | -0.0670761  |      673 |  0.000479488 | False                             | STRONG_EDGE |           nan |    nan |
| receptions     |  167 |   0.51497  |    -0.0595555  |       83 | -0.0456944  |       84 | -0.0732515   | False                             | NO_EDGE     |           nan |    nan |
| rec_yards      |  269 |   0.498141 |    -0.0622037  |      145 | -0.00350812 |      124 | -0.13084     | False                             | NO_EDGE     |           nan |    nan |
| rush_rec_yards |   58 |   0.482759 |    -0.0959916  |       51 | -0.0824927  |        7 | -0.194341    | False                             | LEAN_EDGE   |           nan |    nan |
| rush_yards     |  152 |   0.480263 |    -0.10221    |       86 | -0.0696011  |       66 | -0.144701    | False                             | NO_EDGE     |           nan |    nan |
| rush_rec_yards |   87 |   0.471264 |    -0.119168   |       79 | -0.0536349  |        8 | -0.766304    | False                             | NO_EDGE     |           nan |    nan |
| rec_yards      |  233 |   0.459227 |    -0.13677    |      121 | -0.180453   |      112 | -0.0895773   | False                             | LEAN_EDGE   |           nan |    nan |
| rush_yards     |  159 |   0.45283  |    -0.147178   |       74 | -0.116308   |       85 | -0.174053    | False                             | LEAN_EDGE   |           nan |    nan |
| receptions     |  178 |   0.432584 |    -0.211237   |       97 | -0.161212   |       81 | -0.271144    | False                             | LEAN_EDGE   |           nan |    nan |

## Market x edge-magnitude quartile

| market         |    n |   win_rate |   roi_per_unit |   n_2024 |    roi_2024 |   n_2025 |    roi_2025 | validated_both_seasons_positive   |   signal | prob_edge_q   |   side |
|:---------------|-----:|-----------:|---------------:|---------:|------------:|---------:|------------:|:----------------------------------|---------:|:--------------|-------:|
| receptions     | 1231 |   0.52234  |    0.0284951   |      599 |  0.0294594  |      632 |  0.0275812  | True                              |      nan | EDGE_Q4_HIGH  |    nan |
| receptions     | 1957 |   0.562596 |   -0.000639218 |      972 | -0.0386233  |      985 |  0.0368435  | False                             |      nan | EDGE_Q3       |    nan |
| receptions     | 1427 |   0.542397 |   -0.0169415   |      734 | -0.0266769  |      693 | -0.00663009 | False                             |      nan | EDGE_Q2       |    nan |
| rush_yards     |  475 |   0.52     |   -0.017562    |      248 | -0.0678013  |      227 |  0.0373251  | False                             |      nan | EDGE_Q3       |    nan |
| rec_yards      | 1242 |   0.520129 |   -0.0199419   |      652 | -0.0672801  |      590 |  0.0323709  | False                             |      nan | EDGE_Q3       |    nan |
| rec_yards      | 1609 |   0.52082  |   -0.0202165   |      839 | -0.066746   |      770 |  0.0304825  | False                             |      nan | EDGE_Q2       |    nan |
| rush_yards     |  645 |   0.513178 |   -0.0214242   |      348 | -0.0871315  |      297 |  0.0555661  | False                             |      nan | EDGE_Q4_HIGH  |    nan |
| rec_yards      | 1319 |   0.514026 |   -0.0305573   |      644 | -0.0296639  |      675 | -0.0314097  | False                             |      nan | EDGE_Q4_HIGH  |    nan |
| rush_rec_yards |  531 |   0.516008 |   -0.0333421   |      444 |  0.00395634 |       87 | -0.223693   | False                             |      nan | EDGE_Q2       |    nan |
| rush_yards     |  976 |   0.509221 |   -0.0409915   |      493 | -0.0409457  |      483 | -0.0410382  | False                             |      nan | EDGE_Q1_LOW   |    nan |
| rush_rec_yards | 1048 |   0.509542 |   -0.043402    |      671 | -0.122904   |      377 |  0.0980993  | False                             |      nan | EDGE_Q4_HIGH  |    nan |
| rush_yards     |  676 |   0.498521 |   -0.0555355   |      344 | -0.0615251  |      332 | -0.0493294  | False                             |      nan | EDGE_Q2       |    nan |
| rec_yards      | 1693 |   0.497933 |   -0.0634988   |      890 | -0.0704325  |      803 | -0.0558138  | False                             |      nan | EDGE_Q1_LOW   |    nan |
| rush_rec_yards |  569 |   0.499121 |   -0.0649871   |      414 | -0.0738857  |      155 | -0.0412192  | False                             |      nan | EDGE_Q3       |    nan |
| rush_rec_yards |  478 |   0.493724 |   -0.0768166   |      409 | -0.0450294  |       69 | -0.265237   | False                             |      nan | EDGE_Q1_LOW   |    nan |
| receptions     | 1097 |   0.498633 |   -0.0790856   |      563 | -0.0781904  |      534 | -0.0800295  | False                             |      nan | EDGE_Q1_LOW   |    nan |

## Market x side (OVER / UNDER)

| market         |    n |   win_rate |   roi_per_unit |   n_2024 |    roi_2024 |   n_2025 |    roi_2025 | validated_both_seasons_positive   |   signal |   prob_edge_q | side   |
|:---------------|-----:|-----------:|---------------:|---------:|------------:|---------:|------------:|:----------------------------------|---------:|--------------:|:-------|
| rush_yards     | 1850 |   0.531892 |    0.000918008 |      959 | -0.00639917 |      891 |  0.00879362 | False                             |      nan |           nan | UNDER  |
| receptions     |  792 |   0.550505 |   -0.0092319   |      356 | -0.0134898  |      436 | -0.00575524 | False                             |      nan |           nan | OVER   |
| receptions     | 4920 |   0.53435  |   -0.0141858   |     2512 | -0.0313277  |     2408 |  0.00369646 | False                             |      nan |           nan | UNDER  |
| rec_yards      | 4130 |   0.513075 |   -0.0328239   |     2131 | -0.0650356  |     1999 |  0.0015148  | False                             |      nan |           nan | UNDER  |
| rec_yards      | 1733 |   0.511252 |   -0.040128    |      894 | -0.04817    |      839 | -0.0315588  | False                             |      nan |           nan | OVER   |
| rush_rec_yards | 2186 |   0.506404 |   -0.0507158   |     1557 | -0.0743063  |      629 |  0.00767925 | False                             |      nan |           nan | UNDER  |
| rush_rec_yards |  440 |   0.502273 |   -0.0591394   |      381 | -0.0368056  |       59 | -0.203362   | False                             |      nan |           nan | OVER   |
| rush_yards     |  922 |   0.464208 |   -0.109988    |      474 | -0.173735   |      448 | -0.0425401  | False                             |      nan |           nan | OVER   |

## Side skew

The model's chosen side across all 16973 bets is UNDER 13086 times vs OVER 3887 times (77.1% UNDER). Combined with pooled ROI of -2.40% on UNDER vs -5.26% on OVER, this is consistent with a systematic overprediction bias in the underlying projections, not just noise -- worth investigating as a calibration issue separately from this scorecard's per-slice question.

## Full cut table

All 73 rows across every cut tested written to `docs/research/overnight/BETTING_RELIABILITY_SCORECARD_V1.csv`.

