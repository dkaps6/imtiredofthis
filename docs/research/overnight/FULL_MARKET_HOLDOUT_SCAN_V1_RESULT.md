STATUS: RESEARCH ONLY — NOT PROMOTED — AWAITING USER APPROVAL.

# Full-Market Holdout Scan V1

Extends the receptions-only situational search to rush_yards, rec_yards,
rush_rec_yards, and receptions, with the holdout discipline built in from the
start this time: categorical dimensions (side/home-away/week-bucket) checked
independently in both seasons; quantile dimensions (prob_edge, component_sd)
fit on one season and tested blind on the other, in both directions, requiring
BOTH directions positive with n>=25 to count as a candidate.

Categorical slices tested: 28. Candidates: 2.
Quantile-holdout rules tested: 24. Candidates (both directions positive): 7.

## Categorical candidates

| market     | kind        | dimension   | value    |   n_2024 |   roi_2024 |   wr_2024 |   n_2025 |   roi_2025 |   wr_2025 |   n_pooled |   roi_pooled | candidate   |
|:-----------|:------------|:------------|:---------|---------:|-----------:|----------:|---------:|-----------:|----------:|-----------:|-------------:|:------------|
| receptions | categorical | home_away   | HOME     |     1268 | 0.00968865 |  0.544164 |     1253 |  0.0204222 |  0.555467 |       2521 |    0.0150235 | True        |
| receptions | categorical | week_bucket | MID_7_12 |      877 | 0.00204164 |  0.542759 |      860 |  0.0319522 |  0.561628 |       1737 |    0.0168505 | True        |

## Quantile-holdout candidates (both fit/test directions positive)

| market         | dimension    | value              |   n_fit2024_test2025 |   roi_fit2024_test2025 |   n_fit2025_test2024 |   roi_fit2025_test2024 |
|:---------------|:-------------|:-------------------|---------------------:|-----------------------:|---------------------:|-----------------------:|
| rec_yards      | prob_edge    | top_quartile_OVER  |                  257 |             0.0325647  |                  277 |              0.0389963 |
| receptions     | prob_edge    | top_quartile       |                  713 |             0.00247966 |                  624 |              0.0188903 |
| receptions     | prob_edge    | top_quartile_OVER  |                  193 |             0.0322628  |                   95 |              0.0457608 |
| receptions     | prob_edge    | top_quartile_UNDER |                  509 |             0.0108612  |                  506 |              0.0201632 |
| rush_rec_yards | component_sd | top_quartile       |                  258 |             0.0583593  |                  292 |              0.0262149 |
| rush_rec_yards | component_sd | top_quartile_UNDER |                  218 |             0.0797161  |                  261 |              0.011701  |
| rush_yards     | component_sd | top_quartile_UNDER |                  197 |             0.0640857  |                  225 |              0.0314922 |

Full tables: `full_market_holdout_scan_categorical.csv`, `full_market_holdout_scan_quantile.csv`.

