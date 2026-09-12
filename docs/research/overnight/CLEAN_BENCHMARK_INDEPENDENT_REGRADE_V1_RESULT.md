STATUS: RESEARCH ONLY — NOT PROMOTED. Independent re-grade of the
identity-clean benchmark (PR #541). No production/model/threshold change.

# Clean Benchmark Independent Re-Grade V1

Rows: 17715. Seasons: [2024, 2025].
Markets: ['pass_yards', 'rec_yards', 'receptions', 'rush_rec_yards', 'rush_yards'].

## 1. Model-vs-Vegas MAE/bias by season/market

|   season | market         |    n |   model_mae |   vegas_mae |   model_bias |   vegas_bias | model_beats_vegas   |
|---------:|:---------------|-----:|------------:|------------:|-------------:|-------------:|:--------------------|
|     2024 | pass_yards     |  429 |    60.7993  |    58.9172  |   -13.9654   |    1.20396   | False               |
|     2024 | rec_yards      | 3021 |    21.576   |    19.9694  |   -10.7029   |   -5.18818   | False               |
|     2024 | receptions     | 2863 |     1.73705 |     1.57405 |    -1.05612  |   -0.196822  | False               |
|     2024 | rush_rec_yards | 1937 |    31.2532  |    26.2109  |   -18.9615   |   -4.58725   | False               |
|     2024 | rush_yards     | 1426 |    21.2021  |    19.0042  |   -10.1944   |   -3.23983   | False               |
|     2025 | pass_yards     |  383 |    58.4716  |    54.4243  |   -10.4189   |    0.834204  | False               |
|     2025 | rec_yards      | 2832 |    20.9666  |    19.6624  |    -9.44386  |   -3.75777   | False               |
|     2025 | receptions     | 2839 |     1.62409 |     1.48979 |    -0.884019 |   -0.0368087 | False               |
|     2025 | rush_rec_yards |  654 |    36.3112  |    29.5015  |   -28.2301   |   -2.12232   | False               |
|     2025 | rush_yards     | 1331 |    20.2027  |    19.0823  |    -9.16536  |   -3.23403   | False               |

**Model beats Vegas on raw MAE in 0 of 10 season/market cells.**

## 1b. Model-vs-Vegas MAE by season/position

|   season | position   |    n |   model_mae |   vegas_mae | model_beats_vegas   |
|---------:|:-----------|-----:|------------:|------------:|:--------------------|
|     2024 | QB         |  868 |     37.5632 |    35.9793  | False               |
|     2024 | RB         | 3095 |     20.263  |    16.9691  | False               |
|     2024 | TE         | 1372 |     11.7992 |    10.8156  | False               |
|     2024 | WR         | 4341 |     17.3927 |    15.9851  | False               |
|     2025 | QB         |  806 |     34.1355 |    31.9479  | False               |
|     2025 | RB         | 2768 |     19.2636 |    17.0394  | False               |
|     2025 | TE         | 1355 |     10.7741 |     9.90369 | False               |
|     2025 | WR         | 3110 |     13.3719 |    12.5775  | False               |

**Model beats Vegas on raw MAE in 0 of 8 season/position cells.**


Overall: model_mae=18.2912, vegas_mae=16.5640. Model is consistently, substantially UNDER-biased (model_bias strongly negative in every market/season) while Vegas's own bias is much smaller in magnitude — this is a new, clean-cohort finding not previously isolated this cleanly.

## 2. ROI/side/market decomposition (frozen grading rules)

### tier=ALL_NO_FILTER (n=17715)

| market         | side   |    n |   win_rate |         roi |
|:---------------|:-------|-----:|-----------:|------------:|
| pass_yards     | OVER   |  206 |   0.5      | -0.0605842  |
| pass_yards     | UNDER  |  606 |   0.519802 | -0.021604   |
| rec_yards      | OVER   | 1719 |   0.514834 | -0.0329053  |
| rec_yards      | UNDER  | 4134 |   0.514272 | -0.030626   |
| receptions     | OVER   |  787 |   0.552732 | -0.00671253 |
| receptions     | UNDER  | 4915 |   0.535097 | -0.012651   |
| rush_rec_yards | OVER   |  432 |   0.516204 | -0.0332141  |
| rush_rec_yards | UNDER  | 2159 |   0.508569 | -0.0465783  |
| rush_yards     | OVER   |  919 |   0.457018 | -0.123607   |
| rush_yards     | UNDER  | 1838 |   0.527203 | -0.00783251 |

### tier=LEAN_OR_STRONG (n=16986)

| market         | side   |    n |   win_rate |          roi |
|:---------------|:-------|-----:|-----------:|-------------:|
| pass_yards     | OVER   |  185 |   0.497297 | -0.0653071   |
| pass_yards     | UNDER  |  572 |   0.513986 | -0.0320089   |
| rec_yards      | OVER   | 1583 |   0.517372 | -0.0280848   |
| rec_yards      | UNDER  | 4014 |   0.513951 | -0.0312689   |
| receptions     | OVER   |  696 |   0.557471 | -0.000351704 |
| receptions     | UNDER  | 4814 |   0.535937 | -0.0110736   |
| rush_rec_yards | OVER   |  386 |   0.515544 | -0.0341317   |
| rush_rec_yards | UNDER  | 2127 |   0.508228 | -0.0471443   |
| rush_yards     | OVER   |  840 |   0.461905 | -0.112274    |
| rush_yards     | UNDER  | 1769 |   0.529678 | -0.0030004   |

### tier=STRONG_ONLY (n=16322)

| market         | side   |    n |   win_rate |         roi |
|:---------------|:-------|-----:|-----------:|------------:|
| pass_yards     | OVER   |  169 |   0.485207 | -0.0880648  |
| pass_yards     | UNDER  |  552 |   0.507246 | -0.0445328  |
| rec_yards      | OVER   | 1473 |   0.516633 | -0.0293107  |
| rec_yards      | UNDER  | 3882 |   0.51185  | -0.035286   |
| receptions     | OVER   |  624 |   0.5625   |  0.0100499  |
| receptions     | UNDER  | 4725 |   0.535661 | -0.0111573  |
| rush_rec_yards | OVER   |  356 |   0.505618 | -0.0528023  |
| rush_rec_yards | UNDER  | 2092 |   0.510516 | -0.0427846  |
| rush_yards     | OVER   |  757 |   0.468956 | -0.0976133  |
| rush_yards     | UNDER  | 1692 |   0.530142 | -0.00185091 |

## 3. STRONG coverage + component_sd calibration diagnostics

| market         |    n |   strong_pct |
|:---------------|-----:|-------------:|
| pass_yards     |  812 |     0.887931 |
| rec_yards      | 5853 |     0.914915 |
| receptions     | 5702 |     0.938092 |
| rush_rec_yards | 2591 |     0.944809 |
| rush_yards     | 2757 |     0.888284 |

non-QB STRONG%=0.9230 (n=16903), QB STRONG%=0.8879 (n=812).

STRONG% by component_sd quartile (all markets):

| component_sd_q   |        0 |
|:-----------------|---------:|
| Q1_lowest        | 0.940393 |
| Q2               | 0.929104 |
| Q3               | 0.909214 |
| Q4_highest       | 0.906751 |

Same direction as the pre-rebuild mechanism check (STRONG_GATE_OVERCONFIDENCE_MECHANISM_CHECK.md): overconfidence is worst where model components agree MOST (lowest component_sd), consistent with `component_sd` being the wrong quantity for the Normal-CDF fair-probability formula. This replicates independent of the identity bug — it was never caused by it.

## 4. Holdout scan from scratch (every prior candidate treated as nonexistent)

### Categorical candidates (both seasons positive, n>=25)

None.

### Quantile-holdout candidates (both fit/test directions positive, n>=25)

| market         | dim          | side   |   n_fit2024_test2025 |   roi_fit2024_test2025 |   n_fit2025_test2024 |   roi_fit2025_test2024 |
|:---------------|:-------------|:-------|---------------------:|-----------------------:|---------------------:|-----------------------:|
| rush_yards     | component_sd | UNDER  |                  196 |             0.040572   |                  221 |             0.0157869  |
| rush_rec_yards | component_sd | BOTH   |                  259 |             0.0330842  |                  276 |             0.0248503  |
| rush_rec_yards | component_sd | UNDER  |                  210 |             0.058476   |                  262 |             0.00825589 |
| receptions     | prob_edge    | BOTH   |                  699 |             0.00387183 |                  638 |             0.031233   |
| receptions     | prob_edge    | UNDER  |                  595 |             0.0186788  |                  582 |             0.0295642  |

**Comparison to the pre-fix (corrupted-cohort) holdout candidates (FULL_MARKET_HOLDOUT_SCAN_V1_RESULT.md):**
- Survived on the clean cohort: [('receptions', 'prob_edge', 'UNDER'), ('rush_rec_yards', 'component_sd', 'BOTH'), ('rush_rec_yards', 'component_sd', 'UNDER'), ('rush_yards', 'component_sd', 'UNDER')]
- Vanished on the clean cohort (were only identity-bug artifacts, not real): [('rec_yards', 'prob_edge', 'OVER'), ('receptions', 'prob_edge', 'OVER')]
- New on the clean cohort (did not appear pre-fix): [('receptions', 'prob_edge', 'BOTH')]

**2 of 6 prior candidates disappeared once the identity bug was fixed.** That's direct evidence some of what looked like an 'edge' before was the corrupted game_id join itself, not a football signal — exactly the risk this whole rebuild exists to rule out.


Same disclosed caveat as before the identity fix: this probability layer still uses `Normal(mean=proj, sd=component_sd)`, not production's real simulated distribution, and part 3 above shows that layer is still measurably overconfident. Any candidate surviving this scan is therefore still fidelity-limited, not a confirmed real edge — the identity fix repairs *which game* each row is graded against, not the probability/EV math itself.

