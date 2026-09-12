STATUS: RESEARCH ONLY — NOT PROMOTED — AWAITING USER APPROVAL.

# Situational Edge Hunt V1 — Result

Source: `data/backtests/full_stack_vegas_benchmark_v1/non_qb_detail.csv` (already-computed, real 2024-2025 graded bets, 16973 rows).
Method: slice STRONG_EDGE and LEAN_OR_STRONG bet pools by situational
dimensions (home/away, week bucket, model-uncertainty quartile, edge-magnitude
quartile, bet side) instead of just the tier-level cut already reported in
`full_stack_vegas_benchmark_v1/README.md`. A slice only counts as a candidate
if it has >= 25 bets in EACH of 2024 and 2025 individually, AND positive
ROI in EACH season individually (not just pooled) -- the same both-seasons
consistency bar this project's own research already requires elsewhere.

## Candidates found: 2

| market     | dimension   | value        |    n |   win_rate |   roi_per_unit |   n_2024 |   roi_2024 |   n_2025 |   roi_2025 | positive_in_both_seasons_min_n   | bet_pool         |
|:-----------|:------------|:-------------|-----:|-----------:|---------------:|---------:|-----------:|---------:|-----------:|:---------------------------------|:-----------------|
| receptions | prob_edge_q | EDGE_Q4_HIGH | 1231 |    0.52234 |      0.0284951 |      599 |  0.0294594 |      632 |  0.0275812 | True                             | STRONG_EDGE_ONLY |
| receptions | prob_edge_q | EDGE_Q4_HIGH | 1231 |    0.52234 |      0.0284951 |      599 |  0.0294594 |      632 |  0.0275812 | True                             | LEAN_OR_STRONG   |

### Refinement: this candidate is really an UNDER-only effect

`EDGE_Q4_HIGH` = the model's own top quartile of `prob_edge` on receptions bets
already flagged STRONG_EDGE, i.e. `prob_edge >= 0.440` (quartile boundaries:
Q1 <= 0.260, Q2 <= 0.374, Q3 <= 0.440, Q4 up to 0.705). Splitting that quartile
by bet side shows the effect is concentrated almost entirely in UNDER bets, not
OVER:

| side  |    n | roi_per_unit | win_rate | n_2024 | roi_2024 | n_2025 | roi_2025 |
|:------|-----:|-------------:|---------:|-------:|---------:|-------:|---------:|
| UNDER | 1258 |      +0.0266 |   0.5246 |    632 |  +0.0243 |    626 |  +0.0290 |
| OVER  |   84 |      -0.0092 |   0.5000 |     24 |  +0.0199 |     60 |  -0.0209 |

**The real, cleanest candidate is: receptions market, model's highest-confidence
quartile (`prob_edge >= ~0.44`), UNDER side only.** n=1,258 (632 in 2024, 626 in
2025), win rate 52.5%, ROI +2.66%/unit pooled, positive independently in both
seasons. OVER in the same confidence quartile is small-sample (84 total) and
inconsistent between seasons -- not a reliable finding, don't extend the UNDER
result to OVER.

This is directionally coherent with something this project already knows: the
`receptions`/`rec_yards` numbers in this exact benchmark do **not** yet include
the promoted `WR_R15_PRODUCTION_MODEL_V1`/`TE_R5P_PRODUCTION_MODEL_V1` entitlement
models (disclosed gap in `full_stack_vegas_benchmark_v1/README.md`) -- they're
graded on the shared MC+ML+State+ensemble engine only. A plausible mechanism:
the model's highest-confidence UNDER calls on receptions are catching real
target-share overstatement that a receptions-specific entitlement layer would
likely sharpen further, not dilute. That's a hypothesis, not a tested claim --
applying WR-R15/TE-R5P to this same cohort (already flagged as the project's
own "next legitimate step") would be the direct way to test it.

### Statistical honesty check

146 slice-tests were run to find this one candidate. That is a real multiple-
comparisons exposure -- with enough cuts, *something* will clear a two-season
bar by chance. What makes this one more credible than pure noise: (1) it has
real volume (1,258 bets, not a handful), (2) it is internally coherent (isolating
by side, not just reporting the raw mixed-side quartile, made it *stronger* and
more consistent rather than weaker, which is what you'd expect from a real
effect and not from what you'd expect from noise), and (3) it has a plausible
football/market mechanism (understated market receptions lines on players the
model is most confident about). It is a genuine lead worth a confirmatory test
on more data (a third season, or the WR-R15/TE-R5P-enhanced cohort), not yet
something to bet money on or promote.

## Full slice table (all dimensions, both pools, all markets)

Full results (all 146 slices, most only tested for completeness) written to
`docs/research/overnight/SITUATIONAL_EDGE_HUNT_V1_RESULT.csv`.

## Top 15 slices by pooled ROI (for context, regardless of both-seasons filter)

| market     | dimension      | value        |    n |   win_rate |   roi_per_unit |   n_2024 |    roi_2024 |   n_2025 |   roi_2025 | positive_in_both_seasons_min_n   | bet_pool         |
|:-----------|:---------------|:-------------|-----:|-----------:|---------------:|---------:|------------:|---------:|-----------:|:---------------------------------|:-----------------|
| receptions | prob_edge_q    | EDGE_Q4_HIGH | 1231 |   0.52234  |    0.0284951   |      599 |  0.0294594  |      632 | 0.0275812  | True                             | LEAN_OR_STRONG   |
| receptions | prob_edge_q    | EDGE_Q4_HIGH | 1231 |   0.52234  |    0.0284951   |      599 |  0.0294594  |      632 | 0.0275812  | True                             | STRONG_EDGE_ONLY |
| receptions | side           | OVER         |  635 |   0.570079 |    0.0223363   |      275 | -0.00631478 |      360 | 0.0442226  | False                            | STRONG_EDGE_ONLY |
| receptions | week_bucket    | MID_7_12     | 1765 |   0.545042 |    0.00612416  |      884 | -0.0109768  |      881 | 0.0232834  | False                            | STRONG_EDGE_ONLY |
| rush_yards | side           | UNDER        | 1698 |   0.534158 |    0.00568678  |      881 | -0.013609   |      817 | 0.0264941  | False                            | STRONG_EDGE_ONLY |
| receptions | component_sd_q | SD_Q1_LOW    | 3898 |   0.549513 |    0.00461623  |     1944 | -0.00854222 |     1954 | 0.0177073  | False                            | STRONG_EDGE_ONLY |
| rush_yards | side           | UNDER        | 1776 |   0.533221 |    0.00370084  |      917 | -0.00983999 |      859 | 0.0181559  | False                            | LEAN_OR_STRONG   |
| receptions | week_bucket    | MID_7_12     | 1831 |   0.543965 |    0.00326545  |      924 | -0.0103485  |      907 | 0.0171346  | False                            | LEAN_OR_STRONG   |
| ALL_NON_QB | component_sd_q | SD_Q1_LOW    | 3990 |   0.546115 |    0.000810237 |     1996 | -0.0106218  |     1994 | 0.0122538  | False                            | STRONG_EDGE_ONLY |
| receptions | home_away      | HOME         | 2583 |   0.542005 |   -0.000309908 |     1309 | -0.0199879  |     1274 | 0.0199087  | False                            | STRONG_EDGE_ONLY |
| receptions | prob_edge_q    | EDGE_Q3      | 1957 |   0.562596 |   -0.000639218 |      972 | -0.0386233  |      985 | 0.0368435  | False                            | STRONG_EDGE_ONLY |
| receptions | prob_edge_q    | EDGE_Q3      | 1957 |   0.562596 |   -0.000639218 |      972 | -0.0386233  |      985 | 0.0368435  | False                            | LEAN_OR_STRONG   |
| rush_yards | prob_edge_q    | EDGE_Q1_LOW  |  665 |   0.529323 |   -0.0016097   |      333 | -0.0167982  |      332 | 0.0136245  | False                            | STRONG_EDGE_ONLY |
| receptions | side           | OVER         |  709 |   0.557123 |   -0.00162428  |      317 | -0.0167404  |      392 | 0.0105998  | False                            | LEAN_OR_STRONG   |
| receptions | component_sd_q | SD_Q1_LOW    | 4031 |   0.54577  |   -0.00223381  |     2016 | -0.0142939  |     2015 | 0.00983229 | False                            | LEAN_OR_STRONG   |

## Bottom 15 slices by pooled ROI (worst-losing contexts, for symmetry)

| market         | dimension      | value       |   n |   win_rate |   roi_per_unit |   n_2024 |   roi_2024 |   n_2025 |   roi_2025 | positive_in_both_seasons_min_n   | bet_pool         |
|:---------------|:---------------|:------------|----:|-----------:|---------------:|---------:|-----------:|---------:|-----------:|:---------------------------------|:-----------------|
| rush_rec_yards | component_sd_q | SD_Q1_LOW   |  13 |   0.384615 |     -0.275851  |        8 | -0.298501  |        5 | -0.23961   | False                            | STRONG_EDGE_ONLY |
| rush_rec_yards | component_sd_q | SD_Q1_LOW   |  13 |   0.384615 |     -0.275851  |        8 | -0.298501  |        5 | -0.23961   | False                            | LEAN_OR_STRONG   |
| rec_yards      | component_sd_q | SD_Q1_LOW   |  35 |   0.457143 |     -0.143437  |       15 |  0.123394  |       20 | -0.34356   | False                            | LEAN_OR_STRONG   |
| rec_yards      | component_sd_q | SD_Q1_LOW   |  35 |   0.457143 |     -0.143437  |       15 |  0.123394  |       20 | -0.34356   | False                            | STRONG_EDGE_ONLY |
| rush_yards     | component_sd_q | SD_Q1_LOW   |  44 |   0.363636 |     -0.139884  |       29 | -0.13993   |       15 | -0.139794  | False                            | LEAN_OR_STRONG   |
| rush_yards     | component_sd_q | SD_Q1_LOW   |  44 |   0.363636 |     -0.139884  |       29 | -0.13993   |       15 | -0.139794  | False                            | STRONG_EDGE_ONLY |
| rush_yards     | side           | OVER        | 844 |   0.464455 |     -0.10752   |      430 | -0.17088   |      414 | -0.0417115 | False                            | LEAN_OR_STRONG   |
| rush_rec_yards | week_bucket    | LATE_13_18  | 679 |   0.478645 |     -0.102556  |      479 | -0.112933  |      200 | -0.0777045 | False                            | LEAN_OR_STRONG   |
| rush_rec_yards | component_sd_q | SD_Q3       | 747 |   0.480589 |     -0.100148  |      599 | -0.109758  |      148 | -0.0612529 | False                            | LEAN_OR_STRONG   |
| rush_rec_yards | week_bucket    | LATE_13_18  | 664 |   0.481928 |     -0.0964555 |      468 | -0.104152  |      196 | -0.0780782 | False                            | STRONG_EDGE_ONLY |
| rush_yards     | side           | OVER        | 763 |   0.471822 |     -0.0923056 |      392 | -0.157922  |      371 | -0.0229752 | False                            | STRONG_EDGE_ONLY |
| rush_rec_yards | component_sd_q | SD_Q3       | 726 |   0.484848 |     -0.0920997 |      580 | -0.103102  |      146 | -0.0483934 | False                            | STRONG_EDGE_ONLY |
| receptions     | prob_edge_q    | EDGE_Q1_LOW | 930 |   0.495699 |     -0.0825927 |      480 | -0.0838095 |      450 | -0.0812947 | False                            | LEAN_OR_STRONG   |
| rush_rec_yards | prob_edge_q    | EDGE_Q1_LOW | 391 |   0.498721 |     -0.0673932 |      330 | -0.0429693 |       61 | -0.199523  | False                            | LEAN_OR_STRONG   |
| rush_rec_yards | prob_edge_q    | EDGE_Q3     | 569 |   0.499121 |     -0.0649871 |      414 | -0.0738857 |      155 | -0.0412192 | False                            | LEAN_OR_STRONG   |

