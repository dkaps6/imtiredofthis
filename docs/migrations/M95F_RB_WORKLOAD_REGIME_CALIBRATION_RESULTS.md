# M95F — RB Workload Regime Calibration Results

## Authoritative run

- Workflow: `M95F RB Workload Regime Calibration`
- Run: `33389924330`
- Job: `99480953759`
- Tested SHA: `9f229a63c21c18e933a7f227e21c18dae59d4a2f`
- Branch: `research-rb-m95f-workload-regime-calibration`
- Artifact: `migration-95f-rb-workload-regime-calibration`
- Artifact ID: `9757054258`
- Artifact SHA256: `3d50dd9aaf5157add6c26faeac505edd0b0a39da87b191b80adf2d52b06d7026`
- Artifact size: `1,952,168` bytes
- Execution conclusion: success
- Scientific disposition: `RETAIN_M95F_AS_DIAGNOSTIC_DO_NOT_PROMOTE`
- Production change: `0`
- M94C central carry mean preserved: `1`

## Scientific question

M95E showed strong 20+/25+ workload-state ranking but badly uncalibrated class-balanced probabilities and an over-broad distribution. M95F tested whether that ranking signal could be calibrated into a realistic hurdle / regime distribution while keeping M94C as the official central carry estimate.

The raw M95E logistic tail scorer family was frozen. Temporal 2024 out-of-fold scores were used for calibration. Architecture selection occurred on 2024 W13-18 only, then the chosen calibration was refit without using 2025 outcomes and evaluated on untouched 2025.

Selected calibration:

- 20+ carries: Platt calibration
- 25+ carries: compact football-aware calibration
- frozen 20+ operating threshold: 0.20
- frozen 25+ operating threshold: 0.10

## Probability calibration worked

### 2024 W13-18 holdout

20+ carries:

- actual base rate: `0.093946`
- raw mean score: `0.269363`
- raw Brier: `0.125436`
- calibrated mean probability: `0.083557`
- calibrated Brier: `0.065236`
- calibrated ECE: `0.024917`
- AUC: `0.879263`

25+ carries:

- actual base rate: `0.035491`
- raw mean score: `0.209141`
- raw Brier: `0.109690`
- calibrated mean probability: `0.025503`
- calibrated Brier: `0.031213`
- calibrated ECE: `0.017833`
- calibrated AUC: `0.911001`

The 25+ football-aware calibrator improved ranking as well as probability scale.

### Untouched 2025 validation

20+ carries:

- actual base rate: `0.070352`
- raw mean score: `0.312868`
- raw Brier: `0.164458`
- raw log loss: `0.488688`
- calibrated mean probability: `0.100862`
- calibrated Brier: `0.062636`
- calibrated log loss: `0.208788`
- calibrated ECE: `0.033503`
- AUC: `0.846474`

25+ carries:

- actual base rate: `0.017229`
- raw mean score: `0.225069`
- raw Brier: `0.132016`
- raw log loss: `0.404711`
- calibrated mean probability: `0.030643`
- calibrated Brier: `0.017985`
- calibrated log loss: `0.078526`
- calibrated ECE: `0.015619`
- calibrated AUC: `0.844321`

This is a large and replicated calibration improvement. The class-balanced scorer was useful for ranking but was not a probability estimator. M95F fixed most of that probability-scale error.

## Frozen operating points reduced, but did not solve, false positives

### 2024 holdout

20+ at 0.20:

- 27 true positives
- 38 false positives
- 18 false negatives
- precision `41.5%`
- recall `60.0%`
- 65 flags vs 45 actual positives

25+ at 0.10:

- 7 true positives
- 27 false positives
- 10 false negatives
- precision `20.6%`
- recall `41.2%`
- 34 flags vs 17 actual positives

### Untouched 2025

20+ at frozen 0.20:

- 60 true positives
- 199 false positives
- 38 false negatives
- precision `23.2%`
- recall `61.2%`
- 259 flags vs 98 actual positives

25+ at frozen 0.10:

- 12 true positives
- 139 false positives
- 12 false negatives
- precision `7.95%`
- recall `50.0%`
- 151 flags vs 24 actual positives

This is substantially less over-triggered than M95E's raw class-balanced thresholds, but it is still too broad for production use.

## Hurdle-mixture mean is diagnostic only

The hurdle mixture deliberately adds high-state tail mass but does not replace M94C's central carry estimate.

### 2025 carry MAE

| Slice | M94C | M95F mixture | Gain |
|---|---:|---:|---:|
| All RB | 3.411003 | 3.629266 | -0.218262 |
| 0-5 | 2.559242 | 3.090332 | -0.531091 |
| 6-10 | 3.248217 | 3.697929 | -0.449712 |
| 11-14 | 3.470493 | 3.815317 | -0.344824 |
| 15+ | 5.336313 | 4.548456 | +0.787857 |
| 20+ | 7.876590 | 6.372565 | +1.504025 |
| 25+ | 11.954550 | 10.299993 | +1.654557 |
| Bellcow-60 | 5.309789 | 4.449706 | +0.860083 |

This again confirms that tail mass helps real high-workload games but hurts ordinary workloads when converted into one universal mixture mean. M94C must remain the central estimate.

## Tail diagnostics

### Actual 20+ games in 2025

- actual mean: `22.8673`
- M94C mean: `15.0061`
- M95F mixture mean: `16.5840`
- average p90: `22.9064`
- average p95: `24.9480`
- mean calibrated P(20+): `0.2630`
- mean calibrated P(25+): `0.0928`

### Actual 25+ games in 2025

- actual mean: `27.0000`
- M94C mean: `15.0455`
- M95F mixture mean: `16.7000`
- average p90: `23.2686`
- average p95: `25.3333`
- mean calibrated P(20+): `0.2544`
- mean calibrated P(25+): `0.0905`

M95F puts substantially more realistic workload into the upper distribution than M94C's mean, but even its p95 remains somewhat low in true 25+ games.

## Distribution coverage and failure gate

### 2024 holdout

- p50 coverage: `55.95%`
- p75 coverage: `76.62%`
- p90 coverage: `90.40%`
- p95 coverage: `94.99%`
- p90 >=25 count: 32 vs 17 actual
- p95 >=25 count: 81 vs 17 actual

### 2025 untouched validation

- p50 coverage: `60.01%`
- p75 coverage: `80.83%`
- p90 coverage: `92.96%`
- p95 coverage: `96.63%`
- p90 >=20 count: 444 vs 98 actual
- p95 >=20 count: 589 vs 98 actual
- p90 >=25 count: **154 vs 24 actual**
- p95 >=25 count: **284 vs 24 actual**

The pre-specified M95F coverage gate required the 2025 p90 >=25 count to be no greater than 5x the actual 25+ count (120). The observed count was 154, so `coverage_pass = 0` and `validation_pass = 0`.

## Remaining calibration pattern

The calibrated probability scale is much improved in lower and middle risk buckets, but the highest-risk population remains too optimistic.

The most important stability slice is the apparently stable workhorse group.

2025 stable workhorse:

- actual 20+ rate: `20.87%`
- predicted 20+ probability: `29.53%`
- actual 25+ rate: `4.33%`
- predicted 25+ probability: `11.12%`

The model still assumes that established workhorse structure converts into the extreme workload state much more often than it actually does.

Interestingly, explicit recent role-drop slices were much better calibrated:

Share drop >=15 percentage points:

- actual 20+ `8.72%`, predicted `9.30%`
- actual 25+ `3.36%`, predicted `3.44%`

Carry drop >=5:

- actual 20+ `12.62%`, predicted `12.00%`
- actual 25+ `5.83%`, predicted `5.34%`

This indicates that trailing role-trend information itself is not the main remaining issue.

## Representative false positives

High-risk games that remained ordinary or low volume include:

- Saquon Barkley PHI W12: actual 10, M94C 17.14, mixture 21.09, calibrated P20 ~0.713
- James Cook BUF W10: actual 13, M94C 20.85, mixture 21.68, P20 ~0.615
- Jaylen Warren PIT W17: actual 12, M94C 17.96, mixture 20.24, P20 ~0.602
- Derrick Henry BAL W2: actual 11, M94C 17.32, mixture 21.60, P20 ~0.586, P25 ~0.384
- Quinshon Judkins CLE W8: actual 9, M94C 18.54, mixture 21.26, P20 ~0.577
- Bijan Robinson ATL W12: actual 14, M94C 14.11, mixture 18.80, P20 ~0.553
- Alvin Kamara NO W12: actual 3, M94C 13.77, mixture 17.53, P20 ~0.477

The remaining false-positive problem is not simply low-quality players. Many are true stars/workhorses whose specific game did not enter the high-workload state.

## Representative remaining 25+ misses

- Kareem Hunt KC W12: actual 30, M94C 12.06, mixture 12.66, P20 ~0.068, P25 ~0.004
- Derrick Henry BAL W17: 36, M94C 17.92, mixture 19.71
- James Cook BUF W13: 32, M94C 16.13, mixture 17.25
- Jonathan Taylor IND W10: 32, M94C 14.38, mixture 17.44
- Rico Dowdle CAR W6: 30, M94C 13.84, mixture 15.50
- Cam Skattebo NYG W4: 25, M94C 10.31, mixture 10.97
- Kyle Monangai CHI W9: 26, M94C 11.28, mixture 12.85
- Christian McCaffrey SF W9: 28, M94C 13.58, mixture 15.30
- Emanuel Wilson GB W12: 28, M94C 14.05, mixture 17.01
- Kimani Vidal LAC W13: 25, M94C 12.14, mixture 14.97
- Kyren Williams LAR W9: 25, M94C 13.78, mixture 15.18

A notable subset of these misses are sudden role-transition / injury-replacement / promotion situations that trailing carry-share history cannot fully know. This points toward leakage-safe pregame depth-chart, roster and injury-report information as the next justified workload-state input family.

## Scientific conclusion

M95F validates the central architectural idea from M95E: **high-workload state probability is a real signal and should be modeled separately from the ordinary carry mean.**

Probability calibration itself worked very well. The raw rare-event scorer can be transformed from an exaggerated ranking score into a much more realistic probability scale without sacrificing AUC.

However, the resulting workload distribution is still too broad in the highest-risk region. Adding that tail mass back into one expected carry number still harms normal RB games, and the p90/p95 layers generate too many 20+/25+ possibilities.

Therefore M95F is retained as a diagnostic research component only. The next experiment should focus on **pregame role eligibility / role transition / availability** before any more tail mass is assigned. That means explicitly modeling depth-chart movement, competing-back availability/injuries, weekly roster status, newly promoted/replacement backs, and late-season rest/availability context. The goal is to distinguish a genuine upcoming bellcow opportunity from a player who merely has a historically bellcow-shaped profile.
