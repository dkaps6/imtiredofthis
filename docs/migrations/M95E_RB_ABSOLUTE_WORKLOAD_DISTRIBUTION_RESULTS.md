# M95E — Absolute RB Workload Distribution Results

## Authoritative run

- Workflow: `M95E RB Absolute Workload Distribution v3`
- Run: `33385407820`
- Job: `99466799556`
- Tested SHA: `daaacfb87374ea30cc22f704a4335275fd2013fb`
- Artifact: `migration-95e-rb-absolute-workload-distribution-v3`
- Artifact ID: `9755386298`
- Artifact SHA256: `b84adf070386f0006865baecf30e68074e5e064a4b12cda8e6de34e1d94010e8`
- Artifact size: `1,448,349` bytes
- Execution conclusion: success
- Scientific disposition: `RETAIN_M95E_AS_DIAGNOSTIC_DO_NOT_PROMOTE`
- Production change: `0`

The first two attempts are non-scientific implementation failures. Run 1 exposed that the frozen M94D file named as a 2024 holdout retained all 2024 RB rows while the M94C team trace was correctly W13-18 only. Run 2 fixed that universe alignment, then exposed NA truth labels for low-volume RB/FB rows that were in the M94D comparison universe but not enriched in M95B. V3 only corrected those mechanical identity/truth-universe issues. No model family, feature set, blend grid, threshold grid, selection score, or 2025 decision rule was changed from the pre-specified M95E experiment.

## Selected development architecture

The best fixed-grid architecture was:

- family: Ridge
- mode: explicit decomposition
- blend: 1.00 model / 0.00 prior

It was **not development-eligible**. Every tested architecture failed the pre-specified development protection gate.

### 2024 W13-18 architecture holdout

| Slice | M94C carry MAE | M95E carry MAE | Gain |
|---|---:|---:|---:|
| All RB | 3.435229 | 3.718867 | -0.283638 |
| 0-5 | 2.429739 | 2.985737 | -0.555998 |
| 6-10 | 3.261567 | 3.328645 | -0.067079 |
| 11-14 | 3.316555 | 3.873292 | -0.556736 |
| 15+ | 6.024434 | 5.746423 | +0.278010 |
| 20+ | 7.886385 | 6.991991 | +0.894394 |
| 25+ | 9.557158 | 8.742235 | +0.814924 |
| Bellcow-60 | 6.516953 | 6.059616 | +0.457337 |

This is the exact pattern the protection gate was designed to reject: material tail improvement bought by damaging ordinary/low/middle workloads.

## Untouched 2025 validation

| Slice | M94C carry MAE | M95E carry MAE | Gain |
|---|---:|---:|---:|
| All RB | 3.411003 | 3.705855 | -0.294852 |
| 0-5 | 2.559242 | 3.078378 | -0.519136 |
| 6-10 | 3.248217 | 3.206733 | +0.041485 |
| 11-14 | 3.470493 | 3.616027 | -0.145534 |
| 15+ | 5.336313 | 5.624497 | -0.288184 |
| 20+ | 7.876590 | 8.028468 | -0.151878 |
| 25+ | 11.954550 | 11.677741 | +0.276809 |
| Bellcow-60 | 5.309789 | 5.555759 | -0.245970 |

The 2024 tail gain did not generalize cleanly. Only the rare 25+ slice retained a modest +0.277 carry MAE improvement. Overall, 0-5, 11-14, 15+, 20+, and bellcow results were worse than M94C. This component must not replace the M94C carry mean.

## What the missing RB-room bridge did learn

At the **team RB-pool** level, M95E was useful specifically in high-RB-volume games:

| Team slice | M94C RB-pool MAE | M95E RB-pool MAE | Gain |
|---|---:|---:|---:|
| All team-games | 5.373542 | 5.400049 | -0.026506 |
| Actual RB pool 20+ | 5.128608 | 4.630047 | +0.498561 |
| Actual RB pool 25+ | 7.204408 | 6.597040 | +0.607367 |

So the architectural insight from M94D was real: explicitly separating full-team rushing from RB-room allocation helps recognize some games where a large fraction of the rushing workload belongs to RBs. It is not accurate enough to improve the universal player mean.

RB-room-share calibration in 2025 was monotonic but somewhat high-biased. Predicted quintile means rose from 0.759 to 0.891 while actual means rose from 0.738 to 0.858.

Lead-RB absolute team-rush-share calibration was also informative but compressed/miscalibrated: predicted quintile means rose from 0.352 to 0.692 while actual means rose from 0.395 to 0.625. The highest predicted bucket overstates the lead-back share while lower buckets tend to understate it.

## The 25+ carry problem is not solved by the mean

For the 24 actual 25+ carry games in 2025:

- actual mean: **27.00**
- M94C mean: **15.05**
- M95E mean: **15.32**
- M95E maximum mean projection: **19.41** inside the actual-25+ slice

M95E therefore confirms that a deterministic share-product mean remains too compressed. It cannot be promoted as the solution to the 25-30 carry state.

## Tail-state signal is strong, but badly calibrated

The pre-specified class-balanced logistic tail classifier produced:

### 2024 holdout

- 20+ AUC: **0.8793**
- 25+ AUC: **0.8895**

### 2025 untouched validation

- 20+ AUC: **0.8465**
- 25+ AUC: **0.8428**

This is a real, replicated ranking signal. However, the frozen thresholds are not usable as binary decisions:

- 20+ threshold 0.50: 85/98 true positives, but 329 false positives; precision 20.5%, recall 86.7%.
- 25+ threshold 0.30: 22/24 true positives, but 395 false positives; precision 5.3%, recall 91.7%.

The class-balanced model scores are therefore useful for ranking latent workload risk, not as calibrated event probabilities.

The 2025 top tail-score quintile contained 64 of 98 actual 20+ games and 17 of 24 actual 25+ games, but the raw probability scale substantially overstated actual rates. The top 25+ score quintile averaged roughly 0.747 predicted probability while the actual 25+ rate was only about 0.061.

## Distribution layer

The 2024-frozen beta-binomial style workload distribution produced sensible aggregate coverage in 2025:

- p50 coverage: 55.9%
- p75 coverage: 81.1%
- p90 coverage: 93.6%
- p95 coverage: 97.7%

For actual 25+ games:

- M95E deterministic mean: **15.32**
- p90 average: **24.50**
- p95 average: **27.38**

This is the first M95-series result that places realistic 25-30 carry outcomes inside the modeled distribution without simply raising every lead back's mean.

But the raw distribution is too broad to use directly. In 2025 it placed p90 at 25+ carries for 194 RB-games and p95 at 25+ for 350 RB-games, versus only 24 actual 25+ games. It is therefore diagnostic evidence that the tail belongs in a state/distribution layer, not a production-ready workload distribution.

## Representative failure modes

False high-workload cases include James Cook BUF W18 (actual 2, M95E mean 19.67), Alvin Kamara NO W12 (3 vs 18.70), Kaleb Johnson PIT W4 (6 vs 21.66), and several Derrick Henry/Jaylen Warren weeks. These show that stable role/concentration history can look like a bellcow state even when the specific week collapses because of rest, availability/role transition, injury, or realized game script.

Actual 25+ misses remain extreme: Derrick Henry BAL W17 actual 36 vs M95E 15.88; Kareem Hunt KC W12 30 vs 11.89; Jonathan Taylor IND W10 32 vs 15.27; Rico Dowdle CAR W6 30 vs 13.32; James Cook BUF W13 32 vs 15.64.

## Scientific conclusion

M95E rejects the idea that a better deterministic `team rushes × RB-room share × lead-RB share` mean alone will solve the workload tail. The missing RB-room allocation layer contains real signal, especially for high-volume RB rooms, but applying it continuously to every game hurts ordinary workloads.

At the same time, the replicated 20+/25+ AUC and the p90/p95 behavior show that the model can identify a **latent high-workload regime** substantially better than its deterministic mean expresses it.

The next justified experiment is therefore not a stronger mean multiplier. It is a narrow **hurdle / workload-regime calibration experiment**: preserve M94C as the central carry estimate, use the replicated M95E tail score plus football-environment evidence to estimate the probability of entering a 20+/25+ workload state, calibrate that state probability strictly on pre-2025 data, and add tail mass only through a mixture/distribution layer. False-positive controls for late-season rest, availability/role transitions, and unstable backfields should be investigated before any production use.
