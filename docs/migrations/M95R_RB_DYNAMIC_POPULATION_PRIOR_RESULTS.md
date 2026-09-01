# M95R — Expanded-Panel Dynamic Population-Prior Candidate — Results

## Authoritative run

- Workflow: `M95R RB Dynamic Population Prior`
- Run: **33453141931**
- Job: **99687269044**
- Tested SHA: **272e7537f9642a977e84ffeeda4b9d7d17832c71**
- Artifact: **9780473734**
- Artifact name: `migration-95r-rb-dynamic-population-prior`
- Artifact SHA256: **0d114c1f7cc1d2132c7ebd2c7074bd840b36d669c4a31c33e151cbb85be76821**
- Execution: **success**
- Scientific disposition: **`M95R_RETAIN_DIAGNOSTIC_DO_NOT_PROMOTE`**
- Feature search: `0`
- Hyperparameter search: `0`
- Sportsbook input: `0`
- Production change: `0`

The workflow completed successfully. The disposition is a scientific failure of the frozen M95R candidate, not a mechanical/data failure.

## Frozen candidate tested

M95R kept M95F as a fixed stable-workhorse 20+ backbone and fit only a bounded additive population/workload adjustment in log-odds space:

`logit(P20_R) = logit(P20_M95F) + clipped_dynamic_delta`

Frozen pregame-only inputs:

1. league season-to-date lead-RB 20+ rate;
2. league prior-four lead-RB 20+ rate;
3. team prior-four lead-RB 20+ rate;
4. team prior-four lead-RB 25+ rate;
5. team prior-four mean lead-RB carries.

Other frozen mechanics:

- ridge lambda `10.0`;
- delta cap `[-0.75,+0.75]` log-odds;
- no feature/hyperparameter search;
- strict earlier-season expanding-window training;
- 2023 trained on 2020-2022;
- 2024 trained on 2020-2023;
- 2025 trained on 2020-2024;
- primary scope W13-18 stable workhorses;
- 25+ diagnostic only;
- M94C central carries unchanged;
- M95I vacancy regime unchanged/separate.

## Primary rolling results

| Scope | n | Actual 20+ | M95F mean | R mean | M95F Brier | R Brier | Brier gain | M95F logloss | R logloss | AUC change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2023 W13-18 | 73 | 32.88% | 16.19% | 20.09% | .233221 | .221407 | **+.011814** | .673392 | .638868 | -.024660 |
| 2024 W13-18 | 77 | 35.06% | 27.38% | 27.73% | .206384 | .206530 | -.000147 | .597372 | .597456 | -.001481 |
| 2025 W13-18 | 85 | 28.24% | 29.43% | **43.12%** | .194926 | .216791 | **-.021865** | .568512 | .619034 | +.012295 |
| Pooled | 235 | 31.91% | 24.65% | 30.92% | .210576 | .214863 | **-.004287** | .610548 | .618125 | **-.022417** |

### Calibration

2023:

- M95F calibration gap: `+16.69 pp` underprediction;
- R gap: `+12.79 pp`;
- absolute-gap improvement: `+3.90 pp`.

2024:

- M95F gap: `+7.69 pp`;
- R gap: `+7.33 pp`;
- essentially neutral.

2025 late:

- M95F gap: `-1.20 pp`; the baseline was already almost perfectly centered;
- R gap: **`-14.88 pp`**;
- absolute-gap regression: **`13.69 pp`**.

Pooled mean calibration looked deceptively good: R moved the pooled mean from `24.65%` toward the pooled actual `31.91%`, leaving only a `0.99 pp` pooled gap. But Brier, logloss and AUC all worsened. Matching the pooled event rate is therefore not enough; the adjustment assigned probability mass to the wrong player-games.

## 2025 full-season secondary diagnostic

2025 full stable-workhorse trace (`n=237`, 52 20+ events):

- actual rate: **21.94%**;
- M95F mean probability: **29.60%**;
- M95F gap: `-7.66 pp`;
- R mean probability: **43.48%**;
- R gap: **`-21.54 pp`**;
- M95F Brier `.186593` -> R `.231516` (**regression .044923**);
- M95F logloss `.554301` -> R `.653694` (**regression .099393**);
- AUC `.581185 -> .581809` (essentially unchanged).

This is strong evidence that the R failure is not simply a ranking problem. The multi-season residual population prior injected too much absolute tail mass into 2025 while barely changing discrimination.

## Why 2025 is especially informative

2025 tells us several things simultaneously:

1. **M95F was already close to correctly centered late in the season.** W13-18 actual 20+ was `28.24%` vs M95F `29.43%`.
2. **The historical feed/ceiling concept still had ranking value in 2025.** Frozen M95K improved 2025 stable 20+ AUC `.581185 -> .641164`, and late-season AUC `.646858 -> .732923`.
3. **But adding generic population tail mass is different from reranking players.** M95R raised late-2025 mean probability to `43.12%`, badly overpredicting the population even though AUC improved slightly.
4. **The workload regime moved faster than a cumulative multi-season residual calibrator could adapt.** Training through 2023-2024 taught R to repair historical underprediction; by 2025, that correction was stale and harmful.
5. Therefore the next architecture must separate **population-level calibration/mass** from **player-level discrimination/ranking** instead of letting one slow cross-season residual layer do both.

The casebook makes the failure concrete. Many 2025 stable workhorses hit the +0.75 log-odds cap, including games where the baseline was already moderate/high and the actual workload stayed below 20 carries. Examples include Derrick Henry W15 (11 carries; `.393 -> .578`), Jonathan Taylor W18 (14; `.431 -> .616`), Quinshon Judkins W14 (14; `.433 -> .618`) and Breece Hall W13 (19; `.435 -> .620`).

## Frozen advancement gates

- pooled Brier improves: **FAIL**
- pooled logloss improves: **FAIL**
- pooled AUC regression <= .02: **FAIL** (`-.022417`)
- Brier improves in >=2/3 seasons: **FAIL** (1/3)
- absolute calibration gap improves in >=2/3 seasons: **PASS** (2/3)
- no season Brier regression > .01: **FAIL** (worst `.021865`)
- no season absolute-gap regression > 2.5 pp: **FAIL** (worst `13.685 pp`)

Disposition is therefore frozen as **`M95R_RETAIN_DIAGNOSTIC_DO_NOT_PROMOTE`**.

Do not retune this exact architecture against the now-exposed 2023-2025 rolling outcomes.

## 25+ audit

Late-season exact stable-workhorse 25+ events:

- 2020: 5 / 44
- 2021: 8 / 61
- 2022: 2 / 72
- 2023: 10 / 73
- 2024: 12 / 77
- 2025: 6 / 85

M95R correctly kept 25+ as a diagnostic and did not fit a specialized rare-event candidate.

## Durable historical player source cache

M95Q itself had not permanently stored the raw nflverse weekly player source sheets. Before M95R evaluation, this was corrected with a one-time deterministic source snapshot.

Authoritative cache build:

- Workflow: `M95R Historical Player Source Cache`
- successful run: **33453001235**
- job: **99686833509**
- cache-workflow SHA: **07e0ceb5b6f0e8eda704c4735ba7b2ff2efa32ee**
- persisted cache commit: **402c4a28593e0c79438ea10c88d832a98f0a6f2e**
- path: `data/research_cache/nflverse_player_weekly/`
- seasons: **2018-2025**
- format: one parquet file per season plus checksummed `manifest.csv`
- each source snapshot contains 150 columns.

Future historical research should use this persisted source snapshot unless a deliberate source refresh is required. It should not redownload the entire weekly player universe on every migration.

## Scientific interpretation / next step

M95R falsifies the simple idea that a slowly learned multi-season residual population prior can safely repair M95F across changing workload regimes. It helped 2023, was neutral in 2024, and materially overcorrected 2025.

This does **not** falsify dynamic game/population context. M95N-P-Q already established that regime structure is real. Instead, M95R shows that adaptation speed and decomposition matter.

The next research migration should be **M95S — Population-Mass vs Player-Ranking Decomposition Audit**. It should diagnose, without fitting another tuned candidate, whether contemporaneous within-season workload signals are better suited to controlling population tail mass while M95F/feed-history signals remain responsible for player ordering. The audit should specifically quantify how quickly a population calibration anchor must react to regime changes and whether 2025's shift was visible pregame before the M95R overcorrection.
