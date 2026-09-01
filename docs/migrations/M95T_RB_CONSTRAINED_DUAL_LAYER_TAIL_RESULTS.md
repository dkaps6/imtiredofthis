# M95T — Constrained Dual-Layer Stable-Workhorse Tail Candidate — Results

## Authoritative run

- Workflow: `M95T RB Constrained Dual-Layer Tail`
- Run: **`33455690862`**
- Job: **`99695055863`**
- Tested SHA: **`540edb67d9d5451764e997f19213b80285c15fab`**
- Artifact: **`9781352939`**
- Artifact SHA256: **`bd54485d18de4f4df1f7613d9587281234bf07f8ce9de6df36b68bca26167c70`**
- Execution: **SUCCESS**
- Scientific disposition: **`M95T_FAIL_STOP_NEW_RB_TAIL_CANDIDATES_RETAIN_M94C_M95F_PROCEED_M96`**
- Model fit: `0`
- Feature search: `0`
- Coefficient search: `0`
- Hyperparameter search: `0`
- Sportsbook inputs: `0`
- Production change: `0`

Before the authoritative run, one mechanical implementation error in the team-week lead-RB sort was corrected: the first draft could sort player name before carries. The fix changed only the lead-RB selection ordering so the maximum-carry RB is selected. Candidate formula and gates were already frozen and were not changed after results were exposed.

## Frozen candidate

M95T implemented the M95S decomposition:

1. M95F player-game 20+ probability remained the backbone.
2. A fast pregame population layer used prior-four-week broad lead-RB 20+ prevalence relative to season-to-date prevalence, clipped to `0.70-1.30`, shrunk by `50%`, so total mass could move at most +/-15% relatively.
3. A separate player layer reused M95K `k=4` leakage-safe feed/carry-ceiling semantics. Within each week, feed and M95F ranks had to be on the same half of the distribution. Only aligned observations were reranked; discordant observations were untouched. The log-odds rerank was capped at +/-0.25 and exactly mean-anchored before the population layer.
4. M94C central carries were unchanged.
5. Vacancy logic was unchanged/separate.

## Stable-workhorse 20+ results — W13-18 comparable panel

| Season | n | Actual | M95F mean | M95T mean | M95F AUC | M95T AUC | Brier gain | Logloss gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 44 | 27.27% | 25.15% | 24.36% | .7760 | .7760 | **-.001461** | **-.005114** |
| 2021 | 61 | 36.07% | 31.72% | 33.84% | .4977 | .5035 | -.000142 | +.003807 |
| 2022 | 72 | 29.17% | 26.71% | 26.88% | .6900 | .6863 | **+.000915** | **+.001799** |
| 2023 | 73 | 32.88% | 16.19% | 17.51% | .7270 | **.7415** | **+.004887** | **+.015735** |
| 2024 | 77 | 35.06% | 27.38% | 28.45% | .7044 | .7052 | **+.002334** | **+.005123** |
| 2025 | 85 | 28.24% | 29.43% | 33.54% | .6469 | .6523 | **-.001439** | **-.003055** |
| Pooled | 412 | 31.55% | 26.11% | 27.65% | .6591 | **.6619** | **+.000988** | **+.003447** |

M95T improved pooled Brier, pooled logloss, pooled AUC and pooled absolute calibration gap. It also materially repaired 2023 without recreating the exact frozen M95K failure.

However, the precommitted season-stability gates failed:

- only **3 of 6** seasons had non-negative Brier gain; required >=4;
- 2025 absolute calibration gap worsened by **4.103 pp**, exceeding the 2.5 pp guard;
- the combined 2023/2025 trouble-year guard therefore failed.

The maximum Brier regression was small (`.001461`) and maximum logloss regression was small (`.005114`), but the candidate still fails because the calibration instability remained material in 2025. The gate is not waived.

## What M95T teaches

The M95S decomposition was directionally correct but not sufficient for a durable historical tail overlay.

- The fast population layer helped the badly underpredicted 2023 environment.
- The bounded conditional ranking layer avoided the catastrophic 2023 AUC damage of frozen M95K and actually improved 2023 AUC (`.7270 -> .7415`).
- But the same architecture still assigned too much 20+ probability mass in late 2025 (`29.43% -> 33.54%` vs actual `28.24%`).

Thus there is no adequately stable retrospective evidence to replace M95F for stable-workhorse 20+ probabilities before 2026. Per the frozen stopping rule, **new retrospective RB carry-tail candidate development ends here.**

Selected conservative workload state for downstream RB synthesis:

- M94C remains central carry/opportunity projection.
- M95F remains the stable-workhorse 20+/25+ tail baseline.
- M95I vacancy/role-transition remains a separate diagnostic regime and is not silently promoted by M95T.
- M95K/L/O/R/T failures remain preserved evidence.

## Rushing-yard translation guard

The user's actual end goal is rushing-yard prediction, not carry-count props. M95T therefore added an explicit translation sanity audit using frozen historical player-weekly rushing-yard truth.

Across the 412 comparable stable-workhorse late-season player-games:

- carries vs rushing yards Pearson correlation: **`.789267`**
- Spearman correlation: **`.788957`**
- 75+ rushing-yard events: `173`
- 100+ rushing-yard events: `92`

M95T p20 as a 75+/100+ yard discriminator did not materially collapse:

- pooled 75+ AUC: `.608363 -> .602051` (`-.006312`)
- pooled 100+ AUC: `.596196 -> .592493` (`-.003702`)

Both passed the precommitted `-0.01` translation guards.

This confirms that carry opportunity is strongly tied to rushing yards, but it **does not prove rushing-yard point accuracy is solved**. Efficiency, yards before/after contact, blocking, opponent environment and explosive-play variance still matter.

## Next phase — M96

**M96 — RB Rushing-Yard Synthesis / Opportunity-to-Yardage Translation** is mandatory before RB can be declared complete.

M96 must use the selected conservative workload architecture (M94C + M95F, with separate role-regime evidence) and explicitly combine it with the rushing-efficiency/environment findings retained from M95A-D / the existing production yardage layer.

Primary M96 outputs must include:

- rushing-yard MAE, RMSE and correlation by season and pooled;
- ordinary-game error guards;
- 75+ and 100+ rushing-yard discrimination/calibration;
- decomposition of error into workload vs efficiency;
- explicit comparison of using projected carries vs realized carries (oracle opportunity diagnostic) to quantify remaining efficiency ceiling;
- no sportsbook inputs during football-model construction;
- no production promotion until the final RB synthesis and prospective 2026 validation requirements are satisfied.
