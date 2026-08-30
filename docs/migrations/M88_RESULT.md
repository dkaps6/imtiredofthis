# Migration 88 — Authoritative Result

## Disposition

`M87_REGIMES_NOT_REPLICATED`

Migration 88 completed the preregistered untouched-2023 confirmation of the two conditional pregame regimes isolated by M87. The full current production-style historical stack was rebuilt for 2023 Weeks 1-18 from 2022 plus strictly-prior 2023 information at 2,000 Monte Carlo iterations.

Neither frozen M87-derived regime cleared every preregistered M88 replication gate. No passing-yard correction is authorized from M88.

## Authoritative run

- GitHub Actions workflow: `Migration 88 QB 2023 Regime Replication`
- Run: `33325633231` (Run #1)
- Conclusion: `success`
- Artifact: `m88-2023-regime-replication`
- Artifact ID: `9736226603`
- Artifact SHA256: `2f3a6fa3894807b4e05173d90011051ccd7e857aea255bf11e49ccb7249ee2ef`
- Confirmation season: `2023`
- Prior history season: `2022`
- Stable-primary QB rows: `447`
- Monte Carlo iterations: `2000`
- Bootstrap draws: `2000`
- Bootstrap seed: `88`
- Sportsbook features used: `False`
- Passing-yard correction fit: `False`
- Target-game PBP used as pregame feature: `False`
- Production actionable: `False`

## 2023 clean full-stack scoreboard

| Model | N | MAE | RMSE | Bias | Correlation | 100+ misses |
|---|---:|---:|---:|---:|---:|---:|
| Current MC / Bayesian / rules | 447 | **57.586531** | 72.052364 | -11.988184 | 0.176914 | 77 |
| Current ML | 447 | 60.599372 | 75.632634 | -14.009733 | 0.154948 | 88 |
| Current State | 444 | 60.628135 | 75.545290 | -24.694337 | 0.111776 | 80 |
| OOS ensemble | 447 | 58.170780 | 72.209217 | -15.017164 | 0.163937 | **76** |

Unlike 2024-2025 M82, the 2023 current MC point projection has lower MAE than the M88 OOS ensemble. This is a season-level diagnostic only and does not alter the M82 2024-2025 authoritative production benchmark.

## Frozen regime definitions

Thresholds were frozen from the midpoint of M87 target/control means before 2023 outcomes were opened.

### PASS_FUNNEL_SHORT_INTERMEDIATE_VOLUME

- opponent defense prior 8-game pass rate faced `>= 0.6062143950065955`
- target offense prior 8-game deep-attempt rate `<= 0.18505508535326465`
- expected confirmation event: low-chaos, volume-dominant, 100+ yard underprojection

### EFFICIENCY_SUPPRESSION

- opponent defense prior 8-game success rate allowed `<= 0.42486131276490247`
- opponent defense prior 8-game YPA allowed `<= 6.426889901131469`
- expected confirmation event: low-chaos, efficiency-dominant, 100+ yard overprojection

No 2023 threshold tuning was performed.

## PASS_FUNNEL_SHORT_INTERMEDIATE_VOLUME result

- eligible stable-primary rows: `447`
- regime rows: `85`
- non-regime rows: `362`
- feature coverage: `100%`
- expected-direction events in regime: `3`
- expected-direction events outside regime: `3`
- regime event rate: `3.5294%`
- non-regime event rate: `0.8287%`
- rate ratio: **4.2588x**
- absolute event-rate lift: **+2.7007 percentage points**
- regime mean attempt residual: `+2.1510`
- non-regime mean attempt residual: `+1.6494`
- regime mean ensemble error: `-19.9566` yards
- non-regime mean ensemble error: `-13.8574` yards
- bootstrap support for higher attempt residual: `0.6775`

### Frozen gate verdict

Passed:
- >=90% feature coverage
- >=15 regime rows
- >=3 expected-direction events
- expected component-residual direction
- expected ensemble-error direction

Failed:
- event enrichment gate: rate ratio exceeded 1.50x, but absolute lift was only +2.70pp versus the frozen +5pp requirement
- bootstrap support: 0.6775 versus the frozen 0.70 requirement

Final status: `NOT_REPLICATED_2023`.

The regime shows directionally coherent 2023 evidence, including a 4.26x rare-event ratio and more positive attempt residuals, but the preregistered magnitude/support bar was not met. Thresholds must not be relaxed post hoc.

## EFFICIENCY_SUPPRESSION result

- eligible stable-primary rows: `447`
- regime rows: `162`
- non-regime rows: `285`
- feature coverage: `100%`
- expected-direction events in regime: `7`
- expected-direction events outside regime: `2`
- regime event rate: `4.3210%`
- non-regime event rate: `0.7018%`
- rate ratio: **6.1574x**
- absolute event-rate lift: **+3.6192 percentage points**
- regime mean YPA residual: `-0.3212`
- non-regime mean YPA residual: `+0.2822`
- regime mean ensemble error: `-3.0089` yards
- non-regime mean ensemble error: `-21.8429` yards
- bootstrap support for lower YPA residual: **0.9995**

### Frozen gate verdict

Passed:
- >=90% feature coverage
- >=15 regime rows
- >=3 expected-direction events
- expected component-residual direction
- expected ensemble-error direction relative to non-regime games
- bootstrap support >=0.70

Failed:
- event enrichment gate: rate ratio exceeded 1.50x, but absolute lift was +3.62pp versus the frozen +5pp requirement

Final status: `NOT_REPLICATED_2023`.

This is the stronger of the two directional findings: the expected catastrophic event occurred 6.16x as often and the YPA-residual shift reproduced with 99.95% bootstrap support. However, it still failed the exact preregistered confirmation bar and cannot be promoted by relaxing the rare-event absolute-lift threshold after seeing 2023.

## Scientific interpretation

M88 does not erase M87's 2024-2025 forensic patterns, but it prevents us from treating the exact thresholded regimes as confirmed predictive mechanisms.

The 2023 evidence is best described as **directional continuity without confirmatory replication**:

1. The volume regime moved in the expected direction but missed both the +5pp event-lift and 0.70 bootstrap gates.
2. The efficiency-suppression regime showed strong component-level replication and large relative rare-event enrichment, but missed the frozen +5pp absolute event-lift gate.
3. Neither regime is authorized for a passing-yard correction or production promotion.

Because the event being studied is rare, future work may separately examine whether a preregistered risk-enrichment objective is scientifically better suited than a point-correction objective. That question must be designed prospectively and may not retroactively redefine M88 success.

## Anti-loop consequence

- Do not tune the four M88 thresholds on 2023.
- Do not open a passing-yard correction from either regime under M88.
- Do not relabel the 4.26x / 6.16x relative enrichments as replication after the frozen +5pp gate failed.
- Preserve the efficiency-suppression result as a directional clue only unless a future, prospectively defined experiment establishes deployable value.

`next_predictive_migration_allowed = false`

`production_actionable = false`
