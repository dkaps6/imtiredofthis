# M95P — Dynamic Workload-Regime / Population-Prior Audit Results

## Authoritative run

- Workflow: `M95P RB Dynamic Workload Regime Audit`
- Run: **`33437999902`**
- Job: **`99639034733`**
- Tested SHA: **`a5b36ecb11e4eafabc2d4874e32b22fd9256bc2d`**
- Branch: `research-rb-m95p-dynamic-workload-regime-audit`
- Artifact: `migration-95p-rb-dynamic-workload-regime`
- Artifact ID: **`9775138348`**
- Artifact SHA256: **`01eec3d8a80aee77d1c17fd7c4c56065762b0148e79a9c1b57d877035301a04e`**
- Execution: success
- Disposition: **`M95P_AUDIT_COMPLETE_NO_PRODUCTION_CHANGE`**
- New model fit: `0`
- Feature search: `0`
- Coefficient search: `0`
- Sportsbook inputs: `0`
- Production change: `0`

## Why M95P was run

M95O showed that a fixed 2024 agreement gate does not transfer cleanly across seasons because the stable-workhorse probability prior/calibration itself moves. The user also raised an important concern: the exact RB tail research had leaned heavily on 2023-2025, so perhaps 2023 was simply an unusual RB season.

M95P therefore expanded the historical workload census to **eight modern NFL seasons, 2018-2025**, while keeping the exact M95K stable-workhorse model-trace audit on the comparable 2023-2025 data currently available.

The broad 2018-2025 layer is a **team-week lead-RB census**, not a reconstruction of the exact M95K stable-workhorse label. It is used to diagnose league/team workload regime and year effects. The exact model layer remains the M95K stable-workhorse definition from 2023-2025.

All rolling league/team regime features are shifted by at least one completed week. No target-week outcome enters a pregame regime feature.

## Broad 2018-2025 lead-RB workload census

### Full regular seasons

| Season | Lead RB 20+ | Lead RB 25+ | Mean lead carries | P95 lead carries |
|---|---:|---:|---:|---:|
| 2018 | 17.12% | 5.00% | 14.94 | 24.05 |
| 2019 | 22.69% | 5.96% | 15.54 | 25.00 |
| 2020 | 18.51% | 5.34% | 14.98 | 25.00 |
| 2021 | 21.88% | 6.80% | 15.07 | 26.00 |
| 2022 | 20.85% | 5.17% | 15.08 | 24.95 |
| 2023 | 18.75% | 5.51% | 15.08 | 25.00 |
| 2024 | 22.61% | 7.35% | 15.40 | 25.00 |
| 2025 | 18.01% | 4.41% | 15.12 | 24.00 |

2023 was **not** a broad full-season workload outlier. Its 20+ lead-RB rate was below the eight-year mean and its 25+ rate was ordinary.

### Weeks 13-18

| Season | Lead RB 20+ | Lead RB 25+ | Mean lead carries |
|---|---:|---:|---:|
| 2018 | 18.45% | 5.36% | 15.14 |
| 2019 | 20.24% | 5.36% | 15.74 |
| 2020 | 18.82% | 5.88% | 15.00 |
| 2021 | **26.09%** | 8.70% | 15.58 |
| 2022 | 20.88% | 3.30% | 15.18 |
| 2023 | 23.91% | 7.61% | **15.71** |
| 2024 | 24.19% | **9.14%** | 15.40 |
| 2025 | 21.28% | 4.79% | 15.38 |

For late-season workload, 2023 ranked around the **75th percentile** for both 20+ and 25+ prevalence, with z-scores only `+0.854` and `+0.708`. It did **not** cross the predeclared `1.5 SD` broad-outlier threshold.

This directly answers the "maybe 2023 was just weird" question: **no, not in the broad modern-NFL workload census**. 2021 had a higher late-season 20+ rate and 2024 had higher late-season 20+/25+ rates. 2023 was relatively RB-heavy late, but not uniquely abnormal.

## Exact stable-workhorse model-trace context

The comparable exact M95K stable-workhorse cohort currently exists for 2023-2025.

| Scope | n | Actual 20+ | Actual 25+ | Mean carries | Mean M95F p20 | Actual - predicted p20 |
|---|---:|---:|---:|---:|---:|---:|
| 2023 W13-18 | 73 | 32.88% | 13.70% | 16.97 | 16.19% | **+16.69 pp** |
| 2024 W13-18 | 77 | 35.06% | 15.58% | 16.65 | 27.38% | **+7.69 pp** |
| 2025 W13-18 | 85 | 28.24% | 7.06% | 15.74 | 29.43% | **-1.20 pp** |
| 2025 full available | 237 | 21.94% | 4.64% | 15.50 | 29.60% | **-7.66 pp** |

This is important: the late-season **actual 20+ rate range is only 6.83 percentage points** across 2023-2025, but the M95F calibration-gap range is **17.88 percentage points**.

So the main cross-season problem is not simply that 2023 produced an impossible amount of RB workload. A large part of the instability is that the model's probability scale is not adapting to the contemporaneous workload prior.

`stable_workhorse_nonstationarity_confirmed = 1`.

## Pregame-only dynamic regime signals

M95P created rolling league/team workload state using only prior completed weeks.

Strongest diagnostic correlations across the 2023-2025 exact stable-workhorse rows (`n=371`):

- prior-four-team-weeks 25+ lead-RB rate vs upcoming 20+ event: Spearman **`.2064`**
- prior-four-team-weeks mean lead-RB carries vs upcoming 20+ event: **`.2037`**
- prior-four-team-weeks 20+ lead-RB rate vs upcoming 20+ event: **`.1913`**
- season-to-date league lead-RB 20+ rate vs M95F calibration error: **`.1823`**
- prior-four league mean RB count vs calibration error: **`.1654`**

These are not huge effects, but they are meaningful for an audit and are consistent with the M95O hypothesis that a player's absolute tail score must be interpreted relative to the current team/league workload regime.

### League prior-four-week 20+ regime quartiles

| Pregame regime | n | Prior league lead20 rate | Exact stable actual 20+ | Mean M95F p20 | Calibration gap |
|---|---:|---:|---:|---:|---:|
| Q1 low | 97 | 13.97% | 17.53% | 27.86% | **-10.33 pp** |
| Q2 | 91 | 18.39% | 24.18% | 24.37% | -0.19 pp |
| Q3 | 99 | 23.04% | **34.34%** | 23.68% | **+10.66 pp** |
| Q4 high | 84 | 24.34% | 30.95% | 28.94% | +2.01 pp |

The relationship is not perfectly monotonic at Q4, so this is **not** sufficient to justify a production formula. But low-regime weeks clearly look different from medium/high-regime weeks, and M95F calibration moves in the expected direction.

`pregame_regime_structure_detected_diagnostic = 1`.

## Backtesting-width conclusion

The earlier exact model research was indeed **too narrow to treat three seasons as enough evidence for a rare-event tail architecture**. M95P therefore expands workload-regime context to 2018-2025 and confirms that additional seasons materially improve interpretation.

However, there is an important distinction:

- **Broad workload context:** now eight seasons, 2018-2025.
- **Exact comparable M95K/M95F stable-workhorse model trace:** still only 2023-2025.

The broad census cannot be used as if it were the exact stable-workhorse model backtest. The next step should therefore expand the exact leakage-safe historical model reconstruction before claiming robustness.

## Scientific conclusion

1. **2023 was not a uniquely weird RB year.** It was somewhat high-workload late, but well inside the 2018-2025 modern range.
2. **2024 was also a high-workload environment**, further disproving a simple "discard 2023" explanation.
3. The exact stable-workhorse actual workload rate changes, but the larger instability is **probability calibration relative to the current population prior**.
4. Pregame-only rolling league/team workload state contains modest but real signal for both upcoming 20+ events and model calibration error.
5. A dynamic prior remains worth pursuing, but **not yet as a production candidate**.
6. More exact historical seasons/data points are needed for tail-model robustness.

## Recommended next migration — M95Q

**M95Q — Expanded Historical Stable-Workhorse Backtest Reconstruction**

Primary goal:

> Expand the exact leakage-safe RB stable-workhorse backtest beyond 2023-2025 so future dynamic-prior/mixture candidates are not judged on only three exact model seasons.

Recommended scope:

- reconstruct at least 2020-2022, and older modern seasons if source contracts support comparable inputs;
- preserve the M94C/M95F/M95G role semantics rather than replacing them with the broad lead-RB proxy;
- use only pregame injury/roster/depth/workload information available before each game;
- document source coverage and any historical feature degradation explicitly;
- do not force a season into the exact comparison if its source quality is not comparable;
- after reconstruction, rerun baseline calibration/workload diagnostics across the expanded temporal panel;
- only then precommit a dynamic population-prior candidate;
- treat 2018-2025 broad census as context, not target labels;
- no sportsbook input and no production change.

A future derivative candidate still needs genuinely prospective/untouched confirmation; 2023, 2024 and 2025 are all opened research years.