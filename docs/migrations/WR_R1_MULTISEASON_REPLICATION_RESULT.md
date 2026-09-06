# WR-R1 Multi-Season 2020–2025 Replication Result

## Canonical evidence

- Workflow: `WR-R1 Multi-Season 2020-2025 Replication`
- Run: `34058453941`
- Job: `101554556794`
- Tested SHA: `9e3d31c9cc9d2ab53bce0c61f70c7f80d3f81978`
- Exact M37 comparison ref: `ba83fd05412a36309822cac6aa9cc5003388b073`
- Exact M38 promoted ref: `b98518d97b3038f471aee9ae3201009b2c70bb29`
- Artifact: `9997412312`
- Artifact digest: `sha256:d75bf1233bdd48d119508c0631a0f904e0256239e471bb72804d705d73ed6cfe`
- Iterations: 2000 per walk-forward run.
- Retuning used: false.
- Sportsbook inputs used: false.

## Integrity

The exact 2025 M38 parent parity was reproduced:

- all-receiver receiving-yards n = 4647
- all-receiver receiving-yards MAE = 17.099904733366

The paired WR receiving-yards replication contains 12,396 player-games across 2020–2025.

## Receiving-yards result

M38 beat M37 on receiving-yard MAE in **all six individual seasons**:

| Season | M37 MAE | M38 MAE | M38 - M37 |
|---|---:|---:|---:|
| 2020 | 25.682882 | 24.760720 | -0.922162 |
| 2021 | 24.148680 | 23.254611 | -0.894069 |
| 2022 | 25.090788 | 23.855523 | -1.235265 |
| 2023 | 24.061312 | 23.009415 | -1.051897 |
| 2024 | 23.791613 | 22.726786 | -1.064827 |
| 2025 | 22.653028 | 22.007939 | -0.645089 |

Pooled results:

- 2020–2025: M37 MAE 24.215219 → M38 23.247553; delta **-0.967666**.
- 2021–2025: M37 MAE 23.938070 → M38 22.961811; delta **-0.976259**.
- 2024–2025: M37 MAE 23.220444 → M38 22.366178; delta **-0.854267**.
- Worst single-season MAE change was still an improvement: **-0.645089** in 2025.
- Yearly non-worse count: **6/6**.

M38 also improved pooled receiving-yard RMSE, reduced the systematic negative bias, and increased correlation. On 2020–2025 pooled rows:

- RMSE: 34.188527 → 32.674658.
- Bias: -10.710418 → -8.974384.
- Correlation: 0.485518 → 0.504480.

## Receptions corroboration

The target-hierarchy change also improved receptions in every individual season. Pooled 2020–2025 receptions MAE improved from 1.689205 to 1.597512, with correlation improving from 0.548381 to 0.563505.

## Tail shape note

M38 materially reduced large **underprojection** errors but increased large overprojection counts, consistent with the known post-M38 decomposition rather than invalidating the hierarchy change. Across 2020–2025 receiving yards:

- under by >=25 yards: 3289 → 3068;
- under by >=50 yards: 1516 → 1328;
- over by >=25 yards: 721 → 954;
- over by >=50 yards: 15 → 80.

This is why later WR work must continue to treat mean hierarchy and upper-tail/efficiency mechanics as distinct problems rather than undoing M38.

## Official disposition

**`M38_MULTISEASON_CONFIRMED`**

M38 is no longer supported only by a 2025 result. The exact M37→M38 hierarchy change replicated in the correct direction in every season from 2020 through 2025, with no retuning and no sportsbook input. This strengthens M38 as the canonical WR target-allocation baseline.

This result does not authorize changing the frozen M38 multipliers. It confirms the baseline against which subsequent WR full-stack research must be measured.
