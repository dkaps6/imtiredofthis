# WR-R1 — Multi-Season 2020–2025 Replication Plan

## Status

Frozen before any WR-R1 results are inspected.

## Why this migration exists

The promoted WR hierarchy (M38) and the recent WR ND diagnostic chain were evaluated canonically on 2025, using 2024 only as prior history. That produced a useful 2,130-WR player-game diagnostic casebook, but it is materially narrower than the multi-season evidence standard already used for QB and RB research.

Before more WR feature discovery, WR-R1 expands the canonical WR evidence base to 2020–2025 and asks whether the existing M38 hierarchy improvement is stable outside its 2025 development season.

This is a replication / robustness migration, not a retune.

## Frozen target seasons

Target seasons: **2020, 2021, 2022, 2023, 2024, 2025**.

Each target season uses only its immediately prior season plus completed games strictly before the target game:

- 2020 target -> 2019 prior
- 2021 target -> 2020 prior
- 2022 target -> 2021 prior
- 2023 target -> 2022 prior
- 2024 target -> 2023 prior
- 2025 target -> 2024 prior

Regular season only.

- 2020: Weeks 1–17
- 2021–2025: Weeks 1–18

2020 remains in the canonical aggregate despite the COVID-era environment. A separate 2021–2025 sensitivity result will also be reported, but 2020 may not be removed after results are visible.

## Frozen model refs

### M37 comparison parent

`ba83fd05412a36309822cac6aa9cc5003388b073`

This is the first parent of the M38 merge and represents the canonical pre-M38 simulation path before the promoted WR hierarchy sharpening.

### M38 promoted parent

`b98518d97b3038f471aee9ae3201009b2c70bb29`

Frozen hierarchy multipliers remain exactly:

- WR1 = 1.40
- WR2 = 1.14
- WR3 = 0.91
- WR4+ = 0.78

No multiplier changes are authorized in WR-R1.

## Simulation and leakage contract

- Historical inputs are built with the exact M38 historical input machinery.
- Target-week player universes come from pregame weekly roster/depth information only.
- Team/player form uses only completed games strictly before the target week.
- Target-game outcomes are attached only after predictions are produced.
- Monte Carlo iterations: **2,000**, matching the canonical 2025 M38 reconstruction used by WR-ND1 through WR-ND6.
- Seeds remain the existing walk-forward seeds (`42 + week`).
- No sportsbook/player-prop data.
- No future-season information.
- No post-result feature search, threshold search, multiplier search, or era exclusion.

## Primary questions

1. Does M38 improve WR receiving-yard accuracy versus its M37 parent across the 2020–2025 aggregate?
2. Is the effect directionally stable by season rather than being a 2025-only artifact?
3. Does M38 improve the latest-era 2024–2025 combined slice?
4. What is the resulting multi-season WR sample size available for all subsequent WR research?

## Frozen metrics

For each model ref and each target season, report WR-only:

- receiving-yards N
- receiving-yards MAE
- receiving-yards RMSE
- receiving-yards bias
- receiving-yards correlation
- receptions N
- receptions MAE
- receptions RMSE
- receptions bias
- receptions correlation

Also report:

- absolute receiving-yard errors >= 25 yards
- absolute receiving-yard errors >= 50 yards
- underprojections >= 25 yards
- underprojections >= 50 yards
- overprojections >= 25 yards
- overprojections >= 50 yards

Aggregate the same metrics for:

- 2020–2025
- 2021–2025 sensitivity
- 2024–2025 latest-era slice

All aggregate MAE/RMSE/bias values are calculated from pooled player-games, not averages of yearly summary metrics.

## 2025 parity gate

Before interpreting multi-season science, the M38 2025 run must reproduce the established canonical values within deterministic tolerance:

- all-receiver `rec_yards` rows = 4,647
- all-receiver M38 `rec_yards` MAE = 17.099904733366

If 2025 parity fails, WR-R1 is a mechanical/integrity failure and no multi-season scientific conclusion is allowed.

## Frozen replication disposition

This migration does **not** decide whether M38 remains in production; M38 stays canonical unless a later explicitly authorized production migration changes it. WR-R1 classifies replication strength only.

### `M38_MULTISEASON_CONFIRMED`

All must hold:

- pooled 2020–2025 WR receiving-yard MAE is lower for M38 than M37;
- M38 is non-worse on WR receiving-yard MAE in at least **4 of 6** seasons;
- pooled 2024–2025 WR receiving-yard MAE is lower for M38 than M37;
- no single season worsens by more than **1.00 yard MAE**.

### `M38_MULTISEASON_ERA_DEPENDENT`

- pooled 2020–2025 WR receiving-yard MAE improves, but one or more confirmation conditions above fail.

### `M38_MULTISEASON_NOT_CONFIRMED`

- pooled 2020–2025 WR receiving-yard MAE does not improve versus M37.

No threshold may be changed after results are visible.

## What WR-R1 authorizes next

Regardless of disposition, future WR diagnostics should use the 2020–2025 multi-season casebook whenever the candidate source has adequate historical coverage.

If a candidate source does not cover 2020–2025, its exact available era and coverage must be frozen and documented before testing. We will not silently fall back to 2025-only because it is convenient.

After WR-R1 establishes the multi-season backbone, the next materially-new WR family remains player-level tracking / route-quality information (for example NGS separation, cushion, aDOT, intended-air-yard share, YAC over expectation), subject to a separate frozen source-coverage audit and test plan.

## Sportsbook separation

Sportsbook lines remain downstream only. WR-R1 uses football data exclusively and cannot ingest Vegas lines or prices.