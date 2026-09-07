# WR-R11 Strict-Prior NGS Target Model — Result

## Lineage
- Branch: `research-wr-r11-strict-prior-ngs-target-model`
- Frozen plan commit: `b30d9808e9b8f1e7c5268c8a19972cd85fc13f7f`
- Implementation commit: `f402a01b38d4e5bf18747c6a97e47f13dc5e2ff3`
- Canonical run: `34124533822`
- Job: `101749972264`
- Run head: `d2f251dc267db854e6747734d6cbe692d56f93ae`
- Artifact: `10019545653` (`wr-r11-strict-prior-ngs-target-model`)
- Artifact digest: `sha256:388973f0fc67c6a2f0e4868a7c874d310ff7711cd7b0a6fb38c74128dcd43370`
- OOS seasons: 2022, 2023, 2024, 2025
- Eligible OOS player-games: 6,383
- Ineligible OOS player-games preserved at exact B0 parity: 1,988; max parity gap 0.0
- Sportsbook inputs used: false
- Production changed: false

## Frozen disposition
`WR_NGS_TARGET_MODEL_FAIL`

This is a **scientific failure**, not a mechanical/source failure. Leakage and parity gates passed. The frozen candidate is rejected and is not authorized for production or tuning rescue.

## Pooled OOS scorecard

### Targets
- B0 MAE: **2.323440**
- NGS candidate MAE: **2.287887**
- Delta: **-0.035552** targets; improves, but misses the frozen >=0.05 improvement gate.
- B0 RMSE: 3.157832 -> candidate 2.871322
- Bias: **-1.347681 -> +0.345941**
- Correlation: 0.573104 -> 0.575985
- Median absolute error: **1.685953 -> 1.949557** (worse)
- p90 absolute error: **5.333405 -> 4.650877** (better)

### Receptions
- MAE: **1.689658 -> 1.696855** (slightly worse)
- RMSE: 2.309391 -> 2.160150
- Bias: -0.834167 -> +0.241599
- p90: 3.812200 -> 3.445589

### Receiving yards
- B0 MAE: **24.845863**
- NGS candidate MAE: **26.005654**
- Delta: **+1.159792 yards** (material regression)
- RMSE: 34.492457 -> 33.166032
- Bias: **-10.122787 -> +3.655893**
- Correlation: 0.487026 -> 0.487675
- Median absolute error: **17.263842 -> 22.097131** (material regression)
- p90: **55.675913 -> 51.513891** (improvement)
- 20+ miss rate: **44.399% -> 55.303%** (material regression)
- 30+ miss rate: **28.858% -> 32.775%** (material regression)
- 40+ miss rate: **19.223% -> 18.800%** (slight improvement)

## Temporal replication
Target MAE improved in 3 of 4 seasons, satisfying that frozen gate:
- 2022: 2.440738 -> 2.360301
- 2023: 2.265048 -> 2.287064 (worse)
- 2024: 2.315325 -> 2.278036
- 2025: 2.275185 -> 2.227960

But receiving-yard MAE regressed in **all four** OOS seasons:
- 2022: 25.818934 -> 26.812419
- 2023: 25.029333 -> 26.725436
- 2024: 24.544861 -> 25.290589
- 2025: 24.018681 -> 25.228905

2024–2025 combined receiving-yard MAE: **24.284216 -> 25.260033**.

## High-target tier
For `HIGH_TARGET_Q4` (n=1,596):
- target MAE: 2.788919 -> 2.754063 (small improvement)
- reception MAE: 2.074380 -> 2.057703
- receiving-yard MAE: **31.860292 -> 32.236724** (worse)
- receiving-yard p90: 64.482177 -> 62.998813
- 30+ miss rate: **43.546% -> 47.306%**
- 40+ miss rate: **30.639% -> 32.393%**

The candidate therefore fails the high-target receiving-yard protection gate.

## Diagnostic finding
The candidate's learned correction was **positive for 100% of eligible predictions in every OOS season**:
- 2022 mean correction +1.990 targets
- 2023 +1.697
- 2024 +1.582
- 2025 +1.514

This largely converted B0's systematic target underprediction into a systematic positive target correction rather than learning sufficiently player-specific differential entitlement. The bias flip propagated through the existing reception/YPT mechanics and worsened individual receiving-yard accuracy even while reducing RMSE/p90.

This is directly relevant to the project objective: lower tail/RMSE alone is not enough when median individual errors, MAE, and 20+/30+ miss rates worsen.

## Frozen conclusion
Strict-prior NGS data are **source-eligible** (WR-R10), but this predeclared NGS residual-target implementation is **not an actionable player projection model**.

Do not:
- lower the frozen target-MAE threshold;
- tune the correction cap post hoc;
- recentre the all-positive corrections after seeing the result;
- promote the candidate because target RMSE/p90 improved;
- rerun the same idea under a different label.

A future WR tracking experiment requires genuinely new football information or a different predeclared mechanism—for example, using tracking to condition **role/depth/route archetype or matchup-dependent efficiency**, rather than another generic residual lift.
