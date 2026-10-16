# PLAYER EFFICIENCY TAIL STABILITY V1 — HISTORICAL RESULT

Date: 2026-10-13

Status: **NULL / FAIL CLOSED. NO V2 DISTRIBUTION CANDIDATE AUTHORIZED.**

Frozen contract:
- `docs/research/PLAYER_EFFICIENCY_TAIL_STABILITY_V1_ANTIRETEST_AUDIT.md`
- `docs/research/PLAYER_EFFICIENCY_TAIL_STABILITY_V1_FREEZE.md`

Canonical authority:
- workflow run: `38408536257`
- exact head: `eaafe88efc9428cda295875e400f4f263beccd12`
- artifact: `11819986612`
- digest: `sha256:4bb8bead9f3a40f6465909c2148111d0c2aec9ff0b21a6b85fdc6bb27a0bb1e5`
- workflow conclusion: **SUCCESS**
- strict repository audit: **PASS**

Overall disposition:
`PLAYER_EFFICIENCY_TAIL_STABILITY_NULL`

## Integrity

- target seasons: 2023, 2024, 2025
- target weeks: 5-18
- panel rows: 6,280
- players: 360
- stable identity coverage: 100%
- chronology violations: 0
- sportsbook inputs: 0
- OddsAPI calls: 0
- model residuals: 0
- fitted predictive models: 0
- candidate mean changes: 0
- production changes: false

The experiment was frozen before scoring. Two earlier attempts failed mechanically
before producing a result:
- run `38407631994`: schedule-column naming collision;
- run `38407837847`: non-regular-week source rows reached the regular-season
  schedule join.

Those repairs changed only source plumbing. Run `38408014359` then produced the
same science result but failed the post-result strict audit because PyYAML was
missing from the workflow environment. The canonical run above adds only that
CI dependency and is fully green.

## Frozen cell decisions

### RB rushing YPC

Disposition:
`PLAYER_EFFICIENCY_VOLATILITY_NOT_REPLICATED`

All three seasons fail the frozen controlled-support rule.

Raw player YPC volatility showed positive association with next-game absolute
YPC surprise, but the relationship did not survive the predeclared controls and
clustered inference consistently.

Representative raw / partial Spearman:
- 2023: +0.1411 / +0.0249
- 2024: +0.1637 / +0.0776
- 2025: +0.1426 / +0.0290

No RB rushing efficiency-width candidate advances.

### WR target YPT

Disposition:
`PLAYER_EFFICIENCY_VOLATILITY_NOT_REPLICATED`

This cell contains the most important nuance in V1.

2023 passes the full frozen rule:
- rows: 872
- players: 115
- raw Spearman: +0.1894
- partial Spearman: +0.0805
- game-cluster partial 95% CI low: +0.0140
- player-cluster partial 95% CI low: +0.0176

But replication fails:
- 2024 raw Spearman +0.1462, partial +0.0038;
  partial clustered intervals cross zero;
- 2025 raw Spearman +0.0827, partial +0.0314;
  partial clustered intervals cross zero.

Interpretation:
raw WR YPT volatility contains descriptive signal, but the robust
player-specific component beyond prior efficiency level and opportunity does not
replicate across 2024-2025.

Do not convert the 2023 pass into a majority-season or pooled rescue.

### TE target YPT

Disposition:
`PLAYER_EFFICIENCY_VOLATILITY_NOT_REPLICATED`

The frozen support gate requires >=60 distinct players per season. TE had:
- 2023: 302 rows / 44 players
- 2024: 286 / 42
- 2025: 333 / 52

Therefore all three seasons fail closed on support. Do not lower the player-count
gate after seeing the result.

### RB receiving YPT

Disposition:
`PLAYER_EFFICIENCY_VOLATILITY_NOT_REPLICATED`

2023 had enough support but failed the scientific gate:
- rows: 415
- players: 66
- raw Spearman: +0.0803
- partial Spearman: -0.0018
- raw and partial clustered intervals cross zero.

2024 and 2025 each had 58 distinct players, below the frozen >=60-player support
gate. Do not pool seasons or lower the gate.

## Scientific interpretation

The simple hypothesis

> "a player's own recent per-opportunity efficiency volatility is a stable,
> independent predictor of his next-game efficiency tail risk"

does **not** survive the frozen three-season test.

There is visible raw association in several cells, especially WR YPT and RB
rushing YPC, but much of that association is explained by prior efficiency level
and prior opportunity volume or is not seasonally stable.

This is meaningfully different from the opportunity-share volatility program:
role/share volatility remained strong after finite-count and mean-share controls;
generic efficiency volatility does not.

## Stop rule

This exact family is closed.

Do not rescue with:
- alternate prior windows;
- pooled seasons;
- lowered support gates;
- WR1/WR2/TE1/bellcow slices;
- volatility buckets;
- alternate tail thresholds;
- matchup interactions;
- target-depth/CPOE/run-location recombinations.

No mean change, uncertainty-width promotion, or production integration is
authorized from V1.

The next efficiency investigation must use genuinely different pregame
information or a separately justified regime/mechanism state, not another
version of historical efficiency SD.
