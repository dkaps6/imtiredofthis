# Receiver Targetable-Dropback V1 — Temporal Team Calibration Result

Date: 2026-09-25

Disposition: **RECEIVER_TARGETABLE_DROPBACK_V1_TEAM_CALIBRATION_SUPPORTED**

## Authority

- branch: `research-receiver-targetable-dropback-v1-calibration`
- run: `36144562838`
- job: `108102370840`
- tested head: `a6a597e31c06a1ded9d596288cc9164e44d55dcb`
- artifact: `10868338685`
- digest: `sha256:416c24551afd06b9d0aa8c50ab9f27a04e919484e8e4b83e37ae7d4c1a3332e4`
- parameters fit: 0
- candidate variants on this frozen branch: 1
- sportsbook inputs: 0
- target-game outcomes upstream: 0

## Frozen candidate

For each target team-game:

`targetable_dropback_rate = sum(strict-prior team targets) / sum(strict-prior team dropbacks)`

Eligible history:
- prior regular season;
- completed games in current season strictly before target week.

Candidate target pool:

`projected_plays * 0.57 * targetable_dropback_rate`

No shrinkage, recency weighting, threshold, carveout or sportsbook input.

## 2022

Team-target accuracy:

- MAE: `6.943314 -> 6.417334`
- RMSE: `8.588549 -> 8.220010`
- p90 AE: `14.249708 -> 12.859686`
- absolute bias: `2.669850 -> 2.154121`
- candidate closer rate: `54.24%`

## 2023

Team-target accuracy:

- MAE: `6.316515 -> 5.902369`
- RMSE: `7.683706 -> 7.547814`
- p90 AE: `12.513064 -> 12.425574`
- absolute bias: `2.638170 -> 2.641639` (slightly worse in-season, but not a frozen season-specific bias gate)
- candidate closer rate: `54.04%`

## Pooled 2022-2023

- MAE: `6.629337 -> 6.159377`
- improvement: **0.469960 targets/team**
- RMSE: `8.147865 -> 7.890454`
- p90 AE: `13.480627 -> 12.640581`
- absolute bias: `2.653981 -> 2.398329`
- candidate closer rate: **54.14%**
- 10+ miss rate: `21.55% -> 18.51%`
- targetable-rate median: `0.854982`
- rate range: `0.705305 .. 0.938401`
- fallback rows: **0**

Every frozen gate passed.

## Important duplicate-formulation note

A later independently created duplicate branch
`research-receiver-targetable-dropback-pool-v1` used a different estimator:
an arithmetic mean of prior per-game targetable rates and only a 2023 screen.

That alternate formulation failed its frozen gates:
- MAE improved;
- p90 improved;
- RMSE and absolute-bias gates failed.

It is **not** used to modify, rescue, tune or select parameters for the canonical
candidate above.

The canonical cumulative-count estimator was frozen earlier and passed its own
2022-2023 temporal contract without reference to the later duplicate result.

## Interpretation

This is the first clean temporal support that receiver opportunity should be
modeled from **targetable dropbacks**, not all dropbacks.

It does not yet authorize player-level integration or production.

The exact candidate now requires an unchanged 2024-2025 confirmation before any
player-level full-stack test.
