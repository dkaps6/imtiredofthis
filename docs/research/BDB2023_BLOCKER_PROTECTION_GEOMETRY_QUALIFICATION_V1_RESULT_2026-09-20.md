# BDB2023 Blocker Protection Geometry Qualification V1 — Result

**Status:** CLOSED — ENGINEERING READY / SOURCE THIN  
**Frozen plan:** `docs/research/BDB2023_BLOCKER_PROTECTION_GEOMETRY_QUALIFICATION_V1.md`  
**Implementation commit:** `08e70c2495ede1bd5c021dfe4b81d42f14667796`  
**Materializer-runtime fix:** `4a24d9d0252ce6457ae2d91af79c3083a9aef51f`  
**Stable-GSIS identity fix:** `e70517d477739ef95dc02138e6f51a864f32aaed`  
**Canonical run:** `35514256900`  
**Canonical job:** `106087402711`  
**Canonical artifact:** `10605059418`  
**Artifact digest:** `sha256:e725bdef9e2c461f405ebaf5fd50a2e0d72400834232dfac09fbb6ea5473f929`

## Final verdict

All three frozen blocker-geometry history candidates are valid, stable, identity-safe,
strict-prior and incremental versus basic pregame roster metadata, but only cover
**39.5382%** of the frozen broad 2021 rostered-OL player-week universe.

Final disposition for all three:

`ENGINEERING_READY_SOURCE_THIN`

No predictive protection/QB/rushing experiment is authorized from this V1.

## Candidate evidence

| Candidate | Eligible rows | Observed | Broad coverage | Stability Spearman | Stability pairs | Holdout reconstructibility R2 | Final |
|---|---:|---:|---:|---:|---:|---:|---|
| blocker snap-distance history | 3,465 | 1,370 | 39.5382% | 0.846965 | 246 | 0.749967 | `ENGINEERING_READY_SOURCE_THIN` |
| blocker min-distance history | 3,465 | 1,370 | 39.5382% | 0.487057 | 246 | 0.072124 | `ENGINEERING_READY_SOURCE_THIN` |
| blocker time-to-min history | 3,465 | 1,370 | 39.5382% | 0.575260 | 246 | 0.335707 | `ENGINEERING_READY_SOURCE_THIN` |

Median strict-prior support when present: **89 interactions**.

Redundancy reconstruction:
- train: 2021 Weeks 1-4, **552** rows
- holdout: 2021 Weeks 5-8, **818** rows
- inputs: position one-hot, depth-chart-position one-hot, height, weight, years experience
- target-game participation: not used
- PFF pressure outcomes: not read

The snap-distance R2 of **0.749967** remains below the frozen 0.75 review boundary.
It is not rounded upward or reclassified.

## Identity / integrity

- BDB blocker IDs: **590**
- direct BDB nfl_id -> GSIS -> weekly roster mappings: **590**
- direct stable-ID bridge coverage: **1.0000**
- ambiguous BDB nfl_id mappings: **0**
- ambiguous same-week roster GSIS mappings: **0**
- roster name variants under a stable GSIS: **1**, diagnostic only
- name fallback used: **false**
- rostered OL player-weeks: **3,465**
- unique rostered OL players: **504**
- Week-1 rows retained: **484**
- duplicate published player-week rows: **0**
- join fanout: **0**
- mapped snapshot duplicates: **0**
- chronology violations: **0**
- materializer support-threshold violations: **0**
- blocker-history rows checked: **3,935**
- target-game history rows used: **0**
- target-game snap/participation used for eligibility: **false**
- universal blocker-rusher assignment claimed: **false**

Source and derivative hashes:
- BDB2023 source: `1c3e1eb6fcd0cf85807c649804fa0c72421c76e973e40a742523f1dd3a929182`
- raw interaction derivative:
  `521682a6c2e0e5ced6b4ab96accf793aae061c63a0234dca76ce4459e6dd206d`
- blocker-history derivative:
  `d09077cdb81a4b53676070f213f3ee4662e0eb06f853b1dbf35f3c1a138bcb78`
- weekly roster:
  `fed7d0c54abe1d871b44bdcea8606ab621617ad1d5d93b87625d07f4e0db6650`
- nflverse player crosswalk:
  `35a60c7d63dee4e78c8085ebb933b0b344f5d67b726f05defb1179c1bd0b9be5`

## Mechanical lineage preserved

Run `35514021970` failed before qualification because the repo's legacy pandas 1.5.3
runtime could not execute the already-certified materializer's nullable-Int64 merge.
The certified materializer was then isolated in a temporary modern-pandas runtime;
its code and frozen source hash were unchanged.

Run `35514108207` completed but the first qualification implementation incorrectly
treated a name variant under the same stable GSIS ID as identity ambiguity. The frozen
plan says names are diagnostic only. That mechanical identity interpretation was fixed
without changing coverage, support, geometry, redundancy thresholds or source data.

The final canonical result is run `35514256900`.

## Governance

Do not rescue V1 by:
- restricting to starters/high-snap players;
- excluding Week 1;
- using only later weeks;
- lowering the 10-interaction support threshold;
- lowering the 80% broad coverage floor;
- choosing a favorable OL position;
- using PFF target-game pressure outcomes;
- changing interaction semantics into universal assignment.

Final disposition:

`BDB2023_BLOCKER_PROTECTION_GEOMETRY_QUALIFICATION_V1_SOURCE_THIN_CLOSED`

## Next mechanism

Per the frozen pre-result directive, leave BDB blocker geometry and move to a
different personnel/continuity mechanism.

The next candidate is broad **offensive-line roster continuity**, derived from
nflverse weekly roster identities at scheduled team-game grain. It does not require
tracking geometry and is designed to measure whether the current OL personnel group
is materially continuous with the same team's previous game.
