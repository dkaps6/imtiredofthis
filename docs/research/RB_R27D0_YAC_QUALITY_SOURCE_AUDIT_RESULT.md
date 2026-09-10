# RB R27D0 — YAC-Quality Source Audit Result

Status: `R27D0_YAC_QUALITY_SOURCES_SUPPORT_SEPARATELY_FROZEN_PREDICTIVE_STUDY`

This is source/novelty evidence only. It is not a predictive result and authorizes no production change.

## Canonical authority

- Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- Parent R27C2 result commit: `2cfca82d919290ac18ed01559994d2e0239b0798`
- Audit plan commit: `128e43a857b7d8ce9e63d9b11644ecb3f5f9a795`
- Audit plan blob: `f5b21545bc880f337afa940d20db2205846a3bc3`
- Audit script blob: `48644f59ec10717e3ef7dd3470643fb29520cc23`
- Workflow blob: `821b8b623ff1152e3eaff887f16caa2c7cdd3715`
- Audit lock / workflow head: `28bbe956fb1fda7b820a78239958e827eb45d513`
- Run: `34431705294`
- Job: `102728396342`
- Artifact: `10134726607`
- Artifact name: `rb-r27d0-yac-quality-source-audit`
- Artifact digest: `sha256:b1a2a44d7ed0303300a792bf617ac6c5095aa7a0229186001504ebc2cad44bdf`

## Integrity boundary

PASS.

- model fit performed: false
- candidate projection created: false
- prediction error scored: false
- sportsbook inputs: 0
- production changed: false
- R26 changed: false
- R22 changed: false
- exact R27B V2 parent artifact verified
- exact R27C2 parent artifact verified

## Finding 1 — nflverse PBP expected-YAC is exceptionally complete for RB receiving

The canonical PBP source exposes:
- `xyac_mean_yardage`
- `xyac_median_yardage`
- `xyac_success`
- `xyac_fd`

Completed-RB-catch non-null coverage for `xyac_mean_yardage`:
- 2020: `99.0916%`
- 2021: `99.4870%`
- 2022: `99.0280%`
- 2023: `99.2947%`
- 2024: `99.1497%`
- 2025: `98.9682%`

Raw YAC coverage on completed RB catches is 100% in every season.

Therefore strict-prior PBP-derived quantities can distinguish:
- expected YAC implied by catch/play context; and
- realized YAC above/below that expected level.

This is genuinely different information from R27B V2's raw historical YAC averages.

## Finding 2 — strict-prior PBP YAC-over-expected history is dense in the exact failing cohort

Exact targeted vacancy-RB1 rows: `467`.

All vacancy RB1 targeted games:
- >=1 strict-prior PBP YACOE game: `95.5032%`
- >=3 strict-prior PBP YACOE games: `91.2206%`
- mean prior PBP YACOE games: `25.10`

2023 vacancy RB1 targeted games, n=72:
- >=1 prior PBP YACOE game: `98.6111%`
- >=3 prior PBP YACOE games: `95.8333%`
- mean prior PBP YACOE games: `30.74`

Non-2023 vacancy RB1 targeted games, n=395:
- >=1 prior PBP YACOE game: `94.9367%`
- >=3 prior PBP YACOE games: `90.3797%`
- mean prior PBP YACOE games: `24.08`

So source sparsity is not a plausible reason to avoid a strict-prior YAC-quality experiment.

## Finding 3 — granular RB target-context fields are also essentially complete

Across 2020–2025 RB target rows, all of the following meet >=90% coverage in every season and in practice are roughly 99.6–100%:
- down
- yards to go
- shotgun
- no-huddle
- pass location
- pass length
- score differential
- air yards

These fields may support a separately frozen target-quality-context study, but this audit does not authorize any particular feature set.

## Finding 4 — NFL Next Gen Stats receiving YACOE is not usable for this RB study in the current source feed

The NGS schema itself contains:
- `avg_yac_above_expectation`
- `avg_expected_yac`
- `avg_separation`
- `avg_cushion`
- `avg_intended_air_yards`
- player/team/week identifiers

However, after native player-position resolution, the historical receiving feed contains zero RB rows for every audited season 2020–2025.

Therefore NGS is rejected as an RB-specific historical source for this study. Do not engineer around it, impute WR/TE tracking values to RBs, or claim NGS RB support from schema existence alone.

## Novelty interpretation

The viable new information is **PBP expected YAC / YAC-over-expected plus strict-prior target-context composition**, not another raw historical YAC/YPR average.

R27B V2 already tested raw player/team/opponent YAC context and failed to repair 2023 RB1. R27D must therefore distinguish at least conceptually between:

1. how much YAC the historical catch context was expected to produce; and
2. whether the RB/offense/defense systematically generated YAC above or below that expectation.

The exact predictive construction must be frozen in a new plan before scoring.

## Disposition

`R27D0_YAC_QUALITY_SOURCES_SUPPORT_SEPARATELY_FROZEN_PREDICTIVE_STUDY`

This means the data lane is sufficiently complete and genuinely incremental to justify a new predictive test. It does not mean the information is predictive, and it does not authorize production.
