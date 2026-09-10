# RB R27D0B — 2019 YAC-Quality Training-Source Extension Result

Status: `R27D0B_2019_XYAC_SOURCE_SUPPORTS_2020_OUTER_FOLD`

Source-extension evidence only. No model, candidate, performance scoring, sportsbook use, production change, R26 change, or R22 change.

## Canonical authority

- Parent R27D0 source-audit result commit: `8663b132c9910d07679d7b5e03ee8a8c9542f332`
- Frozen extension document: `docs/research/RB_R27D0B_2019_YAC_QUALITY_TRAINING_SOURCE_EXTENSION.md`
- Exact R27D0 audit-script blob reused: `48644f59ec10717e3ef7dd3470643fb29520cc23`
- Extension lock / workflow head: `3c2133581f549a6f4e251eea7da9a15d81f727b1`
- Run: `34431927369`
- Job: `102729055139`
- Artifact: `10134799189`
- Artifact name: `rb-r27d0b-2019-yac-quality-source-extension`
- Artifact digest: `sha256:13853603029adaf820b3228916c519acfdf98794c70462fbbcebe5f93f97d6cd`
- Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`

## Result

2019 source extension PASS.

2019 regular-season source counts:
- PBP rows: `45,339`
- target rows: `17,223`
- RB target rows: `3,548`
- completed RB targets: `2,709`
- receiver-position resolution: `99.9768%`
- raw YAC non-null on completed RB catches: `100%`
- `xyac_mean_yardage` non-null on completed RB catches: `99.2617%`
- `xyac_median_yardage` non-null: `99.2617%`
- `xyac_success` non-null: `99.2617%`
- `xyac_fd` non-null: `99.2617%`
- down: `99.7745%`
- yards-to-go: `100%`
- shotgun: `100%`
- no-huddle: `100%`
- pass location: `99.7745%`
- pass length: `99.7745%`
- score differential: `100%`
- air yards: `99.7745%`

NGS receiving again resolved to zero RB rows and remains rejected for RB-specific historical use.

## Interpretation

The exact source lane used by R27D0 is sufficiently populated in 2019 to provide legal pre-2020 history for the 2020 outer fold. Therefore a separately frozen 2020–2025 walk-forward R27D predictive study may include all six seasons without an arbitrary 2020 fallback caused by source absence.

This is source feasibility only; it is not evidence that xYAC/YACOE is predictive.
