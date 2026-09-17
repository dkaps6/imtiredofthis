# BDB 2026 Throw-Window Geometry Checkpoint — 2026-09-17

## Scope

Isolated data-frontier geometry research only. No predictive experiment, production change, sportsbook change, Issue #535 change, or frozen WR/RB/QB/TE science change.

- Branch: `data-frontier-bdb2026-throw-window-benchmark`
- Workflow: `.github/workflows/data-frontier-bdb2026-throw-window-benchmark.yml`
- Canonical Actions run: `35268286020`
- Benchmark source commit: `50f577085d8023ab2dd8bc9fc9b6c9ac5aa2919b`
- Run conclusion: **SUCCESS**
- Benchmark version: `BDB2026_THROW_WINDOW_GEOMETRY_V1`
- Official source corpus SHA-256: `228554c6600ac4e73529e5b6309193c8a3371acca7b4bc59792188acd814fb07`
- Raw competition files uploaded to GitHub: **NO**
- Per-play competition rows uploaded to GitHub: **NO**
- Predictive experiment: **NO**
- Exact coverage responsibility claimed: **NO**

The official Kaggle corpus was downloaded ephemerally inside GitHub Actions, fingerprinted, processed, and discarded. GitHub retains the workflow, this research checkpoint, source hashes, and sanitized aggregate benchmark artifacts only.

## Benchmark contract

- **Release frame** = final published pre-throw input frame for the play.
- **Terminal frame** = final published post-release output frame for the play. It is not relabeled as catch/arrival without a separate event marker.
- **Nearest defender** = geometric proximity only. It is not treated as exact coverage responsibility.
- Post-release defender geometry uses only defenders included in the official `player_to_predict = true` output set.

## Structural integrity

Published 2023 play universe used by this benchmark: **14,108 plays**.

- targeted-receiver role non-unique: **0**
- targeted receiver not flagged `player_to_predict`: **0**
- targeted receiver missing from post-release output: **0**
- plays missing final pre-throw target/defense geometry: **1**
- plays with usable release geometry: **14,107 / 14,108 = 99.9929%**
- plays with terminal targeted-receiver geometry: **14,107 / 14,107 = 100%**
- plays with no predicted defender in the post-release output set: **1,141**
- plays with usable post-release predicted-defender geometry: **12,966 / 14,107 = 91.912%**

The 1,141-play post-release defender limitation is an official output-scope limitation, not a targeted-receiver tracking failure.

## Pooled geometry

### Final pre-throw / release frame

Nearest defender to targeted receiver, `n = 14,107`:
- mean: **4.414 yd**
- median: **3.643 yd**
- p10: **1.165 yd**
- p25: **1.939 yd**
- p75: **5.890 yd**
- p90: **8.756 yd**

Second-nearest defender to targeted receiver:
- median: **8.286 yd**

Targeted receiver to ball landing point:
- mean: **6.767 yd**
- median: **5.294 yd**

Nearest defender to ball landing point:
- mean: **8.207 yd**
- median: **7.057 yd**

Defenders within 2 yards of target at release:
- mean: **0.265**
- median: **0**
- p75: **1**
- p90: **1**

Defenders within 3 yards of target at release:
- mean: **0.433**
- median: **0**

### Terminal post-release target geometry

Targeted receiver to ball landing point, `n = 14,107`:
- mean: **1.592 yd**
- median: **1.186 yd**
- p10: **0.379 yd**
- p25: **0.636 yd**
- p75: **1.980 yd**
- p90: **3.080 yd**

### Post-release predicted-defender geometry

Nearest predicted defender to targeted receiver at terminal frame, `n = 12,966`:
- mean: **3.027 yd**
- median: **2.394 yd**
- p10: **0.732 yd**
- p25: **1.242 yd**
- p75: **4.143 yd**
- p90: **6.118 yd**

Minimum nearest predicted-defender distance at any published post-release frame, `n = 12,966`:
- mean: **2.649 yd**
- median: **2.056 yd**
- p10: **0.638 yd**
- p25: **1.027 yd**
- p75: **3.670 yd**
- p90: **5.462 yd**

Nearest predicted defender to ball landing point at terminal frame:
- mean: **3.627 yd**
- median: **3.201 yd**

## Team coverage-family descriptive splits

These are descriptive geometry differences only. They are not causal findings and are not predictive model results.

### Man vs zone

`MAN_COVERAGE`:
- plays: **4,013**
- median release nearest defender: **2.138 yd**
- median terminal nearest predicted defender: **1.655 yd**
- median minimum post-release nearest predicted defender: **1.330 yd**

`ZONE_COVERAGE`:
- plays: **10,092**
- median release nearest defender: **4.295 yd**
- median terminal nearest predicted defender: **2.784 yd**
- median minimum post-release nearest predicted defender: **2.442 yd**

Two benchmark plays have null team man/zone labels.

### Coverage type — median release nearest / terminal nearest predicted defender

- `COVER_0_MAN`: n=601, **2.99 / 1.97 yd**
- `COVER_1_MAN`: n=3,175, **2.05 / 1.62 yd**
- `COVER_2_MAN`: n=237, **1.64 / 1.61 yd**
- `COVER_2_ZONE`: n=1,837, **5.19 / 3.41 yd**
- `COVER_3_ZONE`: n=4,482, **4.27 / 2.75 yd**
- `COVER_4_ZONE`: n=2,346, **3.75 / 2.50 yd**
- `COVER_6_ZONE`: n=1,388, **4.09 / 2.62 yd**
- `PREVENT`: n=39, **4.99 / 2.51 yd**

## Target-route descriptive splits

Median values shown as: release nearest defender / target-to-landing-point at release / terminal nearest predicted defender.

- `GO`: n=1,397, **1.59 / 15.19 / 1.21 yd**
- `POST`: n=769, **2.46 / 8.49 / 2.06 yd**
- `SLANT`: n=1,049, **2.59 / 4.93 / 1.92 yd**
- `WHEEL`: n=76, **2.65 / 13.81 / 2.00 yd**
- `CORNER`: n=509, **2.66 / 12.73 / 2.02 yd**
- `IN`: n=1,107, **2.84 / 5.62 / 2.01 yd**
- `OUT`: n=2,214, **3.24 / 6.29 / 2.30 yd**
- `CROSS`: n=1,496, **3.66 / 6.62 / 2.59 yd**
- `HITCH`: n=2,660, **3.84 / 2.08 / 2.46 yd**
- `ANGLE`: n=544, **6.70 / 2.10 / 3.12 yd**
- `FLAT`: n=1,983, **7.68 / 4.57 / 4.65 yd**
- `SCREEN`: n=301, **9.77 / 2.59 / 5.93 yd**

Two benchmark plays have null targeted-route labels.

## Interpretation

The benchmark demonstrates that the official BDB 2026 Analytics 2023 corpus can support a high-fidelity throw-window geometry laboratory at the final pre-throw frame and for the targeted receiver throughout the published post-release horizon.

The strongest cleanly supported derived families now include:

- targeted-receiver separation/proximity at release;
- second-nearest-defender spacing;
- defensive crowding around the target;
- defensive crowding around the known ball landing point;
- target-to-landing-point displacement at release and terminal output;
- closing geometry through the post-release output window when a defender is included in the official output set;
- descriptive splits by route, man/zone, and coverage family.

The source still does **not** provide exact individual defender responsibility for the targeted receiver. Nearest-defender geometry must remain separate from responsibility semantics.

## Disposition

**`BDB2026_THROW_WINDOW_GEOMETRY_LAB_VALIDATED_WITH_OUTPUT_DEFENDER_SCOPE_LIMIT`**

This is a validated research/geometry lab, not a predictive-model or production-science promotion.

Recommended next isolated research task: derive a versioned throw-window feature dictionary from only the pre-throw/release fields that would be temporally available under an eventual lawful live-data source, while keeping post-release measurements as retrospective ground-truth/validation targets rather than pregame features.
