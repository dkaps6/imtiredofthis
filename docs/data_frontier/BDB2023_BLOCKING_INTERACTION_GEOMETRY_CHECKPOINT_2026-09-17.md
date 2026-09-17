# BDB 2023 Blocking-Interaction Geometry Checkpoint — 2026-09-17

## Scope

Isolated NFL data-frontier engineering only. No predictive experiment, no production-science change, no sportsbook change, no Issue #535 change.

- Branch: `data-frontier-phase0-bdb-contact-v1`
- Official source: Kaggle `nfl-big-data-bowl-2023`
- Source-manifest SHA-256: `1c3e1eb6fcd0cf85807c649804fa0c72421c76e973e40a742523f1dd3a929182`
- Canonical source-audit run: `35264295731`
- Canonical blocking-interaction run: `35264931453`
- Canonical Week-1 geometry run: `35265121166`
- Canonical Week-1 exception run: `35265277140`
- Canonical all-weeks geometry run: `35265440008`
- Raw competition files uploaded: **NO**

## Semantic correction

BDB 2023 should **not** be described as a perfect universal "true blocker-rusher assignment" table.

The exact PFF field `pff_nflIdBlockedPlayer` is better treated as the blocked-player / blocking-interaction identity for this corpus:

- 46,526 rows contain a blocked-player ID.
- source role is `Pass Block` on 44,526 rows and `Pass Route` on 2,000 rows.
- 46,095 interactions point to a defender whose same-play PFF role is `Pass Rush`.
- 429 point to a defender whose same-play role is `Coverage`.
- 2 blocked-player references do not resolve to a same-play PFF row, although both IDs exist in the player table.

The 2,000 `Pass Route` interactions are dominated by `CH` block type (1,508 rows), with 383 `SR` rows and smaller counts of other types. This confirms that route-player contact/chip interactions are represented and must not be discarded merely because the source player's final role is `Pass Route`.

For `Pass Block` rows, after excluding explicit `NB` and null block types, blocked-player identity is populated on 44,526 / 44,542 rows = **99.9641%**.

## Protection-interaction structure

The initial Pass-Block-only audit found:

- Pass Block rows: 46,057
- blocked-player IDs on Pass Block rows: 44,526
- self-assignments: 0
- duplicate blocker-target pair rows: 0
- blocker-target groups with one blocker: 23,057
- groups with multiple blockers: 10,479
- maximum blockers sharing one referenced defender on a play: 4

The same PFF table includes blocker outcome labels (`pff_beatenByDefender`, hit/hurry/sack allowed) and defender outcome labels (`pff_hit`, `pff_hurry`, `pff_sack`). These are role-scoped labels; null values outside the applicable role must never be interpreted as zeros.

## Event reconciliation contract

Tracking exposes both manual and `autoevent_` snap/throw labels. Week-1 audit found 582 plays with both snap label families; most differed by only 1-2 frames, but 16 differed by more than 3 frames and five by more than 10 frames, with extremes up to 140 frames.

The frozen geometry-lab rule is therefore:

**Snap**
1. if manual and auto snap labels are within 3 frames, use the manual snap;
2. if both exist but differ by more than 3 frames, use the earlier labeled snap;
3. if only one exists, use it;
4. if neither exists, abstain — do not infer a snap.

**Terminal protection frame**
1. if manual and auto pass-forward labels are within 3 frames, use manual pass-forward;
2. if both exist but differ by more than 3 frames, use the earlier labeled pass-forward;
3. if only one exists, use it;
4. otherwise use `qb_sack` or `qb_strip_sack` when available;
5. if no pass/sack terminal event exists, use the play's last tracking frame only as an explicitly flagged fallback.

No unlabeled snap is fabricated.

## Week-1 geometry pilot

Week 1 contained 6,310 resolvable blocking interactions.

- reconstructed: 6,295
- geometry coverage: **99.7623%**
- snap geometry among reconstructed: **100%**
- terminal geometry among reconstructed: **100%**
- all 15 unreconstructed interactions came from exactly three plays with no snap label
- none were caused by missing blocker or blocked-player tracking identity

Week-1 medians:

- snap blocker-target distance: 2.482 yd
- minimum distance in protection window: 0.762 yd
- terminal distance: 1.120 yd
- time from snap to minimum distance: 1.9 s
- shared frames in protection window: 29 frames

## All-weeks geometry result

The exact same frozen event/geometry contract was then run across Weeks 1-8.

- resolvable blocked-player interactions: **46,524**
- reconstructed interactions: **46,396**
- pooled geometry coverage: **99.7249%**
- pooled snap geometry among reconstructed: **100%**
- pooled terminal geometry among reconstructed: **100%**
- reconstructed source roles: 44,403 `Pass Block`, 1,993 `Pass Route`
- reconstructed target roles: 45,967 `Pass Rush`, 429 `Coverage`

Per-week geometry coverage:

| Week | Expected | Reconstructed | Coverage |
|---:|---:|---:|---:|
| 1 | 6,310 | 6,295 | 99.762% |
| 2 | 5,797 | 5,773 | 99.586% |
| 3 | 6,154 | 6,142 | 99.805% |
| 4 | 6,106 | 6,075 | 99.492% |
| 5 | 6,059 | 6,044 | 99.752% |
| 6 | 5,452 | 5,442 | 99.817% |
| 7 | 4,998 | 4,983 | 99.700% |
| 8 | 5,648 | 5,642 | 99.894% |

All weeks clear the frozen 99% minimum; pooled coverage clears the frozen 99.5% gate.

Across all reconstructed interactions, geometry quantiles are:

| Metric | p10 | p25 | p50 | p75 | p90 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| Snap distance (yd) | 1.679 | 2.013 | 2.486 | 3.222 | 4.230 | 9.915 |
| Minimum distance (yd) | 0.485 | 0.614 | 0.767 | 0.952 | 1.161 | 1.817 |
| Terminal distance (yd) | 0.634 | 0.808 | 1.137 | 2.440 | 5.403 | 16.406 |
| Seconds to minimum distance | 1.2 | 1.5 | 1.9 | 2.4 | 3.0 | 4.4 |
| Shared frames | 21 | 25 | 30 | 38 | 49 | 75 |

There are 24 plays across the eight weeks with no snap label under either source family. Their interactions are intentionally abstained. There are also 464 plays whose terminal protection frame uses the explicitly flagged last-tracking-frame fallback; downstream work must retain that confidence/provenance distinction.

## Disposition

**BDB2023_PROTECTION_INTERACTION_GEOMETRY_LAB_VALIDATED**

This means the project can deterministically construct high-coverage blocker/blocked-player geometric trajectories from official tracking plus PFF interaction identity on the public 2021 Weeks 1-8 slice.

It does **not** mean:

- the field is a universal primary-responsibility assignment;
- the 2021 slice is a production/live source;
- geometry has predictive value;
- production models should consume these fields;
- withdrawn BDB 2025 richer matchup semantics have been recreated.

The next data-frontier work should preserve this as a ground-truth lab, then continue to the next accessible source family rather than tuning geometry thresholds to force 100% coverage.
