# TE-R5 — Participation Entitlement V1 — Frozen Plan

## Purpose

Test the first player-entitlement candidate under `JOINT_OPPORTUNITY_ENTITLEMENT_V1`.

TE-R2 established that team TE target-pool error is the dominant TE target-error layer. TE-R3 then materially improved the team TE pool and individual target MAE, but failed receiving-yard gates because its pool correction was positive in 99.48% of OOS team-games and the extra pool was allocated by unchanged B0 room shares. TE-R4 independently proved that strict-prior offensive participation is available for nearly the entire historical TE cohort.

The new hypothesis is therefore materially different from TE-R3:

> **Keep the leakage-safe TE-R3 team-pool estimate, but allocate that finite pool through a normalized, participation-aware individual entitlement model instead of proportionally inflating every TE.**

This is research only. It does not promote TE-R3, alter production, or change catch/YPT efficiency mechanics.

## Exact lineage

- Architecture parent: `research-joint-opportunity-entitlement-v1`, plan commit `b69c828098bce9c7fb4a72ee9bc440a2c21d1db9`.
- TE-R3 source run: `34126813280`, job `101757309151`, SHA `c9d1cd1858e34900e33fabc07848fc3f9f86bd1e`, artifact `10020423842`, digest `sha256:99699c7e4745ab8cf678b527fc57325c5f402a5c7be930ff9cb90dba96e63ff8`.
- TE-R3 disposition remains `TE_TARGET_POOL_CONTEXT_MODEL_FAIL`.
- TE-R4 source run: `34127474412`, job `101759439856`, SHA `ffe4e1101193b3502a6b069d2b77347049719b1d`, artifact `10020700686`, digest `sha256:56de7efe302cdf9c329c3a0386d798789d15e6a495c865535291173b6840c163`.
- TE-R4 disposition: `STRICT_PRIOR_TE_PARTICIPATION_ELIGIBLE`.
- Sportsbook inputs: 0.

## Cohort and walk-forward folds

Use the exact TE-R3 OOS player casebook joined one-to-one where available to the TE-R4 strict-prior participation casebook by `season, week, team, player_clean_key/player_key`.

Because TE-R3's team-pool candidate begins in 2022, the entitlement model uses only prior completed OOS seasons:

- train 2022 -> test 2023;
- train 2022-2023 -> test 2024;
- train 2022-2024 -> test 2025.

2022 is training-only for the individual entitlement model. No target-season outcome may enter its own features.

## Frozen baseline and pool

- B0 individual target baseline: exact `b0_targets_recon` / `b0_te_room_share` from TE-R3.
- Team pool for the candidate: exact frozen TE-R3 `candidate_te_pool`. No refit, coefficient change, cap change, or alternate pool candidate is allowed.
- Team candidate target mass must sum to `candidate_te_pool` to numerical tolerance for every team-game with at least one modeled TE.

## Frozen entitlement target

For training rows only:

1. actual room share = `targets / actual_te_pool` when `actual_te_pool > 0`;
2. baseline room share = `b0_te_room_share`;
3. epsilon = `0.02`;
4. residual target = `log(actual_room_share + 0.02) - log(b0_te_room_share + 0.02)`;
5. clip the training residual target to `[-2.0, +2.0]`.

This target is never computed for a future/test season before the model has produced that season's predictions.

## Frozen model

`StandardScaler + Ridge(alpha=20.0)`.

No hyperparameter search and no alternate model family.

### Exact features

All are pregame or derived only from pregame rows. The list below is aligned to the exact qualified TE-R4 artifact schema **before any TE-R5 model result exists**; TE-R4 materializes prior-3 any-team participation plus prior-1 same-team participation, but does not materialize a prior-3 same-team snap value.

1. `b0_te_room_share`
2. `log_b0_te_pool = log1p(b0_te_pool)`
3. `pool_ratio = candidate_te_pool / max(b0_te_pool, 0.25)` clipped `[0.50, 2.00]`
4. `room_size`
5. `prior1_same_team_offense_pct`
6. `prior1_same_team_offense_snaps`
7. `prior1_anyteam_offense_pct`
8. `prior3_anyteam_offense_pct`
9. `prior1_anyteam_offense_snaps`
10. `prior3_anyteam_offense_snaps`
11. `log1p_prior_count_same_team`
12. `log1p_prior_count_anyteam`
13. `prior1_same_team_available`
14. `prior3_same_team_available`
15. `snap_share_prior1_same_team`
16. `snap_share_prior3_anyteam`

Missing numeric participation values are filled with 0 only after the explicit availability indicators/counts are retained. Room-relative snap shares are computed only from same target-game TE rows' strictly-prior participation values.

## Frozen entitlement transform

For each test team-game:

- predict one residual entitlement score per TE;
- clip predicted residual to `[-1.0, +1.0]`;
- score = `log(b0_te_room_share + 0.02) + residual`;
- candidate player share = softmax(score) across modeled TEs on that team;
- candidate targets = `candidate_te_pool * candidate_player_share`.

No hand correction or player-specific override.

## Efficiency isolation

This experiment changes target opportunity only.

- candidate receptions = `candidate_targets * b0_rec_per_target`;
- candidate receiving yards = `candidate_targets * b0_rec_yards_per_target`.

Thus catch conversion and yards-per-target remain the exact B0 pregame efficiency assumptions. Any improvement must come from better pool + entitlement allocation, not an efficiency refit.

## Frozen integrity gates

All must pass before science is interpreted:

1. all 3 OOS test seasons 2023-2025 present;
2. >= 3,000 OOS player-games;
3. TE-R3/TE-R4 join coverage >= 0.90 on eligible 2023-2025 TE-R3 rows;
4. duplicate player-team-week rate = 0 after join;
5. zero same/future participation observations;
6. sportsbook inputs = 0;
7. maximum team candidate target-mass gap <= `1e-9`;
8. exact B0 target/receiving-yard values remain unchanged.

## Frozen scientific gates

`TE_PARTICIPATION_ENTITLEMENT_V1_PASS` requires all:

1. pooled player target MAE improves by >= **0.08 targets** vs B0;
2. pooled player receiving-yard MAE improves by >= **0.25 yards** vs B0;
3. receiving-yard MAE improves in at least **2 of 3** OOS seasons;
4. combined 2024-2025 receiving-yard MAE improves;
5. pooled target p90 absolute error does not worsen;
6. pooled receiving-yard p90 absolute error does not worsen;
7. pooled 30+ yard miss rate does not worsen by > 0.5 percentage points;
8. pooled 40+ yard miss rate does not worsen by > 0.5 percentage points;
9. highest-B0-opportunity quartile receiving-yard MAE improves;
10. no single OOS season receiving-yard MAE worsens by > 1.0 yard;
11. mean signed receiving-yard bias magnitude does not worsen by > 1.0 yard.

## Dispositions / stopping rule

- all integrity + science gates pass: `TE_PARTICIPATION_ENTITLEMENT_V1_PASS` and authorize a separately frozen full-stack TE integration test;
- integrity failure: `MECHANICAL_OR_SOURCE_FAILURE`, repair only the mechanical issue;
- any scientific gate fails: `TE_PARTICIPATION_ENTITLEMENT_V1_FAIL`.

If it fails scientifically, do not search alpha, epsilon, residual caps, alternate snap windows, or nearby thresholds. Diagnose whether remaining error is team-pool, individual entitlement, or per-target efficiency under the joint architecture and require a materially new football mechanism for the next test.
