# TE-R5 PRODUCTION ENTITLEMENT ADAPTER V1 — Frozen Plan

## Purpose
TE-R5 (`TE_PARTICIPATION_ENTITLEMENT_V1_PASS`) is a strong leakage-safe individual-TE entitlement result, but its research candidate combined:
1. TE-R3's corrected finite team TE target pool; and
2. TE-R5's participation-aware within-room entitlement shares.

TE-R3 itself remains a scientific fail and is not a production authority. C2 full-stack receiver integration also remains a scientific fail, and the old TE-R6 plan explicitly prohibited launch unless C2 passed.

This test is therefore a **materially new, production-safe adapter**. It asks only:

> Can the already-proven TE-R5 individual room shares improve player targets and receiving yards when applied to the existing B0/production team TE pool, with no TE-R3 pool correction and no C2 receiver activation?

This is not TE-R6 and does not alter the old TE-R6 disposition or prerequisite.

## Frozen lineage
- TE-R5 authoritative run: `34132127351`
- TE-R5 job: `101774469114`
- TE-R5 launch SHA: `999c29d543e6854a903c5a0a4ee6fecbe69dce61`
- TE-R5 artifact: `10022512461`
- TE-R5 digest: `sha256:4f6d649492d2a08c4deeccd7944a4731e3d463e3f8d1bd40dcf8b9b82797af3d`
- TE-R5 disposition: `TE_PARTICIPATION_ENTITLEMENT_V1_PASS`
- TE-R3 remains `TE_TARGET_POOL_CONTEXT_MODEL_FAIL` and is not activated.
- C2 full-stack integration remains `CONSERVATION_INTEGRATION_CANDIDATE_FAIL` and is not activated for receivers.
- WR production hierarchy remains M38.
- QB point-mean authority remains M89/M90.
- RB Week-1 rushing authority remains RB-P3.
- Sportsbook inputs: 0.

## Exact evaluation cohort
Use the exact authoritative TE-R5 OOS player casebook for 2023-2025 (`te_r5_oos_player_casebook.csv`), with no row filtering based on outcomes.

Expected OOS player-games: 3,214.

## Frozen baseline
For every OOS TE player-game:
- baseline targets = exact `b0_targets_recon`;
- baseline receiving yards = exact `b0_rec_yards`;
- baseline team TE target pool = exact `b0_te_pool`;
- baseline efficiency = exact `b0_rec_per_target` and `b0_rec_yards_per_target`.

These values must remain unchanged.

## Frozen production-safe candidate
Reuse only the already-generated leakage-safe TE-R5 `candidate_room_share` from the authoritative OOS casebook.

For each team-game:
1. `adapter_target = b0_te_pool * candidate_room_share`;
2. `adapter_receptions = adapter_target * b0_rec_per_target`;
3. `adapter_rec_yards = adapter_target * b0_rec_yards_per_target`.

No TE-R3 `candidate_te_pool`, `te_pool_correction`, or C2 pool value may enter the candidate.

The adapter must preserve the exact B0 team TE target mass to numerical tolerance:
`sum(adapter_target by team-game) == b0_te_pool`.

No player-specific override, no coefficient refit, no alpha/epsilon/cap change, no matchup variable, no sportsbook input, and no new efficiency model are allowed.

## Frozen diagnostics
Report pooled and per-season 2023/2024/2025 metrics for:
- target MAE / RMSE / bias / corr / p90 absolute error;
- receiving-yard MAE / RMSE / bias / corr / median / p75 / p90 absolute error;
- 30+ and 40+ receiving-yard miss rates.

Also report:
- combined 2024-2025 receiving-yard metrics;
- highest B0 opportunity quartile metrics using the exact existing TE-R5 `b0_opportunity_quartile`;
- maximum team target-mass gap;
- target-mass preservation for every team-game;
- player/team/week duplicate rate;
- maximum absolute change to B0 team TE pool (must be zero by design);
- explicit count of rows whose candidate share came from TE-R5 OOS predictions.

## Frozen integrity gates
All must pass before science is interpreted:
1. exact 3,214 TE-R5 OOS player-games present;
2. all test seasons 2023, 2024, 2025 present;
3. duplicate player-team-week rate = 0;
4. TE-R5 `candidate_room_share` present for every scored row;
5. maximum candidate team target-mass gap <= `1e-9`;
6. B0 team TE pool is unchanged exactly;
7. B0 target/receiving-yard values are unchanged;
8. sportsbook inputs = 0;
9. same/future outcomes used to generate scored-game shares = 0, inherited from authoritative TE-R5 lineage;
10. production parameters changed = 0.

## Frozen scientific gates
`TE_R5_PRODUCTION_ENTITLEMENT_ADAPTER_ELIGIBLE` only if all are true:
1. pooled target MAE improves by >= **0.05 targets** vs B0;
2. pooled receiving-yard MAE improves by >= **0.10 yards** vs B0;
3. receiving-yard MAE improves in at least **2 of 3** OOS seasons;
4. combined 2024-2025 receiving-yard MAE improves;
5. pooled target p90 absolute error does not worsen;
6. pooled receiving-yard p90 absolute error does not worsen;
7. pooled 30+ yard miss rate does not worsen by > **0.25 percentage points**;
8. pooled 40+ yard miss rate does not worsen by > **0.25 percentage points**;
9. highest B0 opportunity quartile receiving-yard MAE improves by >= **0.10 yards**;
10. no single OOS season receiving-yard MAE worsens by > **0.50 yards**;
11. absolute receiving-yard bias does not worsen by > **1.0 yard**.

## Dispositions / stopping rule
- All integrity + science gates pass: `TE_R5_PRODUCTION_ENTITLEMENT_ADAPTER_ELIGIBLE`.
  - Then freeze a separate final-fit artifact step using the original TE-R3 + TE-R4 source data through 2025, fit the exact frozen TE-R5 StandardScaler+Ridge(alpha=20) model once, persist scaler/coefficients/intercept/feature contract, and run a 2026 Week-1 shadow adapter audit before any Full Slate production wiring.
- Integrity failure: `MECHANICAL_OR_SOURCE_FAILURE`; repair only the mechanical issue and rerun unchanged science.
- Any scientific gate fails: `TE_R5_PRODUCTION_ENTITLEMENT_ADAPTER_NOT_ELIGIBLE`; do not tune a nearby threshold/alpha/epsilon/cap or reactivate TE-R3/C2. Retain TE-R5 as research evidence but do not wire it into production through this adapter.
