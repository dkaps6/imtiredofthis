# Project Continuity Delta — 2026-09-07 11:46 ET

Read this together with `docs/migrations/PROJECT_CONTINUITY_LATEST.md`.

## New since the current continuity ledger was drafted

### TE-R5 participation entitlement — SCIENTIFIC PASS

- Branch: `research-te-r5-participation-entitlement-v1`
- Launch SHA: `999c29d543e6854a903c5a0a4ee6fecbe69dce61`
- Run: `34132127351`
- Job: `101774469114`
- Artifact: `10022512461`
- Digest: `sha256:4f6d649492d2a08c4deeccd7944a4731e3d463e3f8d1bd40dcf8b9b82797af3d`
- OOS player-games: `3214`
- Test seasons: 2023-2025
- Disposition: `TE_PARTICIPATION_ENTITLEMENT_V1_PASS`
- Result documentation commit on TE-R5 branch: `6aea5286296d01746b64ca2a08b64a688db60675`

Pooled TE target metrics:
- MAE `1.682530 -> 1.541270`
- RMSE `2.453207 -> 2.094290`
- bias `-0.944747 -> -0.104208`
- correlation `0.613790 -> 0.648380`
- p90 AE `4.036641 -> 3.326261`

Pooled TE receiving-yard metrics:
- MAE `16.267971 -> 15.858042` (improvement `0.409929` yards)
- RMSE `23.498617 -> 21.620042`
- bias `-6.610834 -> -0.350016`
- correlation `0.511749 -> 0.536554`
- p90 AE `37.227233 -> 33.284738`
- 30+ miss rate `13.8768% -> 12.4144%`
- 40+ miss rate `8.7430% -> 6.5028%`

Season receiving-yard MAE improved 3/3:
- 2023 `16.460921 -> 15.637151`
- 2024 `16.602711 -> 16.284063`
- 2025 `15.765901 -> 15.646121`

Latest 2024-2025 TE receiving-yard MAE:
- `16.178397 -> 15.960587`

High projected-volume TE Q4:
- receiving-yard MAE `24.073076 -> 22.255704`
- p90 AE `55.922507 -> 44.947576`
- 40+ miss rate `18.1592% -> 14.9254%`

All frozen integrity and scientific gates passed. Production was not changed. Authorized next step is a separate TE full-stack integration / confirmation migration that checks 2026 Full Slate availability and interaction with shared pass opportunity + C2 conservation before promotion.

### C2 full-stack pass/receiving integration — ACTIVE

Initial integration run:
- Run `34131591002`
- Job `101772755332`
- SHA `a4f1c3308f159fb462a5ae5e9767338780954fae`
- Artifact `10022570140`
- Digest `sha256:804a49aa46aad7198bbef5d8c5c0a93381065e2d505b48476dd5b5cd205853dc`

The initial run was a mechanical-only failure. All six seasons built; evaluator crashed because intentionally empty 2020-2023 QB distribution files were read with `pd.read_csv`, causing `EmptyDataError`. No science result was taken.

Mechanical repair:
- SHA `647514b7dbeb5cab3223fc603e0a74e721879ba5`
- I/O-only fix; no science/gate/model changes.

Canonical repair rerun:
- Run `34139757238`
- Job `101798849722`
- Head SHA `647514b7dbeb5cab3223fc603e0a74e721879ba5`
- Current state at this timestamp: `in_progress`
- Current step: `Build inputs and run exact B0 vs integrated C2 across 2020-2025`
- Evaluator/gates pending.

### Other position status at this timestamp

- QB: no new scientific candidate beyond QB-PD3. M89/M90 still production. Week-1 football-only attempts/YPA/current-team/opponent/receiver-path audit remains outstanding.
- WR: no WR-R12 run yet. WR-R11 is closed scientific fail. Next authorized lane is relative player entitlement around M38 using finite opportunity + role/participation/competition/injury/transition context, with efficiency separated.
- RB: no new multiseason shared-room entitlement run yet. P3 remains production anchor. PD5 is only exploratory 2025 evidence; M95 tail overlays and depth-rank remap remain closed.
- TE: TE-R5 is now the strongest new individual-player predictive win and should be advanced through separate full-stack confirmation rather than retuned.

## Immediate execution order

1. Finish and formally disposition C2 integration run `34139757238`.
2. Freeze TE-R5 full-stack confirmation / production-integration test.
3. Launch WR relative-entitlement candidate around M38.
4. Build RB multiseason shared-room + player-entitlement + separate-efficiency candidate.
5. Complete QB 2026 Week-1 football-only component/path audit and connect QB attempts to shared pass state.
6. Keep all results documented with exact run/job/SHA/artifact/disposition and update continuity files as the authorized next step changes.