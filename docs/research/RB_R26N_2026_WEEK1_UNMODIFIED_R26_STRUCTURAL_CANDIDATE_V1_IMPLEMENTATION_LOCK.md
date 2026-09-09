# RB R26N 2026 Week-1 Unmodified-R26 Structural Candidate V1 — Implementation Lock

Status: LOCKED BEFORE CANDIDATE EXECUTION
Date: 2026-09-09

Frozen plan:
`docs/research/RB_R26N_2026_WEEK1_UNMODIFIED_R26_STRUCTURAL_CANDIDATE_V1_FROZEN_PLAN.md`

Candidate builder:
`scripts/backtest/build_rb_r26n_2026_week1_unmodified_r26_structural_candidate_v1.py`

Frozen-plan commit:
`919285fd0e2042461a5a89472f6832c01da857b4`

Candidate implementation commit:
`5299ce54575ffcfe33ad203db0ee00285181291f`

Production comparison base:
`f8417f55b04ce0e19baf260e9d532765034c47f1`

## Immutable parent contract

R26M qualification authority:
- run `34390505549`
- artifact `10119429741`
- artifact name `rb-r26m-2026-week1-prospective-qualification-synthesis-v1`
- digest `sha256:1306a3a2e58a0b129ac7e9fe34ad6407d87c491494e8fc27dd0284ba96996b76`
- required disposition `2026_WEEK1_UNMODIFIED_R26_SHADOW_CANDIDATE_DESIGN_QUALIFIED`
- requires `r26n_design_authorized == true`

Current production Full Slate authority:
- run `34317211395`
- artifact `10090547415`
- artifact name `run_34317211395`
- digest `sha256:7eab77e41c5879d4f54d87497eee0d1186010784cfc4ef966178930e16fb7c3b`
- head `f8417f55b04ce0e19baf260e9d532765034c47f1`
- expected current PlayerForm rows `468`
- expected current model-context rows `468`

R26L vacancy-state authority:
- run `34389455694`
- artifact `10119058769`
- artifact name `rb-r26l-2026-week1-regime-transportability-v1`
- digest `sha256:3351dfb5bbf6b571174a94ddf0a03179d70786edbb91d316e4fdf2c3cf005c46`
- required disposition `2026_SOURCE_REGIME_MODERN_LIKE_FOR_PROSPECTIVE_QUALIFICATION`
- exact vacancy state source `r26l_2026_week1_room_state.csv`

R19 prospective serialized R9 authority:
- run `34288244770`
- artifact `10080377483`
- artifact name `rb-r19-deployable-tail-scorer-refit-v1`
- digest `sha256:11432b9d7b7f2367935a862b63c30df9f40955e479806ab67d90921b63a907c7`
- head `6ac1342f737f142acac6a3e4b459f442faf1442a`
- exact inner model SHA-256 `9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba`

The launch workflow must verify all four artifact digests before candidate execution. Parent-integrity failure is a hard workflow failure, not a scientific R26N disposition.

## Locked reconstruction behavior

The candidate builder may reconstruct the current production target entitlement only from the exact downloaded Full Slate artifact and protected production code.

It must:
1. use the exact 468-row current `player_form_consensus.csv` and `model_context_bridge.csv`;
2. construct only a sportsbook-free synthetic lookup stub from player/team/opponent/season/week plus canonical event identity;
3. call the existing protected full-football-universe builder;
4. run the already-promoted M38/explicit entitlement -> TE-R5P -> WR-R15 chain with no new parameter;
5. use the resulting final `entitlement_tgt_share` as baseline;
6. preserve the exact 468-player / 32-team / 16-game current production population.

The older 469-player R22 integration artifact is not allowed as the R26N current baseline.

## Locked R26/R9 behavior

R26N may consume only the serialized R19 `models.r8_r9_identity` payload. It must not call a model `.fit()` or otherwise refit R8/R9.

The locked payload semantics are:
- feature order exactly equals the production-safe identity runtime `FEATURES`;
- `StandardScaler+Ridge`;
- alpha `20.0`;
- train clip `2.0`;
- prediction clip `1.0`;
- training season `2025`;
- R9 reliability `1.0`;
- strict-prior audit true;
- serialization round-trip true with max absolute delta `0.0`.

R26N then applies the frozen historical R26 transform exactly:
- population `RB` + `FB` only;
- preserve each event/team RB+FB target-entitlement pool;
- score R9 manually from the serialized payload;
- inherited R26L vacancy team -> R9 softmax within the preserved RB+FB room;
- non-vacancy team -> exact baseline within-room shares;
- same single floating-point closure correction as R26;
- all non-RB/FB player entitlement shares exact.

No additional role threshold, turnover threshold, 2020 exception, or historical router is allowed.

## Locked authority boundary

R26N is an opportunity/reception-layer structural candidate only.

It may write deterministic baseline/candidate target and reception means for inspection. It may not:
- use 2026 outcomes;
- use sportsbook football inputs;
- use same-week depth;
- fit/refit R9/R8;
- regenerate receiving-yard distributions;
- alter receiving-yard means;
- change R22;
- change production parameters;
- write to production paths;
- activate a live shadow;
- authorize production promotion.

The candidate must explicitly report:
- `2026_outcomes_used = 0`;
- `sportsbook_football_inputs_used = 0`;
- `same_week_depth_used = false`;
- `r9_refit = false`;
- `production_parameters_changed = false`;
- `r22_changed = false`;
- `receiving_yard_means_changed = false`;
- `receiving_distribution_regenerated = false`.

## Frozen structural gate contract

The implementation is bound to all 28 gates in the frozen plan. No gate, threshold, parent, cohort, current-row count, vacancy count, or disposition may be changed after this lock.

Passing disposition:
`R26N_2026_WEEK1_STRUCTURAL_CANDIDATE_PASS_READY_FOR_SHADOW_INTEGRATION_DESIGN`

Failing scientific/structural disposition:
`R26N_2026_WEEK1_STRUCTURAL_CANDIDATE_FAIL_NO_INTEGRATION`

A passing R26N authorizes only design/freeze of a separately governed shadow-integration study. It does not authorize live shadow activation or production promotion.

No candidate result had been materialized when this implementation lock was committed.