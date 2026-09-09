# RB R26N 2026 Week-1 Unmodified-R26 Structural Candidate V1 — Frozen Plan

Status: FROZEN BEFORE CANDIDATE MATERIALIZATION
Date: 2026-09-09
Production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`

## Purpose

R26N is the first prospective 2026 Week-1 candidate build authorized by R26M.

It asks a deliberately narrow question:

> Can the **unmodified R26 Week-1 vacancy/R9 receiving-opportunity mechanism** be materialized on the exact current production football state with all intended structural invariants preserved, without refitting R9, without using 2026 outcomes or sportsbook football inputs, and without touching receiving-yard/R22 production authority?

R26N is an **opportunity/reception-layer structural candidate only**. It is not a live shadow integration and it does not regenerate receiving-yard distributions.

A passing R26N may authorize design of a separately frozen downstream shadow-integration study. It cannot authorize live shadow activation or production promotion.

## Immutable authority parents

### R26M qualification authority

- run `34390505549`
- artifact `10119429741`
- artifact name `rb-r26m-2026-week1-prospective-qualification-synthesis-v1`
- digest `sha256:1306a3a2e58a0b129ac7e9fe34ad6407d87c491494e8fc27dd0284ba96996b76`
- required disposition: `2026_WEEK1_UNMODIFIED_R26_SHADOW_CANDIDATE_DESIGN_QUALIFIED`

R26N must fail closed if R26M does not explicitly report `r26n_design_authorized == true`.

### Current production Full Slate authority

- run `34317211395`
- artifact `10090547415`
- artifact name `run_34317211395`
- digest `sha256:7eab77e41c5879d4f54d87497eee0d1186010784cfc4ef966178930e16fb7c3b`
- head `f8417f55b04ce0e19baf260e9d532765034c47f1`

This exact artifact is the current 2026 Week-1 football-state source. R26N must not substitute the older 469-player R22 integration universe because the post-merge production artifact contains a later 468-player roster/context state.

Known current parent-state counts, frozen as integrity checks rather than fitted parameters:
- `player_form_consensus.csv`: 468 rows;
- `model_context_bridge.csv`: 468 rows;
- 32 teams;
- Week 1;
- sportsbook football inputs absent from the football state.

### 2026 vacancy-state authority

R26L:
- run `34389455694`
- artifact `10119058769`
- artifact name `rb-r26l-2026-week1-regime-transportability-v1`
- digest `sha256:3351dfb5bbf6b571174a94ddf0a03179d70786edbb91d316e4fdf2c3cf005c46`
- required disposition `2026_SOURCE_REGIME_MODERN_LIKE_FOR_PROSPECTIVE_QUALIFICATION`

Required source file:
`data/backtests/r26l_2026_week1_regime_transportability_v1/r26l_2026_week1_room_state.csv`

R26N inherits R26L's canonical weekly-roster vacancy definition. The immutable R26L state contains 31 Week-1 vacancy teams. The one production RB room not in the R26L vacancy table must remain non-vacancy/baseline exact.

R26N must not recompute vacancy from Ourlads or same-week depth.

### Frozen prospective R9 predictor authority

R19 deployable scorer refit:
- run `34288244770`
- artifact `10080377483`
- artifact name `rb-r19-deployable-tail-scorer-refit-v1`
- digest `sha256:11432b9d7b7f2367935a862b63c30df9f40955e479806ab67d90921b63a907c7`
- head `6ac1342f737f142acac6a3e4b459f442faf1442a`

Required serialized model file:
`backtests/rb_r19_deployable_tail_scorer_refit_v1/rb_r19_tail_scorer_model_v1.json`

Required exact inner model SHA-256:
`9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba`

R26N may consume only the serialized `models.r8_r9_identity` payload from this artifact for the R26 opportunity overlay.

Required frozen payload semantics:
- feature order exactly equals the production-safe R8/R9 identity runtime feature order;
- model `StandardScaler+Ridge`;
- `alpha = 20.0`;
- `train_clip = 2.0`;
- `prediction_clip = 1.0`;
- `training_season = 2025`;
- `r9_reliability = 1.0`;
- strict-prior training audit passed;
- serialization max absolute delta `0.0`;
- sportsbook inputs `0`;
- future outcomes `0`.

This serialized 2025-trained payload is the frozen prospective R9 predictor. **R26N may not fit or refit any R9/R8 model.**

## Current production entitlement reconstruction

The post-merge production artifact does not persist `football_simulation_universe.csv` or `target_entitlement_v1_trace.csv`. R26N therefore reconstructs them deterministically from the exact downloaded production artifact using the already-promoted production code from the protected checkout.

The reconstruction must:
1. stage the immutable Full Slate artifact's `data/` files into the isolated runner workspace only after artifact digest verification;
2. build a sportsbook-free synthetic lookup stub from the exact current `player_form_consensus.csv` rows using only player/team/opponent/season/week and canonical game identity;
3. call the existing production full-universe builder, which defines the football universe from PlayerForm/model context rather than sportsbook rows;
4. run the existing promoted entitlement chain:
   - M38/raw baseline -> explicit team target entitlement;
   - TE-R5P inside conserved TE room;
   - WR-R15 inside conserved WR2+ room with M38 WR1 frozen;
5. use the resulting final `entitlement_tgt_share` as the R26N baseline.

No new entitlement parameter may be introduced.

Required reconstruction checks:
- exactly 468 football-player rows;
- exactly 32 teams;
- exactly 16 canonical games;
- exact identity equality between current PlayerForm and model-context roster;
- no forbidden sportsbook/line/odds/team-WP fields in the simulation universe;
- TE-R5P and WR-R15 production audits pass their existing room/team conservation gates;
- all final entitlement shares finite and nonnegative;
- team-level final modeled-player entitlement totals unchanged from the promoted production chain.

The earlier R22 integration artifact `10084118525` may be referenced only as historical certification/parity context. It is not the live R26N baseline because its player universe predates the current post-merge Full Slate state.

## Frozen R26 mechanism

R26N applies exactly the historical R26 allocation transformation, with no new router and no 2020 exception.

Population:
- current 2026 Week-1 production rows whose normalized position family is `RB` or `FB`;
- all other players are carried through exactly unchanged.

For every event/team RB+FB room:

1. Let `baseline_entitlement_tgt_share` be the final promoted production target entitlement.
2. Let `baseline_rb_pool` be the sum of baseline entitlement across RB+FB rows in that event/team.
3. Let `baseline_rb_within_share = baseline_entitlement_tgt_share / baseline_rb_pool`.
4. Build strict-prior receiving-identity features using the production-safe identity runtime with history ending at 2025. No 2026 statistics may be loaded.
5. Score the frozen serialized R9 model manually from its stored scaler/coefficients/intercept.
6. Clip the raw prediction to `[-1.0, 1.0]`.
7. `r9_calibrated_residual = r9_reliability * raw_prediction`.
8. `r9_score = log(baseline_rb_within_share + EPS) + r9_calibrated_residual`.
9. If the team is in the immutable R26L vacancy state, set candidate within-room shares to softmax(`r9_score`).
10. If the team is not in the R26L vacancy state, candidate within-room shares equal baseline within-room shares exactly.
11. Apply only the same floating-point closure correction used by R26: add `1 - sum(candidate_within_share)` to the maximum-share row.
12. `candidate_entitlement_tgt_share = baseline_rb_pool * candidate_within_share` for RB+FB rows.
13. For every non-RB/FB row, `candidate_entitlement_tgt_share = baseline_entitlement_tgt_share` exactly.

No role/state threshold beyond the binary R26L inherited vacancy gate may enter R26N.

## Frozen target/reception mean projection semantics

R26N reports deterministic opportunity/reception means for structural inspection only, using the same R26 expectation semantics:

- `team_pass_attempt_projection = rules_plays_est * rules_pass_rate`;
- `baseline_targets = team_pass_attempt_projection * baseline_entitlement_tgt_share`;
- `candidate_targets = team_pass_attempt_projection * candidate_entitlement_tgt_share`;
- catch-rate authority = current production `rules_catch_rate`, clipped to `[0.35, 0.95]` exactly as in historical R26;
- `baseline_receptions = baseline_targets * catch_rate`;
- `candidate_receptions = candidate_targets * catch_rate`.

These are prospective candidate means only. No 2026 result is used.

## Receiving-yard / R22 boundary

R26N **must not rerun or mutate receiving-yard distributions**.

The candidate overlay contains no changed receiving-yard mean, YPT model, tail model, residual pool, or R22 parameter. The protected production R22 files and model assets must remain clean against production authority.

R26N therefore reports:
- `receiving_yard_means_changed = false`;
- `r22_changed = false`;
- `receiving_distribution_regenerated = false`.

These flags mean R26N does not touch those components. They do **not** certify that a future integrated simulator can accept the R26 opportunity overlay without additional work. That is intentionally deferred to a separately frozen shadow-integration study if R26N passes.

## Frozen structural gates

R26N passes only if all are true:

1. exact R26M, Full Slate, R26L, and R19 artifact digests verified;
2. R26M exact qualifying disposition and `r26n_design_authorized == true`;
3. protected production runtime/model paths clean against `main@f8417f55b04ce0e19baf260e9d532765034c47f1`;
4. exact R19 serialized model inner SHA verified;
5. frozen R19 R8/R9 payload semantics and strict-prior/serialization governance verified;
6. no R9/R8 fitting or refitting occurs in R26N;
7. current production entitlement reconstruction covers exactly 468 players / 32 teams / 16 games;
8. current football universe is sportsbook-independent and matches the current PlayerForm/model-context identity roster exactly;
9. final promoted entitlement is finite/nonnegative and existing TE-R5P/WR-R15 conservation audits pass;
10. current production RB+FB universe covers all 32 teams;
11. every immutable R26L vacancy team maps to exactly one current production RB+FB room;
12. exactly one current production RB+FB team is outside the R26L vacancy set and receives baseline exact;
13. all strict-prior R9 input features are finite for all current RB+FB rows;
14. no strict-prior identity state reaches 2026 Week 1 or later;
15. maximum RB+FB room entitlement-pool conservation gap <= `1e-12`;
16. maximum all-player team entitlement-total delta <= `1e-12`;
17. maximum non-RB/FB entitlement delta <= `1e-12`;
18. maximum non-vacancy RB+FB entitlement delta <= `1e-12`;
19. candidate entitlement shares are finite and nonnegative;
20. no player is added to or removed from the current production football universe;
21. 2026 outcomes used = `0`;
22. sportsbook football inputs used = `0`;
23. same-week depth used = `false`;
24. R9 refit = `false`;
25. production parameters changed = `false`;
26. R22 changed = `false`;
27. receiving-yard means changed = `false`;
28. receiving distribution regenerated = `false`.

There is no performance/accuracy gate because 2026 outcomes do not exist in R26N.

## Frozen dispositions

### `R26N_2026_WEEK1_STRUCTURAL_CANDIDATE_PASS_READY_FOR_SHADOW_INTEGRATION_DESIGN`

All 28 structural/integrity conditions pass.

Maximum consequence: authorize design of a separately frozen shadow-integration study that determines how to carry the R26 target/reception overlay into the production distribution/pricing stack while preserving receiving-yard/R22 authorities.

### `R26N_2026_WEEK1_STRUCTURAL_CANDIDATE_FAIL_NO_INTEGRATION`

Any scientific/structural condition fails after parent-integrity verification.

No integration design is authorized.

If a parent digest or protected production boundary fails, the workflow must fail closed before emitting a scientific R26N disposition.

## Required outputs

- `r26n_current_production_entitlement.csv`
- `r26n_rb_identity_feature_trace.csv`
- `r26n_candidate_entitlement_overlay.csv`
- `r26n_rb_room_structural_audit.csv`
- `r26n_gate_matrix.csv`
- `r26n_disposition.json`

The candidate overlay must include, at minimum:
- event/team/player identity;
- position family;
- vacancy flag;
- baseline and candidate entitlement share;
- baseline and candidate within-RB-room share;
- frozen R9 raw/calibrated residual;
- baseline/candidate target mean;
- baseline/candidate reception mean;
- change delta;
- explicit R22/receiving-yard untouched markers.

## Authority ceiling

Even a passing R26N cannot directly authorize:
- live shadow activation;
- production promotion;
- sportsbook pricing changes;
- receiving-yard distribution regeneration;
- R22 mutation;
- receiving-yard mean mutation;
- R9 refit;
- historical 2020 deletion/reweighting/exemption;
- a new historical router.

It can authorize only **design/freeze of the next shadow-integration study**.