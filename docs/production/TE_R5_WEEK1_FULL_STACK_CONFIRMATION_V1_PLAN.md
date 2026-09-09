# TE-R5 Week 1 Full-Stack Confirmation V1 — Frozen Plan

Date frozen: 2026-09-08 ET
Branch: `production-week1-player-prop-readiness-v1`
Parent scientific winner: `TE_PARTICIPATION_ENTITLEMENT_V1_PASS`
Parent run: `34132127351`
Parent artifact: `10022512461`
Parent digest: `sha256:4f6d649492d2a08c4deeccd7944a4731e3d463e3f8d1bd40dcf8b9b82797af3d`
Parent launch SHA: `999c29d543e6854a903c5a0a4ee6fecbe69dce61`
Status: PLAN FROZEN BEFORE PRODUCTION INTEGRATION

## Objective

Convert the already-supported TE-R5 mechanism into a strict-prior, deployable 2026 Week-1 scorer and prove it can enter the canonical Full Slate target-allocation path without breaking conserved opportunity, current promoted QB/RB/WR behavior, or sportsbook separation.

This is an integration/production-readiness confirmation, not a new TE feature hunt.

## Exact scientific mechanism preserved

TE-R5 is a two-stage opportunity mechanism:

1. **TE-R3 team TE-pool model** estimates the finite team TE target pool.
2. **TE-R5 individual entitlement model** allocates that finite pool among current TEs using strict-prior participation/history and baseline TE room shares.

TE-R3 remains a standalone scientific FAIL at final player receiving-yard level, but its team-pool submodel is an intentional frozen component of the subsequently successful TE-R5 candidate. Do not reinterpret R3 as independently promoted.

No TE catch-rate/YPT efficiency model is refit. The only promoted candidate action is target opportunity allocation.

## Frozen model semantics

### R3 team-pool submodel

Model family: `SimpleImputer(strategy='median') -> StandardScaler -> Ridge(alpha=20.0)`.

Target: `actual_te_pool - b0_te_pool`.

Predicted pool correction clipped to `[-3.0, +3.0]` targets.

Candidate TE pool: `max(0, b0_te_pool + clipped_correction)`.

Exact R3 features:

1. `b0_te_pool`
2. `b0_total_target_pool`
3. `b0_te_target_share`
4. `b0_te_room_size`
5. `b0_top_te_share`
6. `b0_te_hhi`
7. `team_te_pool_prior1`
8. `team_te_pool_prior4`
9. `team_te_pool_season_to_date`
10. `team_total_targets_prior4`
11. `team_total_targets_season_to_date`
12. `team_te_share_prior4`
13. `team_te_share_season_to_date`
14. `team_te_rec_yards_prior4`
15. `opp_te_targets_allowed_prior4`
16. `opp_te_targets_allowed_season_to_date`
17. `opp_total_targets_allowed_prior4`
18. `opp_te_target_share_allowed_prior4`
19. `opp_te_rec_yards_allowed_prior4`
20. `opp_te_receptions_allowed_prior4`

### R5 player-entitlement model

Model family: `StandardScaler -> Ridge(alpha=20.0)`.

Training target:

`clip(log(actual_room_share + 0.02) - log(b0_te_room_share + 0.02), -2, +2)`.

Inference residual clip: `[-1, +1]`.

Exact R5 features:

1. `b0_te_room_share`
2. `log_b0_te_pool`
3. `pool_ratio`
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

Per-team entitlement transform:

- `score = log(b0_te_room_share + 0.02) + clipped_predicted_residual`;
- softmax score within current modeled TE room;
- candidate TE target mass = exact R3 candidate TE pool;
- candidate player targets = candidate TE pool × candidate room share.

## 2026 live baseline bridge

The integration confirmation must use the canonical 2026 Full Slate football inputs, not sportsbook rows.

For each current team:

- canonical expected team pass opportunities are derived from the same pre-simulation `rules_plays_est` / `rules_pass_rate` semantics used by `simulation_v2.py`;
- canonical raw player target shares are taken from the same fallback chain used by `simulation_v2.py`: `rules_tgt_share`, then `bayes_tgt_share`, then `target_share`, then `tgt_share`;
- `b0_te_pool` is the sum of current TE expected targets before any TE-R5 adjustment;
- `b0_total_target_pool` is the modeled-player expected target pool on the same team using those exact raw shares;
- room-level B0 TE features are derived only from current pregame rows;
- strict-prior team/opponent historical features use completed 2020-2025 player-game evidence only for Week 1;
- strict-prior participation uses completed historical snap-count evidence only, with the exact TE-R4 semantics.

No current/future 2026 game outcome may enter the live scorer.

## Current-roster / transition behavior

The scorer operates on the current Full Slate TE room. New-team players retain any-team participation history while same-team history is naturally absent. Rookies/new players with no qualified history retain explicit availability/count states and zero-filled participation values only after those states are recorded, matching R5 training semantics.

No hand-entered player override is allowed.

## Production seam

If confirmation passes, TE-R5 must act **before canonical multinomial target allocation**.

For TE rows only, the production adapter will replace the canonical raw TE target shares with shares implied by the R5 candidate expected targets, while preserving the candidate finite TE target pool in the same team-pass-opportunity units.

M38 WR hierarchy still runs independently on WRs and preserves WR mass. RB/FB target shares are not directly modified by TE-R5.

The full team modeled target probability plus residual bucket must remain valid for canonical allocation. If the candidate TE pool would violate the simulator's probability contract, the integration must fail closed rather than silently rescale the scientific candidate.

## Frozen confirmation gates

All gates must pass before production promotion is authorized.

### Lineage / deployability

1. immutable R3, R4 and R5 source artifact metadata/digests are exact;
2. train-through-2025 R3 and R5 models fit with the exact frozen model families/features/caps;
3. serialization/deserialization prediction parity max absolute delta <= `1e-12`;
4. historical reproduction/parity checks demonstrate the deployable feature builder/model semantics reproduce the parent R3/R5 logic to numerical tolerance on sampled/qualified historical rows;
5. zero sportsbook inputs to either model.

### Strict-prior / Week-1 availability

6. all 32 teams and the authoritative Week-1 current TE universe are uniquely identified;
7. every scored TE has finite required B0/derived features after the model's frozen missing-value treatment;
8. participation builder uses only observations strictly before 2026 Week 1;
9. team/opponent historical features use only completed seasons/weeks strictly before the target game;
10. current/future 2026 outcome reads = 0;
11. current-roster new-team/rookie cases are represented fail-closed with explicit availability states, not guessed identities.

### Opportunity conservation

12. candidate TE target mass equals the R3 candidate TE pool for every team to <= `1e-9` expected targets;
13. R5 player room shares sum to 1.0 within every non-empty TE room to <= `1e-12`;
14. candidate target shares are finite and nonnegative;
15. canonical modeled target-share contract including residual bucket remains valid without hidden rescaling;
16. no WR, RB, FB or QB raw target share is directly altered by the TE adapter;
17. M38 WR hierarchy output is exactly unchanged for identical non-TE inputs.

### Full-stack protection

18. canonical simulation remains deterministic at fixed seed;
19. non-TE rushing components are exact;
20. promoted QB M89/M90 final passing-yard mean path remains exact;
21. promoted RB P3 Week-1 rushing-yard mean/path remains exact;
22. no sportsbook input enters football projections;
23. no production parameter is changed by the confirmation run itself;
24. current production validation/audit suite passes before and after the shadow adapter test.

### Scientific-mechanism parity

25. TE-R5 changes only TE target entitlement before simulation; catch-rate and YPT inputs remain canonical/frozen;
26. resulting TE receptions and receiving yards respond through the canonical joint simulation rather than post-hoc independent output patches;
27. the Week-1 confirmation artifact exposes baseline vs candidate TE expected targets, target shares, receptions means, receiving-yard means, and team-pool conservation for every TE/team.

## Dispositions

All gates pass:
`TE_R5_WEEK1_FULL_STACK_CONFIRMATION_PASS_PROMOTION_ELIGIBLE`

Mechanical/source/integrity failure:
`TE_R5_WEEK1_FULL_STACK_CONFIRMATION_MECHANICAL_FAIL`

Any scientifically material incompatibility with canonical opportunity/conservation architecture:
`TE_R5_WEEK1_FULL_STACK_CONFIRMATION_NOT_PROMOTABLE`

## Promotion rule

A confirmation PASS authorizes a **separate explicit production commit** wiring the exact frozen scorer/adapter into canonical Full Slate for 2026 Week 1. It does not itself change `main` production.

The production commit must retain fail-closed version/provenance audits and must be followed by canonical no-odds Full Slate validation before live-odds pricing.
