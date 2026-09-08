# RB R19 Deployable Tail Scorer Refit V1 — Frozen Plan

Date frozen: 2026-09-08
Parents:
- `RB_R9_RECEIVING_IDENTITY_SHRINKAGE_V1` scientific PASS
- `RB_R11_HIGH_STATE_PROBABILITY_V1` supported diagnostic
- `RB_R16_UPSIDE_TAIL_STATE_V1` supported diagnostic
- `RB_R17_MEAN_PRESERVING_TAIL_MIXTURE_V1` supported distribution research
- `RB_R18_CANONICAL_MC_TAIL_ADAPTER_PARITY_V1` parity PASS
Status: PLAN FROZEN BEFORE EXECUTION

## Purpose

R16-R18 established that strict-prior RB receiving identity/state information contains useful upside-tail signal, that the signal can improve a mean-preserving receiving-yard distribution, and that the resulting tail shape can be layered onto canonical Monte Carlo output without altering the receiving mean, target allocation, non-RB outcomes, or other RB components.

The remaining blocker is deployability for a real 2026 slate.

R19 is a refit/serialization/parity migration. It is NOT a new football hypothesis and it does NOT promote any RB receiving mean or production path.

Research/deployability question:

> Can the exact upstream research chain `R8 identity -> R9 reliability/target delta -> R11 high-state probability -> R16 p30/p50 tail risk -> R17 residual pools` be refit through the final completed 2025 season and serialized into a versioned football-only scorer artifact for prospective 2026 use, while reproducing the immutable historical lineage exactly where overlap exists?

## Frozen lineage inputs

Immutable parents required:
- R10 modern-stability artifact from run `34269618181`, artifact `10073920506`, digest `sha256:e76840ad1f915a18e729319c6a5d76b4786dff2e215da7de63522acbbb33ac6b`
- mechanically corrected R12/state artifact from run `34273055095`
- R16 artifact from run `34286363931`, artifact `10079630404`, digest `sha256:ca44a174dafadcaf27496b00efe4e933941211037b0eacea0431d72a9b4fa099`

Freshly reconstructed football-only training surface:
- completed 2025 historical inputs using the same historical-input builders and strict-prior player-log machinery used by the R8/R9/R10 lineage
- 2025 outcomes may be labels/training targets because 2025 is fully completed before 2026 scoring
- no 2026 outcomes may enter any feature or fitted parameter

## Frozen final 2026 upstream fit architecture

### A. R8/R9 identity model

Natural next-season extension of R10 annual folds:
- previous folds: 2021 -> 2022, 2022 -> 2023, 2023 -> 2024, 2024 -> 2025
- R19 final fit: **2025 -> score 2026**

Use the exact R8 19-feature strict-prior identity snapshot definitions:
1. prior_targets_pg
2. prior_receptions_pg
3. prior_target_share
4. prior_rb_room_share
5. prior_5plus_target_rate
6. prior_7plus_target_rate
7. last8_targets_pg
8. last8_receptions_pg
9. last8_target_share
10. last8_rb_room_share
11. prev_season_targets_pg
12. prev_season_receptions_pg
13. prev_season_target_share
14. prev_season_rb_room_share
15. same_team_prior_targets_pg
16. same_team_prior_rb_room_share
17. log1p_prior_games
18. log1p_same_team_prior_games
19. prev_season_available

Exact R8 model class:
- StandardScaler
- Ridge(alpha=20)
- training target = clipped log actual within-RB share minus log baseline within-RB share, clip [-2,2]
- model prediction clip [-1,1]

Exact R9 reliability:
- fit from training-season rolling-origin OOF blocks using existing frozen `_fit_reliability`
- raw slope `dot(pred,y)/dot(pred,pred)`, clipped [0,1]
- do not assume reliability = 1.0 merely because prior R10 folds were 1.0

R9 final target delta is computed by applying calibrated identity residual only inside the fixed baseline RB target room. The live/production receiving mean remains unchanged by R19; this R9 target delta is a scorer feature only.

### B. R11 high-state model

Final 2026 state model is the natural cumulative extension of the immutable R11 walk-forward design.

Training feature rows:
- use strict-OOS R10 prediction rows from seasons 2022-2025
- TOP20 receiving-identity backs only

Exact features:
- baseline_pred_targets
- prior_rb_room_share
- target_delta
- r9_raw_r8_residual

Exact model:
- StandardScaler
- LogisticRegression(C=1.0, penalty='l2', random_state=911)
- high5 label: actual_targets >=5

For prospective scoring:
- identity percentile is computed within the current slate/team-week RB universe using strict-prior `prior_rb_room_share`
- `identity_top20 = identity_pct > 0.80`
- state probability is model probability for TOP20 only; REST80 state probability = 0 exactly

### C. R16 tail models

Final 2026 R16 fits are the natural extension of the immutable R16 OOS folds:
- immutable folds: train 2023 -> test 2024; train 2023-2024 -> test 2025
- R19 final fit: train **2023-2025 -> score 2026**

Fit separate cat30 and cat50 logistic models using exact R16 full feature order:
1. baseline_pred_targets
2. baseline_pred_rec_yards
3. state_probability
4. prior_rb_room_share
5. r9_raw_r8_residual
6. frozen_ypt
7. identity_top20

Exact model class must match R16:
- StandardScaler
- LogisticRegression with the same frozen solver/penalty/C/random-state settings as the R16 script

Labels for final training:
- cat30: actual_rec_yards - baseline_pred_rec_yards >=30
- cat50: actual_rec_yards - baseline_pred_rec_yards >=50

### D. R17 residual pools

Freeze final completed-history residual pools from corrected baseline rows 2023-2025:
- NON_TAIL: residual <30
- TAIL_30_49: 30 <= residual <50
- TAIL_50_PLUS: residual >=50

Store exact sorted values, counts, and SHA256 hashes in the scorer artifact.

## Frozen historical parity tests

### 1. R11 2025 parity

Refit the exact R11 model on 2022-2024 strict-OOS R10 rows, score 2025 TOP20 rows, and compare against the corrected R12 2025 `state_probability` values on exact matching player-games.

Gate:
- matched rows >=95% of eligible corrected R12 TOP20 2025 rows
- max absolute probability difference <=1e-10

### 2. R16 OOS probability parity

Reproduce immutable R16 folds from corrected R12 rows:
- train 2023 -> score 2024
- train 2023-2024 -> score 2025

Compare generated cat30 and cat50 probabilities against immutable `rb_r16_tail_predictions.csv` on exact matching player-games.

Gate for each label/fold:
- exact-key match coverage = 100%
- max absolute probability difference <=1e-10

### 3. R17 residual-pool lineage audit

Regenerate the historical training pools used by R17 and require the known fold counts:
- 2023 train: NON_TAIL 1272, TAIL_30_49 56, TAIL_50_PLUS 28
- 2023-2024 train: NON_TAIL 2580, TAIL_30_49 122, TAIL_50_PLUS 48

Final 2023-2025 pool counts/hashes are recorded, not tuned.

## Frozen final-fit integrity gates

R19 passes only if ALL are true:

1. `r11_2025_parity_coverage`: matched eligible rows >=0.95.
2. `r11_2025_probability_parity`: max abs state-probability delta <=1e-10.
3. `r16_probability_parity`: cat30/cat50, 2024/2025, each 100% key coverage and max abs probability delta <=1e-10.
4. `r17_pool_count_parity`: both known historical R17 pool-count triplets reproduce exactly.
5. `final_r9_feature_complete`: all 19 final 2025 training features finite for every row used by the final R8/R9 fit after the same historical eligibility rules as the original evaluator.
6. `final_r9_reliability_range`: final 2025 R9 reliability is finite and within [0,1].
7. `final_r11_fit_valid`: both high5 classes are present and all serialized scaler/model values finite.
8. `final_r16_cat30_fit_valid`: both classes present; serialized scaler/model values finite.
9. `final_r16_cat50_fit_valid`: both classes present; serialized scaler/model values finite.
10. `residual_pools_valid`: each final 2023-2025 residual pool is nonempty, finite, sorted, and hashed.
11. `serialization_roundtrip`: serialized model parameters reproduce in-memory probabilities/predictions to max abs delta <=1e-12 on a deterministic audit sample.
12. `strict_prior_audit`: 2025 identity features use only games strictly before each training row; violations = 0.
13. `future_outcome_zero`: 2026/current outcomes used as features = 0.
14. `sportsbook_zero`: sportsbook inputs added = 0.
15. `production_parameters_zero`: production parameters changed = 0.

## Frozen scorer artifact contract

R19 must materialize a versioned JSON/NPZ-compatible scorer artifact containing at minimum:
- candidate/version identifier
- source run IDs and artifact digests
- git commit SHA
- R8/R9 19-feature order
- final R8 StandardScaler means/scales
- final R8 Ridge coefficients/intercept/alpha
- final R9 reliability
- final R11 feature order, scaler means/scales, logistic coefficients/intercept/C/random_state
- final R16 feature order, scaler means/scales, cat30 coefficients/intercept, cat50 coefficients/intercept, model settings
- final 2023-2025 residual pool counts and SHA256 hashes
- residual pool values in a companion artifact file
- strict fail-closed required live input fields
- explicit `sportsbook_inputs_added = 0`
- explicit `production_parameters_changed = 0`

## Required live scorer inputs (fail closed)

The later prospective 2026 scorer must require enough football-only inputs to derive:
- event/team/player identity and position
- current finite RB target entitlement / target share
- projected plays and pass rate or already-certified team pass attempts
- current/frozen YPT
- strict-prior historical RB target/reception/team-room logs needed for the exact R8 snapshot

R19 must not silently substitute generic PlayerForm features for missing R8 identity fields.

## Governance

PASS means a faithful, versioned 2026 tail-scorer artifact exists and is historically parity-verified. It remains SHADOW-ONLY.

PASS does NOT:
- promote R12 or any RB receiving mean,
- alter canonical target entitlement,
- alter `scripts/simulation_v2.py`,
- activate the R18 adapter in Full Slate,
- change WR/TE/QB production models,
- introduce sportsbook inputs.

After PASS, the next separately frozen migration must score a real 2026 slate in shadow mode and run full-slate fail-closed parity/data-quality checks before any promotion discussion.

FAIL means stop and diagnose the failed parity/refit/serialization gate before changing the scorer architecture or model definitions.
