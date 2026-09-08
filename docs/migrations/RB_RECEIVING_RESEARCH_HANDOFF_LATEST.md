# RB Receiving Research — Canonical Handoff

Last updated: 2026-09-08
Active research branch: `research-cross-position-catastrophic-casebook-v1`
Repository: `dkaps6/imtiredofthis`

## Purpose

This is the durable continuity ledger for the RB receiving-yards / receptions research lane. Future chats/agents should read this file before extending the lane. Do not infer production promotion from a research PASS. Preserve frozen gates, strict-prior feature rules, exact RB-room conservation, and zero sportsbook inputs upstream.

## User research principles carried forward

1. Project real football statistics accurately first; Vegas is downstream evaluation/pricing context, not an upstream football-projection feature.
2. Do not discard a football theory merely because a first implementation fails a composite gate; diagnose *why* it failed before changing the hypothesis.
3. Large/catastrophic misses matter. Do not win MAE by merely compressing projections if the model becomes worse on meaningful tails.
4. Distinguish receiving identity (who can command RB targets) from weekly receiving state (when that role activates).
5. Do not manufacture opportunity. RB receiving allocation is a finite room and conservation is mandatory.
6. Do not change frozen science gates after seeing results.
7. Distinguish coded/tested research, scientifically supported research, shadow/deployable artifacts, and certified/production-active code.

## Research lineage

### Identity hypothesis -> R8

A small set of RBs were persistent passing-game weapons and were disproportionately responsible for large underprojections. The preceding identity diagnostic showed the top 20% by prior RB-room receiving share had roughly 15.21 receiving-yard MAE, -6.15 yard bias, ~53% of 30+ misses and ~71% of 50+ misses.

R8 run `34263401496`, artifact `10071067712`, disposition `RB_R8_RECEIVING_IDENTITY_OOS_FAIL`.

Combined:
- targets MAE 1.618081 -> 1.541722
- receiving-yards MAE 12.781963 -> 12.474528
- RMSE 18.725660 -> 18.016724
- bias -2.958796 -> -1.421041
- p90 27.858 -> 28.427 (worse)
- 30+ miss 0.088608 -> 0.092586 (worse)
- 50+ miss 0.032911 -> 0.024231 (better)

R8 failed fresh p90, fresh 30+ guard, and fresh top-20 MAE.

### R8 forensic -> R9 shrinkage

The R8 FAIL was diagnosed rather than discarded. Q1-Q4 adjustment sizes improved while the largest-adjustment quintile Q5 caused the failure. Q5 MAE worsened ~20.027 -> ~20.638 and bias crossed from ~-9.18 to +1.65, while 50+ misses improved. Sparse history was not the primary explanation.

Conclusion: identity signal was real but amplitude was too aggressive. Apply reliability/shrinkage.

R9 run `34268245735`, artifact `10072877150`, scientific PASS.

Fresh 2016:
- target MAE 1.5549 -> 1.4970
- rec-yard MAE 12.1065 -> 11.7930
- RMSE 18.4157 -> 17.8832
- p90 27.7321 -> 26.4276
- 30+ 8.77% -> 8.05%
- 50+ 2.92% -> 2.64%
- bootstrap improvement probability 1.000
- 4/4 phases

2017 overall replicated; top-20 MAE slightly worsened in replication. R9 is a research scientific PASS, not automatic production promotion.

### R10/R11 -> weekly receiving-state theory

Identity answers who is a receiving weapon, not when a high-usage receiving game occurs. R10 performed state forensic/stability work. R11 tested strict-prior high-state probability and supported that high receiving states are partly predictable pregame.

R11 high-state features were frozen as:
- baseline_pred_targets
- prior_rb_room_share
- target_delta (R9 minus baseline projected targets)
- r9_raw_r8_residual

Only TOP20 receiving-identity RBs receive nonzero R11/R12 state probabilities; REST80 state probability is explicitly zero in R12.

### R12 -> state-gated identity

Theory: identity should express more strongly when the pregame state model says a high-usage receiving game is plausible. R12 improved multiple research metrics, including modern receiving-yard MAE roughly 10.79 -> 10.54 versus R9 and top receiving-back target MAE, but did not clear every frozen promotion requirement. R12 remains research-only/unpromoted.

R12 creates `frozen_ypt` from the baseline receiving-yards / baseline-target mapping and conserves the exact team RB target pool.

### R13-R15B -> efficiency hypotheses rejected

R13 tested persistent player efficiency / YPT history: unsupported.

R14B tested richer PBP player role/YAC/aDOT/explosive efficiency: unsupported.

R15/R15B moved efficiency to strict-prior team/opponent context. R15B run `34275426259`, artifact `10075521598`, disposition `RB_R15_EFFICIENCY_CONTEXT_SIGNAL_NOT_SUPPORTED_DIAGNOSTIC_ONLY`. Mechanical team-code correction `LA -> LAR`; football features changed 0; science thresholds changed 0.

ALL_RB combined:
- context_yac_per_target Spearman 0.072405; high8 AUC 0.524745
- context_explosive_target_rate Spearman 0.065150; high8 AUC 0.521492
- state_probability Spearman 0.028826; high8 AUC 0.500600
- frozen_ypt Spearman 0.085977; high8 AUC 0.537756

Conclusion after R13-R15B: stop blindly tuning mean YPT. Test distribution/tail behavior instead.

## R16 — upside-tail state diagnostic — SUPPORTED

Script: `scripts/backtest/diagnose_rb_r16_upside_tail_state_v1.py`
Workflow: `.github/workflows/research-rb-r16-upside-tail-state-v1.yml`
Workflow commit: `90ee0dd2448e24401b99a088f8b797646d4ec89a`
Actions run: `34286363931`
Job: `102262785132`
Artifact: `10079630404`
Artifact digest: `sha256:ca44a174dafadcaf27496b00efe4e933941211037b0eacea0431d72a9b4fa099`
Disposition: `RB_R16_UPSIDE_TAIL_SIGNAL_SUPPORTED_DIAGNOSTIC_ONLY`

R16 kept the baseline receiving mean frozen and tested whether strict-prior opportunity/state/identity information predicts upside receiving-yard tails.

R16 full features:
- baseline_pred_targets
- baseline_pred_rec_yards
- state_probability
- prior_rb_room_share
- r9_raw_r8_residual
- frozen_ypt
- identity_top20

Primary label: baseline actual-minus-prediction >=30 receiving yards.

Primary result:
- n 2,787
- events 169 (6.064%)
- combined AUC 0.673042 vs frozen 0.65 gate
- mean-only AUC 0.639598
- gain vs mean-only +0.033444 AUC
- Brier 0.055708 vs pooled 0.056963
- top-quintile lift 1.625469x
- top-quintile capture 32.544%
- 2/2 OOS folds >=0.58

Primary folds:
- 2024 AUC 0.657964
- 2025 AUC 0.689046

Secondary:
- 50+ underprojection AUC 0.766455; top-quintile lift 2.742x; capture 54.902%
- actual 40+ rec yards AUC 0.764674
- actual 60+ rec yards AUC 0.802164; top-quintile lift 2.933x; capture 58.730%
- >=20-yard reception label AUC 0.674779

PBP exact-target integrity on positive-target rows: 98.867%.
Sportsbook inputs added: 0.
Production parameters changed: 0.

Every frozen R16 support gate passed.

Important nuance: target-only was already informative (primary AUC 0.655782) and had stronger top-quintile capture on the 30+ label than the full model. The full state/identity model nevertheless improved overall primary discrimination and showed materially larger incremental value for the more extreme 50+ tail. Interpretation: R16 is strongest as incremental extreme-tail information layered on opportunity, not as a replacement for opportunity.

## R17 — mean-preserving tail mixture — SUPPORTED

Frozen plan: `docs/migrations/RB_R17_MEAN_PRESERVING_TAIL_MIXTURE_V1_PLAN.md`
Plan commit: `f803fd44b2bb8fe42a63ed211f44f373cc140d2f`
Implementation: `scripts/backtest/evaluate_rb_r17_mean_preserving_tail_mixture_v1.py`
Implementation commit: `9e58b6501bb2a6c1c769c2379d2ee581399f1835`
Workflow: `.github/workflows/research-rb-r17-mean-preserving-tail-mixture-v1.yml`
Workflow commit: `310288320feece05367dbefdd76f52f952d20a3c`
Actions run: `34286785433`
Job: `102264103483`
Artifact: `10079782239`
Artifact digest: `sha256:b3feaf53d882238d1cd3446e646742d0fffa6bc30c589db6aa045d69e5c0d878`
Disposition: `RB_R17_DISTRIBUTION_SIGNAL_SUPPORTED_RESEARCH_ONLY`
Science PASS: true

Combined comparator -> candidate:
- CRPS: 7.525223 -> 7.524093
- Brier30: 0.0574860 -> 0.0571548
- Brier50: 0.01802747 -> 0.01802536
- q90 pinball: 3.665299 -> 3.590688
- q95 pinball: 2.484965 -> 2.404085
- 80% coverage: 75.924% -> 77.144%
- 90% coverage: 86.473% -> 87.585%
- max mean delta: ~1.07e-14 yards

Every frozen R17 support gate passed.

Caution: the global CRPS gain is tiny and combined 50+ Brier gain is nearly zero. 2025 Brier50 slightly worsened. The clearest R17 value is upper-quantile loss and interval calibration, not a large global score change.

## R18 — canonical MC tail-adapter parity — SUPPORTED

Frozen plan: `docs/migrations/RB_R18_CANONICAL_MC_TAIL_ADAPTER_PARITY_V1_PLAN.md`
Plan commit: `1398eed191994079f9e5f1050adccd097cce633e`
Adapter: `scripts/modeling/rb_r17_tail_distribution_adapter_v1.py`
Adapter commit: `551ea7772310dcbf67d0671e74e5f069f71ad37c`
Evaluator: `scripts/backtest/evaluate_rb_r18_canonical_mc_tail_adapter_parity_v1.py`
Evaluator commit: `9e5f3de0eecca7de9adef2de0f60862d148e45ed`
Workflow: `.github/workflows/research-rb-r18-canonical-mc-tail-adapter-parity-v1.yml`
Workflow commit: `1fc889728ce3afdef22f6638203b0b9eda76099d`
Actions run: `34287356990`
Job: `102265907467`
Artifact: `10079996458`
Artifact digest: `sha256:3d2ded174147b5b2e5b8ecc7550b8600a1d3e930fce4eb0220ef9fe653e35949`
Disposition: `RB_R18_CANONICAL_MC_TAIL_ADAPTER_PARITY_PASS_RESEARCH_ONLY`
Science PASS: true

R18 did NOT modify canonical `scripts/simulation_v2.py`. The adapter is a post-simulation shadow transformation.

Architecture:
- generate the R17 tail-mixture target distribution around each canonical RB receiving-yard sample mean;
- mean-preserve it;
- rank-preserving quantile-map it onto the canonical RB receiving-yard draws so the canonical dependence ordering is retained;
- change RB `rush_rec_yards` only by the exact receiving-yard draw delta;
- leave all other outputs untouched.

Canonical fixture result (4 RBs; 10,000 iterations):
- canonical_mean_parity: PASS; max mean delta 0.0 yards
- non_rb_exact: PASS
- rb_component_exact: PASS
- rush_rec_identity: PASS
- allocation_trace_exact: PASS
- nonnegative receiving yards: PASS
- finite draws: PASS
- rank preservation: PASS; minimum Spearman 0.9999999999999999
- deterministic replay: PASS
- adapted RB count: 4
- canonical simulation blob exactly unchanged: `887e9c776ab112276ec8281195b0fed790ea0551`

Historical R17 code-path replay was exact:
- combined CRPS 7.524092768426105
- Brier30 0.05715478937926085
- Brier50 0.018025357463222102
- q90 pinball 3.590688254090627
- q95 pinball 2.4040846023906104

2024 replay Brier50 0.014120358321377332; 2025 0.021933159906676238. The explicit frozen 50+ fragility guard passed.

Every frozen R18 gate passed. Sportsbook inputs added 0. Production parameters changed 0.

R18 proves the R17 distribution shape can be composed mechanically with the canonical MC output without mutating the mean, target allocation, non-RB markets, or other RB component markets. It still does NOT make the mechanism live.

## R19 — deployable 2026 tail scorer refit — PASS, SHADOW ONLY

Frozen plan: `docs/migrations/RB_R19_DEPLOYABLE_TAIL_SCORER_REFIT_V1_PLAN.md`
Plan commit: `30d3a18310c955d960ad393b5e44464b1dc3ba25`
Implementation: `scripts/backtest/build_rb_r19_deployable_tail_scorer_refit_v1.py`
Implementation commit: `d71725597a44d8b3b78c20c5eaa79a9dc51f63b8`
Workflow: `.github/workflows/research-rb-r19-deployable-tail-scorer-refit-v1.yml`
Workflow commit/head SHA: `6ac1342f737f142acac6a3e4b459f442faf1442a`
Actions run: `34288244770`
Job: `102268690877`
Artifact: `10080377483`
Artifact digest: `sha256:11432b9d7b7f2367935a862b63c30df9f40955e479806ab67d90921b63a907c7`
Disposition: `RB_R19_DEPLOYABLE_TAIL_SCORER_REFIT_PASS_SHADOW_ONLY`
PASS: true

R19 answered the deployability question without changing production. It recreated and serialized the exact forward chain needed to score a 2026 slate:
`R8 identity -> R9 reliability/target-delta feature -> R11 high5 state probability -> R16 cat30/cat50 tail probability -> R17 residual pools`.

### Historical parity

R11 2025 parity:
- eligible rows 285
- matched rows 285
- coverage 1.0
- max abs probability delta `2.220446049250313e-16`
- mean abs probability delta `2.858337348486697e-17`

R16 probability parity, all exact-key coverage 1.0:
- train 2023 -> test 2024, cat30: 1,394/1,394; max abs delta `9.71445146547012e-17`
- train 2023 -> test 2024, cat50: 1,394/1,394; max abs delta `9.974659986866641e-17`
- train 2023-2024 -> test 2025, cat30: 1,393/1,393; max abs delta `9.71445146547012e-17`
- train 2023-2024 -> test 2025, cat50: 1,393/1,393; max abs delta `9.974659986866641e-17`

R17 historical pool-count lineage reproduced exactly:
- 2023: non-tail 1,272; tail30-49 56; tail50+ 28
- 2023-2024: non-tail 2,580; tail30-49 122; tail50+ 48

Final completed-history 2023-2025 residual pools serialized:
- non-tail: 3,890; SHA256 `f677e91cd25cdbd6db044e9decccec6312943b99f8ffc25a827527e54d4d7b1d`
- tail30-49: 174; SHA256 `3da7bf656fcab3c111c8d5fb60e38fb7f735d7fcce639dabad899431f613c225`
- tail50+: 79; SHA256 `ee203d3ffa01b8687e7774d817d0dce6cdc8b56ca62321a5583d5287dfd5ee43`

### Final 2026 refits

R8/R9 identity:
- final training season 2025
- training rows 2,134
- OOF reliability rows 1,502
- raw reliability slope 1.2984460655735952
- frozen reliability clip produced final reliability 1.0
- OOF raw/shrunk residual MAE 0.9780920658968174
- OOF raw correlation 0.5053735994294682

R11 high5:
- training rows 1,141
- training seasons 2022-2025
- high5 event rate 0.32602979842243646
- TOP20 only; REST80 state probability remains exactly 0

R16 final tail fits:
- training rows 4,143
- training seasons 2023-2025
- cat30 training rate 0.06106685976345643
- cat50 training rate 0.019068307989379675

Strict-prior audit:
- training rows checked 2,134
- player time violations 0
- same-team time violations 0

Serialization roundtrip:
- R8 Ridge max abs delta 0.0
- R11 high5 max abs delta 0.0
- R16 cat30 max abs delta 0.0
- R16 cat50 max abs delta 0.0
- overall max abs delta 0.0

Every frozen R19 gate passed:
- r11_2025_parity_coverage
- r11_2025_probability_parity
- r16_probability_parity
- r17_pool_count_parity
- final_r9_feature_complete
- final_r9_reliability_range
- final_r11_fit_valid
- final_r16_cat30_fit_valid
- final_r16_cat50_fit_valid
- residual_pools_valid
- serialization_roundtrip
- strict_prior_audit
- future_outcome_zero
- sportsbook_zero
- production_parameters_zero

Sportsbook inputs added: 0.
Production parameters changed: 0.

### R19 artifact contract actually materialized

Artifact includes:
- `rb_r19_tail_scorer_model_v1.json`
- `rb_r19_residual_pools_v1.npz`
- final R8/R9 coefficient audit
- final R9 OOF reliability audit
- R11 parity CSV
- R16 parity CSV
- result JSON and historical input audits

Serialized scorer:
- candidate `RB_R19_DEPLOYABLE_TAIL_SCORER_REFIT_V1`
- version 1
- status `SHADOW_ONLY`
- fit_for_season 2026
- git SHA `6ac1342f737f142acac6a3e4b459f442faf1442a`
- exact R8/R9, R11, and R16 scaler/model parameters
- exact residual-pool hashes and companion NPZ
- explicit fail-closed live-input contract
- `sportsbook_inputs_added = 0`
- `production_parameters_changed = 0`

Required live inputs fail closed on:
- event_id
- team
- player_clean_key
- position/position_family
- certified finite RB target entitlement or target share
- certified team pass-attempt projection OR rules_plays_est + rules_pass_rate
- rules_ypt/frozen_ypt
- strict-prior R8 identity history source

R9 target delta remains a scorer feature only. R19 does not alter live RB receiving mean or target entitlement.

## Production boundary

Do not state that R9/R11/R12/R13/R14/R15/R16/R17/R18/R19 are active production RB receiving code unless a later explicit promotion ledger says so. R19 is a parity-verified **2026 SHADOW scorer artifact**, not a production promotion. Canonical `scripts/simulation_v2.py` remains unchanged by this lane.

## Exact next step

Freeze and execute **R20: real-2026-slate shadow scoring + full-slate fail-closed parity/data-quality validation** before any promotion discussion.

R20 must, without changing production:
1. load the immutable R19 model JSON and residual-pool NPZ and verify their hashes/lineage;
2. build the exact strict-prior R8 identity features for a real 2026 slate using only completed games before the scored slate;
3. derive current baseline RB targets and receiving-yard means only from the certified football projection surfaces;
4. score R11 state probability, R16 p30/p50, and R17/R18 tail distribution in shadow mode;
5. fail closed on missing/duplicate/nonfinite player, team, entitlement, YPT, or history inputs rather than substituting generic PlayerForm proxies;
6. preserve every RB receiving mean exactly after the shadow adapter;
7. preserve canonical target allocation, receptions, rushing outputs, all non-RB outputs, and simulation dependence/rank ordering exactly as required by R18;
8. run the current certified full-slate stack validator and prove the existing production/certified outputs are byte/value-identical when the shadow lane is disabled/unconsumed;
9. emit a player-level 2026 shadow casebook with identity percentile/TOP20, R9 residual/target delta feature, R11 state probability, R16 p30/p50, canonical mean, and adapted distribution diagnostics;
10. use zero sportsbook inputs and change zero production parameters.

R20 is a prospective deployment/parity gate, not permission to tune on 2026 outcomes. A PASS may justify a separately governed promotion discussion or prospective grading plan; it must not silently activate the adapter.