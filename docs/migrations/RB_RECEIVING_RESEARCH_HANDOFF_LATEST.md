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
7. Distinguish coded/tested research, scientifically supported research, and certified/production-active code.

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

## Deployability audit after R18

The next blocking problem is scoring R16 tail probabilities on a real 2026 slate.

Directly available/derivable from current production-football surfaces:
- `baseline_pred_targets`: current finite M38 target entitlement multiplied by projected team pass attempts (`rules_plays_est * rules_pass_rate`)
- `baseline_pred_rec_yards`: baseline predicted targets multiplied by current `rules_ypt`
- `frozen_ypt`: current/baseline YPT mapping (`rules_ypt`), matching the R12 frozen-efficiency concept

Deployable but requires carrying/refitting frozen research mechanisms into a scored artifact:
- `prior_rb_room_share`: strict-prior RB receiving-room history from the R8 identity snapshot machinery
- `r9_raw_r8_residual`: exact R8 Ridge identity residual prediction using the frozen 19-feature set
- `state_probability`: exact R11/R12 TOP20 high5 logistic probability using baseline targets, prior RB-room share, R9 target delta, and raw R8 residual
- `identity_top20`: current-slate percentile classification from strict-prior RB-room share

Important: these are reconstructable using pregame football information, but they are NOT yet first-class certified full-slate fields. Do not approximate them with ad hoc current PlayerForm columns.

## Production boundary

Do not state that R9/R11/R12/R13/R14/R15/R16/R17/R18 are active production RB receiving code unless a later explicit promotion ledger says so. RB receiving research has not been silently integrated into the certified full-slate production stack.

## Exact next step

Freeze an R19 deployable-feature/refit contract before code execution. R19 should:
1. reproduce the exact R8 identity snapshot feature definitions using strict-prior 2023-2025 completed-game history;
2. refit the R8 Ridge identity residual on the final allowed training window, derive R9 reliability/target delta without changing the live mean;
3. refit the R11 high5 state model for TOP20 backs;
4. refit R16 cat30/cat50 tail classifiers on the final allowed 2023-2025 training window;
5. freeze the 2023-2025 residual pools required by R17;
6. materialize a versioned, football-only 2026 scorer artifact with coefficients/scalers/pool hashes and fail-closed feature requirements;
7. prove historical feature parity against the existing R16/R17 lineage before scoring any 2026 slate;
8. keep the scorer shadow-only until a separately frozen full-slate prospective/parity gate passes.
