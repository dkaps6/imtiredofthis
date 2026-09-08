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

### R12 -> state-gated identity

Theory: identity should express more strongly when the pregame state model says a high-usage receiving game is plausible. R12 improved multiple research metrics, including modern receiving-yard MAE roughly 10.79 -> 10.54 versus R9 and top receiving-back target MAE, but did not clear every frozen promotion requirement. R12 remains research-only/unpromoted.

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

R16 support authorizes only a separately frozen distribution/MC candidate. It does not promote an RB receiving mean or R12.

## R17 — mean-preserving tail mixture — CURRENT

Frozen plan: `docs/migrations/RB_R17_MEAN_PRESERVING_TAIL_MIXTURE_V1_PLAN.md`
Plan commit: `f803fd44b2bb8fe42a63ed211f44f373cc140d2f`
Implementation: `scripts/backtest/evaluate_rb_r17_mean_preserving_tail_mixture_v1.py`
Implementation commit: `9e58b6501bb2a6c1c769c2379d2ee581399f1835`
Workflow: `.github/workflows/research-rb-r17-mean-preserving-tail-mixture-v1.yml`
Workflow commit: `310288320feece05367dbefdd76f52f952d20a3c`
Actions run: `34286785433`
Job: `102264103483`
Status at this update: RUNNING/QUEUED

### R17 hypothesis

Can OOS R16 probabilities for >=30-yard and >=50-yard baseline underprojection improve the RB receiving-yard predictive distribution while preserving the frozen baseline receiving mean exactly?

Comparator: unconditional historical empirical residual bootstrap, nonnegative and rescaled to the frozen player-game mean.

Candidate: nested R16-conditioned residual mixture using training pools:
- residual <30
- residual 30-49
- residual >=50

For each OOS test player-game:
- w50 = min(R16 p50, R16 p30)
- w30 = max(R16 p30 - w50, 0)
- wnon = 1 - R16 p30

Both comparator and candidate use 2,000 deterministic draws per player-game, seed 917, floor at zero, and are rescaled so the simulated player-game mean equals the frozen baseline receiving mean.

Frozen R17 support gates:
1. candidate max mean delta <=1e-6 yards
2. combined CRPS <= comparator
3. each fold CRPS no worse by >1%
4. combined 30+ Brier strictly better
5. combined 50+ Brier strictly better
6. q90 pinball strictly better
7. q95 pinball strictly better
8. 80% coverage absolute error no worse than comparator +2pp
9. 90% coverage absolute error no worse than comparator +2pp
10. sportsbook inputs 0

Do not modify these gates after seeing the result.

## Production boundary

Do not state that R9/R11/R12/R13/R14/R15/R16/R17 are active production RB receiving code unless a later explicit promotion ledger says so. RB receiving research has not been silently integrated into the certified full-slate production stack.

## Exact next step

Let R17 run `34286785433` complete. Retrieve the immutable `rb-r17-mean-preserving-tail-mixture-v1` artifact, record every metric/gate here, then decide whether R17 earns a separately frozen production-parity/distribution integration test. Do not modify R17 based on partial output.
