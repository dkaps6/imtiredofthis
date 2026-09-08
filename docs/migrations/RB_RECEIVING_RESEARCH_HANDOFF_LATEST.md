# RB Receiving Research — Canonical Handoff

Last updated: 2026-09-08
Active research branch: `research-cross-position-catastrophic-casebook-v1`
Repository: `dkaps6/imtiredofthis`

## Purpose

This file is the durable continuity ledger for the RB receiving-yards / receptions research lane. Future chats/agents should read this file before extending the lane. Do not infer production promotion from a research PASS. Preserve frozen gates, strict-prior feature rules, exact RB-room conservation, and zero sportsbook inputs upstream.

## User research principles carried forward

1. Project real football statistics accurately first; Vegas is downstream evaluation/pricing context, not an upstream football-projection feature.
2. Do not discard a football theory merely because a first implementation fails a composite gate; diagnose *why* it failed before changing the hypothesis.
3. Large/catastrophic misses matter. Do not win MAE by merely compressing projections if the model becomes worse on meaningful tails.
4. Distinguish player receiving identity (who can command RB targets) from weekly receiving state (when that role activates).
5. Do not manufacture opportunity. RB receiving allocation is a finite room and conservation is mandatory.
6. Do not change frozen science gates after seeing results.
7. Distinguish clearly between: coded/tested research, scientifically supported research, and certified/production-active code.

## Intellectual lineage

### Identity hypothesis -> RB R8

Observation/theory: a small set of RBs are persistent passing-game weapons and the baseline was systematically under-projecting that group. Historical targets, receptions, target share, RB-room share, same-team history and 5+/7+ target-game frequency should contain strict-prior information about receiving identity.

The preceding identity diagnostic showed the top 20% by prior RB-room receiving share had roughly 15.21 receiving-yard MAE, -6.15 yard bias, ~53% of 30+ misses and ~71% of 50+ misses. This motivated a deployable identity correction rather than treating all RB target shares as interchangeable.

R8 immutable run: `34263401496`
Artifact: `10071067712`
Candidate: `RB_R8_RECEIVING_IDENTITY_V1`
Disposition: `RB_R8_RECEIVING_IDENTITY_OOS_FAIL`

Combined metrics:
- targets MAE 1.618081 -> 1.541722
- receiving-yards MAE 12.781963 -> 12.474528
- receiving-yards RMSE 18.725660 -> 18.016724
- receiving-yards bias -2.958796 -> -1.421041
- p90 27.858 -> 28.427 (worse)
- 30+ miss rate 0.088608 -> 0.092586 (worse)
- 50+ miss rate 0.032911 -> 0.024231 (better)

Frozen gates failed on fresh p90, fresh 30+ guard, and fresh top-20 MAE.

### R8 forensic -> shrinkage theory

The response to the R8 FAIL was diagnostic, not abandonment. The forensic showed the largest adjustment quintile Q5 was the failure source while Q1-Q4 improved.

Q5:
- MAE ~20.027 -> ~20.638
- bias ~-9.18 -> +1.65
- 30+ miss rate worsened
- 50+ miss rate improved materially

Sparse history was not the primary explanation. Conclusion: receiving identity signal was real but the adjustment amplitude was too aggressive. This motivated reliability/shrinkage rather than removing the signal.

### R9 receiving-identity shrinkage

Run: `34268245735`
Artifact: `10072877150`
Scientific result: PASS

Fresh 2016:
- target MAE 1.5549 -> 1.4970
- receiving-yard MAE 12.1065 -> 11.7930
- RMSE 18.4157 -> 17.8832
- p90 27.7321 -> 26.4276
- 30+ 8.77% -> 8.05%
- 50+ 2.92% -> 2.64%
- bootstrap improvement probability 1.000
- 4/4 phases

2017 overall replicated; top-20 MAE slightly worsened in replication. R9 is a scientific research PASS, not automatic production promotion.

Conclusion: receiving identity is useful when reliability-weighted/shrunk.

### R10/R11 -> weekly receiving-state theory

Identity answers *who* is a receiving weapon, not necessarily *when* a high-usage receiving game will occur. Research moved to whether strict-prior information can predict target-state occupancy (0-2, 3-4, 5-6, 7+ targets / high-state probability).

R10 performed receiving-state forensic/stability work.
R11 tested high-state probability and supported that high receiving states are partially predictable pregame.

Conclusion: carry a receiving-state probability as research information rather than assuming a stationary weekly role.

### R12 -> state-gated identity

Theory: receiving identity should express more strongly when the pregame state model says a high-usage receiving game is plausible, and be muted when it does not.

R12 state-gated entitlement improved multiple research metrics, including modern receiving-yard MAE roughly 10.79 -> 10.54 versus R9 and improved top receiving-back target MAE, but did not clear every frozen promotion requirement. Therefore it remains research-only/unpromoted.

Exact RB-room conservation remains mandatory.

### R13 -> player historical efficiency identity

Theory: after improving opportunity, remaining receiving-yard error might be recoverable through persistent player-specific efficiency (YPT / receiving efficiency history).

Result: signal not supported strongly enough. Do not apply static player YPT boosts as a production mean solution.

### R14B -> PBP player role / YAC / aDOT / explosive efficiency

Theory: richer play-by-play role and efficiency information (YAC, target depth/type, explosive receiving tendency) might separate equal-target RBs.

Result: signal not supported strongly enough for mean-model use. Player historical YAC/aDOT/explosive tendencies were not sufficiently stable/predictive.

### R15 / R15B -> team/opponent efficiency context

Theory: if efficiency is not a stable player trait, perhaps offensive scheme / opponent context creates predictable RB target value or YAC/explosive environments.

R15B immutable executed run: `34275426259`
Head SHA: `5f96dc45b7736d4ee5f0971a3f6a851c83f4f68b`
Artifact: `10075521598`
Disposition: `RB_R15_EFFICIENCY_CONTEXT_SIGNAL_NOT_SUPPORTED_DIAGNOSTIC_ONLY`

Mechanical correction only: canonical team code `LA -> LAR`. Football features changed: 0. Science thresholds changed: 0.

ALL_RB combined signals:
- context_yac_per_target: Spearman 0.072405; high8 AUC 0.524745; high10 AUC 0.505526
- context_explosive_target_rate: Spearman 0.065150; high8 AUC 0.521492; high10 AUC 0.525563
- state_probability: Spearman 0.028826; high8 AUC 0.500600; high10 AUC 0.498308
- frozen_ypt: Spearman 0.085977; high8 AUC 0.537756; high10 AUC 0.523583

Conclusion after R13-R15B: multiple increasingly rich mean-efficiency hypotheses failed. Stop blindly tuning YPT. The next question should target distribution/tail behavior rather than forcing weak efficiency information into the mean.

## R16 — upside-tail state diagnostic

Script committed before execution: `scripts/backtest/diagnose_rb_r16_upside_tail_state_v1.py`
Original script commit/head before workflow: `f2b0f4f8e8ab9bec20aeba85d05ebb5506b9bb85`
Workflow commit: `90ee0dd2448e24401b99a088f8b797646d4ec89a`
Workflow: `.github/workflows/research-rb-r16-upside-tail-state-v1.yml`
Actions run: `34286363931`
Job: `102262785132`
Status at this handoff update: IN PROGRESS

### R16 hypothesis

R13-R15 did not support forcing a new mean YPT mechanism. Keep the frozen baseline receiving mean unchanged and test whether pregame opportunity/state/identity information predicts the probability of a large upside receiving-yard tail.

Primary label:
- baseline underprojection by >=30 receiving yards

Secondary labels:
- >=50-yard underprojection
- actual >=40 receiving yards
- actual >=60 receiving yards
- actual >=20-yard reception from PBP (label only)

Current-game outcomes/PBP are labels only. No current-game outcome enters predictors.

R16 features:
- baseline_pred_targets
- baseline_pred_rec_yards
- state_probability
- prior_rb_room_share
- r9_raw_r8_residual
- frozen_ypt
- identity_top20

Folds are frozen:
- train 2023 -> test 2024
- train 2023-2024 -> test 2025

Frozen primary support gates (set before run):
- combined primary AUC >= 0.65
- each of 2 OOS folds contributes to at least 2 folds with AUC >= 0.58 (effectively both)
- top-quintile lift >= 1.50
- top-quintile capture >= 0.30
- Brier score strictly better than pooled-base probability
- PBP exact-target integrity >= 0.90
- sportsbook inputs = 0

Support, if achieved, authorizes only a separately frozen distribution/Monte Carlo candidate. It does NOT promote an RB receiving mean or R12.

## Production boundary

Do not state that R9/R11/R12/R13/R14/R15/R16 are active production RB receiving code unless a later explicit promotion ledger says so. The active certified full-slate branch work is a separate governed stack. RB receiving research has not been silently integrated.

## Exact next step

Wait for R16 run `34286363931` to complete. Retrieve and inspect the immutable artifact `rb-r16-upside-tail-state-v1`. Record every gate and the primary/secondary/fold metrics here before deciding any R17 hypothesis. Do not design or tune R17 based on partial R16 output.
