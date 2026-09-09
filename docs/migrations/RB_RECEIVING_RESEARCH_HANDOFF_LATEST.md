# RB Receiving Research — Canonical Handoff

Last updated: 2026-09-08/09 UTC
Active research branch: `research-cross-position-catastrophic-casebook-v1`
Repository: `dkaps6/imtiredofthis`

## Purpose

This is the durable continuity ledger for the RB receiving-yards / receptions research lane. Read this before extending the lane.

Do **not** infer production promotion from a research PASS. Preserve frozen gates, strict-prior feature rules, exact RB-room conservation, and zero sportsbook inputs upstream.

## Source-of-truth rule for the R8-era narrative gap

The project chat **R8 Run Watch** contains decision rationale that was not fully written into GitHub at the time. The underlying implementations, commits, workflows, runs, and artifacts are present in the repository.

For that interval:

- GitHub is authoritative for execution facts, code, frozen gates, runs, artifacts, and metrics.
- The R8 Run Watch project conversation is authoritative for missing decision rationale: why a run was attempted, how a failure was interpreted, and why the next experiment followed.
- Do not reconstruct the research story from run outcomes alone when the chat explains the causal reasoning.

The key recovered narrative is:

`R8 identity hypothesis -> R8 over-adjustment failure -> diagnose amplitude rather than discard identity -> R9 shrinkage PASS -> identity answers who, not when -> R10/R11 weekly-state work -> R12 state-gated identity -> test remaining efficiency explanations -> R13/R14/R15B efficiency avenues unsupported -> deliberately stop forcing mean YPT -> R16 distribution/tail hypothesis`.

## User research principles carried forward

1. Project real football statistics accurately first. Vegas is downstream evaluation/pricing context, not an upstream football-projection feature.
2. Do not discard a football theory merely because a first implementation fails a composite gate; diagnose why it failed before changing the hypothesis.
3. Large/catastrophic misses matter. Do not improve MAE by merely compressing projections if meaningful tails get worse.
4. Distinguish receiving identity (`who`) from weekly receiving state (`when`).
5. Do not manufacture opportunity. RB receiving allocation is a finite room and exact conservation is mandatory.
6. Do not move frozen science gates after results are visible.
7. Distinguish coded/tested research, scientifically supported research, shadow/deployable artifacts, and certified/production-active code.
8. RB receiving-yard tail work and RB receptions/target-entitlement work are related but not interchangeable. R16-R21 primarily improve/evaluate the **receiving-yard distribution**; they do not separately solve or promote receptions.

---

# Research lineage

## R8 — receiving identity — OOS FAIL, signal retained

Preceding diagnostic:

- top 20% by prior RB-room receiving share had roughly 15.21 receiving-yard MAE;
- bias roughly -6.15 yards;
- accounted for about 53% of 30+ misses;
- accounted for about 71% of 50+ misses.

Run: `34263401496`
Artifact: `10071067712`
Disposition: `RB_R8_RECEIVING_IDENTITY_OOS_FAIL`

Combined baseline -> R8:

- target MAE: `1.618081 -> 1.541722`
- receiving-yard MAE: `12.781963 -> 12.474528`
- RMSE: `18.725660 -> 18.016724`
- bias: `-2.958796 -> -1.421041`
- p90 error: `27.858 -> 28.427` (worse)
- 30+ miss rate: `0.088608 -> 0.092586` (worse)
- 50+ miss rate: `0.032911 -> 0.024231` (better)

R8 failed the fresh p90, fresh 30+ guard, and fresh top-20 MAE gates.

### R8 forensic conclusion

The identity hypothesis was **not** discarded.

Q1-Q4 adjustment sizes generally improved. The largest-adjustment quintile Q5 caused the failure:

- Q5 MAE roughly `20.027 -> 20.638`;
- Q5 bias moved from roughly `-9.18` to `+1.65`;
- 50+ misses still improved.

Sparse history was not the main explanation.

Conclusion: the receiving-identity signal was real, but the amplitude was too aggressive. Apply reliability/shrinkage rather than abandon the signal.

## R9 — shrunk receiving identity — scientific PASS

Run: `34268245735`
Artifact: `10072877150`
Status: scientific research PASS, not automatic production promotion.

Fresh 2016 baseline -> R9:

- target MAE: `1.5549 -> 1.4970`
- receiving-yard MAE: `12.1065 -> 11.7930`
- RMSE: `18.4157 -> 17.8832`
- p90: `27.7321 -> 26.4276`
- 30+ miss: `8.77% -> 8.05%`
- 50+ miss: `2.92% -> 2.64%`
- bootstrap probability of improvement: `1.000`
- phases passed: `4/4`

2017 overall replicated, although top-20 MAE slightly worsened in replication.

## R10/R11 — weekly receiving-state theory

R9 established persistent identity, but identity answers **who** can command receiving work, not **when** the high-usage state activates.

R10 performed the state forensic/stability work.

R11 tested a strict-prior high-receiving-state probability and supported that high receiving states are partly predictable pregame.

Frozen R11 state features:

- `baseline_pred_targets`
- `prior_rb_room_share`
- `target_delta` (R9 minus baseline projected targets)
- `r9_raw_r8_residual`

Only TOP20 receiving-identity RBs receive nonzero state probability in the R11/R12 mechanism. REST80 is explicitly zero in R12.

R19 later reproduced the R11 2025 parity exactly:

- eligible rows: 285
- matched rows: 285
- coverage: 1.0
- max absolute probability delta: `2.220446049250313e-16`

## R12 — state-gated identity — research-only/unpromoted

Theory: express the identity signal more strongly when the pregame state model indicates a high-usage receiving game is plausible.

R12 improved multiple research metrics, including modern receiving-yard MAE roughly `10.79 -> 10.54` versus R9 and top receiving-back target MAE, but it did not clear every frozen promotion requirement.

R12 remains research-only/unpromoted.

Important architecture:

- `frozen_ypt` comes from the baseline receiving-yards / baseline-target mapping;
- exact team RB target-pool conservation is preserved.

The corrected R12 lineage used later by R19/R20 points to run `34273055095`, artifact `10074589299`.

## R13-R15B — efficiency avenues unsupported

R13 tested persistent player efficiency / YPT history: unsupported.

R14B tested richer PBP player role, YAC, aDOT, and explosive-efficiency information: unsupported.

R15/R15B moved the efficiency hypothesis to strict-prior team/opponent context.

R15B run: `34275426259`
Artifact: `10075521598`
Disposition: `RB_R15_EFFICIENCY_CONTEXT_SIGNAL_NOT_SUPPORTED_DIAGNOSTIC_ONLY`

Mechanical correction only: `LA -> LAR`; football features changed 0; science thresholds changed 0.

ALL_RB combined:

- context YAC/target: Spearman `0.072405`, high8 AUC `0.524745`
- context explosive-target rate: Spearman `0.065150`, high8 AUC `0.521492`
- state probability: Spearman `0.028826`, high8 AUC `0.500600`
- frozen YPT: Spearman `0.085977`, high8 AUC `0.537756`

Conclusion after R13-R15B: stop blindly tuning mean YPT. Test the receiving-yard **distribution/tail** instead.

---

# Tail/distribution lane

## R16 — upside-tail state diagnostic — SUPPORTED

Script: `scripts/backtest/diagnose_rb_r16_upside_tail_state_v1.py`
Workflow: `.github/workflows/research-rb-r16-upside-tail-state-v1.yml`
Workflow commit: `90ee0dd2448e24401b99a088f8b797646d4ec89a`
Run: `34286363931`
Job: `102262785132`
Artifact: `10079630404`
Digest: `sha256:ca44a174dafadcaf27496b00efe4e933941211037b0eacea0431d72a9b4fa099`
Disposition: `RB_R16_UPSIDE_TAIL_SIGNAL_SUPPORTED_DIAGNOSTIC_ONLY`

R16 froze the receiving mean and tested whether strict-prior opportunity/state/identity information predicts large positive residuals.

Features:

- `baseline_pred_targets`
- `baseline_pred_rec_yards`
- `state_probability`
- `prior_rb_room_share`
- `r9_raw_r8_residual`
- `frozen_ypt`
- `identity_top20`

Primary label: `actual_rec_yards - baseline_pred_rec_yards >= 30`.

Primary result:

- n: 2,787
- events: 169 (`6.064%`)
- combined AUC: `0.673042` vs frozen `0.65` gate
- mean-only AUC: `0.639598`
- gain vs mean-only: `+0.033444`
- Brier: `0.055708` vs pooled `0.056963`
- top-quintile lift: `1.625469x`
- top-quintile capture: `32.544%`
- OOS folds >=0.58: `2/2`
- 2024 AUC: `0.657964`
- 2025 AUC: `0.689046`

Secondary:

- 50+ underprojection AUC: `0.766455`; top-quintile lift `2.742x`; capture `54.902%`
- actual 40+ receiving yards AUC: `0.764674`
- actual 60+ receiving yards AUC: `0.802164`; top-quintile lift `2.933x`; capture `58.730%`
- >=20-yard reception label AUC: `0.674779`

PBP exact-target integrity on positive-target rows: `98.867%`.
Sportsbook inputs added: 0.
Production parameters changed: 0.

Every frozen R16 support gate passed.

Nuance: target-only was already informative (primary AUC `0.655782`). The full state/identity model was most compelling in the more extreme tail. Treat R16 as **incremental extreme-tail information layered on opportunity**, not a replacement for opportunity.

## R17 — mean-preserving tail mixture — SUPPORTED, research-only

Frozen plan: `docs/migrations/RB_R17_MEAN_PRESERVING_TAIL_MIXTURE_V1_PLAN.md`
Plan commit: `f803fd44b2bb8fe42a63ed211f44f373cc140d2f`
Implementation: `scripts/backtest/evaluate_rb_r17_mean_preserving_tail_mixture_v1.py`
Implementation commit: `9e58b6501bb2a6c1c769c2379d2ee581399f1835`
Workflow: `.github/workflows/research-rb-r17-mean-preserving-tail-mixture-v1.yml`
Workflow commit: `310288320feece05367dbefdd76f52f952d20a3c`
Run: `34286785433`
Job: `102264103483`
Artifact: `10079782239`
Digest: `sha256:b3feaf53d882238d1cd3446e646742d0fffa6bc30c589db6aa045d69e5c0d878`
Disposition: `RB_R17_DISTRIBUTION_SIGNAL_SUPPORTED_RESEARCH_ONLY`
PASS: true

Comparator -> candidate:

- CRPS: `7.525223 -> 7.524093`
- Brier30: `0.0574860 -> 0.0571548`
- Brier50: `0.01802747 -> 0.01802536`
- q90 pinball: `3.665299 -> 3.590688`
- q95 pinball: `2.484965 -> 2.404085`
- 80% coverage: `75.924% -> 77.144%`
- 90% coverage: `86.473% -> 87.585%`
- max mean delta: about `1.07e-14` yards

Every frozen R17 support gate passed.

Caution: global CRPS gain was tiny and combined Brier50 gain nearly zero; 2025 Brier50 slightly worsened. The clearest R17 value is upper-quantile loss and interval calibration.

## R18 — canonical Monte Carlo tail-adapter parity — PASS, research-only

Frozen plan: `docs/migrations/RB_R18_CANONICAL_MC_TAIL_ADAPTER_PARITY_V1_PLAN.md`
Plan commit: `1398eed191994079f9e5f1050adccd097cce633e`
Adapter: `scripts/modeling/rb_r17_tail_distribution_adapter_v1.py`
Adapter commit: `551ea7772310dcbf67d0671e74e5f069f71ad37c`
Evaluator: `scripts/backtest/evaluate_rb_r18_canonical_mc_tail_adapter_parity_v1.py`
Evaluator commit: `9e5f3de0eecca7de9adef2de0f60862d148e45ed`
Workflow: `.github/workflows/research-rb-r18-canonical-mc-tail-adapter-parity-v1.yml`
Workflow commit: `1fc889728ce3afdef22f6638203b0b9eda76099d`
Run: `34287356990`
Job: `102265907467`
Artifact: `10079996458`
Digest: `sha256:3d2ded174147b5b2e5b8ecc7550b8600a1d3e930fce4eb0220ef9fe653e35949`
Disposition: `RB_R18_CANONICAL_MC_TAIL_ADAPTER_PARITY_PASS_RESEARCH_ONLY`
PASS: true

R18 does **not** modify canonical `scripts/simulation_v2.py`.

Architecture:

1. generate the R17 target distribution around each canonical RB receiving-yard sample mean;
2. mean-preserve it;
3. rank-preserving quantile-map onto canonical draws to retain canonical dependence ordering;
4. change RB `rush_rec_yards` only by the exact receiving-yard draw delta;
5. leave every other output untouched.

Canonical fixture (4 RB, 10,000 iterations):

- mean parity: PASS, max delta 0.0
- non-RB exact: PASS
- other RB components exact: PASS
- rush+rec identity: PASS
- allocation trace exact: PASS
- nonnegative/finite: PASS
- rank preservation: minimum Spearman `0.9999999999999999`
- deterministic replay: PASS
- canonical simulation blob unchanged: `887e9c776ab112276ec8281195b0fed790ea0551`

Historical R17 replay exact:

- CRPS `7.524092768426105`
- Brier30 `0.05715478937926085`
- Brier50 `0.018025357463222102`
- q90 `3.590688254090627`
- q95 `2.4040846023906104`
- 2024 Brier50 `0.014120358321377332`
- 2025 Brier50 `0.021933159906676238`

Frozen 50+ fragility guard passed.
Sportsbook inputs 0.
Production parameters 0.

## R19 — deployable strict-prior 2026 tail scorer — PASS, SHADOW ONLY

Frozen plan: `docs/migrations/RB_R19_DEPLOYABLE_TAIL_SCORER_REFIT_V1_PLAN.md`
Plan commit: `30d3a18310c955d960ad393b5e44464b1dc3ba25`
Implementation: `scripts/backtest/build_rb_r19_deployable_tail_scorer_refit_v1.py`
Implementation commit: `d71725597a44d8b3b78c20c5eaa79a9dc51f63b8`
Workflow: `.github/workflows/research-rb-r19-deployable-tail-scorer-refit-v1.yml`
Workflow/head commit: `6ac1342f737f142acac6a3e4b459f442faf1442a`
Run: `34288244770`
Job: `102268690877`
Artifact: `10080377483`
Digest: `sha256:11432b9d7b7f2367935a862b63c30df9f40955e479806ab67d90921b63a907c7`
Disposition: `RB_R19_DEPLOYABLE_TAIL_SCORER_REFIT_PASS_SHADOW_ONLY`
PASS: true

R19 serialized the forward chain:

`R8 identity -> R9 reliability/target-delta feature -> R11 high-state probability -> R16 cat30/cat50 probability -> R17 residual pools`.

Historical parity:

- R11 2025: 285/285 matched, coverage 1.0, max probability delta `2.220446049250313e-16`.
- R16 2024 and 2025 exact-key coverage 1.0; max probability deltas on the order of `1e-16`.
- R17 pool-count lineage reproduced exactly.

2026 refit summary:

- R8/R9 train-through-2025 rows: 2,134
- OOF reliability rows: 1,502
- raw reliability slope: about `1.298446`; clipped final reliability `1.0`
- OOF residual MAE: `0.978092`
- OOF residual correlation: `0.505374`
- R11 rows: 1,141 across 2022-2025
- high-state event rate: about `0.32603`
- R16 rows: 4,143 across 2023-2025
- cat30 rate: about `0.0610669`
- cat50 rate: about `0.0190683`

Strict-prior audit:

- rows: 2,134
- player violations: 0
- same-team violations: 0

Serialization roundtrip: max delta 0 for all models.

R19 file hashes used by R20:

- model SHA256: `9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba`
- residual-pools file SHA256: `c69a268a5a1683e846bcb5f59fe55bcae20d70c679792449ea77e6548b37a362`

All frozen gates passed: parity, feature completeness, reliability range, valid fits/pools, serialization, strict-prior, zero future outcomes, zero sportsbook inputs, zero production parameters.

Status remains SHADOW_ONLY.

Required live inputs include event/team/player identity, certified RB target entitlement/share, pass-volume projection, rules/frozen YPT, and strict-prior R8 identity history.

Important: R9 target delta is a **feature only** in this shadow tail chain. It does not change the production target entitlement or receiving-yard mean.

## R20 — real 2026 full-slate shadow integration — PASS, SHADOW ONLY

Implementation commit: `587bf2a89ca16f11361016df3915361390289a7e`
Workflow: `.github/workflows/research-rb-r20-real-slate-shadow-integration-v1.yml`
Run: `34291433027`
Artifact: `10081502774`
Artifact digest: `sha256:853c0d3aea971c058ae6cc3b80c99ae2a5a0f681fae642835bbfa544da8283ca`
Disposition: `RB_R20_REAL_2026_SLATE_SHADOW_INTEGRATION_PASS_SHADOW_ONLY`
PASS: true

R20 used the immutable governed 2026 Week-1 no-credit Full Slate replay plus immutable R19 scorer. It did not refetch OddsAPI and did not read 2026 outcomes.

Real-slate shape:

- 16 games
- 32 teams
- 469 players
- 94 RB
- 13 FB
- 10,000 simulations

All frozen gates passed, including:

- immutable artifact/source hashes
- strict-prior history
- exact R19 model/pool hashes
- deterministic replay
- exact RB-room conservation
- non-RB exactness
- certified-stack validation before/after
- no future/current outcomes
- no sportsbook input to football distributions
- zero production parameter changes

Parity:

- adapted RBs: 94
- max receiving-mean change: `3.552713678800501e-15` yards
- minimum Spearman: `0.9999999999999999`
- R9 shadow RB-pool conservation gap: `2.78e-17`
- production parameters changed: 0
- sportsbook inputs added: 0
- current/future outcomes used: 0

Week-1 R16 probabilities in the real slate:

- +30 yard upside range: about `1.69%` to `26.85%`
- +50 yard upside range: about `0.27%` to `12.53%`

Example: Bijan Robinson had roughly 60.4% high-receiving-state probability, 17.2% +30 probability, 7.68% +50 probability, while the canonical receiving-yard mean stayed unchanged.

R20 governance note: PASS certifies real-slate SHADOW deployability/parity only. It does not activate the mechanism in production.

### R20 handoff housekeeping note

A one-time canonical-handoff helper attempt around run `34291691149` failed before a job was created. That did not invalidate R20 science. The helper was later removed (`c6094bcbd298875cb431efe382017afd77fb721b`). This canonical handoff now supersedes that stale-paper-trail state.

---

# R21 — prospective evidence lane

## Frozen R21 plan

Plan: `docs/migrations/RB_R21_2026_PROSPECTIVE_TAIL_FORECAST_LOCK_AND_GRADE_V1_PLAN.md`
Plan commit: `cdd503919e56860c6f17df12e01f4f34072d888c`
Status at freeze: PLAN FROZEN BEFORE 2026 REGULAR-SEASON OUTCOMES.

Research question:

> Does the already-frozen R19/R17/R18/R20 SHADOW receiving-yard distribution outperform the canonical CONTROL distribution on genuinely unseen 2026 regular-season outcomes when the forecasts themselves are immutably sealed before kickoff?

Frozen first-kickoff boundary for Week 1: `2026-09-10T00:20:00Z`.

R21 is not a feature-search migration and not a retrospective tuning pass.

### Frozen Week-1 grading metrics

- CRPS
- Brier for actual >= frozen mean + 30
- Brier for actual >= frozen mean + 50
- q90 pinball
- q95 pinball
- central 80% interval coverage
- central 90% interval coverage
- upper-tail casebook diagnostics
- R16 p30/p50 risk-stratified event rates as diagnostics

CONTROL and SHADOW have the same mean by construction; mean MAE/RMSE/bias are football-model monitoring metrics but cannot demonstrate tail-layer improvement.

### Frozen Week-1 support gates

All must pass:

1. exact pre-kickoff lock and draw hashes;
2. valid fail-closed outcome/participation audit;
3. SHADOW CRPS <= CONTROL CRPS;
4. SHADOW Brier30 <= CONTROL Brier30;
5. SHADOW Brier50 <= CONTROL Brier50;
6. SHADOW q90 pinball < CONTROL q90;
7. SHADOW q95 pinball < CONTROL q95;
8. SHADOW absolute 80% coverage error <= CONTROL error + 0.03;
9. SHADOW absolute 90% coverage error <= CONTROL error + 0.03;
10. sportsbook inputs remain zero upstream;
11. production parameter changes remain zero.

Week 1 is an early prospective checkpoint only.

### Frozen cumulative production-evidence floor

A production-promotion decision is not eligible until the prospective ledger contains at least:

- 4 distinct completed 2026 regular-season weeks;
- 250 eligible locked RB player-games;
- 15 observed 30+ underprojection events;
- 5 observed 50+ underprojection events.

Every included weekly forecast must be sealed before outcomes using the same frozen scorer/adapter version, unless a formally versioned reset starts a new evidence ledger.

Even after the evidence floor is met, production requires a separate explicit governed promotion ledger/commit. Nothing auto-promotes.

## R21 Phase A — Week-1 pre-outcome forecast lock — PASS

Implementation: `scripts/backtest/lock_rb_r21_2026_prospective_tail_forecasts_v1.py`
Workflow: `.github/workflows/research-rb-r21-2026-prospective-tail-forecast-lock-v1.yml`
Implementation/workflow head: `eefa0a4fc0b272de03db804acc4e77d7cbe39b92`
Run: `34294227588`
Job: `102287208335`
Run started: `2026-09-09T00:15:29Z`
Artifact: `10082525892`
Artifact name: `rb-r21-week1-prospective-forecast-lock-v1`
Artifact digest: `sha256:302df98f83c461d8abd74da9985dacc80c9477596b8bba85cc4d4a27c4fbc6f3`
Result record: `docs/migrations/RB_R21_WEEK1_PROSPECTIVE_FORECAST_LOCK_V1_RESULT.md`
Disposition: `RB_R21_WEEK1_PROSPECTIVE_FORECAST_LOCK_PASS_SHADOW_ONLY`
PASS: true

Timing proof:

- run start: `2026-09-09T00:15:29Z`
- frozen first kickoff: `2026-09-10T00:20:00Z`
- therefore the lock was created before any 2026 regular-season outcome.

Locked forecast shape:

- 94 RB rows
- 10,000 CONTROL receiving-yard draws per RB
- 10,000 SHADOW receiving-yard draws per RB

Exact raw float64 matrix hashes:

- CONTROL: `0149d2bc2fcd4cd9f401b1a46c1c27ecc9489c3ce00c40ec8f33880bb8b5181e`
- SHADOW: `6be1c67952f465ffb6d01bf3bbdbbdb0ea3e5758684e10c2f075715f58b582d9`

Additional hashes:

- forecast ledger: `b6f528217f1f14057730f52a64174f59497354256a5037ccbf4a43c8bd4a980a`
- compressed draw archive: `8a255f76ac5a327833c220bd134d31d73ad06b566672dd38584fb0c7220918a6`

R20 parent/reexecution integrity:

- immutable parent result SHA256: `955a647375f5c27920e24790c65b5605a76ecf99da8b5ebe1cc756eccc2c11d5`
- R21 unchanged-R20 reexecution result SHA256: exact same hash
- parent player-level summary parity max delta: `7.105427357601002e-15`
- CONTROL-vs-SHADOW max mean delta: `3.552713678800501e-15` yards

Every frozen Phase-A gate passed:

- pre-kickoff lock
- exact R20 parent
- unchanged R20 reexecution
- parent draw-summary parity
- exact 94 x 10,000 shape
- unique player keys
- finite/nonnegative draws
- mean parity
- draw hashes present
- sportsbook zero upstream
- 2026 outcome reads zero
- production parameters zero

R21 Phase A does **not** activate production. It creates immutable prospective evidence that later grading must consume. Future Week-1 grading must use these sealed arrays; it may not regenerate a post-outcome forecast.

---

# Current status / production boundary

Scientific/deployability state:

- R16 tail predictability: SUPPORTED
- R17 mean-preserving distribution mechanism: SUPPORTED
- R18 canonical MC adapter: PASS
- R19 strict-prior deployable scorer: PASS, SHADOW_ONLY
- R20 real 2026 full-slate integration: PASS, SHADOW_ONLY
- R21 Week-1 pre-outcome forecast lock: PASS, SHADOW_ONLY

Production state:

**The RB receiving-yard tail layer is NOT production-active.**

Do not confuse this lane with the separate RB rushing production lane (RB P3 Week-1 rushing synthesis and canonical pricing wiring).

Do not claim R16-R21 separately solved or promoted RB receptions. The tail adapter preserves the existing target/reception architecture and receiving-yard mean.

# Next controlled step

Do **not** tune the R19/R17/R18/R20 model based on Week-1 outcomes.

Once Week-1 games are final and governed weekly player outcomes are available, execute the frozen R21 Phase-B grader against the exact sealed R21 draw matrices.

The first Week-1 outcome grade is evidence only, not production authorization. Continue accumulating pre-outcome weekly locks under the frozen R21 evidence rules until the cumulative evidence floor is reached or a failure forces a formally versioned diagnostic/reset.
