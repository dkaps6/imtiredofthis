# NFL HANDOFF — 2026-09-20 — FOOTBALL CONTEXT / EXACT ASSIGNMENT CLOSED, RB FORWARD CONFIRMATION NEXT

**Repository:** `dkaps6/imtiredofthis`  
**GitHub is canonical; chat memory is secondary.**  
**Football-context branch:** `research-football-context-event-redundancy-v1`  
**Production main:** `f0dad2c6711e85104eeffedfa5f5112fd172cbf5`

## Newly closed exact-assignment frontier

### BDB2026 Prediction source expansion audit

Run `35518578013` established:
- existing Analytics archive remains accessible/certified;
- Prediction archive returned HTTP 403 under the current Kaggle credential;
- no scientific "no data" claim was made from the access failure;
- public competition structure did not justify assuming a new 2024 downloadable training corpus.

### BDB2025 exact-assignment acquisition attempt

Runs `35518769786` and `35518845392` preserved the failed acquisition attempt.
Individual files returned 404 and the full-archive fallback did not materialize the expected
files under the existing workflow. This is an acquisition result only, not a scientific
assignment-signal result.

### BDB2023 exact blocker-rusher source qualification — PASS

Canonical:
- implementation/workflow SHA: `5b6a2a820abd46cbe642e80b2fcac124921c60b9`
- run: `35519025957`
- job: `106099764039`
- artifact: `10607537977`
- digest: `sha256:852c9c8b6dec11817d6d72ebf1222fb0e666641a3c78a4ca59b9520daa8db974`

Evidence:
- exact PFF blocker->blocked-defender assignment edges: **46,396**
- unique blockers: **590**
- unique blocked defenders: **665**
- unique pairs: **12,536**
- stable-ID coverage: **100%**
- Week 5+ edges: **22,111**
- Week 5+ edges with >=10 strict-prior edges for both players: **19,436 (87.9019%)**
- prior same-pair coverage: **3.3784%**
- final: `VALUE_OF_INFORMATION_LAB_READY_SOURCE_SLICE`

### BDB2023 exact-assignment VOI V1 — FAILED CLOSED

Frozen plan:
`docs/research/BDB2023_EXACT_BLOCKER_RUSHER_ASSIGNMENT_VOI_EXPERIMENT_V1.md`

Canonical:
- frozen plan SHA: `7590d5d867a4a368624ef923b72da8146dd7a7ee`
- run: `35519336205`
- job: `106100587284`
- artifact: `10607218889`
- digest: `sha256:18cd686c11573bb483400e016747c45d91bde2ddade26468f2b6bb935f2e30db`
- execution SHA: `7a86516f70a85608528105dc8db9a2ff7c22455d`

Holdout Weeks 7-8:
- denominator offense-team rows: **56**
- scored rows: **46**
- coverage: **82.1429%**
- baseline MAE: **0.02064617**
- candidate MAE: **0.02065083**
- baseline RMSE: **0.02440605**
- candidate RMSE: **0.02443043**
- baseline p90 AE: **0.03738783**
- candidate p90 AE: **0.03767568**
- pairing-covariance coefficient: **+0.758973** (expected sign)
- 5,000-rep game-cluster AE-gain CI:
  **[-0.00005814, +0.00005209]**

Only integrity, coverage and coefficient direction passed. Support (<50), MAE, bootstrap,
RMSE and p90 gates failed.

Final:
`BDB2023_EXACT_ASSIGNMENT_VOI_NO_ACTIONABLE_SIGNAL_V1`

Do not rescue with another pairing transform, lower support, favorable subsets, alternate
models, block type or geometry appended to this V1.

Result:
`docs/research/BDB2023_EXACT_BLOCKER_RUSHER_ASSIGNMENT_VOI_EXPERIMENT_V1_RESULT_2026-09-20.md`

## Football-context interpretation

Recent OL cohesion, defensive-front cohesion and exact blocker-rusher pairing mechanisms
have now all failed their frozen predictive/VOI tests despite valid source qualification.

Do not continue manufacturing variants of the same protection/pressure information.
The non-WR source-ready football-context frontier is materially narrowed.

## RB is explicitly unresolved and back in scope

The user explicitly re-opened RB priority on 2026-09-20.

Do not describe RB as scientifically empty. The strongest positive unresolved result is:

`RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED`

Authority:
- merged PR #562
- qualification run `35039152022`
- all **28** gates passed
- pooled CRPS improvement: **1.218%**
- high-difficulty-quartile CRPS improvement: **2.624%**
- player-clustered and crossed player x game bootstraps: **p=1.0**
- point MAE: unchanged by mean-neutral design
- 4/4-season robustness
- production change: none

The frozen plan requires a **separate forward/shadow confirmation** before production.
None has been started.

RB production gaps remain:
- P3 rushing specialist authority is Week-1-only;
- R22/R26 promoted specialist scopes are Week-1-only;
- Weeks 2-18 RB rushing authority remains unresolved;
- RB receiving-yard mean remains unresolved, with conventional YPR/YPT/YAC/xYAC
  transformation family closed by R23-R27D.

## Exact next task

Leave this football-context branch scientifically closed at the exact-assignment result.

Move to current-main-derived RB work and design the required forward/shadow confirmation
for the already-qualified PD2 yard-difficulty MC-width signal. Do not restart RB feature
search and do not retune the qualified width mapping.

Do not touch Issue #535 / the separate WR lane.
