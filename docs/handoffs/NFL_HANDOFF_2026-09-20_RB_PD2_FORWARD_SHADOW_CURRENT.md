# NFL HANDOFF — 2026-09-20 — RB PD2 FORWARD / SHADOW CONFIRMATION CURRENT

**Repository:** `dkaps6/imtiredofthis`  
**GitHub is canonical; chat memory is secondary.**  
**User explicitly re-opened RB as an unresolved priority on 2026-09-20.**

## Read order

1. `AGENTS.md`
2. this handoff
3. `CURRENT_NFL_RESEARCH_HANDOFF.md`
4. `docs/research/RB_PD2_YARD_DIFFICULTY_MC_WIDTH_V1_PLAN.md`
5. `docs/research/RB_PD2_YARD_DIFFICULTY_MC_WIDTH_V1_RUN.md`
6. `docs/research/overnight/RB_POST_WEEK1_GAP_FINDINGS.md`
7. `docs/research/overnight/RB_PD_CHAIN_STATUS.md`
8. the latest GPT-5.6 / Claude checkpoints in GitHub Issue #535

## Branch / production boundary

Active RB branch:

`research-rb-pd2-forward-shadow-confirmation-v1`

Branch was created directly from current production main:

`f0dad2c6711e85104eeffedfa5f5112fd172cbf5`

Do not mutate production while designing the shadow. Do not use sportsbook data as football-model input.

Separate football-context branch:

`research-football-context-event-redundancy-v1`

Closed-context head at handoff creation:

`c963355b270ce3b733ea9d9b6b01536c4d858adc`

Do not merge that context branch into this RB branch. The RB branch intentionally starts from clean current main.

## Why RB is back in scope

The user explicitly said RB is still not reliably solved and wants it treated as a live priority rather than left as "paused."

Important distinction:

RB is **not scientifically empty**. There is a real positive distribution-calibration result already merged into main, but it is not production-authorized because its required forward/shadow confirmation has never been run.

### Positive RB authority already established

PR #562:

`Research: RB-PD2 yard-difficulty MC-width V1`

Merged on 2026-09-16.

Qualification branch head:

`8faa18fedaaeec8a073c0275b2542396c4b2e04e`

Canonical qualifying run:

`35039152022`

Final disposition:

`RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED`

All 28 hard gates passed.

Key result:
- pooled CRPS improvement: **+1.218%**
- high-difficulty-quartile CRPS improvement: **+2.624%**
- player-clustered bootstrap probability candidate better: **1.0**
- crossed player x game bootstrap probability candidate better: **1.0**
- point MAE: **identical by design**
- coverage improved
- Brier >=100 improved
- robustness: **4/4 seasons**
- production changed: **false**
- sportsbook used as model input: **false**

This was a mean-neutral uncertainty-width result, not a new rushing-yard mean model.

Frozen mapping:
- `WIDTH_CAP = 0.30`
- no widening below difficulty score 0.50
- linearly increases from 1.00x to 1.30x as difficulty percentile moves 0.50 -> 1.00
- widening is applied around the already-aligned football mean
- candidate is renormalized to preserve the exact baseline mean

Do **not** retune 0.30, the 0.50 onset, percentile construction, historical window, reference floor, or mean-neutral transform.

The plan explicitly requires a **separate forward/shadow confirmation** before production promotion.

No such forward/shadow confirmation has yet been started.

## RB production state that remains unresolved

Week-1 specialist authorities exist, but they do not solve the full-season RB problem.

Current boundaries:
- RB-P3 rushing synthesis is promoted for its qualified 2026 Week-1 route only.
- R26 receptions/opportunity specialist scope is Week-1 only.
- R22 receiving-yard tail/distribution specialist scope is Week-1 only.
- Weeks 2-18 RB rushing authority remains unresolved.
- RB receiving-yard mean remains unresolved.
- conventional historical receiving-efficiency transformations (YPR / YPT / YAC / xYAC family, R23-R27D) are closed; do not reopen them.
- the strongest unfinished RB lead is distributional calibration, not another mean-feature hunt.

## Immediate work completed immediately before this handoff

GPT-5.6 created the clean branch:

`research-rb-pd2-forward-shadow-confirmation-v1`

from:

`main@f0dad2c6711e85104eeffedfa5f5112fd172cbf5`

Then audited the live pricing seam in `scripts/run_pricing_v2.py`.

Important production-order finding:

1. current football metrics are built;
2. ML / State / Bayesian / canonical rules are applied;
3. Monte Carlo `base_outcomes` are generated;
4. `mc_proj = mean(base_outcomes)`;
5. current ensemble / position-specific mean authority determines `target_mean`;
6. the MC distribution is multiplicatively rescaled so its mean equals the final football `target_mean`;
7. only **after that football-distribution step** does sportsbook information enter for line probability / edge comparison.

That is exactly the seam needed by the qualified RB width mechanism.

Therefore a research shadow can be implemented **downstream of the final football mean and upstream of sportsbook comparison**, preserving production outputs while producing a separate candidate distribution artifact.

No production probability or board should change during shadow.

## Frozen historical width implementation already in main

Reference evaluator:

`scripts/research/evaluate_rb_pd2_yard_difficulty_mc_width_v1.py`

Key frozen functions:
- `strict_prior_difficulty_scores(...)`
- `width_multiplier(score)`
- `widen_mean_neutral(draws, multiplier)`

Key historical contracts:
- history window: last **8** strictly-prior same-player games
- minimum prior games: **4**
- `prior8_yard_mae` is the raw difficulty statistic
- difficulty score is an empirical strictly-prior percentile against the accumulated eligible reference population
- reference minimum: **100**
- same-week target rows do not enter their own percentile reference
- width onset: score **0.50**
- width cap: **1.30x**
- candidate mean equals baseline mean at machine precision

Historical baseline distribution:
- exact canonical football MC distribution
- then multiplicatively aligned to the final football projection
- candidate width transform occurs only after this mean alignment

The live shadow must preserve the same semantic order.

## Critical design question now being audited

The next blocker is not the width formula.

It is constructing the **2026 pregame difficulty score** correctly and prospectively.

For a 2026 RB target game, the shadow needs:
- the player's last up to 8 completed prior-game rushing-yard prediction errors;
- those historical predictions must correspond to the legitimate football mean route available for each historical game;
- no target-week or future information;
- enough reference history to compute the same strict-prior percentile;
- no sportsbook inputs.

### 2025 use must be handled carefully

The original MC-width qualification explicitly excluded 2025 as a qualification / replication cohort because the broader PD chain had already observed 2025.

That does **not automatically mean 2025 must be erased from 2026 predictor history**.

For an actual 2026 pregame shadow, completed 2025 games are temporally prior and are naturally candidates for the player's last-8 error history.

However, this must be treated as a source/chronology contract, not silently assumed.

Claude + GPT-5.6 should independently verify:

1. Which canonical football projection lineage should define the 2025 RB rushing-yard prediction errors used as 2026 history?
2. Can that 2025 history be rebuilt authority-exact without fabricating P3 outside its qualified Week-1 scope?
3. Is it scientifically clean to use completed 2025 outcomes **only as predictor history for 2026**, while keeping 2026 as the sole forward confirmation cohort?
4. What exact historical reference population should compute the 2026 difficulty percentile so the frozen percentile meaning is preserved without outcome-informed retuning?

The expected conceptual answer is that 2025 may be strict-prior history but is **not** a new confirmation cohort. This still needs to be proven against repo lineage before implementation.

## Proposed shadow architecture — not yet frozen

Do not treat this section as final until Claude + GPT-5.6 review it.

Candidate direction:

### Stage A — build pregame RB difficulty state

For each live 2026 RB rush-yards row:
- resolve stable player identity;
- gather last 8 completed strictly-prior eligible games;
- use the appropriate historical football projection and realized rush yards;
- compute `prior8_yard_mae`;
- require >=4 prior games;
- map to the frozen empirical difficulty percentile using a chronology-safe reference pool.

### Stage B — generate the live baseline football distribution

Use exactly the current production football distribution before sportsbook comparison:
- canonical MC draws;
- current final football mean;
- same multiplicative mean-alignment contract already used in production.

Do not rebuild a parallel alternate football model.

### Stage C — shadow-only width candidate

Apply exactly:

`width_mult = 1 + 0.30 * clip((difficulty_score - 0.50) / 0.50, 0, 1)`

Then apply the existing `widen_mean_neutral` semantics.

Hard invariants:
- candidate draws finite
- candidate draws nonnegative
- candidate mean == baseline football mean
- no sportsbook used to construct the candidate
- production fair probabilities unchanged
- production recommendations unchanged
- candidate written only to a separate shadow artifact

### Stage D — lock before games

For each 2026 slate being confirmed, persist before kickoff:
- season
- week
- event/game
- team/opponent
- player stable identity
- market
- final football mean
- baseline distribution hash / summary
- prior-games count
- prior8 yard MAE
- difficulty score
- difficulty reference N
- difficulty reference max chronology marker
- width multiplier
- baseline distribution metrics
- candidate distribution metrics
- explicit lock timestamp / source lineage

No realized outcome may be attached before the lock.

### Stage E — grade only after games

Primary scoring should remain football-distribution scoring, not sportsbook ROI.

At minimum:
- CRPS baseline vs candidate
- 80% interval coverage / gap
- 90% interval coverage / gap
- tail Brier at frozen 50 / 75 / 100 yard thresholds
- point mean parity assertion

The exact prospective pass/fail sample-size and dependence-aware bootstrap gates must be frozen **before first 2026 shadow outcomes are scored**.

Do not choose thresholds after looking at Week 1/2 results.

## Important current-date / sample issue

Current date is 2026-09-20.

The 2026 regular season is already underway. Any shadow confirmation started now cannot honestly claim to have prospectively locked Weeks 1-2 if no candidate artifacts were persisted before those games.

Therefore:
- do not reconstruct Weeks 1-2 after the fact and call them forward confirmation;
- the forward/shadow cohort starts only with the first future week for which the candidate is locked pregame under the frozen protocol;
- previously played 2026 weeks may potentially become strict-prior history for later 2026 weeks only if their football prediction lineage can be reconstructed without using the new candidate and without violating chronology;
- they cannot count as prospectively locked confirmation observations.

This point should be explicitly reviewed before the plan is frozen.

## Separate football-context work just completed — do not reopen

Immediately before returning to RB, GPT-5.6 exhausted another genuinely different context frontier.

### Defensive-front pairwise cohesion

Qualification was recovered after forensic identity repair:
- corrected run `35517035459`
- artifact `10607480543`
- pregame coverage **99.1448%**
- stability Spearman **0.813629**
- redundancy R2 **0.401720**
- qualification: `READY_FOR_FROZEN_EXPERIMENT`

Frozen pressure-generation mechanism then failed closed:
- run `35517405126`
- artifact `10607820423`
- 2024 rows 544 / 544
- coefficient wrong direction
- MAE / RMSE / p90 all slightly worse
- bootstrap CI crossed zero
- 2025 remained sealed
- final `DEFENSIVE_FRONT_PAIRWISE_COHESION_PRESSURE_MECHANISM_V1_FAILED_CLOSED_PRIMARY`

### Exact blocker-rusher assignment source

BDB2023 exact PFF blocker->blocked-defender assignment qualified strongly:
- run `35519025957`
- artifact `10607537977`
- **46,396** assignment edges
- **590** blockers
- **665** blocked defenders
- **12,536** unique pairs
- stable ID **100%**
- Week 5+ edges **22,111**
- Week 5+ both-player prior-10 support **87.9019%**
- disposition `VALUE_OF_INFORMATION_LAB_READY_SOURCE_SLICE`

One frozen hindsight value-of-information experiment then tested whether exact realized
pairing structure improved pressure prediction beyond the same blocker/defender marginal
exposure.

Frozen plan:
`docs/research/BDB2023_EXACT_BLOCKER_RUSHER_ASSIGNMENT_VOI_EXPERIMENT_V1.md`

Canonical run:
- `35519336205`
- artifact `10607218889`
- digest `sha256:18cd686c11573bb483400e016747c45d91bde2ddade26468f2b6bb935f2e30db`

Holdout Weeks 7-8:
- denominator rows 56
- scored rows 46
- coverage **82.1429%**
- baseline MAE **0.02064617**
- candidate MAE **0.02065083**
- baseline RMSE **0.02440605**
- candidate RMSE **0.02443043**
- baseline p90 AE **0.03738783**
- candidate p90 AE **0.03767568**
- pairing-covariance coefficient **+0.758973**
- bootstrap AE-gain 95% CI **[-0.00005814, +0.00005209]**

Support, MAE, bootstrap, RMSE and p90 gates failed.

Final:
`BDB2023_EXACT_ASSIGNMENT_VOI_NO_ACTIONABLE_SIGNAL_V1`

Do not rescue it.

### BDB2026 / BDB2025 source notes

BDB2026 Prediction archive returned HTTP 403 under the current Kaggle credential.
Do not interpret that as a scientific "no signal" result.

BDB2025 exact-assignment acquisition attempts hit 404 / archive-materialization issues.
Again, acquisition result only.

Because BDB2023 exact assignment itself already qualified and then failed its frozen VOI
mechanism, do not spend time fighting those source endpoints to rescue the same idea.

## Other anti-retest boundaries

Do not reopen:
- QB Conditional Analog V1: `NO_ACTIONABLE_QB_CONDITIONAL_ANALOG_RELIABILITY`
- OL pairwise cohesion pressure mechanism
- defensive-front pairwise cohesion pressure mechanism
- exact blocker-rusher pairing-covariance V1
- generic aggregate pressure/mismatch direct QB family
- M77 exact personnel discontinuity
- M80-M81 FTN tactical pressure families
- BDB2023 geometry broad source-thin experiment
- BDB2026 receiver release broad source-thin experiment
- Historical Analog State V1
- RB receiving mean historical efficiency transforms R23-R27D
- failed role-room concentration family
- failed event-regime reliability family

Failures remain failures.

## Claude collaboration request

The user explicitly wants Claude reconnected to the GPT-5.6 back-and-forth now.

Claude should **not restart from scratch**.

Requested independent review:

1. Audit the proposed RB forward/shadow architecture against the frozen PR #562 plan/result.
2. Determine the exact authority-exact 2025 RB rushing projection lineage suitable for use as strict-prior 2026 history.
3. Challenge whether completed 2025 / already-played 2026 rows may be used as predictor history without contaminating the future 2026 confirmation cohort.
4. Recommend exact frozen sample-size / dependence-aware prospective confirmation gates **before any future outcome is opened**.
5. Audit the cleanest non-production integration seam for persisting baseline + candidate RB rush-yard distributions during the live Full Slate workflow without changing production probabilities.
6. Post concerns / corrections back to Issue #535 before GPT-5.6 freezes the forward-shadow plan.

No production promotion is authorized yet.

## Immediate GPT-5.6 checkpoint

GPT-5.6 had just begun auditing how to construct 2025 component/projection history for the 2026 predictor state when the user requested this Claude handoff.

Resume there after Claude review.

