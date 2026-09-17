# RB Rushing Anti-Retest Map — 2026-09-17

**PURPOSE:** prevent circular RB research. This file is a hard precondition for any new RB rushing/workload/allocation experiment after Workhorse-Transition-Gate V1.

**RULE:** no new candidate may be implemented, fit, or run until its proposal explicitly maps itself against the closed/tested families below and demonstrates genuine novelty. If the distinction is only a new threshold, blend, window, classifier, interaction, subgroup, or renamed version of an exposed historical idea, stop.

This map is built from the repository's canonical continuity records, terminal synthesis docs, relevant result commits, and the current 142-branch `research-rb-*` inventory. It is intentionally conservative: ambiguity counts as potential overlap until disproved.

## 1. Newly closed family — Workhorse-Transition-Gate V1

Canonical terminal result:
- run `35253048097`
- head `bc56c80664a93b8691baf932312b9ccb6405749c`
- artifact `10513292695`
- digest `sha256:e8d632b2b602ae108090e74fe08ceb0d3bd0ec8bf88fef4b12af079b67e7a78c`
- result record: `docs/research/RB_WORKHORSE_TRANSITION_GATE_V1_RESULT.md`
- disposition: `RB_WORKHORSE_TRANSITION_GATE_V1_NOT_QUALIFIED`

Already tested in this family:
- binary target: post-transition active RB/FB/HB room produces at least one `>=20 actual carries` back;
- scored loss/vacancy transition population;
- strictly-prior top/second RB role share;
- pre-transition backfield HHI;
- departed-room prior workload sum/max;
- active/prior RB-room size;
- MC projected plays and dropback rate;
- derived raw team rush volume;
- historical RB-room rush share;
- top/second raw MC projected rush attempts;
- standardized L2 logistic regression with balanced classes;
- F0.5 cutoff selection on a fixed `{0.50,...,0.90}` grid;
- two independent historical rotations.

Permanent boundary: no rescue of this exact binary gate on the exposed 2019-2023 chronology.

## 2. The broad `team volume x player allocation/concentration` concept is NOT new

This is the most important anti-circle finding after Workhorse V1.

### RB-ND1 forensic decomposition

Canonical ND1 audit established that carry error already decomposes into:
- team-volume absolute contribution share: `39.87%`;
- player-share/backfield-allocation contribution share: `60.13%`.

Rushing-yard error decomposed into:
- opportunity: `51.51%`;
- efficiency: `48.49%`.

ND1 therefore already established the architecture `team rush opportunity -> player share/allocation -> efficiency`. Re-discovering that architecture is not a new experiment.

### RB-R1 room volume vs individual allocation

Branch: `research-rb-r1-room-volume-vs-individual-allocation`
- run `34070566812`
- artifact `10000309225`
- result commit `087ac6dee3af0eba41eee03f35c22dfcb0a2d28a`
- disposition: `RB_CARRY_ERRORS_REMAIN_MIXED_ROOM_AND_ALLOCATION`

All player-games:
- room-volume abs component `2.200468` carries;
- individual-allocation abs component `2.514077`;
- ratio `1.142519`.

CARRIES-dominant slice still failed the preregistered universal routing gates. Player-level heterogeneity was the result: some RBs were room-volume dominant; others allocation dominant.

Therefore a universal new model described only as `team rushing volume x player share` would repeat RB-R1/ND1 architecture rather than create new information.

### RB-R2 allocation subgroup role context

Branch: `research-rb-r2-allocation-subgroup-role-context`
- run `34070930143`
- artifact `10000420714`
- result commit `65201dbb8acb12b016008f8c4d6b277e453b5ed0`
- disposition: `NO_ACTIONABLE_RB_ALLOCATION_SUBGROUP_ROLE_CONTEXT_SIGNAL`

Already tested inside allocation-dominant players:
- depth vs projected carry-order mismatch;
- injury-created context;
- no prior same-team game;
- rookie status;
- limited prior history.

Coarse depth/rookie/injury/continuity flags did not explain enough allocation error. Do not repeat them as a supposedly new allocation router.

### RB-R3 dynamic workload allocation

Branch: `research-rb-r3-dynamic-workload-allocation`
- run `34072849006`
- artifact `10001031461`
- result commit `1042cd9d3ce7e1f9a2f4a257b4315651f11a000f`
- disposition: `NO_ACTIONABLE_RB_DYNAMIC_WORKLOAD_ALLOCATION_SIGNAL`

Already tested:
- prior-1 RB-room carry share;
- prior-1 carries;
- room-share acceleration 1-vs-4;
- carries acceleration 1-vs-4.

`PRIOR1_ROOM_CARRY_SHARE` passed 9/10 pooled gates but failed within-player consistency badly (2/17 positive). The scientific conclusion was structural between-player heterogeneity, not a reusable latest-game momentum signal.

Permanent boundary: no last-game usage / carry acceleration / short-term momentum retry.

## 3. Major historical opportunity/concentration families already explored

The RB rushing program already includes these branch families and must be checked before any adjacent proposal:

- `research-rb-m91-temporal-baseline`
- `research-rb-m92-opportunity-decomposition`
- `research-rb-m93-backfield-concentration`
- `research-rb-m93b-role-aware-concentration*`
- `research-rb-m94-team-rush-volume`
- `research-rb-m94b-explicit-game-state`
- `research-rb-m94c-game-environment`
- `research-rb-m94d-joint-opportunity*`
- M95A-M95T workload/matchup/environment/workload-regime/role/vacancy/concentration/feed/carry-ceiling family;
- M96A-M96E opportunity/efficiency/router/guard family.

Canonical continuity interpretation:
- M94C became a central opportunity/carry reference;
- M95 repeatedly explored workload regimes, vacancy/transition, feed tendency and carry ceiling;
- M95T formally stopped detached retrospective tail-overlay invention;
- multiple residual-calibration variants improved central MAE while damaging p90, which is a stopping signal rather than permission to tune until a pass appears.

Any future `team volume`, `backfield concentration`, `workload regime`, `lead-back entitlement`, `vacancy`, `feed tendency`, or `carry ceiling` proposal must first identify the exact historical branch/result it differs from.

## 4. STACK family — team rush context / allocation already heavily explored

Historical families include STACK1-STACK7 and STACK6 variants through 6T, including:
- production-equivalent baseline;
- enriched allocation;
- frozen state composition;
- efficiency portability;
- market-gap forensics;
- secondary-back role state/models;
- contraction/availability/inactive-competitor context;
- team pool;
- regime change;
- team-rush mechanics;
- state occupancy/tendency;
- trail/urgency/run-event context;
- designed run-call/context and conditional run advantage.

Canonical stop rule from the repository synthesis: **do not reopen STACK6 team-rush-context slicing.**

The existence of a new wording for score state, pace, game script, run tendency or team-rush mechanics does not make it a new family if it reuses the same information class.

## 5. Depth/role remapping is closed as direct workload authority

Role-Order Remap V1:
- run `34063904515`
- artifact `9998334300`
- carry MAE `3.483 -> 4.106` (worse)
- rush-yard MAE `20.424 -> 22.837` (worse)

Permanent lesson: depth chart can be contextual evidence but cannot be treated as direct carry/workload authority.

R2 further showed simple depth-order mismatch was not a strong allocation-error router.

## 6. PD residual / uncertainty family

### PD2 player-error persistence

The original diagnostic found real strictly-prior player-specific residual persistence. This is already-known information, not a new discovery target.

### PD3 / PD4 / PD5 specific residual-calibration designs

Recovered historical results showed the specific mean/residual calibration designs failed their frozen safety gates, commonly by improving central MAE while worsening yard p90. Do not rerun these exact designs or relax their gates.

### PD2 yard-difficulty MC-width V1

This is a separate, later distributional-width study and is **qualified research**, not an open idea to rediscover:
- branch `research-rb-pd2-yard-difficulty-mc-width-v1`;
- canonical rerun `35039152022`;
- disposition `RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED`;
- pooled CRPS improvement `+1.218%`;
- high-difficulty-Q4 CRPS improvement `+2.624%`;
- point MAE exactly mean-neutral;
- 4/4 season robustness;
- production remains unchanged pending separate forward/shadow confirmation.

Forward-confirmation planning already exists on `research-rb-pd2-yard-width-forward-confirmation-v1`. Do not create a second MC-width candidate that merely repackages PD2.

## 7. Lane-A allocation family is terminal

### Lane-A V1

Terminal constructibility failure before outcome exposure due to unavailable production comparator values. No outcome result was opened.

### Lane-A V2

Terminal disposition: `RB_LANE_A_TRANSITION_ALLOCATION_INSUFFICIENT_EVIDENCE`.

It used:
- projected team rush pool;
- historical RB-room rush share;
- prior3 role weights;
- pre-transition HHI concentration exponent;
- post-transition production-scoreable active recipient universe;
- incumbent/comparator YPC translation;
- exact conservation.

It produced large descriptive improvements in postgame-defined heavy-workload slices but failed overall transition MAE, whole-season MAE, bootstrap protection, per-season protection and 2025 p90.

Permanent boundary: no Lane-A V2 rescue on exposed 2024-2025 outcomes.

Workhorse-Gate V1 was a separately frozen attempt to predict when to activate that frozen V2 mechanism and is now also terminal.

## 8. Receiving-side RB lanes that must not be confused with rushing research

The receiving-yard mean family R23 -> R27D exhausted historical YPR/YPT/YAC/xYAC/YACOE transformations for the lead-back receiving-yard mean. R27D explicitly closed another historical-efficiency transformation path.

Production authorities remain separate:
- R26 receptions/opportunity authority where its promotion scope applies;
- R22 mean-preserving receiving-yard tail authority where its promotion scope applies;
- generic production mean outside separately promoted scope.

Do not recycle a failed receiving-efficiency transformation as a rushing-efficiency idea without genuinely new football information.

## 9. What the Workhorse-V1 result actually adds

It adds **diagnostic evidence**, not a new architecture entitlement:

- Rotation B showed non-random discrimination (ROC-AUC `0.637`, PR-AUC `0.289` vs 2023 prevalence `0.187`, recall `0.565`), so strictly-pregame transition/role information carries some relationship to future high workload.
- But high-precision activation failed: precision only `0.255` in Rotation B and zero confirmation fires in Rotation A.

The correct inference is NOT `try a better classifier on the same 13 features`. The correct inference is `there may be information in this neighborhood, but the exact binary gate/calibration architecture is closed`.

## 10. Mandatory novelty gate before the next RB experiment

Before implementation, a proposal must include an explicit table with one row for every nearby prior family and answer all of the following:

1. **Target novelty:** Is the response variable genuinely different from already tested carries, room share, HHI/concentration, workload regimes, carry residuals, >=20 workload, yard residuals or width? If merely re-thresholded, FAIL novelty.
2. **Mechanism novelty:** What causal layer is new relative to ND1/R1/R2/R3, M92-M96, STACK1-7, Lane-A and Workhorse V1?
3. **Information novelty:** Which pregame information is genuinely new? If it is just prior carries/share, HHI, depth rank, injury/vacancy, room size, projected attempts, team rush context, score/game-script, pace/run tendency or already-used historical participation, presume overlap until lineage proves otherwise.
4. **Temporal novelty / evidence:** Which years are discovery-exposed? Do not present a retuned 2019-2025 result as independent confirmation.
5. **No rescue:** Is the proposal motivated by a new mechanism/data family rather than changing thresholds/model family after a failure?
6. **Source constructibility:** Can every proposed feature be reconstructed strictly pregame with known historical semantics before fitting?
7. **Production relationship:** Is this a mean, allocation, efficiency or distribution candidate? It must not silently overwrite an already-qualified authority such as PD2 width or a protected production route.
8. **Stop rule:** Freeze the exact failure condition before outcomes open.

If any of these cannot be answered clearly, do not run the experiment.

## 11. Immediate process from here

**No next RB model is authorized by this document.**

The next action is an independent white-space audit, not candidate fitting:

1. Use the full 142-branch `research-rb-*` inventory plus canonical continuity/result records.
2. For any candidate concept, identify all nearest historical analogues and their exact disposition.
3. Separate `already tested`, `tested but with a different target`, `planned but never executed`, and `genuinely untested`.
4. Only the `genuinely untested` portion may become a new prospective family.
5. Freeze that family before any outcome relationship is inspected.

This process is intended to stop the project from repeatedly rediscovering `volume x allocation`, `role concentration`, `latest usage`, `vacancy`, or `tail width` under new names.
