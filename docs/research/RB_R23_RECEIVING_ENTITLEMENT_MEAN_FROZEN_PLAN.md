# RB R23 — Receiving Entitlement / Receptions / Receiving-Yard Mean

Status: **FROZEN BEFORE OUTCOME INSPECTION**
Date: 2026-09-09
Base production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Research branch: `research-rb-r23-receiving-entitlement-mean-v1`

## Why R23 exists

Week-1 RB receiving science is intentionally split into separate authorities.

- **RB P3** owns promoted Week-1 rushing point means.
- **RB R22** owns Week-1 RB receiving-yard **distribution shape / tail behavior only** and is mean-preserving.
- The current production stack still uses the canonical finite-pool RB receiving entitlement, reception conversion, and receiving-yard point mean beneath R22.

R23 is authorized to investigate only that remaining point-mean/opportunity gap. It must not reopen or retune solved R22 tail science.

## Frozen scientific question

Can strictly-prior, sportsbook-free football evidence improve individual RB receiving opportunity and point prediction versus the current production baseline while preserving finite team target conservation and every already-promoted RB rushing/tail authority?

The target chain is:

`finite team target pool -> RB-room entitlement -> individual RB targets -> receptions -> receiving-yard mean -> existing R22 tail adapter`

R23 is successful only if the upstream opportunity/mean model improves out-of-sample player predictions without requiring any modification to R22.

## Non-negotiable protected authorities

The following are frozen and cannot be changed to rescue R23:

1. `RB_P3_SYNTHESIS_V1` rushing authority and its Week-1 routing.
2. `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_V1` transformation, including R19 scorer assets, residual pools, thresholds, routing logic, seeds, rank-preservation contract, and mean-preservation contract.
3. M89/M90 QB passing-yard point-mean authority.
4. QB C2 distribution selector.
5. M38 WR1 + WR-R15 WR2+ receiving entitlement.
6. TE-R5P entitlement.
7. Explicit finite team target pool and residual mass.
8. Sportsbook independence upstream: lines, odds, market probabilities, books, consensus lines, and market-derived win probabilities are forbidden model inputs.

If an R23 candidate needs any protected authority changed to pass, R23 fails.

## Phase R23-A — Historical contract / baseline audit

Before fitting a new model, build one reproducible historical table in which every row is an RB player-game and every feature is available strictly before kickoff.

Required labels where source coverage allows:

- targets
- receptions
- receiving yards
- team pass attempts / dropbacks
- team RB targets
- player target share
- player share of RB-room targets
- routes or route proxy when genuinely available
- snaps or snap proxy when genuinely available

Required strictly-prior feature families:

- lagged player targets/receptions/receiving yards
- lagged team RB target rate / RB-room share
- lagged player share of RB-room targets
- lagged catch rate
- lagged yards per target / yards per reception, appropriately shrunk
- current roster/depth/role information that was available pregame
- injury availability/status only when timestamp-safe
- opponent/team context already certified as pregame-safe
- game-environment features only from football context, never sportsbook prices

Every feature column must receive an explicit provenance classification. Any feature with ambiguous timing is excluded rather than inferred.

## Phase R23-B — Frozen candidate family

R23 tests a small mechanistic candidate family; it does not perform broad feature fishing.

### Candidate 0 — Production baseline

Recreate the current finite-pool canonical RB receiving entitlement and current reception/receiving-yard mean logic exactly. This is the control and must reproduce the production baseline before any candidate is judged.

### Candidate 1 — Shrunk RB-room entitlement

Predict each RB's share of the already-finite team RB receiving room from strictly-prior role evidence. The candidate may redistribute **only the existing RB-room receiving mass**; it may not create additional team target mass or take opportunity from WR/TE rooms unless a separately frozen future study authorizes that.

Core signals are limited to:

- lagged player RB-room target share
- lagged targets per team dropback
- lagged receptions per team dropback
- current depth/role ordering
- recent participation/route or snap proxy when timestamp-safe
- injury-driven availability of same-team RB competitors when timestamp-safe

All empirical rates must be shrunk toward a role/position prior. No post-result window tuning is allowed.

### Candidate 2 — Opportunity + conversion decomposition

Use Candidate 1 entitlement for expected targets, then estimate expected receptions through a strictly-prior shrunk catch-conversion component.

`E[receptions] = E[targets] * shrunk catch rate`

Candidate 2 is evaluated separately for targets and receptions so a yardage improvement cannot hide a worse reception process.

### Candidate 3 — Opportunity + conversion + efficiency mean

Use Candidate 2 expected receptions and a strictly-prior shrunk receiving-efficiency component.

Primary construction:

`E[receiving yards] = E[receptions] * shrunk yards per reception`

A yards-per-target decomposition may be logged as a diagnostic challenger, but it cannot silently replace the frozen primary candidate after outcomes are seen. A replacement requires a new R24 plan.

## Frozen priors and windows

To avoid post-result tuning:

- Primary recent-form window: previous **6 available player games**.
- Secondary stabilizing history: previous **16 available player games** where present.
- If fewer than 3 prior player games exist, use role/position priors plus available current depth information rather than extrapolating tiny samples.
- Recent and stabilizing history are blended by sample opportunity count, not by a tuned outcome-derived alpha.
- Catch-rate and efficiency estimates use empirical-Bayes shrinkage toward historical RB role priors.
- No candidate-specific thresholds or caps may be changed after seeing confirmation results.

If the historical source cannot reconstruct one of these windows without leakage, that signal is dropped and the deviation is recorded before scoring.

## Walk-forward evaluation

Primary confirmation seasons are frozen as:

- 2023
- 2024
- 2025

For each game, training/priors/features may use only information strictly earlier than that kickoff. 2025 is not a tuning set; all three seasons are confirmation folds under the same frozen candidate.

2026 Week-1 outcomes are forbidden from R23 research and promotion decisions.

## Evaluation population

Primary population: RB/FB player-games with valid receiving labels and sufficient team-game identity to enforce room conservation.

Report separately:

- all RB/FB rows
- RB1 / lead-back role
- RB2+ / complementary role
- high-opportunity receiving backs
- low-history/new-player rows
- injury/role-change rows when timestamp-safe classification exists

No cohort may be excluded after outcomes are observed merely because it performs poorly.

## Frozen metrics

For each of targets, receptions, and receiving yards report:

- MAE
- RMSE
- mean signed error / bias
- Pearson correlation
- Spearman correlation
- median absolute error
- p75 absolute error
- p90 absolute error

Additional receiving-yard tail diagnostics:

- miss >= 20 yards
- miss >= 30 yards
- miss >= 40 yards

Additional structural diagnostics:

- per-team predicted RB-room entitlement sum
- per-team total explicit target-pool conservation gap
- individual negative-projection count
- finite-value failures
- change in WR/TE entitlement (must be exactly zero for R23)
- change in P3 rushing projection (must be exactly zero)
- R22 mean-preservation after upstream mean replacement

## Frozen promotion gates

A candidate is eligible for further integration only if **all** gates pass.

### Integrity gates

1. Zero sportsbook inputs to feature generation, fitting, selection, or football projection.
2. Zero 2026 outcome usage.
3. Strict-prior feature timing verified.
4. Team target pool remains physically conserved.
5. WR/TE entitlement is exactly unchanged.
6. P3 rushing outputs are exactly unchanged.
7. R22 transformation remains byte/parameter/logic unchanged and remains mean-preserving around the candidate's upstream receiving mean.

### Scientific gates

The primary promoted candidate is Candidate 3, but it may advance only if the upstream stages are also coherent.

1. **Targets:** pooled 2023-2025 MAE must improve versus production baseline and RMSE may not worsen.
2. **Receptions:** pooled 2023-2025 MAE must improve versus baseline and RMSE may not worsen.
3. **Receiving yards:** pooled 2023-2025 MAE must improve by at least **1.0%** versus baseline and RMSE must not worsen.
4. Directional replication: receiving-yard MAE must improve in at least **2 of 3** confirmation seasons, with no single confirmation season worsening by more than **1.0%**.
5. Tail protection: pooled p90 absolute receiving-yard error may not worsen by more than **2.0%**, and 30+ yard miss rate may not worsen by more than **1.0 percentage point**.
6. Bias protection: absolute pooled receiving-yard bias may not worsen by more than **1.0 yard**, and absolute receptions bias may not worsen by more than **0.10 receptions**.
7. Role robustness: neither RB1 nor RB2+ receiving-yard MAE may worsen by more than **1.0%**.
8. Mechanism coherence: improved yards cannot be accepted if targets or receptions become materially worse or if improvement comes from violating finite opportunity.

These gates are frozen now. A near miss is a scientific failure; thresholds will not be relaxed afterward.

## Promotion disposition rules

- **PASS:** every integrity and scientific gate passes. R23 may proceed to a separately frozen full-stack integration test beneath unchanged R22.
- **MIXED / NO PROMOTION:** aggregate improvement exists but one or more frozen gates fail. Preserve results and diagnose; do not retune R23.
- **FAIL:** no reliable improvement or any leakage/conservation violation. Preserve the null and move to a genuinely different future hypothesis under a new plan.

## Required artifacts

R23 must leave:

- frozen-plan copy/hash
- historical feature provenance table
- baseline-reproduction audit
- fold-level prediction CSV
- pooled and season metrics CSV/JSON
- role-tier metrics
- tail diagnostics
- finite-pool conservation audit
- protected-authority parity audit
- exact run/job/commit/artifact lineage
- final disposition document

## What R23 does not authorize

R23 does **not** authorize:

- retuning R22 tail thresholds or residual pools
- changing P3 rushing
- changing WR/TE entitlement
- using sportsbook lines/odds as features or correction targets
- selecting features/windows after looking at 2023-2025 outcomes
- using 2026 results
- changing frozen promotion gates after a near miss
- direct promotion to `main` without a separate full-stack integration/certification gate

This plan was committed before inspecting R23 historical candidate outcomes.