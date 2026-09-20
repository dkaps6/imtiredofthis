# BDB2023 Exact Blocker-Rusher Assignment Value-of-Information Experiment V1 — Result

**Status:** CLOSED — NO ACTIONABLE SIGNAL  
**Date:** 2026-09-20  
**Frozen plan:** `docs/research/BDB2023_EXACT_BLOCKER_RUSHER_ASSIGNMENT_VOI_EXPERIMENT_V1.md`  
**Frozen-plan SHA:** `7590d5d867a4a368624ef923b72da8146dd7a7ee`  
**Evaluator commit:** `4653d87403850e3eed7dbfdcc581491baa8fe0c1`  
**Test commit:** `6eec186581e0dc3dc6ae82541a6b086c49558d7f`  
**Canonical execution SHA:** `7a86516f70a85608528105dc8db9a2ff7c22455d`  
**Canonical run:** `35519336205`  
**Canonical job:** `106100587284`  
**Artifact:** `10607218889`  
**Artifact digest:** `sha256:18cd686c11573bb483400e016747c45d91bde2ddade26468f2b6bb935f2e30db`

The preceding run `35519289850` failed at frozen-plan verification because the default
GitHub checkout was shallow. It stopped before Kaggle download and before outcome
inspection. The only repair was fetching the already-frozen plan object before
verification. No scientific rule, feature, split, threshold, or gate changed.

## Final disposition

`BDB2023_EXACT_ASSIGNMENT_VOI_NO_ACTIONABLE_SIGNAL_V1`

This was an explicitly hindsight/nondeployable value-of-information test. It does not
authorize production use of realized assignment information.

## Source qualification retained

The source-only qualification remains valid:

- qualification run: `35519025957`
- job: `106099764039`
- artifact: `10607537977`
- artifact digest: `sha256:852c9c8b6dec11817d6d72ebf1222fb0e666641a3c78a4ca59b9520daa8db974`
- exact blocker->blocked-defender edges: **46,396**
- unique blockers: **590**
- unique blocked defenders: **665**
- unique pairs: **12,536**
- stable-ID coverage: **100%**
- ambiguous IDs: **0**
- Week 5+ edges: **22,111**
- Week 5+ both-player prior-10 support: **19,436 / 87.9019%**
- prior same-pair coverage: **3.3784%**

Thus the scientific failure below is not an acquisition/identity failure. The exact
assignment slice was sufficiently rich to run the frozen laboratory.

## Frozen experiment

Question:

> Conditional on the same realized target-game blocker and defender participation
> volumes, does the exact realized pairing structure improve target-game
> pressure-allowed prediction?

History:
- Weeks 1-4: strict-prior history only
- Weeks 5-6: fit
- Weeks 7-8: frozen holdout

Baseline:
- blocker prior pressure-allow mean
- defender prior pressure-generation mean
- scoreable assignment-edge count
- week

Candidate adds exactly:
- `assignment_pairing_covariance`

Outcome:
- offense-team target-game edge pressure-allowed rate
- pressure defined only from `pff_hitAllowed`, `pff_hurryAllowed`, and
  `pff_sackAllowed`
- `pff_beatenByDefender` was not used

## Primary result

Holdout:
- denominator offense-team rows: **56**
- scored offense-team rows: **46**
- coverage: **82.1429%**
- game clusters: **28**

Baseline:
- MAE: **0.020646169641423812**
- RMSE: **0.024406046877737002**
- p90 absolute error: **0.03738782937087038**
- bias: **-0.006508500559728014**
- correlation: **0.04883274683332676**

Candidate:
- MAE: **0.02065083105751703**
- RMSE: **0.024430427525823042**
- p90 absolute error: **0.03767567735929137**
- bias: **-0.006657017499644935**
- correlation: **0.05183504692471174**

Baseline-minus-candidate gains:
- MAE: **-0.000004661416093218462** — candidate worse
- RMSE: **-0.000024380648086040624** — candidate worse
- p90: **-0.0002878479884209917** — candidate worse

Candidate pairing-covariance coefficient:
- **+0.7589731320232647**
- expected direction was positive

Game-cluster bootstrap:
- reps: **5,000**
- seed: **92028**
- mean AE gain: **-0.0000049822698092189154**
- 95% CI: **[-0.000058144333479468504, +0.00005209063320249488]**

## Frozen gates

Passed:
- integrity
- holdout coverage >=80%
- pairing-covariance coefficient positive

Failed:
- holdout rows >=50 (**46**)
- candidate MAE lower
- bootstrap CI lower bound >0
- candidate RMSE nonincrease
- candidate p90 nonincrease

Even absent the support miss, the candidate did not improve any frozen primary error
metric and the paired bootstrap centered essentially at zero.

## Interpretation

The exact assignment information is real and the matchup-covariance coefficient points
in the hypothesized direction, but V1 provides **no evidence that this simple pairing
structure materially improves offense-team pressure prediction beyond the same blockers
and defenders considered separately**.

This is therefore not a reason to pay for, scrape, infer, or manufacture an exact
blocker-rusher feed for this V1 mechanism.

Do not rescue V1 with:
- a different pairing interaction transform;
- geometry or block-type additions;
- a lower support floor;
- Weeks 5-8 pooling after seeing the holdout;
- selected blockers/rushers/teams;
- alternate regression models;
- target-game PFF labels as upstream production features.

A future blocker-rusher study would require a genuinely different football mechanism,
not a reformulation of this failed pairing-covariance experiment.

## Production / governance

- target-game realized assignment used: **true**, by design for hindsight VOI only
- deployable pregame claim: **false**
- sportsbook read: **false**
- production changed: **false**
- Issue #535 touched: **false**

## RB continuity

RB remains unresolved outside its qualified scopes.

The repo's strongest live positive RB research result remains
`RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED`:
- all 28 gates passed;
- pooled CRPS improved **1.218%**;
- high-difficulty quartile CRPS improved **2.624%**;
- dependence-aware bootstraps reported **p=1.0**;
- point mean remained unchanged.

That result still requires the separately required forward/shadow confirmation before
production. Weeks 2-18 RB production authority remains unresolved.
