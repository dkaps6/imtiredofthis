# BDB2023 Exact Blocker-Rusher Assignment Value-of-Information Experiment V1

**Status:** FROZEN BEFORE OUTCOME INSPECTION  
**Date:** 2026-09-20  
**Research only:** yes  
**Production change authorized:** no  
**Deployable pregame experiment:** no — target-game exact assignment is realized hindsight information  
**Issue #535 touched:** no  
**Sportsbook input:** forbidden

## Authority / source qualification

Source qualification:

`BDB2023_EXACT_BLOCKER_RUSHER_ASSIGNMENT_QUALIFICATION_V1`

Canonical qualification:
- run: `35519025957`
- job: `106099764039`
- artifact: `10607537977`
- artifact digest: `sha256:852c9c8b6dec11817d6d72ebf1222fb0e666641a3c78a4ca59b9520daa8db974`
- active implementation/workflow SHA: `5b6a2a820abd46cbe642e80b2fcac124921c60b9`
- certified materializer SHA: `47bcd58aecf453f54b3f5db06a9dbdc94b000ad2`
- certified BDB2023 source hash: `1c3e1eb6fcd0cf85807c649804fa0c72421c76e973e40a742523f1dd3a929182`

Qualification evidence:
- exact assignment edges: **46,396**
- weeks: **2021 Weeks 1-8**
- unique blockers: **590**
- unique blocked defenders: **665**
- unique blocker-defender pairs: **12,536**
- stable-ID coverage: **1.0000**
- ambiguous BDB nfl IDs: **0**
- duplicate assignment keys: **0**
- Week 5+ assignment edges: **22,111**
- Week 5+ edges where both blocker and defender have >=10 strictly-prior assignment edges: **19,436 (87.9019%)**
- Week 5+ edges with prior same-pair history: **747 (3.3784%)**
- qualification disposition: `VALUE_OF_INFORMATION_LAB_READY_SOURCE_SLICE`

The source relationship is PFF `pff_nflIdBlockedPlayer`, already consumed by the certified BDB2023 materializer as an exact realized blocker -> blocked-player relationship.

## Scientific question

Conditional on knowing the same target-game blocker and defender participation volumes, does the **exact realized pairing structure** contain incremental information about target-game pressure allowed?

This is a value-of-information / acquisition-priority experiment. It is intentionally not a deployable backtest because the target-game exact blocker-defender assignment is not available to the current production system before kickoff.

A positive result means only:

> exact weekly assignment information appears valuable enough to justify acquisition / construction work for a historical + live pregame-predictable source.

A negative result means this exact V1 pairing-covariance mechanism does not justify acquisition by itself.

## Outcome

Target grain:

`game_id, possession_team`

Target:

`target_pressure_allowed_edge_rate`

For every exact blocker->defender assignment edge in the target offense-game, define a pressure-allowed event from the blocker PFF scouting labels:

- `pff_hitAllowed`
- `pff_hurryAllowed`
- `pff_sackAllowed`

An edge is positive when any of the three is explicitly positive. Rows where all three labels are unavailable are excluded from outcome scoring.

The target is the mean edge-event rate within the offense-game over scoreable assignment edges.

`pff_beatenByDefender` is diagnostic only and is not part of the frozen primary outcome.

## Strictly-prior player quality

Before each target week W, using only completed weeks < W:

For blocker i:

`blocker_prior_pressure_allow_rate_i = prior positive pressure-allowed assignment edges / prior scoreable assignment edges`

For defender j:

`defender_prior_pressure_generation_rate_j = prior positive pressure-allowed edges credited to blockers assigned to defender j / prior scoreable assignment edges involving defender j`

Minimum prior support for an edge to be scoreable:

- blocker >= **10** prior scoreable assignment edges
- defender >= **10** prior scoreable assignment edges

Same-week target outcomes never enter these histories.

No same-pair history is required.

## Realized-participation conditioning

Both baseline and candidate are conditioned on the **same realized target-game blocker and defender edge volumes**.

This is intentional. The experiment asks only whether knowing *which blocker is paired with which defender* adds information beyond knowing the target game's marginal blocker/defender participation exposure.

Because realized participation is hindsight information, neither arm is production-deployable. The only permitted interpretation is incremental value of exact pairing structure conditional on participation.

## Baseline features

For each target offense-game, among scoreable exact assignment edges:

1. `blocker_prior_allow_mean`  
   edge-weighted mean of the blocker strictly-prior pressure-allow rate.

2. `defender_prior_pressure_mean`  
   edge-weighted mean of the defender strictly-prior pressure-generation rate.

3. `scoreable_assignment_edge_count`

4. target week numeric.

No target-game assignment interaction statistic enters the baseline.

## Candidate feature

Candidate adds exactly one feature:

`assignment_pairing_covariance`

where:

`pair_product_mean = mean_edges(blocker_prior_allow_rate_i * defender_prior_pressure_generation_rate_j)`

and:

`assignment_pairing_covariance = pair_product_mean - blocker_prior_allow_mean * defender_prior_pressure_mean`

This statistic is zero when realized pairing carries no information beyond the marginal blocker and defender exposure distributions. Positive values mean higher-risk blockers are disproportionately paired with higher-pressure defenders; negative values mean the opposite.

No alternate matchup transform, nonlinear search, pair-history term, geometry term, block-type term, alignment term, or interaction family may be inspected in V1.

Frozen expected coefficient direction: **positive**.

## Temporal split

BDB2023 contains 2021 Weeks 1-8.

Primary fit:
- **Weeks 5-6**

Primary holdout:
- **Weeks 7-8**

Weeks 1-4 are history only.

No random split.

Weeks 7-8 outcomes remain unread until implementation and frozen feature construction are complete.

## Estimator

Ordinary least squares with intercept.

Baseline:
- the four frozen baseline features above.

Candidate:
- the same baseline plus `assignment_pairing_covariance`.

Numeric missingness is not imputed for primary rows. A target offense-game must satisfy the coverage rule below.

No Ridge, HGB, XGBoost, random forest, neural network, ensemble, hyperparameter search, or alternate link function.

## Evaluation coverage

For an offense-game to enter training/evaluation:

- at least **20** scoreable target assignment edges;
- at least **80%** of its target assignment edges with known pressure outcome must have both blocker and defender >=10 strictly-prior scoreable edges;
- finite target pressure-allowed edge rate;
- finite frozen features.

Primary holdout support gate:
- at least **50** offense-team-games in Weeks 7-8.

## Metrics

Report baseline and candidate:
- MAE
- RMSE
- signed bias
- p90 absolute error
- Pearson correlation when defined

Report:
- candidate minus baseline / gain convention explicitly
- candidate assignment covariance coefficient

## Bootstrap

Paired game-cluster bootstrap on the Weeks 7-8 holdout:
- cluster unit: `game_id`
- all two offense-team rows from a sampled game remain together
- 5,000 replicates
- seed `92028`
- statistic: mean baseline absolute error minus mean candidate absolute error

Report percentile 95% CI.

## Frozen primary gate

`BDB2023_EXACT_ASSIGNMENT_VOI_SIGNAL` requires every item:

1. holdout offense-team rows >= **50**
2. eligible holdout coverage over scheduled offense-team rows represented in the BDB slice >= **0.80**
3. candidate MAE < baseline MAE
4. bootstrap 95% CI lower bound for mean AE gain > **0**
5. candidate RMSE <= baseline RMSE
6. candidate p90 absolute error <= baseline p90 absolute error
7. assignment covariance coefficient > **0**

Failure of any gate:

`BDB2023_EXACT_ASSIGNMENT_VOI_NO_ACTIONABLE_SIGNAL_V1`

Integrity/leakage failure:

`BDB2023_EXACT_ASSIGNMENT_VOI_INTEGRITY_FAILURE`

## Interpretation boundary

Even a full PASS does **not** authorize:
- production integration;
- target-game realized assignment as a football predictor;
- an inferred/fake assignment proxy;
- a paid data purchase automatically;
- any sportsbook use.

A PASS authorizes only a separate acquisition/source program to seek a real historical + live assignment contract or a leakage-safe pregame assignment prediction mechanism.

## No-rescue rule

After outcome inspection do not change:
- Weeks 5-6 / 7-8 split;
- 10-edge prior support;
- 20-edge offense-game floor;
- 80% scoreable coverage floor;
- pressure outcome definition;
- pairing covariance formula;
- estimator;
- bootstrap unit/reps/seed;
- gate thresholds;
- favorable team/player/position subsets;
- block type or geometry additions.

A failed V1 closes this exact value-of-information mechanism.

## RB continuity note

RB remains an unresolved production/research priority outside its qualified scopes.

Separately, `RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED` is a real positive RB research result:
- all 28 gates passed;
- pooled CRPS improved 1.218%;
- high-difficulty-quartile CRPS improved 2.624%;
- player-clustered and crossed player x game bootstraps reported p=1.0;
- point MAE was unchanged by design.

That RB width result still requires a separately frozen forward/shadow confirmation before production. Weeks 2-18 RB rushing authority and RB receiving-yard mean remain unresolved. This BDB assignment experiment does not alter RB science.
