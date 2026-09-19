# Event/Regime Reliability Experiment V1

**Status:** FROZEN BEFORE OUTCOME INSPECTION
**Frozen branch:** `research-football-context-event-redundancy-v1`
**Parent:** `a5e19a08de9ea48918e32ec15626cafdcf09a50d`

## Question
Do qualified strict-prior regime-change signals identify player-games where the canonical historical opportunity state is materially less reliable, without treating the event as a direct mean boost?

## Scientific boundary
This is a reliability / stale-history mechanism test. It does not authorize direct carry, target, yard, fantasy-point, price, or sportsbook adjustments. No sportsbook field may be read.

## Candidate families
Only event families already dispositioned `READY_FOR_FROZEN_EXPERIMENT` by the outcome-free event qualification may enter. Primary families are RB and WR joint player+room transition; supported target-room/rush-room churn families are separately scored replication families. Source-thin team-change and TE joint-transition signals cannot rescue a family.

## Baseline
Use the same canonical strict-prior PlayerForm-style opportunity representation used by the prior redundancy/mechanism work. Fit the baseline opportunity estimator on 2019-2023 only. No candidate event is allowed to alter the baseline mean prediction in V1.

## Outcomes
RB reliability is scored on next-game rush-share baseline residuals. WR reliability is scored on next-game target-share baseline residuals. Churn families use the matching opportunity domain. Outcomes are read only after cohort labels, split assignments, estimator, metrics and gates are fixed.

## Temporal split
- Train / baseline fit: 2019-2023
- Primary holdout: 2024
- Untouched temporal replication: 2025
No random split.

## Metrics
For event and non-event cohorts separately report: rows, MAE, RMSE, signed bias, median absolute error, p75 absolute error, p90 absolute error, p95 absolute error, residual standard deviation, and catastrophic-miss rate. Catastrophic miss is frozen as absolute residual >= the 90th percentile of TRAINING-period absolute residual for the matching position/domain baseline. The training threshold is then held fixed for 2024 and 2025.

Also report event/non-event ratios for MAE, RMSE, p90, p95, residual SD, and catastrophic-miss rate.

## Primary reliability gate (2024)
A family earns `RELIABILITY_SIGNAL_PRIMARY_PASS` only if:
1. event rows >= 50 in 2024;
2. event MAE is >= 5% higher than non-event MAE;
3. event RMSE is >= 5% higher than non-event RMSE;
4. event p90 absolute error is >= 5% higher than non-event p90;
5. event catastrophic-miss rate is >= 20% relatively higher than non-event catastrophic-miss rate;
6. absolute event signed bias is not used as a rescue criterion and must be reported;
7. no split/cohort definition is changed after outcome inspection.

## Temporal replication gate (2025)
A primary-passing family reaches `RELIABILITY_SIGNAL_REPLICATED` only if:
1. event rows >= 50 in 2025;
2. event MAE > non-event MAE;
3. event RMSE > non-event RMSE;
4. event catastrophic-miss rate > non-event catastrophic-miss rate;
5. at least two of MAE, RMSE, p90, p95, residual SD show >= 3% event/non-event degradation.

2025 cannot rescue a 2024 failure.

## Global safety / interpretation
This V1 is diagnostic: the candidate does not modify predictions, so there is no candidate global mean-regression gate. A replicated result only establishes that a pregame event identifies elevated baseline uncertainty. It does not authorize production by itself.

## Multiplicity
Each position/domain/event family is independent. A pass in one cannot authorize another. No choosing a preferred event after outcomes to create a composite V1 result.

## No-retest rule
A failed family closes under this V1 definition. Do not rescue it with different thresholds, event combinations, nonlinear models, alternative windows, selected seasons, target outcomes, or final-yard outcomes. A materially different mechanism requires a new pre-outcome plan.

## Production rule
`RELIABILITY_SIGNAL_REPLICATED` is necessary but not sufficient. Any production use must separately freeze an implementation hypothesis (for example uncertainty widening or confidence/abstention), pass leakage/identity QA, replay/integration tests, full-stack calibration/regression checks, and existing repository governance. No direct mean boost is authorized here.

## Hard prohibitions
- no sportsbook or odds input;
- no paid odds pull;
- no target-game features in pregame state;
- no Issue #535 changes;
- no rescue of `ROLE_ROOM_CONCENTRATION_OPPORTUNITY_V1`;
- no post-hoc cohort/threshold selection;
- no production merge from this diagnostic alone.

## Required artifact
One row per event family x evaluation season x cohort plus a gate summary with exact feature/event name, train/evaluation seasons, event/non-event rows, all frozen metrics and ratios, training catastrophic threshold, each gate boolean, final disposition, source commit/input hashes, and `sportsbook_read=false`.

**Frozen disposition:** `EVENT_REGIME_RELIABILITY_EXPERIMENT_V1_FROZEN_PRE_OUTCOME`
