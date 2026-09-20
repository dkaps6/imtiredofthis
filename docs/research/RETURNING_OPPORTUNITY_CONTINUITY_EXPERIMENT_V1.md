# Returning Opportunity Continuity Experiment V1

**Status:** FROZEN BEFORE OUTCOME INSPECTION
**Parent result:** `EVENT_REGIME_RELIABILITY_EXPERIMENT_V1` closed with no replicated family.

## Distinct question

Does strict-prior returning-opportunity overlap improve next-game opportunity entitlement beyond the canonical PlayerForm-style baseline as a continuous personnel-continuity signal?

This is not a rescue of event-regime reliability V1 and does not use binary transition/churn labels. It tests a different mechanism: how much of the prior opportunity pool is represented by returning participants may change the expected distribution of opportunity before the next game.

## Candidate families

Only continuous descriptors already dispositioned `READY_FOR_FROZEN_EXPERIMENT` by the outcome-free qualification may enter:

- RB rush: `prior_rush_share_game_returning_overlap`
- RB target: `prior_tgt_share_game_returning_overlap`
- WR target: `prior_tgt_share_game_returning_overlap`
- TE target: `prior_tgt_share_game_returning_overlap`

QB returning-rush overlap is descriptive replication only and cannot rescue a primary family.

## Baseline and candidate

Fit the same canonical strict-prior PlayerForm-style opportunity baseline on 2019–2023 only. Candidate adds exactly the matching continuous returning-opportunity-overlap descriptor as one additive linear term. No binary churn/transition flags, room concentration, sportsbook fields, target-game state, nonlinear transforms, interactions, thresholding, or post-hoc feature selection are allowed.

## Outcomes

- RB rush family: next-game rush share.
- RB/WR/TE target families: next-game target share.

## Temporal split

- Fit: 2019–2023
- Primary holdout: 2024
- Untouched replication: 2025

No random split. 2025 cannot rescue a 2024 failure.

## Metrics

For baseline and candidate report rows, MAE, RMSE, signed bias, p90 absolute error and p95 absolute error. Report candidate-vs-baseline relative deltas using `(candidate - baseline) / baseline`; negative is improvement.

## 2024 primary gate

A family passes only if all are true:

1. evaluation rows >= 500;
2. candidate MAE improves by at least 1.0%;
3. candidate RMSE does not worsen by more than 0.5%;
4. candidate p90 absolute error does not worsen by more than 1.0%;
5. absolute signed bias does not worsen by more than 0.002 opportunity-share units;
6. no cohort, estimator, feature, threshold or metric definition changes after outcome inspection.

## 2025 replication gate

A 2024-passing family reaches `CONTINUITY_SIGNAL_REPLICATED` only if:

1. evaluation rows >= 500;
2. candidate MAE is better than baseline;
3. candidate RMSE is no worse than baseline by more than 0.5%;
4. candidate p90 is no worse than baseline by more than 1.0%;
5. absolute signed bias is no worse by more than 0.002.

2025 is inspected only for a 2024-primary pass and cannot rescue failure.

## No-retest rule

A failed family closes under V1. Do not rescue with overlap cutoffs, selected seasons, nonlinear models, interactions, event combinations, alternative windows, yards/fantasy outcomes, or relaxed gates. Any materially different mechanism requires a new frozen plan.

## Production rule

Replication is necessary but not sufficient. A replicated family must separately pass leakage/identity QA, integration/replay, global and subgroup calibration/regression checks, and existing repository promotion governance before production use.

## Prohibitions

No sportsbook/odds input or paid odds pull. No Issue #535 changes. No target-game information. No rescue of either failed room-concentration V1 or event-regime reliability V1.

**Frozen disposition:** `RETURNING_OPPORTUNITY_CONTINUITY_EXPERIMENT_V1_FROZEN_PRE_OUTCOME`
