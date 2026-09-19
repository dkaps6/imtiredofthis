# Role/Room Concentration Opportunity Experiment V1

**Status:** FROZEN BEFORE OUTCOME INSPECTION  
**Frozen from branch:** `research-football-context-execution-v1` @ `f04511f140e470dfa18c2f2eabc00f951ee8da15`  
**Scope:** first predictive mechanism test for context signals that survived engineering qualification.  

## Scientific question

Does strict-prior position-room concentration add out-of-sample information about a player's next-game opportunity entitlement beyond the canonical pregame PlayerForm opportunity state already available to production?

This experiment tests opportunity first. It does **not** test yards, fantasy points, sportsbook lines, bet results, or target-game tracking.

## Qualified candidate family

The only V1 candidate fields are:

- `prior_rush_share_game_top1`
- `prior_rush_share_game_top2`
- `prior_tgt_share_game_top1`
- `prior_tgt_share_game_top2`

These fields reached this plan because the 2019–2025 historical qualification run established broad pregame coverage, stable identity, zero duplicate/fanout defects, strong strict-prior persistence, a direct opportunity mechanism, and holdout reconstructibility below the frozen 0.75 redundancy-review boundary.

Rolling 3/5 player-share fields are explicitly excluded because the outcome-free redundancy audit found them highly reconstructible from existing production opportunity state. Returning-opportunity-overlap fields are excluded from V1 because they require an event-specific qualification path.

## Hypotheses

### RB rushing entitlement

H1-RB: room top-1/top-2 rushing concentration contains incremental information about a running back's target-game rushing opportunity beyond canonical pregame player rush-share state.

### WR receiving entitlement

H1-WR: room top-1/top-2 target concentration contains incremental information about a wide receiver's target-game target opportunity beyond canonical pregame player target-share state.

### TE receiving entitlement

H1-TE: the same target-room concentration family may contain incremental information for tight-end target opportunity, but TE is a separately scored replication family and may not rescue WR or vice versa.

QB is excluded from V1.

## Historical source and temporal rules

Use the deterministic canonical historical player-game reconstruction already qualified for 2019–2025. Do not reacquire ordinary game logs.

All candidate and baseline inputs for target game `(season, week, player)` must be constructed strictly from information before that target game. Target-game outcomes may be read only by the evaluator after the feature matrix and frozen split assignments exist.

No sportsbook field may be read anywhere in this experiment.

## Cohorts

Score three independent families:

1. RB rushing opportunity: position `RB`; target-game `rush_share = rushes / team_rushes` when `team_rushes > 0`.
2. WR receiving opportunity: position `WR`; target-game `target_share = targets / team_targets` when `team_targets > 0`.
3. TE receiving opportunity: position `TE`; same target-share definition.

Rows require:
- stable player identity;
- canonical unique player-game key;
- known baseline production state;
- both relevant room concentration candidates known;
- target-game denominator > 0.

Do not impute unknown room concentration to zero.

## Fixed temporal split

- **Train:** 2019–2023
- **Primary holdout:** 2024
- **Temporal replication:** 2025

No random split is allowed.

## Baseline information set

The baseline is the exact outcome-free canonical opportunity-state representation used in the redundancy audit for the relevant domain:

- prior-season share;
- prior-season games;
- current-season strict-prior share;
- current-season games;
- PlayerForm-style prior/current blend.

The evaluator may include an intercept. No other new feature may be added in V1.

## Candidate model

Candidate = baseline information set plus exactly two room fields for the domain:

- rushing: room top-1 and top-2 strict-prior rush-share concentration;
- receiving: room top-1 and top-2 strict-prior target-share concentration.

Use the same deterministic linear estimator for baseline and candidate. Fit coefficients on 2019–2023 only. Do not tune regularization, transforms, interactions, thresholds, weights, or nonlinear forms after outcomes are inspected.

This is a mechanism test, not a claim that linear regression is the eventual production implementation.

## Metrics

For each family and each holdout season separately report:

- rows;
- MAE of opportunity share;
- RMSE;
- signed bias;
- median absolute error;
- p75 absolute error;
- p90 absolute error;
- Pearson correlation;
- Spearman correlation.

Also report pooled 2024–2025 metrics descriptively, but pooled results cannot rescue a failed annual gate.

## Frozen pass/fail gates

A family earns `MECHANISM_PASS_REPLICATION_REQUIRED` on 2024 only if all are true:

1. candidate MAE improves by at least **1.0% relative** to baseline;
2. candidate RMSE does not worsen by more than **0.25% relative**;
3. candidate p90 absolute error does not worsen by more than **1.0% relative**;
4. absolute signed bias does not worsen by more than **0.0025 opportunity-share points**;
5. candidate Pearson and Spearman correlations each do not fall by more than **0.005**;
6. eligible holdout rows >= **300** for that family.

A family reaches `MECHANISM_REPLICATED` only if 2025 independently satisfies:

1. candidate MAE is strictly better than baseline;
2. candidate RMSE does not worsen by more than **0.25% relative**;
3. candidate p90 absolute error does not worsen by more than **1.0% relative**;
4. absolute signed bias does not worsen by more than **0.0025**;
5. eligible replication rows >= **300**.

The 1% primary MAE threshold is intentionally material rather than accepting microscopic noise. Replication requires directional MAE improvement rather than another 1% threshold so a genuine but smaller second-year effect can validate the mechanism without post-hoc threshold relaxation.

## Transition diagnostic (secondary, non-rescuing)

After global family gates are scored, report metrics for rows tagged with a known player/room transition state, including the narrower joint player+room transition cohort where support permits.

These are diagnostics only. A transition subgroup may explain mechanism behavior but **cannot rescue a failed global family in V1** and cannot trigger production promotion by itself.

No subgroup threshold may be chosen after seeing outcomes.

## Family independence / multiplicity rule

RB, WR and TE are independent mechanism families. A pass in one does not authorize another.

Within a family, top-1 and top-2 are treated as one preregistered two-field candidate bundle. No choosing whichever field looked best after the test.

## No-retest rule

If a family fails its frozen gates, V1 is closed for that family. Do not retune coefficients, thresholds, windows, transformations, interactions, or cohort definitions and rerun under the V1 name.

A materially different hypothesis requires a new version with a new pre-outcome plan and a scientific reason beyond rescuing the failure.

## Production rule

`MECHANISM_REPLICATED` is necessary but not sufficient for production.

A replicated family must still undergo:

1. implementation design compatible with the actual production component;
2. leakage and identity QA in that implementation;
3. integration replay against existing production artifacts where possible;
4. full-stack regression/calibration checks;
5. existing repository promotion/governance rules.

No production promotion is authorized by this document alone.

## Hard prohibitions

- no sportsbook/odds input;
- no target-game outcome in feature construction;
- no target-game routes/tracking/snaps as pregame input;
- no Issue #535 changes;
- no failed-family rescue;
- no paid odds pull;
- no post-hoc feature selection;
- no yards test in V1;
- no production merge based only on subgroup improvement.

## Required evaluator artifact

One row per `family × evaluation_season × model` plus a gate-summary table containing:

- exact train/evaluation seasons;
- feature names;
- eligible row counts;
- all frozen metrics;
- relative deltas;
- each individual gate boolean;
- final family disposition;
- source commit and input hashes;
- explicit `sportsbook_read=false`.

## Frozen disposition

`ROLE_ROOM_CONCENTRATION_OPPORTUNITY_EXPERIMENT_V1_FROZEN_PRE_OUTCOME`
