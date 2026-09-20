# Historical Analog State Experiment V1

**Status:** `HISTORICAL_ANALOG_STATE_EXPERIMENT_V1_FROZEN_PRE_OUTCOME`

**Parent qualification:** `HISTORICAL_ANALOG_STATE_QUALIFICATION_V1`

**Freeze parent:** `4d8d3f2ef0466f0c78efd8a86f20a3e26989680b`

## Scientific question

Does strict-prior historical analog support identify stale-history / reliability regimes in the existing PlayerForm opportunity forecasts, without changing the production mean model?

This V1 tests reliability conditioning only. It does not authorize production integration and does not test a direct analog-derived mean correction.

## Frozen candidate family

To avoid post-outcome selection among the 16 qualified descriptors, V1 uses exactly one preregistered composite mechanism across each position/domain: **analog novelty risk**.

For every eligible target player-game with valid analog support, construct a risk score from the already-qualified, outcome-free analog state using only:

1. `mean_k_distance` — larger means less historically comparable;
2. `effective_analog_support` — smaller means less historical support.

Within each position, transform each component to an expanding strict-prior percentile using only rows with kickoff strictly earlier than the target kickoff. Define:

`analog_novelty_risk = 0.5 * pct(mean_k_distance) + 0.5 * (1 - pct(effective_analog_support))`

No target outcome may influence percentile construction. `NO_ANALOG_SUPPORT` rows are not silently scored; they remain an explicit abstention state and are reported descriptively, not used to tune thresholds.

The high-novelty event cohort is frozen as `analog_novelty_risk >= 0.80`. The comparison cohort is `analog_novelty_risk < 0.80`. No alternate cutoff may be inspected in V1.

## Outcome / baseline definitions

Use the existing canonical PlayerForm-style opportunity forecast and the corresponding realized opportunity outcome already used by the repository's historical opportunity evaluators:

- RB rushing: predicted rush opportunity vs realized rush attempts;
- RB/WR/TE receiving: predicted target opportunity vs realized targets;
- QB only if the canonical historical evaluator already exposes an equivalent opportunity forecast/outcome pair without introducing a new model. Otherwise QB remains qualification-only in V1.

No sportsbook, odds, EV, final-score conditioning, or target-game feature is allowed.

For each row define signed error `prediction - actual`, absolute error, squared error, and catastrophic miss using the same position/domain catastrophic-miss definition already frozen by the repository's reliability experiments. Do not invent a new catastrophic threshold for this experiment.

## Frozen temporal protocol

- Fit / historical state: 2019-2023 only where a fitted component is required.
- Primary holdout: 2024.
- Temporal replication: 2025, inspected only for a family that passes every 2024 primary gate.
- 2025 may validate a 2024 pass and may never rescue a 2024 failure.

No retest, cutoff change, descriptor substitution, subgroup rescue, or post-hoc family combination is allowed after 2024 outcomes are inspected.

## Primary hypothesis

High analog novelty marks games where existing historical PlayerForm is stale or less reliable. Therefore forecast error dispersion/tails should be worse in the high-novelty cohort than in the comparison cohort.

For each eligible position/domain family, report high-novelty and comparison cohort row counts plus:

- MAE;
- RMSE;
- p90 absolute error;
- mean signed error / bias;
- catastrophic-miss frequency.

## Frozen 2024 pass gate

A family passes the primary reliability gate only if all of the following hold in 2024:

1. high-novelty cohort has at least 100 rows and comparison cohort has at least 250 rows;
2. high-novelty MAE is at least 3% worse than comparison MAE;
3. at least two of the other three dispersion/tail metrics — RMSE, p90 absolute error, catastrophic-miss frequency — are at least 3% worse in high novelty;
4. no integrity, chronology, identity, or target-leakage violation exists.

Bias is diagnostic and is not a standalone pass gate.

A 2024 failure is final for that family under V1: `FAILED_CLOSED_PRIMARY`.

## Frozen 2025 replication gate

Only a 2024-primary pass may expose its 2025 result. It reaches `ANALOG_RELIABILITY_SIGNAL_REPLICATED` only if all of the following hold in 2025:

1. high-novelty cohort has at least 75 rows and comparison cohort at least 200 rows;
2. high-novelty MAE is at least 3% worse than comparison MAE;
3. at least two of RMSE, p90 absolute error, catastrophic-miss frequency are at least 3% worse in high novelty;
4. the direction of the MAE effect matches 2024;
5. all leakage/integrity gates remain clean.

Otherwise disposition is `FAILED_CLOSED_REPLICATION`. 2025 cannot trigger threshold or feature changes.

## Multiplicity / selection rule

The composite and 0.80 event threshold are fixed globally before outcome inspection. Position/domain families are evaluated independently. A pass in one family does not authorize borrowing its threshold, effect size, or outcome information to modify another family.

The 16 individual qualified analog descriptors remain qualification evidence only in V1; their outcome associations may not be ranked to choose a winner.

## Production rule

`ANALOG_RELIABILITY_SIGNAL_REPLICATED` is research evidence, not production authority. Before any operational use, freeze a separate implementation hypothesis and require:

- leakage and identity QA;
- no-credit/offline replay where possible;
- full-stack calibration and regression checks;
- existing repository promotion governance;
- no unacceptable regressions.

Production mean forecasts remain unchanged throughout this experiment.

## Hard prohibitions

- no paid odds/OddsAPI pull;
- no sportsbook-derived teacher or feature;
- no Issue #535 changes;
- no direct mean correction in V1;
- no outcome-driven descriptor selection;
- no alternative novelty threshold after outcome inspection;
- no failed-family rescue by combination with concentration, transition/churn, returning continuity, or other previously failed V1 mechanisms;
- no production merge from this experiment alone.

## Immediate execution task

Implement this exact frozen evaluator against the already-materialized canonical historical forecasts, outcomes, and qualified analog-state artifact. Lock chronology, percentile construction, cohort threshold, metrics, row floors, and conditional 2025 exposure in tests before executing the 2024 primary holdout.
