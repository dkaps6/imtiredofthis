# OL Pairwise Cohesion -> Pressure Mechanism Experiment V1

**Status:** FROZEN PRE-RESULT  
**Qualified parent:** `OL_ROSTER_PAIRWISE_COHESION_QUALIFICATION_V1`  
**Candidate:** `ol_roster_pairwise_cohesion_prior_share`  
**Target:** target-game team `pressure_rate_allowed`  
**Production changes authorized:** false  
**QB/RB/WR/TE projection changes authorized:** false

## Scientific question

Does accumulated shared OL roster history contain leakage-safe information about a
team's next-game pass-protection pressure environment beyond:

- the already-qualified immediate OL roster-continuity share;
- prior completed team-game pressure allowed;
- other frozen prior team state;
- team identity and target week?

This is a mechanism-validation experiment, not a QB mean experiment.

## Why this is distinct from closed work

M77 tested exact personnel-discontinuity features directly as QB attempt/YPA/passing-yard
corrections and failed.

M71 tested QB efficiency uncertainty/risk and failed with the then-available pregame
information.

This experiment does neither. It tests a newly qualified accumulated pairwise-cohesion
state against the intermediate football mechanism it is supposed to describe:
target-game pressure allowed.

Repository search found no prior pairwise-cohesion pressure-target experiment.

Generic aggregate pressure as a direct QB feature remains closed and is not reopened here.

## Frozen source semantics

Reuse:

- canonical 2019–2025 regular-season schedule builder;
- nflverse weekly rosters;
- `build_team_weekly_from_pbp`;
- the exact frozen pairwise-cohesion materializer;
- the exact frozen immediate OL roster-continuity materializer.

Target `pressure_rate_allowed` is the existing team-week PBP observation:

`mean(sack == 1 OR qb_hit == 1)` over team offensive dropbacks.

It is an outcome used only after this experiment plan is frozen.

No sportsbook information is read.

## Cohort and chronology

Training seasons: **2019–2023**.

Primary holdout: **2024**.

Conditional replication: **2025**, exposed/scored only if the full frozen 2024
primary gate passes.

All predictors for a target team-game must be available before that game.

Current weekly roster is a pregame source. Every team-state predictor is from the
same team's prior completed scheduled regular-season game.

## Frozen baseline feature set

Baseline OLS with intercept:

- `ol_roster_continuity_share_prev_game`
- prior `pressure_rate_allowed`
- prior `success_rate_off`
- prior `dropback_rate`
- prior `plays_est`
- prior `proe`
- target week numeric
- team one-hot learned from 2019–2023 only

Numeric missingness uses 2019–2023 training medians only.

## Frozen candidate model

Exactly the baseline feature set plus:

- `ol_roster_pairwise_cohesion_prior_share`

No interactions.

No alternate lookback.

No feature subset search.

No hyperparameter search.

No model substitution.

Fit by ordinary linear least squares with intercept.

The 2019–2023 coefficients are frozen for both 2024 and any conditionally exposed
2025 replication. Do not refit on 2024 before replication.

## Evaluation population

One row per scheduled regular-season team-game with:

- known target `pressure_rate_allowed`;
- known pairwise cohesion;
- stable team key.

Frozen support floors:

- 2024 primary evaluation rows >= **400**
- 2025 replication rows >= **400** if exposed
- target coverage within each evaluated season >= **0.80**

Unknown cohesion is excluded from scoring and reported, never imputed as zero.

## Metrics

For baseline and candidate:

- MAE
- RMSE
- p90 absolute error
- Pearson correlation
- bias

Incremental metrics:

- row-level absolute-error gain = baseline abs error - candidate abs error;
- MAE gain;
- RMSE gain;
- p90 absolute-error gain;
- correlation gain;
- fitted standardized? **No**. Report the raw OLS coefficient on cohesion from the
  candidate model; expected football direction is **negative**.

## Cluster bootstrap

Use **5,000** bootstrap replicates, seed **92026**.

Cluster by team within the evaluated season. Sample teams with replacement and include
all rows belonging to each sampled team.

Report the 2.5% and 97.5% quantiles of mean row-level absolute-error gain.

No alternate bootstrap is inspected in V1.

## Frozen 2024 primary gate

2024 passes only if all are true:

1. support/coverage floors pass;
2. candidate MAE is lower than baseline MAE;
3. team-cluster bootstrap 95% CI lower bound for MAE gain is **> 0**;
4. candidate RMSE <= baseline RMSE;
5. candidate p90 absolute error <= baseline p90 absolute error;
6. fitted cohesion coefficient is **< 0**.

If any primary gate fails:

`OL_PAIRWISE_COHESION_PRESSURE_MECHANISM_V1_FAILED_CLOSED_PRIMARY`

and 2025 is not scored.

## Frozen 2025 replication gate

Only after full 2024 passage, score 2025 using the already-fitted 2019–2023 models.

2025 passes only if all are true:

1. support/coverage floors pass;
2. candidate MAE is lower than baseline MAE;
3. team-cluster bootstrap 95% CI lower bound for MAE gain is **> 0**;
4. candidate RMSE <= baseline RMSE;
5. candidate p90 absolute error <= baseline p90 absolute error.

The coefficient direction is not refit in 2025; the 2019–2023 fitted coefficient
remains the mechanism-direction authority.

If primary passes but replication fails:

`OL_PAIRWISE_COHESION_PRESSURE_MECHANISM_V1_FAILED_CLOSED_REPLICATION`

If both pass:

`OL_PAIRWISE_COHESION_PRESSURE_MECHANISM_V1_VALIDATED`

Validation still does **not** authorize a player projection or production change.

## Integrity gates

Must remain zero/false:

- target-game PBP entering any predictor;
- target-game snap/participation entering cohesion;
- future-week roster use;
- duplicate published team-week keys;
- schedule/cohesion/target join fanout;
- sportsbook read;
- production change;
- Issue #535 touch.

## No-rescue rule

After the first scored result do not:

- change the 20-game cohesion lookback;
- drop backups;
- switch to starter-only cohesion;
- add interactions;
- add opponent pass-rush fields;
- change the pressure target;
- change the regression family;
- change bootstrap type/seed;
- lower support gates;
- weaken the all-gates primary/replication rule.

A failure is preserved and the mechanism closes.

## Pre-result disposition

`OL_PAIRWISE_COHESION_PRESSURE_MECHANISM_EXPERIMENT_V1_FROZEN_PRE_RESULT`
