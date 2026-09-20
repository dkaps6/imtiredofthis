# Defensive Front Pairwise Cohesion -> Pressure Generation Mechanism Experiment V1

**Status:** FROZEN PRE-RESULT  
**Qualified parent:** `DEFENSIVE_FRONT_PAIRWISE_COHESION_QUALIFICATION_V1`  
**Candidate:** `def_front_pairwise_cohesion_prior_share`  
**Target:** target-game team `pressure_rate_generated`  
**Production changes authorized:** false  
**QB/RB/WR/TE projection changes authorized:** false

## Scientific question

Does accumulated shared defensive-front roster history contain leakage-safe information
about a defense's next-game pressure generation beyond:

- immediate defensive-front roster continuity;
- the defense's own strictly prior completed-game state;
- the opponent offense's strictly prior completed-game protection/context state;
- defense identity, opponent identity, and target week?

This is an intermediate football-mechanism validation experiment. It is not a direct
QB or player-projection experiment.

## Anti-retest authorization

Repository history was audited before freezing this plan.

### Closed work that must remain closed

M77 exact-personnel discontinuity already tested last-game personnel turnover/addition,
replacement deficit, role delta, defensive rush pressure/sack quality and deltas, plus
selected discontinuity interactions directly as QB attempt/YPA/passing-yard corrections.
That family failed prospectively.

M80-M81 tested novel FTN tactical pressure/decision families, including explicit
blitz/pressure-response observables, and failed frozen development gates.

Generic aggregate defense pressure/mismatch as a direct QB feature was already explored
across earlier migrations; richer aggregate variants are closed.

### Why this experiment is distinct

The qualified candidate is a long-memory **relational unit state**: mean historical
co-roster experience across every unordered pair in the current broad defensive front,
using up to 20 strictly prior scheduled team-games.

It is not:
- a last-game turnover count;
- a starter replacement count;
- a player pressure-quality sum/delta;
- a blitz tendency;
- an aggregate pressure-rate remix;
- an exact blocker-rusher assignment.

The no-retest ledger explicitly requires a new mechanism beyond discontinuity counts to
reopen personnel work. This candidate satisfies that information requirement.

Using `pressure_rate_generated` as the **outcome of a mechanism check** does not reopen
aggregate pressure as a direct QB predictor. A failure here closes this mechanism; a
success only permits a later separately frozen integration question.

## Frozen source semantics

Reuse:

- canonical 2019-2025 regular-season schedule builder;
- nflverse weekly rosters;
- the mechanically corrected defensive-front identity contract from
  `DEFENSIVE_FRONT_PAIRWISE_COHESION_QUALIFICATION_V1`;
- `build_team_weekly_from_pbp`;
- the exact frozen defensive-front pairwise-cohesion materializer;
- the exact frozen immediate defensive-front continuity definition.

The semantic-GSIS quarantine remains mandatory:
a GSIS proven to map to multiple nonblank ESB IDs or multiple nonblank Smart IDs is
excluded from stable-ID use. No team/person selection is allowed.

Target `pressure_rate_generated` is the existing team-week PBP observation for the
defense. Target-game PBP is outcome-only and may not enter any predictor.

No sportsbook information is read.

## Cohort and chronology

Training seasons: **2019-2023**.

Primary holdout: **2024**.

Conditional replication: **2025**, exposed/scored only if every frozen 2024 primary gate
passes.

All predictors for a target game must be available before that game.

Current weekly roster is treated as the frozen pregame roster source. Team-state
predictors come only from the strictly prior scheduled regular-season game for the
relevant defense or opponent offense.

## Frozen baseline feature set

OLS with intercept.

### Defensive unit state

- `def_front_roster_continuity_share_prev_game`
- prior defense `pressure_rate_generated`
- prior defense `success_rate_def`
- prior defense `def_pass_epa`
- prior defense `explosive_play_rate_allowed`

### Opponent offensive state

From the opponent's strictly prior scheduled game:

- prior offense `pressure_rate_allowed`
- prior offense `success_rate_off`
- prior offense `dropback_rate`
- prior offense `plays_est`
- prior offense `proe`

### Fixed context controls

- target week numeric
- defense-team one-hot learned from 2019-2023 only
- opponent-team one-hot learned from 2019-2023 only

Numeric missingness uses 2019-2023 training medians only.

## Frozen candidate model

Exactly the baseline plus:

- `def_front_pairwise_cohesion_prior_share`

No interactions.

No alternate lookback.

No position-subset search.

No starter-only variant.

No hyperparameter search.

No model substitution.

Fit by ordinary least squares with intercept.

The 2019-2023 coefficients are frozen for 2024 and any conditionally exposed 2025
replication. Do not refit on 2024 before replication.

## Evaluation population

One row per scheduled regular-season defense team-game with:

- known target `pressure_rate_generated`;
- known pairwise cohesion;
- stable team/opponent keys.

Frozen support floors:

- 2024 primary evaluation rows >= **400**
- 2025 replication rows >= **400** if exposed
- target coverage within each evaluated season >= **0.80**

Unknown cohesion is excluded and reported, never imputed as zero.

## Metrics

For baseline and candidate:

- MAE
- RMSE
- p90 absolute error
- Pearson correlation
- bias

Incremental:

- row-level absolute-error gain = baseline absolute error - candidate absolute error;
- MAE gain;
- RMSE gain;
- p90 absolute-error gain;
- correlation gain;
- raw fitted OLS coefficient on cohesion.

Expected football direction: **positive**. More accumulated front cohesion should be
associated with higher pressure generation, conditional on the frozen baseline.

## Cluster bootstrap

Use **5,000** bootstrap replicates, seed **92027**.

Cluster by defense team within the evaluated season. Sample defense teams with
replacement and include all rows for each sampled team.

Report 2.5% and 97.5% quantiles of mean row-level absolute-error gain.

No alternate bootstrap is inspected in V1.

## Frozen 2024 primary gate

2024 passes only if all are true:

1. support/coverage floors pass;
2. candidate MAE < baseline MAE;
3. team-cluster bootstrap 95% CI lower bound for MAE gain > 0;
4. candidate RMSE <= baseline RMSE;
5. candidate p90 absolute error <= baseline p90 absolute error;
6. fitted cohesion coefficient > 0.

If any primary gate fails:

`DEFENSIVE_FRONT_PAIRWISE_COHESION_PRESSURE_MECHANISM_V1_FAILED_CLOSED_PRIMARY`

and 2025 is not scored.

## Frozen 2025 replication gate

Only after full 2024 passage, score 2025 with the already-fitted 2019-2023 models.

2025 passes only if all are true:

1. support/coverage floors pass;
2. candidate MAE < baseline MAE;
3. team-cluster bootstrap 95% CI lower bound for MAE gain > 0;
4. candidate RMSE <= baseline RMSE;
5. candidate p90 absolute error <= baseline p90 absolute error.

The coefficient is not refit; the 2019-2023 coefficient remains the direction authority.

If primary passes but replication fails:

`DEFENSIVE_FRONT_PAIRWISE_COHESION_PRESSURE_MECHANISM_V1_FAILED_CLOSED_REPLICATION`

If both pass:

`DEFENSIVE_FRONT_PAIRWISE_COHESION_PRESSURE_MECHANISM_V1_VALIDATED`

Validation still does not authorize a player projection or production change.

## Integrity gates

Must remain zero/false:

- target-game PBP entering a predictor;
- target-game snap/participation entering cohesion;
- future-week roster use;
- duplicate published team-week keys;
- feature/target join fanout;
- chronology violations for defense or opponent prior state;
- sportsbook read;
- production change;
- Issue #535 touch.

The frozen semantic-GSIS quarantine must remain active. Any same-person/multi-team
ambiguity that survives the quarantine remains a hard failure.

## No-rescue rule

After the first scored result do not:

- change the 20-game lookback;
- drop backups;
- switch to starter-only cohesion;
- change the broad front position set;
- alter the semantic-GSIS quarantine;
- add interactions;
- add player pressure-quality fields;
- add blocker-rusher assignment proxies;
- change the pressure target;
- change the baseline controls;
- change regression family;
- change bootstrap type/seed;
- lower support gates;
- weaken the all-gates primary/replication rule.

A failure is preserved and the mechanism closes.

## Pre-result disposition

`DEFENSIVE_FRONT_PAIRWISE_COHESION_PRESSURE_MECHANISM_EXPERIMENT_V1_FROZEN_PRE_RESULT`
