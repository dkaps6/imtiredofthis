# DEFENSIVE FRONT PAIRWISE COHESION -> PRESSURE GENERATION MECHANISM V1 — RESULT — 2026-09-20

## Final disposition

`DEFENSIVE_FRONT_PAIRWISE_COHESION_PRESSURE_MECHANISM_V1_FAILED_CLOSED_PRIMARY`

The frozen 2024 primary gate failed. The physically separate 2025 replication outcome
file was not opened, scored, or hashed by the experiment.

Do not rescue this mechanism.

## Frozen plan

`docs/research/DEFENSIVE_FRONT_PAIRWISE_COHESION_PRESSURE_MECHANISM_EXPERIMENT_V1.md`

The experiment was frozen before target outcomes were inspected.

Qualified parent:

`DEFENSIVE_FRONT_PAIRWISE_COHESION_QUALIFICATION_V1`

Candidate:

`def_front_pairwise_cohesion_prior_share`

Target:

target-game team `pressure_rate_generated`

## Anti-retest basis

This experiment was authorized only as an intermediate mechanism test because accumulated
pairwise front experience was new information beyond:

- M77 last-game personnel discontinuity / role / pressure-quality deltas;
- M80-M81 tactical pressure/blitz observables;
- closed aggregate pressure as a direct QB feature.

The experiment did not reopen a direct QB or player-projection test.

## Canonical execution

- implementation/workflow commit: `21fd6ced6e52fdfff379c1f5eeda55b6e7874980`
- run: `35517405126`
- job: `106095556525`
- artifact: `10607820423`
- artifact digest: `sha256:232758a13a42e4f05571daf8768e2bebd62c5382a902d6d4bebcccd19b183c40`
- frozen schedule SHA256: `60db4d57a7132b4f00d7f51996dab19b4d171e8e90393f3f95c8fa8b19b14f04`
- frozen weekly-roster SHA256: `f2b791d47b146fe703a73d3111d609504779c7e9d2dc0ad47b3bd1996776f18a`

Focused tests passed before the scored run.

## Frozen model

Training: 2019-2023.

Primary holdout: 2024.

Baseline OLS included:

- immediate defensive-front roster continuity;
- prior defense pressure generated;
- prior defense success rate;
- prior defense pass EPA;
- prior defense explosive-play rate allowed;
- opponent prior pressure allowed;
- opponent prior offensive success rate;
- opponent prior dropback rate;
- opponent prior plays;
- opponent prior PROE;
- target week;
- defense-team one-hot;
- opponent-team one-hot.

Candidate model added only:

`def_front_pairwise_cohesion_prior_share`

No interaction search, model substitution, lookback change, starter subset, or threshold
change was permitted.

## Fit

2019-2023 train rows: **2,622**

Raw fitted cohesion coefficient:

**-0.0006477229**

Frozen expected direction: **positive**.

The fitted sign therefore failed the mechanism-direction gate before any replication.

## 2024 primary

- scheduled team-games: **544**
- scored rows: **544**
- scoring coverage: **100.0%**
- defense-team bootstrap clusters: **32**
- bootstrap replicates: **5,000**
- bootstrap seed: **92027**

### Error metrics

| Metric | Baseline | Candidate | Gain |
|---|---:|---:|---:|
| MAE | 0.058079 | 0.058080 | **-0.000001** |
| RMSE | 0.071599 | 0.071602 | **-0.000003** |
| p90 absolute error | 0.116396 | 0.116437 | **-0.000041** |
| Pearson correlation | 0.161363 | 0.161313 | **-0.000050** |

Bias:

- baseline: +0.001722
- candidate: +0.001756

Mean absolute-error gain team-cluster bootstrap 95% CI:

**[-0.000011, +0.000007]**

The interval crosses zero and is centered essentially at no effect.

## Frozen primary gates

Passed:

- support rows >=400
- scoring coverage >=0.80

Failed:

- candidate MAE lower than baseline
- bootstrap CI lower bound >0
- RMSE nonincrease
- p90 absolute-error nonincrease
- cohesion coefficient >0

Therefore:

`PRIMARY_PASS=False`

and:

`REPLICATION_EXPOSED=False`

## 2025 seal

The experiment manifest records:

`team_weekly_replication_sha256 = NOT_READ_NOT_HASHED`

No 2025 result exists for this experiment and none should be inferred.

## Identity and integrity

The mechanically corrected identity contract remained active:

- semantic GSIS collision IDs detected: 14
- front-roster rows quarantined: 136
- stable-ID coverage: 99.8141%
- same-week GSIS/team ambiguity: 0

All experiment integrity checks were clean:

- chronology violations: 0
- opponent prior-state chronology violations: 0
- missing opponent schedule keys: 0
- feature/target join fanout: 0
- duplicate published team-week rows: 0
- target-game PBP used as predictor: false
- target-game PBP used in cohesion: false
- sportsbook read: false
- production changed: false
- Issue #535 touched: false
- player projection changed: false

## Scientific interpretation

Defensive-front pairwise roster cohesion is a real, stable, incremental **descriptive
context state**, but this frozen test found no evidence that it improves next-game
pressure generation beyond immediate continuity, prior defense state, opponent offense
state, and team/opponent controls.

The result is not borderline under the frozen rules: all four performance/direction
gates beyond support failed, and the fitted coefficient had the wrong sign.

## No-rescue closure

Do not rerun this family with:

- alternate cohesion lookbacks;
- starter-only/front-position subsets;
- a different pressure target definition;
- player pressure-quality additions;
- interactions;
- different regression families;
- different bootstrap seeds/types;
- favorable team/season subsets;
- 2025 exposure.

Any future defensive-front research must use materially different information or a
different football mechanism, not a retuned version of pairwise-cohesion -> pressure.
