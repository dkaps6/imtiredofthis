# OL Pairwise Cohesion -> Pressure Mechanism Experiment V1 — Result — 2026-09-20

**Status:** FAILED CLOSED — PRIMARY HOLDOUT  
**Frozen plan:** `docs/research/OL_PAIRWISE_COHESION_PRESSURE_MECHANISM_EXPERIMENT_V1.md`  
**Execution commit:** `ba1fc29a478461080836a8d6c1bc610b7384c945`  
**Production changes authorized:** false

## Canonical execution

- workflow: `OL Pairwise Cohesion Pressure Mechanism V1`
- run: `35515454686`
- job: `106090515886`
- artifact: `10607110400`
- artifact digest: `sha256:90d3021fb88169b95f7a640967ace4449d01510792c109032251dad53ba4fb2b`
- conclusion: success

## Frozen design

- train: 2019–2023
- primary holdout: 2024
- conditional replication: 2025
- baseline: immediate OL continuity + prior team state + team + week
- candidate: baseline + `ol_roster_pairwise_cohesion_prior_share`
- model: fixed OLS
- bootstrap: 5,000 team-cluster replicates, seed 92026
- target: team target-game `pressure_rate_allowed`

2025 outcomes were stored separately and were authorized to be opened only after a full
2024 primary passage.

## Fit

- training rows: **2,622**
- raw fitted cohesion coefficient: **-0.0170807593**

The fitted sign matched the football hypothesis: more accumulated OL cohesion was
associated with lower predicted pressure allowed, conditional on the frozen baseline.

## 2024 primary result

- scheduled team-games: **544**
- scored rows: **544**
- scoring coverage: **1.0000**
- baseline MAE: **0.057509**
- candidate MAE: **0.057436**
- MAE gain: **+0.000074**
- baseline RMSE: **0.071154**
- candidate RMSE: **0.071023**
- RMSE gain: **+0.000131**
- baseline p90 absolute error: **0.113903**
- candidate p90 absolute error: **0.114629**
- p90 gain: **-0.000727** (worse)
- baseline correlation: **0.158744**
- candidate correlation: **0.166074**
- correlation gain: **+0.007329**
- baseline bias: **+0.001695**
- candidate bias: **+0.002583**
- bootstrap 95% CI for mean absolute-error gain: **[-0.000113, +0.000269]**
- team clusters: **32**

## Frozen gate

Passed:

- support rows >=400
- scoring coverage >=0.80
- candidate MAE lower
- RMSE nonincrease
- fitted cohesion coefficient <0

Failed:

- team-cluster bootstrap 95% CI lower bound >0
- p90 absolute error nonincrease

Therefore:

`PRIMARY_PASS = false`

`REPLICATION_EXPOSED = false`

`FINAL_DISPOSITION = OL_PAIRWISE_COHESION_PRESSURE_MECHANISM_V1_FAILED_CLOSED_PRIMARY`

## 2025 seal

2025 was **not scored or opened by the experiment script**.

Manifest authority:

`team_weekly_replication_sha256 = NOT_READ_NOT_HASHED`

This preserves the conditional replication holdout.

## Integrity

- stable-ID coverage: **0.9999086440956679**
- ambiguous same-week GSIS/team conflicts: **0**
- chronology violations: **0**
- immediate-continuity chronology violations: **0**
- prior-team-state chronology violations: **0**
- duplicate published team-week rows: **0**
- schedule join fanout: **0**
- feature join fanout: **0**
- target/prior-state join fanout: **0**
- future-week roster used: false
- target-game PBP used as predictor: false
- target-game PBP used in cohesion: false
- target-game snap/participation used: false
- sportsbook read: false
- production changed: false
- player projection changed: false
- Issue #535 touched: false

## Input fingerprints

- schedule SHA256: `60db4d57a7132b4f00d7f51996dab19b4d171e8e90393f3f95c8fa8b19b14f04`
- weekly roster SHA256: `f2b791d47b146fe703a73d3111d609504779c7e9d2dc0ad47b3bd1996776f18a`
- primary team-weekly SHA256: `6f26f9bf935ef51ff1b86f10edf71bf70b5eaaf970997fa62160ea1fa94f494e`
- replication team-weekly: `NOT_READ_NOT_HASHED`

## Interpretation

The already-qualified pairwise cohesion state remains real, stable and incrementally
nonredundant context information. This experiment does **not** validate it as a
target-game pressure predictor under the frozen V1 gates.

The tiny mean-error improvements are not statistically reliable under the frozen
team-cluster bootstrap, and the p90 tail worsened. The correct disposition is failure,
not a threshold/model/subgroup rescue.

Do not:

- expose 2025 for this V1;
- change the 20-game lookback;
- switch to starter-only cohesion;
- change the pressure target;
- add interactions or opponent pass-rush fields;
- reuse the result as a QB mean or uncertainty adjustment.

## Final disposition

`OL_PAIRWISE_COHESION_PRESSURE_MECHANISM_V1_FAILED_CLOSED_PRIMARY`
