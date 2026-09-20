# NFL HANDOFF — 2026-09-20 — FOOTBALL CONTEXT / OL PAIRWISE COHESION CURRENT

**Repository:** `dkaps6/imtiredofthis`  
**GitHub is canonical; chat memory is secondary.**  
**Active branch:** `research-football-context-event-redundancy-v1`  
**Production main remains:** `f0dad2c6711e85104eeffedfa5f5112fd172cbf5`

## Newly qualified OL roster continuity V1

Canonical execution:

- implementation commit: `d5ce896c4ff34fe48e76b6bcb8f87c29b55e9848`
- run: `35514546250`
- job: `106088156234`
- artifact: `10606771062`
- digest: `sha256:e7688b9abd6ac683ca19674f412b11f18dc77109e785d15420d60609c03d33ee`

Result:

- 3,742 eligible scheduled team-games
- 3,518 known
- 94.0139% broad coverage
- 99.9909% stable-ID coverage
- 0 duplicate/fanout/chronology violations
- holdout reconstructibility R2 0.005216
- qualification: `READY_FOR_FROZEN_EXPERIMENT`

## Anti-retest decision

Do **not** immediately run a QB predictive experiment with
`ol_roster_continuity_share_prev_game`.

M77 already tested exact personnel-discontinuity counts/role changes for QB attempts/YPA/passing yards and failed. M71 already closed the QB efficiency-volatility/risk family. The V1 roster-overlap feature is excellent new engineering information, but a QB use would still be too close to the closed discontinuity-count family under the repository's no-retest rule.

Current authorization state:

`PREDICTIVE_AUTHORIZATION_WITHHELD_QB_ANTI_RETEST`

No new QB/RB predictive outcome has been inspected.

Result authority:

`docs/research/OL_ROSTER_CONTINUITY_QUALIFICATION_V1_RESULT_2026-09-20.md`

## Active next mechanism

Frozen outcome-free plan:

`docs/research/OL_ROSTER_PAIRWISE_COHESION_QUALIFICATION_V1.md`

Candidate:

`ol_roster_pairwise_cohesion_prior_share`

Mechanism:

Accumulated shared historical OL roster experience among the current OL group across up to 20 strictly prior scheduled team-games. This is not last-game turnover and is explicitly designed to test a mechanism beyond discontinuity counts.

Qualification remains outcome-free.

## Exact next task

Implement the frozen pairwise-cohesion qualifier, focused tests and an isolated Actions workflow.

Required evidence:

- broad coverage
- stable identity
- chronology / duplicate / fanout integrity
- current-pair support diagnostics
- adjacent-game stability (hard gate: Spearman >=0.50 with >=500 pairs)
- redundancy against immediate OL continuity plus prior team state
- sanitized artifact only
- no predictive outcomes

If it qualifies, perform another anti-retest authorization audit before freezing any predictive plan.

## Closed lineages to preserve

- `ROLE_ROOM_CONCENTRATION_OPPORTUNITY_V1`
- `EVENT_REGIME_RELIABILITY_EXPERIMENT_V1`
- `RETURNING_OPPORTUNITY_CONTINUITY_EXPERIMENT_V1`
- `HISTORICAL_ANALOG_STATE_EXPERIMENT_V1`
- `BDB2026_RECEIVER_RELEASE_GEOMETRY_QUALIFICATION_V1`
- `BDB2023_BLOCKER_PROTECTION_GEOMETRY_QUALIFICATION_V1`
- M71 QB efficiency uncertainty/risk
- M77 exact personnel discontinuity predictive correction

## Boundaries

- no production change
- no paid odds pull
- sportsbook remains downstream only
- RB predictive research remains pinned/paused
- do not touch Issue #535 / the separate WR lane
- preserve failed experiments; no threshold rescue


## Frozen mechanism experiment — next authorized execution

Pairwise cohesion qualified cleanly on the first frozen run:

- implementation commit `638f7cc59f97e007d05407ff68172454754c203d`
- run `35515200623`
- job `106089849313`
- artifact `10606781837`
- digest `sha256:3ecd4ddea00e46686bdd00e181b20c6dea6fdd7c8e1cc9d3e8b17cf0071d8bf1`
- broad coverage **99.1448%**
- stability Spearman **0.822048** on **3,678** adjacent pairs
- redundancy holdout R2 **0.425764** even after immediate OL continuity + prior team state
- qualification `READY_FOR_FROZEN_EXPERIMENT`

The next authorized experiment is deliberately an intermediate-mechanism test, not
a QB mean test:

`docs/research/OL_PAIRWISE_COHESION_PRESSURE_MECHANISM_EXPERIMENT_V1.md`

It asks whether accumulated OL cohesion improves target-game team pressure-rate
prediction beyond prior pressure/team state and immediate OL continuity.

2024 is primary; 2025 is conditionally exposed only after full 2024 passage.


## OL cohesion pressure mechanism result

Canonical run:

- run `35515454686`
- job `106090515886`
- artifact `10607110400`
- digest `sha256:90d3021fb88169b95f7a640967ace4449d01510792c109032251dad53ba4fb2b`

Disposition:

`OL_PAIRWISE_COHESION_PRESSURE_MECHANISM_V1_FAILED_CLOSED_PRIMARY`

2024 had a tiny MAE/RMSE improvement and negative cohesion coefficient, but the frozen
team-cluster bootstrap CI crossed zero and p90 error worsened. 2025 remained sealed and
was not scored.

Do not rescue this mechanism.

## Next outcome-free mechanism

Frozen plan:

`docs/research/DEFENSIVE_FRONT_PAIRWISE_COHESION_QUALIFICATION_V1.md`

The next task is qualification only. It tests accumulated front-seven shared roster
history, not another OL variant and not a QB outcome.


## Defensive front pairwise cohesion — identity audit and recovered qualification

The first frozen qualification run was correctly rejected on the identity gate:

- original implementation: `dc34825ebd0c7a2c37bf530bf369d5c92fd32ba4`
- original run: `35515764090`
- original job: `106091315051`
- original artifact: `10606857557`
- original digest: `sha256:c3a5d19b130569f30f85068fa4da5562aee5630e8ab2c1728b30eb4f53657c65`
- ambiguous same-week GSIS/team groups: **13**
- disposition: `REJECTED_INTEGRITY`

A source-only forensic audit reproduced the exact frozen weekly-roster hash and showed
all 13 groups were one GSIS (`00-0035718`) incorrectly shared by two different people:
Quinnen Williams (NYJ) and Isaiah Searight (NYG). Their upstream ESB and Smart IDs differ,
so this is a person-identity collision, not an in-week team transaction.

Enhanced audit:

- commit: `c63bbbbf5caf593bcc4d21bbeb5f5d2192b8cc31`
- run: `35516818596`
- job: `106094055976`
- artifact: `10606947000`
- digest: `sha256:ef8b0d3f164e1ebaf7052602da4b9acb261cac9f664bac3a684ac4e5ff708414`

Mechanical correction:

- commit: `2aeacc2004a42cda9a21282d21d2d09f2dfce15c`
- rule: quarantine any GSIS proven to map to multiple nonblank ESB IDs or multiple
  nonblank Smart IDs across the frozen roster horizon
- do not select a team/person
- same-person multi-team ambiguity still fails the original gate
- no formula, position, lookback, threshold, outcome, or sportsbook change

Corrected canonical qualification:

- run: `35517035459`
- job: `106094605432`
- artifact: `10607480543`
- digest: `sha256:367a608fbf6f37d83b63befbc405d38579702fda98b399c16147e80606363fa0`
- same weekly-roster SHA as original: `f2b791d47b146fe703a73d3111d609504779c7e9d2dc0ad47b3bd1996776f18a`
- 14 semantic GSIS collisions detected globally
- 136 front-roster rows quarantined
- stable-ID coverage: **99.8141%**
- same-week ambiguity after quarantine: **0**
- pregame coverage: **99.1448%**
- stability Spearman: **0.813629** on **3,678** adjacent pairs
- redundancy holdout R2: **0.401720**
- integrity/fanout/chronology: clean
- final: `READY_FOR_FROZEN_EXPERIMENT`

Result authority:

`docs/research/DEFENSIVE_FRONT_PAIRWISE_COHESION_QUALIFICATION_V1_RESULT_2026-09-20.md`

### Exact next task

Do **not** score an outcome casually. First perform the no-retest / mechanism-authorization
audit for defensive-front pairwise cohesion, then freeze one distinct mechanism
experiment if a non-duplicative target remains justified.

Preserve the original rejected run and the forensic lineage. Do not waive identity gates,
do not re-key the candidate to ESB/Smart IDs, and do not touch production or Issue #535.


## Defensive front cohesion pressure mechanism — failed closed

After the recovered defensive-front qualification, the repository no-retest audit
authorized one narrow intermediate mechanism experiment. It did **not** authorize a
direct QB/player use.

Frozen plan:

`docs/research/DEFENSIVE_FRONT_PAIRWISE_COHESION_PRESSURE_MECHANISM_EXPERIMENT_V1.md`

Canonical execution:

- implementation/workflow commit: `21fd6ced6e52fdfff379c1f5eeda55b6e7874980`
- run: `35517405126`
- job: `106095556525`
- artifact: `10607820423`
- digest: `sha256:232758a13a42e4f05571daf8768e2bebd62c5382a902d6d4bebcccd19b183c40`

Frozen baseline controlled for immediate front continuity, prior defense state, opponent
prior offensive/protection state, target week, defense identity and opponent identity.

2019-2023 fit:

- train rows: **2,622**
- cohesion coefficient: **-0.000647723**
- frozen expected direction: positive

2024 primary:

- scored rows: **544 / 544**
- coverage: **100%**
- baseline MAE: **0.058079**
- candidate MAE: **0.058080**
- MAE gain: **-0.000001**
- RMSE gain: **-0.000003**
- p90 absolute-error gain: **-0.000041**
- correlation gain: **-0.000050**
- 5,000-replicate defense-team bootstrap CI for mean AE gain:
  **[-0.000011, +0.000007]**

Only support and coverage passed. MAE, bootstrap, RMSE, p90 and coefficient-direction
gates all failed.

Final:

`DEFENSIVE_FRONT_PAIRWISE_COHESION_PRESSURE_MECHANISM_V1_FAILED_CLOSED_PRIMARY`

2025 remained sealed:

`team_weekly_replication_sha256 = NOT_READ_NOT_HASHED`

Result authority:

`docs/research/DEFENSIVE_FRONT_PAIRWISE_COHESION_PRESSURE_MECHANISM_EXPERIMENT_V1_RESULT_2026-09-20.md`

### Updated next-action boundary

Do not rescue OL or defensive-front pairwise cohesion with alternate lookbacks,
starter-only subsets, target substitutions, interactions or model changes.

Both pairwise-cohesion contexts may remain descriptive/engineering information, but
their frozen pressure-mechanism paths are closed.

The next football-context research step must use **materially different information or
a different pre-registered mechanism**, after another no-retest audit.

Production remains untouched. RB predictive research remains pinned/paused. Do not
touch Issue #535.
