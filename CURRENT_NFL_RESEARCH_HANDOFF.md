# CURRENT NFL RESEARCH HANDOFF — READ FIRST
## ACTIVE CHECKPOINT — 2026-09-20 — EXACT ASSIGNMENT CLOSED / RB FORWARD CONFIRMATION NEXT

Read first:

`docs/handoffs/NFL_HANDOFF_2026-09-20_EXACT_ASSIGNMENT_CLOSED_RB_FORWARD_NEXT.md`

The football-context exact blocker-rusher source qualified strongly, but its frozen
hindsight value-of-information experiment failed closed on Weeks 7-8. Do not rescue it.

The user has explicitly re-opened RB as an unresolved priority. The strongest positive
RB result is already `RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED` (PR #562 / run
`35039152022`), all 28 gates passed. It has **not** received the separately required
forward/shadow confirmation. That is now the next task.

Production main remains `f0dad2c6711e85104eeffedfa5f5112fd172cbf5`.
Do not touch GitHub Issue #535 / the separate WR lane.

---

## ACTIVE FOOTBALL-CONTEXT UPDATE — 2026-09-20 — DEFENSIVE FRONT PRESSURE MECHANISM FAILED CLOSED

Defensive Front Pairwise Cohesion V1 was mechanically recovered from its original
identity-integrity rejection and remains a clean, qualified descriptive/context feature.

The original qualification run `35515764090` is preserved as `REJECTED_INTEGRITY`.
Forensic audit proved its 13 same-week conflicts were a source GSIS collision between
different people, not transaction timing. A general ESB/Smart-ID semantic quarantine
was added without selecting a team/person or changing the frozen feature definition.

Corrected qualification authority:

- implementation: `2aeacc2004a42cda9a21282d21d2d09f2dfce15c`
- run: `35517035459`
- job: `106094605432`
- artifact: `10607480543`
- digest: `sha256:367a608fbf6f37d83b63befbc405d38579702fda98b399c16147e80606363fa0`
- pregame coverage: **99.1448%**
- stable-ID coverage after quarantine: **99.8141%**
- ambiguity after quarantine: **0**
- stability Spearman: **0.813629**
- redundancy holdout R2: **0.401720**
- qualification: `READY_FOR_FROZEN_EXPERIMENT`

A no-retest audit then authorized one narrow intermediate mechanism test only:
whether accumulated defensive-front cohesion predicts next-game
`pressure_rate_generated` beyond immediate front continuity, prior defense state,
opponent prior offensive/protection state, target week, defense identity and opponent
identity.

Frozen pressure-mechanism authority:

- plan: `docs/research/DEFENSIVE_FRONT_PAIRWISE_COHESION_PRESSURE_MECHANISM_EXPERIMENT_V1.md`
- implementation/workflow: `21fd6ced6e52fdfff379c1f5eeda55b6e7874980`
- run: `35517405126`
- job: `106095556525`
- artifact: `10607820423`
- digest: `sha256:232758a13a42e4f05571daf8768e2bebd62c5382a902d6d4bebcccd19b183c40`

2024 primary result:

- scored rows: **544 / 544**
- coverage: **100%**
- cohesion coefficient: **-0.000647723** (frozen expected direction was positive)
- baseline MAE: **0.058079**
- candidate MAE: **0.058080**
- MAE gain: **-0.000001**
- RMSE gain: **-0.000003**
- p90 absolute-error gain: **-0.000041**
- correlation gain: **-0.000050**
- team-cluster bootstrap 95% CI for mean absolute-error gain:
  **[-0.000011, +0.000007]**

Only support and coverage passed. MAE, bootstrap, RMSE, p90 and coefficient-direction
gates all failed.

Final:

`DEFENSIVE_FRONT_PAIRWISE_COHESION_PRESSURE_MECHANISM_V1_FAILED_CLOSED_PRIMARY`

2025 remained physically sealed and was not read or hashed:

`team_weekly_replication_sha256 = NOT_READ_NOT_HASHED`

Result authority:

`docs/research/DEFENSIVE_FRONT_PAIRWISE_COHESION_PRESSURE_MECHANISM_EXPERIMENT_V1_RESULT_2026-09-20.md`

### Active boundary / next action

Do **not** rescue either OL or defensive-front pairwise cohesion with alternate
lookbacks, starter-only subsets, target substitutions, interactions, favorable
subsets, model changes, or 2025 exposure.

Both pairwise-cohesion features may remain useful descriptive/engineering context,
but their frozen pressure-mechanism paths are closed.

The next football-context research step must use **materially different information or
a materially different pre-registered football mechanism** after a fresh no-retest
audit. Do not treat another cohesion target as the default next experiment.

Production main remains `f0dad2c6711e85104eeffedfa5f5112fd172cbf5`.
RB predictive research remains pinned/paused.
Do not touch GitHub Issue #535 / the separate WR-data-frontier lane.

---

## ACTIVE FOOTBALL-CONTEXT UPDATE — 2026-09-20 — DEFENSIVE FRONT COHESION QUALIFIED

Defensive Front Pairwise Cohesion V1 is now **READY_FOR_FROZEN_EXPERIMENT** after a
source-identity forensic correction.

The original qualification run `35515764090` remains preserved as
`REJECTED_INTEGRITY` because it found 13 same-week GSIS/team conflicts. A dedicated
audit showed all 13 were the same corrupted GSIS shared by two distinct people rather
than a transaction-timing ambiguity.

A general, outcome-free rule now quarantines any GSIS proven to map to multiple upstream
ESB IDs or Smart IDs. It does not choose a team/person and it preserves the original
same-person multi-team ambiguity gate.

Corrected canonical run:

- implementation: `2aeacc2004a42cda9a21282d21d2d09f2dfce15c`
- run: `35517035459`
- job: `106094605432`
- artifact: `10607480543`
- digest: `sha256:367a608fbf6f37d83b63befbc405d38579702fda98b399c16147e80606363fa0`
- same frozen weekly-roster SHA: `f2b791d47b146fe703a73d3111d609504779c7e9d2dc0ad47b3bd1996776f18a`
- pregame coverage: **99.1448%**
- stable-ID coverage after semantic quarantine: **99.8141%**
- same-week ambiguity: **0**
- stability Spearman: **0.813629**
- redundancy holdout R2: **0.401720**
- final: `READY_FOR_FROZEN_EXPERIMENT`

Result authority:

`docs/research/DEFENSIVE_FRONT_PAIRWISE_COHESION_QUALIFICATION_V1_RESULT_2026-09-20.md`

**Immediate next action:** perform a defensive-front cohesion no-retest/mechanism
authorization audit before freezing or executing any predictive/mechanism outcome test.

Production main remains `f0dad2c6711e85104eeffedfa5f5112fd172cbf5`.
Do not touch Issue #535. RB predictive research remains pinned/paused.

---

## ACTIVE FOOTBALL-CONTEXT UPDATE — 2026-09-20

OL pairwise cohesion qualified as context but **failed** its frozen target-game pressure
mechanism on the 2024 primary holdout. 2025 remained sealed.

Canonical mechanism run: `35515454686`; job `106090515886`; artifact
`10607110400`; digest
`sha256:90d3021fb88169b95f7a640967ace4449d01510792c109032251dad53ba4fb2b`.

Final:
`OL_PAIRWISE_COHESION_PRESSURE_MECHANISM_V1_FAILED_CLOSED_PRIMARY`.

Immediate next action: execute the frozen outcome-free
`DEFENSIVE_FRONT_PAIRWISE_COHESION_QUALIFICATION_V1`.

GitHub is canonical; chat memory is secondary.

## ACTIVE FOOTBALL-CONTEXT HANDOFF — 2026-09-20

OL roster continuity V1 has qualified outcome-free, but its QB predictive use is
withheld by the anti-retest ledger because it remains a personnel-discontinuity
count/share mechanism overlapping failed M77, while QB uncertainty/risk was closed
in M71.

**Read this first for the active context lane:**

`docs/handoffs/NFL_HANDOFF_2026-09-20_FOOTBALL_CONTEXT_OL_COHESION_CURRENT.md`

Active branch:

`research-football-context-event-redundancy-v1`

Latest canonical OL roster-continuity qualification:
- implementation commit `d5ce896c4ff34fe48e76b6bcb8f87c29b55e9848`
- run `35514546250`
- job `106088156234`
- artifact `10606771062`
- digest `sha256:e7688b9abd6ac683ca19674f412b11f18dc77109e785d15420d60609c03d33ee`
- broad coverage: **94.0139%**
- stable ID: **99.9909%**
- redundancy holdout R2: **0.005216**
- qualification: `READY_FOR_FROZEN_EXPERIMENT`
- QB predictive authorization: `WITHHELD_ANTI_RETEST`

**Immediate next action:** implement and execute
`OL_ROSTER_PAIRWISE_COHESION_QUALIFICATION_V1` exactly as frozen, without reading
predictive outcomes.

---


