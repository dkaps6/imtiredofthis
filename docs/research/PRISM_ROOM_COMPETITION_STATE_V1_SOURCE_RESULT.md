# PRISM Room Competition State V1 — Source Result

Date: 2026-10-17

Status: **SOURCE BLOCKED / FAIL CLOSED / NO PREDICTIVE SCORE**

Model:
**PRISM — Player Role & Individual Situation Model**

Frozen source plan:
`docs/research/PRISM_ROOM_COMPETITION_STATE_V1_SOURCE_PLAN.md`

Canonical authority:
- run `39039540742` — **SUCCESS**;
- job `117416030937`;
- exact head `73542d0b0eed3fd212352e0d51c7d550decd04e2`;
- artifact `12103282541`;
- artifact digest
  `sha256:98bbfb4d75727e5d49985f1ba0e4195e1bdacb53ac7cbd3a66c7495c401f64f0`.

Disposition:

`PRISM_ROOM_COMPETITION_STATE_SOURCE_BLOCKED`

## Question

Before kickoff, can PRISM quantify how much competitive room participation has
disappeared around the named player using:
- exact official inactive identity; and
- strictly-prior same-team offensive snap state,

without assuming that the missing player's opportunities transfer to any
specific successor?

This was intentionally a **state-source** question, not an opportunity-transfer
model.

## Frozen state

For each active named RB/WR/TE in a room-change event:
- active room player count;
- inactive room player count;
- named-player prior snap mean, last up to 3 same-team games;
- active-room prior snap mass;
- inactive-room prior snap mass;
- total observed room prior snap mass;
- competition-release fraction;
- named player's active-room snap share.

No target-game output was used.

No carry/target transfer was estimated.

## Result

The active player's own prior snap state is source-strong.

The inactive competitor's prior snap state is not sufficiently complete under
the frozen source contract.

| Family | Season | Active snap coverage | Inactive snap coverage | Finite room state | Event team-games | Active players |
|---|---:|---:|---:|---:|---:|---:|
| RB | 2024 | 94.27% | 78.37% | 68.58% | 235 | 144 |
| RB | 2025 | 93.61% | 76.73% | 60.16% | 264 | 153 |
| WR | 2024 | 94.47% | 79.42% | 73.78% | 298 | 242 |
| WR | 2025 | 94.80% | 83.33% | 83.26% | 340 | 248 |
| TE | 2024 | 93.85% | 72.41% | 66.61% | 187 | 119 |
| TE | 2025 | 94.91% | 83.56% | 81.17% | 218 | 118 |

Frozen requirements included:
- active named-player prior snap >=90%;
- inactive competitor prior snap >=85%;
- finite room-state construction >=85%.

No family-season satisfied the complete frozen contract.

## Interpretation

This result does **not** say teammate competition or room change is irrelevant.

It says the exact V1 definition:

> official inactive room identity + quantitatively complete prior snap mass for
> both active and inactive room members

does not have enough historical source support to authorize a predictive test.

The asymmetry is informative:
- the named active player's own participation history is consistently strong;
- reconstructing the missing competitor's historical participation mass is the
  source bottleneck.

Therefore PRISM should not pretend it knows the exact amount of competition
removed when the historical source cannot support that quantity.

## Anti-rescue

Do not now:
- lower inactive snap coverage below 85%;
- lower finite-room coverage below 85%;
- treat missing inactive snap history as zero;
- replace snap mass with player count after seeing this result;
- infer missing participation from target-game usage;
- use sportsbook availability;
- convert this into the previously blocked vacancy-transfer coefficient;
- use target/carry share as a substitute.

Any materially different room-change source requires a new frozen source plan.

## Relationship to protected science

Unchanged:
- Stage-1 active-role participation remains QUALIFIED ALL;
- Stage-2 conditional allocation remains QUALIFIED for RB rushing, RB
  receiving and TE receiving;
- raw recent-share authority replacement remains CLOSED;
- teammate-availability Role Elasticity V1 remains SOURCE BLOCKED;
- WR/TE target-share trajectory remains supported/protected;
- RB carry/snap state work remains protected;
- opponent/environment generic mechanisms remain closed.

## Integrity

Canonical run:
- target-game outcomes used: **0**;
- 2026 outcomes used: **0**;
- sportsbook inputs used: **0**;
- OddsAPI calls: **0**;
- predictive models fit: **0**;
- projection values changed: **0**;
- production changed: **false**;
- strict repository audit: **PASS**.

## Forward consequence

Do not spend another trial trying to rescue this exact room-state quantity.

The higher-value current-season question is now:

> If the already-qualified PRISM Stage-1/Stage-2 player mechanisms had been
> applied to the preserved Week-5 pregame football state, would they have
> improved the named-player opportunity projections that the Week-5 forensic
> replay identified as the dominant failure seam?

Stage 1 and Stage 2 were frozen/scored from 2024/2025 evidence without reading
2026 outcomes. Replaying those unchanged mechanisms on Week 5 is therefore the
clean next current-season diagnostic.

No Week-5 retuning is authorized.
