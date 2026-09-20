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
