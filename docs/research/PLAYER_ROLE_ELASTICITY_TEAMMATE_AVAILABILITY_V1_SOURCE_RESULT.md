# Player Role Elasticity from Teammate Availability V1 — Source Audit Result

Date: 2026-10-15

Status: **COMPLETE / SOURCE BLOCKED / NO PREDICTIVE TEST AUTHORIZED**

Version:
`PLAYER_ROLE_ELASTICITY_TEAMMATE_AVAILABILITY_V1_SOURCE_AUDIT`

Canonical successful run:
- run: `38847501580`
- exact head: `fb516aea9e8af21563af71a6f1506edd154d29af`
- artifact: `12024335417`
- digest: `sha256:1a22efad89da4593124d366d21176d8c70d0d5f2cc0e7498a6216fab57353186`

Frozen source authority:
- corrected M78 official-inactive snapshot
- SHA-256 `d39aaf0feea101f3e0d2721ebd4118ef33fb1a4d3c76670e2a4f17734e37b609`
- 1,088 team-weeks across 2024-2025
- official inactive identity only; fixed pre-kickoff fact
- strictly-prior player usage + snap evidence only

No target-game outcome was scored.
No sportsbook input.
No OddsAPI.
No predictive model fit.
No projection mutation.
No production mutation.

## Overall disposition

`PLAYER_ROLE_ELASTICITY_SOURCE_BLOCKED`

Position dispositions:
- RB: `RB_ROLE_ELASTICITY_SOURCE_BLOCKED`
- WR: `WR_ROLE_ELASTICITY_SOURCE_BLOCKED`
- TE: `TE_ROLE_ELASTICITY_SOURCE_BLOCKED`

This is a source-readiness failure, **not** evidence that named-player role elasticity is false.

## What was strong

The historical room-change source is much richer than expected.

### RB
2024:
- 245 inactive RB/FB rows
- 235 inactive team-games
- 786 candidate successor rows
- 144 distinct successors
- 94 successors with >=2 prior room states
- inactive identity bridge 96.73%
- strict-prior snap coverage 96.95%

2025:
- 275 inactive RB/FB rows
- 264 inactive team-games
- 876 candidate successor rows
- 153 distinct successors
- 119 successors with >=2 prior room states
- inactive identity bridge 97.09%
- strict-prior snap coverage 97.03%

### WR
2024:
- 345 inactive WR rows
- 298 inactive team-games
- 1,518 candidate successor rows
- 242 distinct successors
- 167 successors with >=2 prior room states
- inactive identity bridge 96.81%
- strict-prior snap coverage 97.17%

2025:
- 414 inactive WR rows
- 340 inactive team-games
- 1,732 candidate successor rows
- 248 distinct successors
- 198 successors with >=2 prior room states
- inactive identity bridge 96.62%
- strict-prior snap coverage 97.86%

### TE
2024:
- 203 inactive TE rows
- 187 inactive team-games
- 569 candidate successor rows
- 119 distinct successors
- 72 successors with >=2 prior room states
- inactive identity bridge 99.51%
- strict-prior snap coverage 95.43%

2025:
- 225 inactive TE rows
- 218 inactive team-games
- 648 candidate successor rows
- 118 distinct successors
- 97 successors with >=2 prior room states
- inactive identity bridge 99.11%
- strict-prior snap coverage 97.99%

Every frozen identity, position, snap, event-count, distinct-player,
multi-room-state and chronology gate passed.

## Exact blocker

The only failed frozen gate was:

`strict_prior_same_team_opportunity_history >= 0.90`

Observed:
- RB 2024: 88.68%
- RB 2025: 88.36%
- WR 2024: 88.67%
- WR 2025: 89.26%
- TE 2024: 85.59%
- TE 2025: 84.88%

The frozen 90% gate is not lowered after seeing the result.

## Interpretation

The repo has enough trustworthy historical information to know:
- which same-position teammates were officially inactive;
- which named successors were in the room;
- their stable identities;
- their prior snap participation;
- whether they had experienced multiple prior room states.

But the exact V1 contract required >=90% of candidate successor rows to have
strictly-prior same-team opportunity history. That requirement was not met.

Therefore the exact V1 predictive/mechanism experiment is not authorized.

Do not:
- lower the 90% gate;
- substitute postgame participation;
- infer opportunity from target-game outcomes;
- pool positions;
- convert snap-only rows into fake carry/target history;
- retune the old RB vacancy transfer formula.

## Scientific consequence

The hypothesis remains conceptually plausible but this exact historical source
contract is blocked.

This also reinforces the player-first principle:
availability identity alone is not enough. The missing information is the
named player's **own prior opportunity history in that exact room/team context**.

The next research lane should use a different player-identity information family,
not another vacancy coefficient rescue.

Canonical status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.
