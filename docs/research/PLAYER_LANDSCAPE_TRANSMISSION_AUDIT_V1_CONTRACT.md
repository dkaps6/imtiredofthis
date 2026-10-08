# PLAYER LANDSCAPE TRANSMISSION AUDIT V1 — FROZEN CONTRACT

Date: 2026-10-07
Branch: `research-player-landscape-transmission-audit-v1`

## Purpose

Verify that the NFL projection stack is genuinely operating at the individual
player level across the complete football landscape, not merely producing a
player-named row from position/team averages.

The target question for every projected player is:

> Given this player's current role, recent workload, room hierarchy, efficiency,
> health, offense, opponent, defensive matchup, and likely game environment,
> which of those facts materially reach this player's final usage and output
> projection?

This is an architecture/consumption audit first. It does not fit a new model.

## Parent authorities

Individual-player replay:
- run `37683439543`
- artifact `11509468659`

RB receiving-room retrospective impact:
- run `37703522415`
- Weeks 1-4 individual RB improvement confirmed

RB receiving-room Week-5 lock:
- run `37705464974`
- 98 frozen RB identities / 30 rooms / 0 Week-5 outcomes

Football Matchup Transmission Phase A:
- run `37504432928`

Football Matchup Transmission Phase B/C:
- run `37514137803`

Frozen matchup integration scoring:
- run `37519246784`
- all three simple V1 integration candidates failed closed

Do not restart or rerun these parent studies.

## Scope

Positions:
- QB
- RB / FB
- WR
- TE

Primary player-output families:
- QB pass attempts / pass yards
- RB carries / rush yards
- RB targets / receptions / rec yards / rush+rec yards
- WR targets / receptions / rec yards
- TE targets / receptions / rec yards

Primary unit:
**individual player-game-market**

No sportsbook conditioning is allowed upstream.

## Required player landscape layers

Every position/market must be audited against the following layers.

### 1. Identity / availability
Examples:
- current active identity
- starter / depth role
- inactive / injury availability
- team and opponent identity
- roster changes / trade continuity where supported

### 2. Individual recent usage
Examples:
- carries
- targets
- target share
- rush share
- routes / route participation
- snaps where authorized
- recent room share
- same-team continuity
- prior-season role persistence

### 3. Room hierarchy / competition
Examples:
- RB committee concentration
- WR1 / WR2 / slot hierarchy
- TE room hierarchy
- vacancy/injury redistribution
- competition for team opportunities

### 4. Team opportunity environment
Examples:
- expected plays
- pass / rush volume
- PROE / pass tendency
- game-script opportunity
- QB quality / passing environment where authorized

### 5. Individual efficiency
Examples:
- YPC
- YPT
- catch rate
- YPRR
- target depth / air-yards profile
- QB passing efficiency
- authorized specialist efficiency state

### 6. Opponent defense
Examples:
- pass-defense quality
- rush-defense quality
- success allowed
- pressure
- explosive-play allowed
- box rates
- yards before contact / stuff rate
- position-specific receiving efficiency allowed
- coverage shape where source parity exists

### 7. Individual matchup interaction
Examples:
- player role × defensive environment
- slot/outside role × opponent coverage if supported
- RB receiving role × opponent receiving defense
- focal workload × expected game environment

This layer must distinguish:
- actually consumed interaction;
- available data but no interaction;
- previously tested/closed interaction;
- source parity blocked.

### 8. Injuries / vacancies
Examples:
- player's own status
- teammate absence affecting workload
- opponent defender absence where a valid source exists

Do not treat unqualified defender injury data as valid merely because it exists.

### 9. Distribution / uncertainty
Examples:
- right-tail asymmetry
- target-depth distribution state
- count-distribution behavior
- player-specific uncertainty treatment

Mean and distribution layers must be distinguished.

## Consumption classifications

Every audited feature/layer must receive exactly one primary status:

- `CONSUMED_DIRECT_PLAYER`
- `CONSUMED_PLAYER_VIA_SPECIALIST`
- `CONSUMED_TEAM_CONTEXT`
- `CONSUMED_OPPONENT_CONTEXT`
- `CONSUMED_INDIRECT`
- `AVAILABLE_BUT_DROPPED`
- `AVAILABLE_BUT_NOT_CONSUMED`
- `SOURCE_PARITY_BLOCKED`
- `PROSPECTIVE_ONLY_FROZEN`
- `TESTED_AND_CLOSED`
- `NOT_AVAILABLE`

Do not count a feature as consumed merely because the player's row contains the
column. The audit must trace it into the final usage/efficiency/simulation seam.

## Required outputs

1. `player_landscape_feature_inventory.csv`
2. `player_landscape_transmission_matrix.csv`
3. `player_landscape_position_market_summary.csv`
4. `player_landscape_trace_examples.csv`
5. `player_landscape_audit_summary.json`

## Transmission matrix

At minimum report one row per:
- position family
- market
- landscape layer
- concrete feature/mechanism

Required columns:
- source artifact/module
- feature scope: PLAYER / TEAM / OPPONENT / ROOM
- strict-prior/live availability
- current-season freshness
- production consumption status
- exact consuming module/function if consumed
- whether it affects:
  - opportunity
  - efficiency
  - mean
  - variance/distribution
- historical validation status
- current-season validation status
- closed/reopen rule
- notes

## Required trace examples

Produce pregame traces for representative current players from every position.

Each trace must show the chain:

`identity -> availability -> player state -> room state -> team environment ->
opponent environment -> opportunity allocation -> efficiency -> final mean ->
distribution`

This is not a hand-picked "good example" test. Trace players must be selected
deterministically from the current football universe:
- highest projected workload player by position on a deterministic ordering;
- median projected workload player;
- low projected workload active player.

No outcome-based selection.

## Current known facts that must be preserved

### Individual-player opportunity
The W1-4 replay confirmed workload compression across positions.

Do not reopen generic positional mean calibration before individual allocation is
resolved.

### RB receiving
Strict-prior RB receiving-room share is a genuine individual-player mechanism.

Retrospective W1-4 impact:
- targets MAE 7.60% better
- receptions MAE 3.83% better
- rec yards MAE 2.89% better
- rush+rec yards MAE 1.66% better
- rush yards unchanged

Prospective Week-5 lock is frozen.

### WR / TE target share
Target Share Trajectory V1 remains exact-rule prospective from Week 5+.

Do not weaken its four-prior-same-season-game gate.

### Target depth
The full symmetric target-depth distribution transform did not improve pooled
W1-4 CRPS.

Do not promote it universally.

### Football matchup transmission
Production collects more matchup data than generic RB/WR/TE rules consume.

Known upstream-but-undertransmitted families include:
- defensive rush EPA
- explosive play allowed
- position-specific WR/TE/RB YPT allowed
- YBC/stuff fields
- some offense pass-tendency state
- richer game-script context

However, the three frozen simple V1 integration candidates all failed closed:
- RB opponent pass-rate-faced
- WR true PROE
- TE pass-success allowed

Therefore the audit may identify missing transmission but may not simply revive
those exact formulas.

### QB
QB has a richer specialist matchup path than generic RB/WR/TE.

Do not conflate the QB specialist stack with generic skill-position rules.

## Interpretation

The audit must answer, by position and market:

1. Is this projection truly individualized at the usage layer?
2. Is it individualized at the efficiency layer?
3. Does the opponent materially affect this player's projection?
4. Is the matchup effect position-generic or player-role-specific?
5. Which known player/defense facts exist but fail to reach the final number?
6. Which apparent gaps are already tested and closed?
7. Which gaps remain legitimate candidates for new research?

## Prohibited

- no coefficient fitting
- no post-hoc threshold search
- no sportsbook inputs upstream
- no paid OddsAPI
- no target-game outcome features
- no reopening closed M95A/M95B by renaming them
- no arbitrary defense-vs-position multiplier
- no automatic production promotion

## Next-step rule

Only after the matrix is complete may a new mechanism be opened.

A new mechanism is eligible only if:
- it is tied to a concrete missing player-level transmission seam;
- it is not a closed prior family;
- its input is available pregame with valid source parity;
- it can be tested without target-game leakage;
- it has a clear effect path into individual opportunity or efficiency.

The goal is not more features.

The goal is a final projection whose football logic is explainable for the
specific player being projected.
