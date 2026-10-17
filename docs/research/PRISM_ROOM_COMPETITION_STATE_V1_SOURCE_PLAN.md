# PRISM Room Competition State V1 — Frozen Source Plan

Date: 2026-10-17

Status: **FROZEN SOURCE / PARITY AUDIT BEFORE ANY PREDICTIVE SCORING**

Model:
**PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_ROOM_COMPETITION_STATE_V1_SOURCE_AUDIT`

## Player-first question

PRISM must distinguish stable player identity from the current role regime.

This source audit asks only:

> Before kickoff, can PRISM identify that the named player's competitive room
> has materially changed, and quantify the amount of prior participation removed
> from or still present in that room, without assuming where the missing
> opportunities will go?

This is **room competition state**.

It is not:
- a vacancy-transfer coefficient;
- an injury boost;
- raw recent opportunity share;
- a target/carry allocator;
- a projection adjustment.

## Why this is genuinely distinct

Protected / closed work remains protected:

### Role Elasticity V1
`PLAYER_ROLE_ELASTICITY_SOURCE_BLOCKED`.

That V1 required enough strictly-prior same-team **opportunity history** to learn
how a named successor responded to a changed room. Its >=90% opportunity-history
gate failed.

This V1 does **not** estimate the successor response.

It only asks whether the room-change state itself can be observed.

### RB Vacancy Opportunity V1
Do not recreate:
`vacated rush share -> successor snap-weight transfer`.

Room Competition State does not transfer one player's workload to another.

### Opportunity Authority Priority V1
Raw current share replacement remains CLOSED.

No recent target/carry share is used to replace the player's entitlement here.

### Active-role / Stage-2 allocation
Qualified PRISM participation and active-set allocation remain unchanged.

Room Competition State is contextual evidence about the competitive room around
that player. It is not a replacement for those layers.

## Historical target seasons

Source audit only:
- 2024
- 2025.

Strict-prior participation history may use:
- 2023
- 2024
- 2025,
but only rows earlier than the target game.

No 2026 outcomes.

## Current room-change authority

Use the frozen official game-day inactive semantics already audited by M78:
- fixed pre-kickoff official inactive identity;
- no target-game performance used to infer absence;
- exact player/team/position identity bridge.

Rooms:
- RB = RB/HB/FB;
- WR = WR;
- TE = TE.

Candidate named players:
- current target-week roster members in the same room;
- exclude players on the official inactive list for that team-game.

## Strict-prior participation state

For every active named player and every inactive same-room competitor, construct
offensive-snap state using only completed games before the target week.

Frozen participation summary:
- latest up to 3 strictly-prior offensive snap percentages;
- arithmetic mean;
- minimum 1 observed strictly-prior snap game.

No target, carry, reception or yard outcome is required to construct this source
state.

## Frozen room-state fields

For each active named player in a room-change event:

- `active_room_player_count`;
- `inactive_room_player_count`;
- `player_prior_snap_mean3`;
- `active_room_prior_snap_mass`;
- `inactive_room_prior_snap_mass`;
- `total_observed_room_prior_snap_mass`;
- `competition_release_fraction` =
  inactive prior snap mass / total observed room prior snap mass;
- `player_active_room_snap_share` =
  player's prior snap mean / active-room prior snap mass;
- source coverage flags.

These are descriptive pregame state.

They are **not** projection multipliers.

## Live semantic boundary

The exact inactive identity is available prospectively only when the current
official-inactive authority is valid.

Current Full Slate already enforces:
- >T-75: official inactives are not yet required;
- inside T-75: official inactive sections are required and missing evidence
  fails closed;
- current player-level definitive-unavailable facts remain authoritative.

Therefore this V1 room state may be used prospectively only when the exact
official inactive authority is certified.

It must not manufacture an inactive state earlier in the week from sportsbook
availability or target-game participation.

The snap feature itself is strictly-prior completed-game state and may be
rebuilt with the same PFR/nflverse snap semantics used by the historical audit.

## Frozen source gates

Adjudicate RB, WR and TE separately in both 2024 and 2025.

Require ALL per family-season:

1. official inactive -> roster identity bridge >=90%;
2. active candidate stable identity >=90%;
3. inactive competitor stable identity >=90%;
4. active named-player strictly-prior snap-state coverage >=90%;
5. inactive competitor strictly-prior snap-state coverage >=85%;
6. finite room-state construction >=85% of active candidate rows;
7. event team-game support:
   - RB >=20;
   - WR >=40;
   - TE >=15;
8. distinct active named players with room-change exposure:
   - RB >=30;
   - WR >=50;
   - TE >=20;
9. chronology violations = 0;
10. target-game outcomes used = 0;
11. sportsbook inputs used = 0;
12. OddsAPI calls = 0;
13. predictive models fit = 0;
14. production changes = 0.

Do not lower these gates after running the audit.

## Readiness dispositions

Per position:
- `RB_ROOM_COMPETITION_STATE_SOURCE_READY`
- `WR_ROOM_COMPETITION_STATE_SOURCE_READY`
- `TE_ROOM_COMPETITION_STATE_SOURCE_READY`

or corresponding `..._SOURCE_BLOCKED`.

Overall:
- all ready:
  `PRISM_ROOM_COMPETITION_STATE_SOURCE_READY_ALL`;
- partial:
  `PRISM_ROOM_COMPETITION_STATE_SOURCE_READY_PARTIAL`;
- none:
  `PRISM_ROOM_COMPETITION_STATE_SOURCE_BLOCKED`.

## If source-ready

Do **not** immediately change projections.

Freeze a separate predictive state-arbitration experiment.

That future experiment must ask whether this named-player room state provides
incremental information beyond:
- stable player identity;
- active-role probability;
- qualified Stage-2 allocation;
- existing WR/TE target-share trajectory where applicable;
- protected RB carry/snap state;
- named-player reliability.

It must keep markets separate.

## If blocked

Fail closed.

Do not:
- lower source gates;
- infer missing inactive identity from target-game snaps;
- use sportsbook listings as availability;
- substitute raw target/carry share;
- convert this into the old vacancy transfer formula.

## Integrity

This source audit authorizes:
- target outcomes: 0;
- sportsbook predictors: 0;
- OddsAPI calls: 0;
- model fits: 0;
- projection changes: 0;
- production changes: 0.
