# Player Role & Environment Evidence V1 — Engineering Contract

**Status:** frozen evidence schema; source engineering only.

Machine-readable authority:

`docs/research/player_role_environment_evidence_v1.json`

Validator:

`scripts/research/validate_player_role_environment_evidence_v1.py`

## What this layer is

This layer records the football evidence needed to decide whether a player's current opportunity-generating environment is materially different from the history being used as his prior.

It sits between current roster/availability state and later player-entitlement modeling.

It does not assign carries, targets, receptions or yards.

## Existing qualified backbone

The project already has certified current football availability plumbing:

- current Ourlads depth/status;
- weekly injury evidence;
- official game-day inactive evidence;
- timing certification;
- `current_player_availability.csv`;
- `roles_current_production_eligible_v1.csv`;
- strict-prior `player_game_logs.csv` / PlayerForm safeguards.

Those assets remain authoritative for current player existence and availability.

The regime program **extends** them. It does not create a competing roster universe.

## Evidence families frozen in V1

### Availability
Current authoritative availability state and eligibility.

### Hierarchy
Current depth rank / role evidence, with the existing rule that WR lane labels are not automatically a unique WR1/WR2 hierarchy.

### Team continuity
Whether the player changed teams and how much same-team history is available.

### Room continuity
Competitors added/departed, current room size, and how much prior opportunity was vacated or returned.

### Vacancy
Current injury/inactive-created opportunity removal.

### Current-season strict-prior role
Completed prior-game usage that may supersede stale prior-season information.

### QB environment
Whether the primary quarterback environment changed and whether the QB/player pairing has prior overlap.

### Coaching / play caller
Head coach, offensive coordinator and primary play-caller transitions. This family remains source-blocked until a stable historical/current contract is frozen.

### Trench environment
Current OL / front continuity and key absences. Exact target-game blocker-rusher assignment is not implied.

### Authoritative qualitative role evidence
Structured pre-kickoff football evidence such as lead role, committee, primary target, workload expansion/contraction, pass-down, goal-line, outside/slot role, starter status and limited role.

## Qualitative evidence rule

A statement must preserve:

- source;
- publication time;
- speaker/authority;
- direct statement vs reporter interpretation;
- normalized role concept;
- player/team identity;
- available-before-kickoff flag;
- confidence.

A statement such as "bell cow" may support `LEAD_ROLE` or `WORKLOAD_EXPANSION`.

It does not translate directly into a numerical carry adjustment.

## Current source status

| Source family | V1 status |
|---|---|
| current roster/depth/availability | qualified current production authority |
| strict-prior player history | qualified |
| historical roster/team history | available; regime engineering required |
| coaching/play-caller history | source contract required |
| authoritative role statements | forward archive required |
| transaction-event history | source contract required or audited derivation |
| current trench personnel | partial current context available |

## Next engineering

1. Build current/prior team-history bridge at stable player identity.
2. Define credible positional competitor logic.
3. Compute room additions/departures.
4. Compute vacated/returning opportunity share from strict-prior data.
5. Build QB-player continuity bridge.
6. Source-audit coaching/play-caller history.
7. Start timestamped authoritative role-statement archive.
8. Build current OL/front personnel continuity evidence.
9. Only then implement the deterministic regime-state classifier.

## Hard boundary

V1 authorizes no:
- production integration;
- projection adjustment;
- predictive coefficients;
- sportsbook input;
- target-game usage leakage.

Disposition:

`PLAYER_ROLE_ENVIRONMENT_EVIDENCE_V1_FROZEN_FOR_SOURCE_ENGINEERING`
