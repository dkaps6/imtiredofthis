# Football Context Engineering Checkpoint — 2026-09-18

**Status:** ENGINEERING CHECKPOINT — NO PRODUCTION SCIENCE CHANGE

## Canonical state inspected

- `main`: `f0dad2c6711e85104eeffedfa5f5112fd172cbf5`
- football-context branch before this checkpoint: `2905f61480b2d07007baaf91c8f9b6681246ef6f`
- branch relation: 7 commits ahead of `main`, 0 behind
- no Actions runs currently attached to the football-context branch

## Completed football-context foundation

The branch now contains the frozen program specification, historical-data reuse policy, deterministic historical-base manifest builder/tests, and the leakage-safe context join contract.

The join contract establishes three canonical grains:

1. player-game context: `season, week, game_id, player_id`;
2. team-game context: `season, week, game_id, team`;
3. player-opponent exposure context: `season, week, game_id, player_id, opponent_player_id, context_family`.

It also freezes strict-prior provenance, explicit missingness, family namespaces, and semantic firewalls for proximity vs assignment and interaction vs responsibility.

## Parallel advanced-data state observed

Open PR #623 (`Data frontier: materialize advanced feature layer V1`) reports a hardened engineering-only materialization across the three validated advanced-data labs. Its PR record reports:

- 47/47 contracted fields implemented;
- BDB 2021: 12/12 fields, 78,343 route player-plays, 58,484 strict-prior player×route history rows;
- BDB 2023: 13/13 fields, 46,396 reconstructed protection interactions, 3,935 strict-prior blocker histories;
- BDB 2026 Analytics: 22/22 fields, 14,107 targeted-receiver plays, 7,161 strict-prior receiver histories, 35,417 strict-prior receiver×route histories;
- target-game history rows used: 0;
- landing/post-release fields used pregame: 0.

This is treated as a parallel data-frontier dependency, not copied or modified here. No interference with its branch/PR was performed.

## Next authorized engineering slices

In dependency order, without predictive experiments:

1. **Role/environment event ledger**
   - timestamped team/room/depth/injury/coaching state transitions;
   - source provenance and strict-prior eligibility;
   - derived opportunity-change formulas versioned separately from sourced statements.

2. **Personnel continuity table**
   - player/team continuity and room turnover descriptors at player-game/team-game grain;
   - no outcome-trained regime labels.

3. **Historical analog index contract/materializer**
   - descriptors only for neighbor selection;
   - outcome attachment only after neighbors are frozen;
   - explicit cutoff, candidate count, neighbor count, and season distribution.

4. **WR/TE defender proximity exposure adapter**
   - consume BDB geometry as exposure/proximity, never authoritative coverage assignment;
   - produce player-opponent exposure rows with coverage and missingness fields.

5. **OL/DL interaction exposure adapter**
   - consume blocker/rusher geometry as interaction evidence;
   - preserve distinction between observed interaction, inferred exposure, and authoritative responsibility.

## Frozen boundaries re-verified

This checkpoint does not:

- alter production science;
- change projection weights/calibration/bet selection;
- run candidate predictive experiments;
- reopen failed-closed research families;
- touch Issue #535;
- make a sportsbook/paid odds pull.

## Disposition

`FOOTBALL_CONTEXT_ENGINEERING_FOUNDATION_READY_V1`

The program is ready for source/materializer implementation at the frozen join surfaces. Predictive use remains unauthorized.
