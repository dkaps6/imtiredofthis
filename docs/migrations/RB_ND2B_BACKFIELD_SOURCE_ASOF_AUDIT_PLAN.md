# RB-ND2B — Backfield Allocation Source-As-Of Audit

## Purpose

Before fitting a new backfield allocation engine, prove that the missing role hierarchy can be reconstructed without target-game leakage.

## Sources

- frozen M94C 2025 RB trace only for target player identity/coverage;
- nflverse depth charts 2024 and 2025;
- nflverse schedules for scheduled kickoff;
- PFR/nflverse snap counts 2024/2025.

## Rules

- No model fit and no outcome-based source selection.
- 2025 depth rows use the new date-bearing schema (`dt`, `pos_slot`, `pos_rank`). For each team-game choose only the latest snapshot with `dt < scheduled kickoff`.
- 2024 depth rows remain week-tagged and are audited separately; no assumption that 2024 and 2025 schemas are interchangeable.
- Target-game snap counts are forbidden. Snap shares may only be used lagged from prior games.
- A source that lacks adequate pregame coverage is rejected rather than backfilled from future information.

## Outputs

- 2025 depth min/max timestamp and RB-row count under the new schema.
- Team-game pre-kick depth coverage and snapshot-age distribution.
- M94C player-game depth-rank coverage, including Week 1.
- 2024 depth team-week coverage and rank field values.
- Lagged PFR offensive-snap percentage coverage on M94C rows.
- Player-level as-of depth and lag-snap traces for deterministic follow-up.

## Next model if sources pass

The first allocation candidate must keep the M94C team/RB carry pool fixed and replace only within-backfield distribution. Candidate shares must normalize to one across the available RB/FB universe. Depth rank, lagged snap/carry share, roster continuity, injury/availability and competitor state may enter only after their timestamp semantics are audited.
