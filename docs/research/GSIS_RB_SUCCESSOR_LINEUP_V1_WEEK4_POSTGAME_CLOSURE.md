# GSIS RB Successor Lineup V1 — Week 4 Postgame Closure

Status: **PREGAME ALLOCATION LOCK VALID / SCIENTIFIC GRADE NOT SCOREABLE**  
Date reviewed: 2026-10-06

## Frozen public authority

Parent:
- `docs/research/GSIS_RB_SUCCESSOR_LINEUP_V1_PLAN.md`
- `docs/research/GSIS_RB_SUCCESSOR_LINEUP_V1_PREGAME_LOCK_CONTRACT.md`
- `docs/research/GSIS_RB_SUCCESSOR_LINEUP_V1_WEEK4_PROSPECTIVE_LOCK_CHECKPOINT.md`

The corrected private V2 allocation lock remains the pregame authority:
- allocation SHA256: `38a79c295a85f5f58c7a77473aae9ea62122da0c5f06d99920f868f1ef3e6b4e`
- event-audit SHA256: `2d01fa956319210a420ac0617e9d2a422abb7dbc101e8610974df8c5536c0461`
- finalized: `2026-10-01T17:14:21.114582+00:00`
- target outcomes read at lock: false
- sportsbook inputs read: false
- 1 prospective week / 2 qualifying vacancy team-games / 5 locked successor player-games

## Postgame integrity review

The private V2 allocation and event-audit files are still available and reproduce the recorded authority. Raw/private GSIS rows remain outside public GitHub.

However, the frozen grading contract requires a **pregame three-arm projection lock** containing BASELINE, VACANCY_V1_SNAP, and GSIS_LINEUP_V1 football projections before outcomes are attached.

The repository contains the pregame projection-lock implementation:
`scripts/research/lock_gsis_rb_successor_projection_v1.py`

But the postgame audit found:
- no private three-arm projection-lock file in the authorized private Week-4 GSIS Library folder;
- no public-safe three-arm projection manifest;
- no Week-4 GitHub Actions execution that persisted that projection lock before kickoff.

The allocation lock alone is not enough to reconstruct frozen attempt/yards predictions after outcomes are known. The parent contract explicitly forbids postgame reconstruction.

## Disposition

Therefore Week 4 is closed as:

`GSIS_RB_SUCCESSOR_LINEUP_V1_WEEK4_VALID_ALLOCATION_LOCK_NOT_SCOREABLE_PROJECTION_LOCK_MISSING`

This is **not** a scientific PASS or FAIL of the GSIS successor mechanism.

Do not reconstruct the three arms now.
Do not change the formula, cohort, support floors, or source semantics.

Operational repair for the next eligible event must ensure the three-arm projection lock is executed and persisted privately before kickoff, with a public-safe manifest/hash.

For scientific support, Week 4 contributes a valid locked allocation/event observation but **zero scored successor player-games** to the terminal paired-AE gate.
