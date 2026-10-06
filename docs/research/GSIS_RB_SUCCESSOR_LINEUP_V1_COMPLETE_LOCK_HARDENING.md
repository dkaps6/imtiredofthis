# GSIS RB Successor Lineup V1 — Complete Pregame Lock Hardening

Status: **OPERATIONAL HARDENING — SCIENCE/FROZEN FORMULA UNCHANGED**  
Date: 2026-10-06  
Branch: `research-gsis-rb-successor-lineup-v1`

## Why this exists

Week 4 produced a valid private V2 allocation/event lock before kickoff, but the
required private three-arm projection lock was not persisted.

Postgame reconstruction is forbidden by the frozen contract, so Week 4 is a
valid allocation observation but cannot be scientifically graded for
BASELINE vs VACANCY_V1_SNAP vs GSIS_LINEUP_V1 attempt/yard error.

This is an execution-completeness problem, not a scientific failure.

## Hardening

New fail-closed verifier:

`scripts/research/verify_gsis_rb_successor_complete_lock_v1.py`

A future prospective GSIS observation may not be described as a **complete
pregame scientific lock** unless this verifier succeeds before kickoff.

The verifier requires all four private/public-safe authorities:

1. private allocation lock;
2. private event audit;
3. private three-arm projection lock;
4. public-safe projection manifest emitted by
   `lock_gsis_rb_successor_projection_v1.py`.

It verifies:
- target season/week parity;
- event coverage;
- every locked successor appears in the projection lock;
- both required markets are present: rush_att and rush_yards;
- projection manifest disposition is
  `GSIS_RB_SUCCESSOR_LINEUP_V1_THREE_ARM_PROJECTION_LOCKED`;
- sportsbook inputs used = 0;
- target outcomes attached = 0;
- private-row protection is asserted;
- exact SHA256s for all required lock files.

Public output contains hashes/counts only. No player identity or raw GSIS cells
are emitted.

Success disposition:

`GSIS_RB_SUCCESSOR_LINEUP_V1_COMPLETE_PREGAME_LOCK_READY`

## Mandatory next-event order

For every Week-5+ eligible vacancy event:

1. capture private GSIS point-in-time snapshot before kickoff;
2. build/finalize private allocation + event audit;
3. run `lock_gsis_rb_successor_projection_v1.py` from the exact pregame
   no-odds Full Slate football source;
4. persist the private projection CSV outside public GitHub;
5. persist the public-safe projection manifest;
6. run `verify_gsis_rb_successor_complete_lock_v1.py`;
7. preserve the public-safe completeness receipt/hash;
8. only then count the event as a complete prospective scientific observation.

If step 3-6 does not complete before kickoff:
- keep the allocation lock if valid;
- mark the event incomplete/not-scoreable;
- never reconstruct the projection postgame.

## Scientific firewall

No candidate formula, cohort, weight, support floor, YPC authority, production
mean, sportsbook rule or outcome attachment rule changed.

Week-4 remains unscoreable and is not retroactively repaired.
