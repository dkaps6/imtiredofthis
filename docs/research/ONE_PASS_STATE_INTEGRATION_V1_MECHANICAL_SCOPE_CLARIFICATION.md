# One-Pass-State Integration V1 — Mechanical Scope Clarification

Date: 2026-09-24

Status: **FROZEN BEFORE OUTCOME SCORING**

This clarification changes no football mechanism and is recorded before any
candidate outcome scoring.

The parent candidate explicitly states:

- C2-selected team-games receive the shared completed-pass receiver state;
- unselected team-games remain exact current canonical production.

Therefore mechanical semantic gates must respect that scope.

## Correct scope of the completed-pass semantic gate

The requirement:

> candidate has zero player-iterations with zero receptions and positive receiving yards

applies to **C2-selected team-games whose receiver arrays are replaced by
ONE_PASS_STATE_INTEGRATION_V1**.

It does not apply to C2-unselected team-games, because those rows are frozen
bit-identical to the existing canonical receiver state by candidate definition.

Required gates are therefore:

1. selected-team candidate receiver arrays:
   zero `receptions == 0 && rec_yards > 0` iterations;
2. unselected-team receiver arrays:
   bit-identical baseline vs candidate;
3. selected-team pass identity:
   QB C2 passing yards = candidate modeled receiver yards + C2 residual yards
   within `1e-10` per draw.

No result has been inspected and no scientific threshold has changed.
