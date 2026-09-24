# RB Rush + Receiving Conservation V2 — Scope Amendment

Date: 2026-09-24

Status: FROZEN PRODUCTION-SCOPE CORRECTION BEFORE MERGE

## Why

The scientific mean qualification was explicitly run on `position == RB`.

The initial draw-integration and production wrapper allowed `RB/FB` because the legacy Week-1 P3 conservation path historically treated RB/FB together. That broader scope was not supported by the V2 historical qualification.

The Week-2 integration/certification sample contained:
- 32 matched rush+receiving player-games;
- position counts: **32 RB, 0 FB**.

Therefore the broader code path did not affect the observed V2 results, but leaving FB enabled would silently generalize beyond the tested population.

## Frozen correction

V2 production scope is narrowed to:
- position **RB only**;
- non-Week-1;
- `rush_rec_yards`.

FB is an explicit V2 no-op unless a future separately frozen FB study qualifies it.

The formula, historical result, candidate-applied Week-2 rows, sportsbook separation, and all RB gates are unchanged.

## Required re-certification

After this scope correction:
- rerun the frozen Week-2 integration A/B;
- rerun production certification;
- require identical RB scientific/mechanical results;
- require the new FB no-op unit test to pass.

No scientific rescue or parameter change is permitted.
