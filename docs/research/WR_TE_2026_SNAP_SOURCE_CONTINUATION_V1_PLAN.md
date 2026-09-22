# WR/TE 2026 Snap Source Continuation V1 — Frozen Plan

**Status:** FROZEN BEFORE ANY CANDIDATE OUTPUT  
**Branch:** `research-wr-te-2026-snap-source-continuation-v1`  
**Production change authorized:** NO  
**Coefficient/model refit:** PROHIBITED  
**Sportsbook inputs:** PROHIBITED  
**2026 outcomes:** PROHIBITED

## Trigger

Current-Season State Persistence V1 established that:
- current-season WR/TE target-share state materially persists;
- nflverse already publishes 2026 offensive snap counts for Weeks 1-2 across all 32 teams;
- TE-R5P's shared production snap loader is hardcoded to `SOURCE_SEASONS=[2020..2025]`;
- WR-R15 imports the same loader.

Thus the frozen WR-R15 / TE-R5P models expect strict-prior participation features,
but the live 2026 adapters currently cannot consume completed 2026 participation.

## Question

Can 2026 be appended to the existing snap-count source contract without changing
model science, while preserving historical parity and strict-prior safety?

## Candidate

No coefficients, intercepts, features, clipping rules, entitlement pools, or
redistribution logic change.

Candidate source history for a target season S is:

`2020..S`

with the existing feature constructors still requiring source ordinal
`< target_season*100 + target_week`.

## Required gates

### A. Frozen historical parity
For target season 2025, dynamic source `2020..2025` must reproduce the current
frozen source exactly after the production loader's normalization/deduplication.

### B. Week-1 invariance
For synthetic/current 2026 Week-1 target rows, adding 2026 to the source may not
use any 2026 snap row. Feature vectors must equal the current 2020-2025 loader.

### C. Week-2 strict-prior inclusion
For 2026 Week-2 target rows:
- candidate may use Week-1 2026 snaps;
- candidate may not use Week-2 or later snaps;
- current frozen loader must have zero 2026 source rows;
- report how many WR/TE target identities gain a same-team and any-team prior-1
  participation observation from the candidate.

### D. Week-3 readiness
Using a target frame based only on identities known from completed Weeks 1-2,
show that candidate Week-3 features can consume Weeks 1-2 only and never a
Week-3/future row. This is a source-readiness proof, not a Week-3 projection.

### E. Source quality
Report:
- weeks available in 2026;
- teams;
- WR/TE rows;
- non-null offense_pct/offense_snaps;
- duplicate rate under production keys.

### F. No science mutation
The frozen model JSON checksums/parameters are not edited and no production file
is modified.

## Interpretation

If all gates pass:
`WR_TE_2026_SNAP_SOURCE_CONTINUATION_READY`

This means the omission is an operational source-continuation gap and authorizes
a separate production-integration patch with exact parity/regression tests.

If historical parity or strict-prior safety fails:
`WR_TE_2026_SNAP_SOURCE_CONTINUATION_BLOCKED`

No rescue/tuning.

## Disposition

`WR_TE_2026_SNAP_SOURCE_CONTINUATION_V1_PLAN_FROZEN`
