# INDIVIDUAL OPPORTUNITY — FUTURE-WEEK LOCK PACK V1

Frozen: 2026-10-11

Status: **MECHANICAL ORCHESTRATION ONLY. NO NEW SCIENCE. MANUAL EXECUTION. FAIL CLOSED ON INCOMPLETE PRIOR STATE OR NON-CURRENT TARGET-WEEK IDENTITY.**

Parent:
- `docs/research/INDIVIDUAL_OPPORTUNITY_ALLOCATION_ROADMAP_V1.md`
- `docs/research/INDIVIDUAL_OPPORTUNITY_PROSPECTIVE_WEEK6_LOCK_CONTINUATION_V1.md`

## Purpose

Create one reproducible future-week research artifact containing the already-frozen
individual opportunity mechanisms:

1. current target-week player-state / identity snapshot;
2. RB carry/snap allocation shadow;
3. RB receiving-room share shadow;
4. WR/TE target-share trajectory shadow.

This pack does not create a new predictor and does not alter any formula.

## Immutable scientific rules

### Player-state authority
- exact target-week roster source required;
- immediately prior scheduled-week stats and snaps must be complete for target-week teams;
- all usage/snap features strictly prior to target week;
- no target-week outcomes;
- no sportsbook input.

### RB carry/snap
- exact frozen formula:
  `0.50 * normalized recent carry share + 0.50 * normalized recent snap fraction`;
- preserve team rushing volume and efficiency;
- no fitted weights.

### RB receiving
- exact frozen strict-prior `prior_rb_room_share`;
- normalize within exact current RB room;
- preserve total RB+FB receiving target mass and receiving efficiency;
- no fitted fallback.

### WR/TE trajectory
- same-season / same-team only;
- >=4 completed target-team games;
- RECENT2 vs all EARLIER games, with >=2 EARLIER games;
- `weight_i = baseline_entitlement_i * exp(trajectory_delta_i)`;
- M38 WR1 frozen;
- WR2+ pool conserved;
- full TE room mass conserved;
- team modeled target mass conserved.

## Execution boundary

The live pack workflow is `workflow_dispatch` only.

It must fail closed if:
- exact target-week roster identity is unavailable;
- prior-week stats or snaps are incomplete;
- any component parent is not READY/FROZEN;
- any target-week or later feature leaks into a lock;
- any sportsbook field is consumed;
- any conservation invariant fails.

A failed readiness run is not a scientific failure.

## Pack manifest

A successful pack must preserve:
- season / target week;
- component versions and statuses;
- row counts / team counts;
- component row-file SHA256 digests;
- prior-week source-completion state;
- exact-target-week roster readiness;
- chronology/conservation checks;
- contamination counters;
- production-change flag.

## Promotion boundary

This lock pack is prospective research evidence only.

It does not:
- grade the prior prospective week;
- change production;
- authorize a new formula;
- promote RB carry/snap or WR/TE trajectory before their original multi-week gates;
- use sportsbook data.

