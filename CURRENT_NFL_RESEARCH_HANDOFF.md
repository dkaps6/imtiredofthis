# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## ACTIVE PRODUCTION CHECKPOINT — 2026-09-11

Before doing anything else, read in this order:

1. `docs/handoffs/NFL_HANDOFF_2026-09-11_SEASON_LONG_RB_PHASE_A_RECOVERY_CHECKPOINT.md`
2. `docs/handoffs/NFL_HANDOFF_2026-09-11_SEASON_LONG_PRODUCTION_READINESS_CURRENT.md`
3. `docs/handoffs/NFL_HANDOFF_2026-09-11_FULL_SLATE_LIVE_REPAIR_MERGED_CURRENT.md`
4. `NFL_MASTER_CONTINUITY_RECORD.md`

GitHub is canonical; chat memory is secondary.

## Current highest priority

**Season-long 2026 production readiness is the active critical path.**

The objective is one stable, week-aware production system that can run every week of the season. Weekly work refreshes pregame football state; it does not invent a new Week-N formula.

The Week-1 live Full Slate mechanical incident is closed and merged. Do not restart it.

The current Week-1 WR, TE, QB and qualified RB football stack must **not** be described as unusable. Week-1 passed its documented production/mechanical gates. The current task is to convert remaining Week-1-specific adapters/guards into explicit season-long routing while preserving settled model science.

## Newly recovered RB verdict

STACK2/STACK3 + ND2A/ND2B lineage has now been recovered and recorded in the Phase-A checkpoint.

The recovered season-long P3 rushing architecture is:

- Week 1 initialization -> STACK1 full-stack rushing-yard projection;
- Weeks 2-18 -> frozen STACK2 enriched RB opportunity/allocation × STACK1 implied efficiency.

STACK3's additional M95F/M95I routing did not improve the Weeks 2-18 central architecture enough to replace STACK2. Do not invent additional week-specific rushing routes.

Current role/availability plumbing already exists and is timestamped/certified. PlayerForm already enforces active-season `week < target_week` evidence. The primary remaining P3 productionization gap is to freeze/materialize the exact historical STACK2 allocation learner and feed it current depth/availability plus lagged usage every week.

Active branch:
- `production-season-long-readiness-2026`

Latest checkpoint commit:
- `193feebc1d8dbb16339033be59960b05b341e582`

## Exact immediate task

1. verify the 2026 `load_rosters_weekly` metadata contract required for STACK2 rookie/draft/status features;
2. materialize the exact frozen 2024-fit STACK2 allocation learner and prove prediction parity against the archived 2025 STACK2 casebook;
3. freeze its artifact hash/feature contract;
4. build one generic target-week RB state adapter for Weeks 2-18;
5. replace Week-1-only downstream P3 guards with explicit target-week routing invariants;
6. then close season-long routing decisions for R26/R22 and audit WR/TE/QB for hidden Week-1 assumptions;
7. finish with a generic no-paid-odds weekly dry run and an immutable in-season grading/calibration harness.

No paid odds acquisition is authorized for this work.

## Parked lanes

Until season-long production readiness is closed, park QB/WR public-intent V1B, new game ML/spread/total science, new anytime-TD science, and broad betting-card optimization unrelated to production sanity.

Important continuity correction: prior chat notes carried an `M108 = 26/26 PASS` label, but repository search did not recover authoritative M108 workflow/script/run/PR evidence. Do not invent or require M108 by name unless concrete GitHub lineage is recovered.