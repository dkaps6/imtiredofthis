# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## ACTIVE PRODUCTION CHECKPOINT — 2026-09-11

Before doing anything else, read:

1. `docs/handoffs/NFL_HANDOFF_2026-09-11_SEASON_LONG_PRODUCTION_READINESS_CURRENT.md`
2. `docs/handoffs/NFL_HANDOFF_2026-09-11_FULL_SLATE_LIVE_REPAIR_MERGED_CURRENT.md`
3. `NFL_MASTER_CONTINUITY_RECORD.md`

GitHub is canonical; chat memory is secondary.

## Current highest priority

**Season-long 2026 production readiness is now the active critical path.**

The objective is one stable, week-aware production system that can run every week of the season. Weekly work should refresh pregame football state, not invent a new Week-N model.

The Week-1 live Full Slate mechanical incident is closed and merged. Do not restart that repair.

The current Week-1 WR, TE, QB and qualified RB football stack must **not** be described as unusable. The engineering/science issue now is converting Week-1-specific production adapters into explicit season-long routing contracts while preserving settled research lineage.

RB P3 already contains a Weeks 2-18 research route (`enriched RB carries × STACK1 implied YPC`). R26 receptions and R22 receiving-tail adapters were promoted only for Week 1 and therefore need a season-long W2-18 authority decision rather than week-by-week reinvention.

Active branch:
- `production-season-long-readiness-2026`

Active handoff:
- `docs/handoffs/NFL_HANDOFF_2026-09-11_SEASON_LONG_PRODUCTION_READINESS_CURRENT.md`

## Season-long operating contract

Each target week should use the same orchestration:

- authoritative target-week schedule and roster;
- current injury/availability/depth state;
- lagged-only 2026 usage and team context from weeks before the target week;
- historical/preseason priors according to frozen rules;
- stable position-model science;
- joint simulation/distributions;
- frozen football projections;
- sportsbook attachment downstream only;
- preserved pregame artifacts for postgame grading.

Completed 2026 weeks form a separate calibration/monitoring stream. Grade frozen predictions, diagnose systematic component-level errors, and only promote prospectively frozen changes for future weeks after predeclared gates pass. Never rescue completed weeks retroactively.

## Exact immediate task

- recover STACK2/STACK3 plus ND2A/ND2B lineage;
- reconstruct the exact winning W2-18 RB opportunity state;
- produce a feature/source matrix showing what is reproducible live every week and what remains provenance-blocked;
- freeze one season-long W2-18 RB state contract before production implementation changes;
- then audit season-long routing for R26/R22 and WR/TE/QB;
- build a generic target-week no-paid-odds dry run plus in-season grading/calibration harness.

## Parked lanes

Until season-long production readiness is closed, park:

- QB/WR shared-opportunity / first-down public-intent V1B research;
- broad Week-1 betting-card optimization except production sanity checks;
- game ML/spread/total model development;
- anytime-TD model development.

The QB/WR science checkpoint remains preserved at:
- `docs/handoffs/NFL_HANDOFF_2026-09-11_QB_WR_SHARED_OPPORTUNITY_CURRENT.md`

Important continuity correction: prior chat notes carried an `M108 = 26/26 PASS` label, but repository search did not recover authoritative M108 workflow/script/run/PR evidence. Do not invent or require M108 by name unless concrete GitHub lineage is recovered.