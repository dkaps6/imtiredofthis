# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## ACTIVE PRODUCTION CHECKPOINT — 2026-09-11

Before doing anything else, read:

1. `docs/handoffs/NFL_HANDOFF_2026-09-11_WEEK2_PRODUCTION_READINESS_CURRENT.md`
2. `docs/handoffs/NFL_HANDOFF_2026-09-11_FULL_SLATE_LIVE_REPAIR_MERGED_CURRENT.md`
3. `NFL_MASTER_CONTINUITY_RECORD.md`

GitHub is canonical; chat memory is secondary.

## Current highest priority

**Season-long 2026 player-prop production continuity (Weeks 1-18) is now the active critical path.**

Week 2 is only the first live out-of-sample invocation of the future-week contract. The goal is not a Week-2 model and not weekly model reinvention. The goal is one stable, prospectively defined production architecture that can run every regular-season week, ingesting legitimate current-week pregame state while keeping formula/routing/model science frozen unless a separately predeclared recalibration study earns a future promotion.

The intended in-season operating loop is:

- run the same weekly model contract;
- grade completed weeks afterward;
- maintain an expanding 2026 calibration/backtest ledger;
- use accumulated completed-week evidence to identify systematic calibration errors or regime drift;
- make only prospectively frozen, evidence-backed changes for future weeks.

The Week-1 live Full Slate mechanical incident is closed and merged. Do not restart that repair.

The current Week-1 WR, TE, QB and qualified RB football stack must **not** be described as unusable. The urgent continuity issue is that RB P3 rushing already has a frozen Weeks 2-18 research formula, but its live `enriched_att` production source path was not promoted because the historical availability/injury timestamp contract remained unresolved. R26 receptions and R22 receiving-tail adapters are also explicitly Week-1 production-gated and require an explicit W2-18 authority decision.

Do not redesign settled model science merely because these production gates exist. Recover and qualify the existing research lineage first.

Active branch:
- `production-week2-readiness-2026`

Active handoff:
- `docs/handoffs/NFL_HANDOFF_2026-09-11_WEEK2_PRODUCTION_READINESS_CURRENT.md`

Exact immediate task:
- recover STACK2/STACK3 result lineage;
- identify the precise source/provenance gap for W2-18 `enriched_att`;
- freeze a leakage-safe **Weeks 2-18 recurring production bridge** before implementation/results;
- close W2-18 routing for R26/R22;
- prove a no-paid-odds Full Slate can run generically for arbitrary future regular-season weeks;
- establish the rolling completed-week 2026 calibration ledger.

## Parked lanes

Until season-long player-prop production continuity is closed, park:

- QB/WR shared-opportunity / first-down public-intent V1B research;
- broad Week-1 betting-card evaluation except production sanity checks;
- game ML/spread/total model development;
- anytime-TD model development.

The QB/WR science checkpoint remains preserved at:
- `docs/handoffs/NFL_HANDOFF_2026-09-11_QB_WR_SHARED_OPPORTUNITY_CURRENT.md`

Important continuity correction: prior chat notes carried an `M108 = 26/26 PASS` label, but repository search did not recover authoritative M108 workflow/script/run/PR evidence. Do not invent or require M108 by name unless concrete GitHub lineage is recovered.