# CURRENT NFL RESEARCH HANDOFF — R27 FRONTIER DELTA

**Date:** 2026-09-09 / 2026-09-10 UTC execution window  
**Repository:** `dkaps6/imtiredofthis`  
**Parent canonical handoff:** `CURRENT_NFL_RESEARCH_HANDOFF.md` on main commit `d63ba0216d43e4763954e0be7351ece06f8fa6b4`  
**Current production authority remains unchanged:** `bb76ba9eabb08e2f0875a9af49301c3877f4141f`

This delta exists so a new ChatGPT session can resume the active R27 work even if the current chat ends before R27 reaches a scientific checkpoint. Read the parent canonical handoff first, then this file.

## Active scientific lane

**RB R27 Receiving-Yard Mean Decomposition V1**

Research branch:
`research-rb-r27-receiving-yard-mean-decomposition-v1`

Current branch head at this checkpoint:
`efebb966922ea6703236bd2ec60f1351fdc1a08c`

Frozen plan authority:
- plan commit: `5333d7e1cc33dcb567d03c924c774afb6877e932`
- scientific question: with R26 as the fixed upstream RB opportunity/receptions authority, does the exact R26 vacancy-gated redistribution improve RB receiving-yard mean when the existing production YPT efficiency assumption is held fixed?
- baseline receiving-yard mean: baseline targets × production YPT
- candidate receiving-yard mean: exact R26 candidate targets × the same production YPT
- R22 remains untouched during mean discovery
- sportsbook inputs upstream = 0
- 2026 Week1 outcomes = 0
- production changes = 0

The candidate is deliberately narrower than prior R24. R24 tested a broader receiving-entitlement mechanism with existing production efficiency and did not improve overall receiving-yard MAE. R27 tests whether the specifically qualified R26 vacancy/role-transition opportunity improvement carries into receiving yards before any new efficiency model is added.

## Frozen R27 implementation identity

The latest mechanical-repair note explicitly preserves:
- R27 evaluator blob: `a284592a0b2f948b8f12f973ca9eb7d9c08d6b23`
- R27 finalizer blob: `4f8bf99b25210e8c88a6c8c4511ff9d565d47652`

No repair may change the R27 cohorts, candidate formula, thresholds, gates, R26 vacancy/R8/R9 science, production YPT/catch-rate fallback hierarchy, R22, sportsbook separation, or outcome handling.

## Workflow status — NO SCIENTIFIC R27 RESULT YET

Workflow:
`RB R27 Receiving Yard Mean Decomposition V1`

There have been five branch workflow runs at this checkpoint. The latest is:
- run: `34421405933`
- job: `102697424190`
- run number: 5
- head: `efebb966922ea6703236bd2ec60f1351fdc1a08c`
- conclusion: `failure`

Latest run step status:
1. setup / dependencies: PASS
2. verify frozen R27 lock and exact R26 parent mechanism: PASS
3. build canonical leakage-safe transition state: PASS
4. build historical football bundles 2019-2025: FAIL
5. exact frozen R26 opportunity folds: SKIPPED
6. R27 receiving-yard translation: SKIPPED
7. all 27 R27 scientific gates: SKIPPED
8. R27 artifact upload: SKIPPED

Therefore **no R27 football/scientific disposition exists yet**. The failures to date are plumbing/scope failures before candidate scoring.

## Preserved Run 4 root cause and frozen repair V2

Preserved failed execution before the latest note:
- run: `34421267285`
- job: `102696999664`
- head: `4fe26a31cd988fad84c0ac3d00426ad69fe86c71`
- conclusion: `failure`

Run 4 proved the first schedule-scoping repair was not enough. The validated 2018 REG schedule correctly ends at Week 17, but the inherited current-branch `scripts/backtest/historical_player_logs.py` uses a generic `REGULAR_SEASON_MAX_WEEK = 18`. nflreadpy exposes 2018 postseason rows labelled Week 18, so those rows survive the generic player-stat filter and later create unresolved opponent rows.

This is a historical observation-scope plumbing defect, not R27 scientific evidence.

Frozen repair V2 authority:
- repair note commit: `efebb966922ea6703236bd2ec60f1351fdc1a08c`
- exact already-authorized R25 player-log builder blob to stage: `b993bd46fd44c88785bb37344c756c51b5d39afa`
- authoritative R25 repair commit: `a74f8091edf12d3652345a945003509c58c9561c`

That exact R25 builder makes validated REG `(season, week, team)` schedule keys authoritative by inner-joining weekly player stats to the regular-season schedule and excluding out-of-scope postseason rows.

The repair note classification is:
`R27_RUN4_PLAYER_LOG_REG_SCOPE_MECHANICAL_FAILURE_EXACT_R25_REPAIR_REUSE_AUTHORIZED`

## Important nuance about run 5

The repair-note commit itself triggered run 5 automatically. At this checkpoint the authorized R25 player-log blob has **not yet been staged into the R27 branch**, so run 5 again failed in the historical-football-bundle stage before any R27 candidate scoring. Do not count run 5 as a scientific failure.

## Exact next action

1. Stage **exactly** R25 `historical_player_logs.py` blob `b993bd46fd44c88785bb37344c756c51b5d39afa` into the R27 research branch.
2. Update the R27 workflow only as needed to verify that exact blob hash before execution.
3. Do not modify the frozen R27 evaluator/finalizer/science.
4. Rerun the workflow.
5. If historical bundle construction clears, execute the exact frozen R26 opportunity folds.
6. Translate R26 targets into receiving-yard means using the same production YPT.
7. Apply the 27 frozen R27 gates.
8. Only then assign a scientific R27 disposition.

## Production boundary

Production is untouched by the R27 work at this checkpoint.

Current production RB tree remains:
- RB-P3 rushing
- R26 receptions/opportunity refinement
- existing receiving-yard mean machinery
- R22 mean-neutral receiving-yard tail/distribution

R27 is research-only until separately qualified and promoted.

## One-sentence frontier

**R27 is fully frozen and implemented scientifically, but no scientific result exists yet because historical player-log regular-season scoping is mechanically blocking the workflow; the exact previously authorized R25 REG-keyed player-log builder is now frozen as the only allowed repair, after which the unchanged R27 candidate can finally run through its 27 gates.**
