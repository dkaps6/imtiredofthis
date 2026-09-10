# RB R27 Run 4 — Historical Player-Log REG Scope Mechanical Repair V2

Status: **FROZEN MECHANICAL REPAIR BEFORE RERUN**  
Date: 2026-09-10 UTC

## Preserved failed execution

- workflow: `RB R27 Receiving Yard Mean Decomposition V1`
- intended repaired run: `34421267285`
- job: `102696999664`
- head: `4fe26a31cd988fad84c0ac3d00426ad69fe86c71`
- workflow run number: 4
- conclusion: `failure`

This run produced **no R27 scientific result**.

Completed before failure:

1. frozen R27 plan / implementation / exact historical R25-R26 parent blob verification: PASS;
2. protected production-code boundary: PASS;
3. canonical leakage-safe R26 transition-state build: PASS;
4. 2019 historical input bundle build/enrichment/validation: PASS for REG Weeks 1-17;
5. R27 schedule-staging helper audit: PASS and demonstrated that the supplied 2018/2019 schedule artifact already contained exactly 512 REG team-week rows for each season and zero out-of-scope rows.

Skipped because of the failure:

- exact R26 historical opportunity folds;
- R27 receiving-yard translation;
- all 27 R27 disposition gates;
- R27 evidence artifact upload.

## Exact root cause

The first repair correctly scoped the **schedule**, but the schedule was not the source of the remaining Week-18 row.

The current branch inherited `scripts/backtest/historical_player_logs.py` from the current production lineage, where the player-log builder uses a generic constant:

`REGULAR_SEASON_MAX_WEEK = 18`

and filters nflreadpy normalized weekly player stats with `week.between(1, 18)` before left-joining them to the REG schedule.

For 2018, nflreadpy exposes postseason rows labelled Week 18. Those rows survive the generic filter even though the validated 2018 REG schedule correctly ends at Week 17, and the subsequent left join therefore reports unresolved 2018 Week-18 opponent rows.

The canonical R25 mechanical repair already solved this exact class of problem. At authoritative R25 repair commit:

- commit: `a74f8091edf12d3652345a945003509c58c9561c`
- `scripts/backtest/historical_player_logs.py` blob: `b993bd46fd44c88785bb37344c756c51b5d39afa`

that file makes exact `(season, week, team)` REG schedule keys authoritative by inner-joining normalized weekly player stats to the season's validated REG schedule and reporting excluded non-REG schedule rows. R25 documented this as a scope/integrity repair, not a football-model change.

## Authorized Repair V2

Stage **exactly** the already-authorized R25 repaired `historical_player_logs.py` blob:

`b993bd46fd44c88785bb37344c756c51b5d39afa`

into the R27 research branch at:

`scripts/backtest/historical_player_logs.py`

No edits to that blob are authorized.

The R27 workflow must verify the staged file's Git blob hash equals the exact R25 authority before execution.

The existing R27 schedule-staging helper may remain as an independent no-op/value-preservation audit at the schedule seam; the staged R25 player-log builder then uses exact validated REG schedule keys as the actual player-observation scope authority.

## Explicitly forbidden changes

This repair does **not** authorize changes to:

- the R27 frozen plan;
- R27 evaluator blob `a284592a0b2f948b8f12f973ca9eb7d9c08d6b23`;
- R27 finalizer blob `4f8bf99b25210e8c88a6c8c4511ff9d565d47652`;
- any R27 cohort, formula, gate, threshold or disposition logic;
- exact R26 vacancy/R8/R9 science;
- production YPT / catch-rate fallback hierarchy;
- R22;
- protected production runtime code;
- sportsbook separation;
- actual-outcome handling after predictions.

## Classification

**`R27_RUN4_PLAYER_LOG_REG_SCOPE_MECHANICAL_FAILURE_EXACT_R25_REPAIR_REUSE_AUTHORIZED`**

This is a historical observation-scope plumbing repair only. It contains no evidence about whether the R27 receiving-yard candidate improves or worsens predictive accuracy.
