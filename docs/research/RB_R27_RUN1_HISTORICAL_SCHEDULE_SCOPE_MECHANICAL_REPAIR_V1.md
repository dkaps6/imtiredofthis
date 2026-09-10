# RB R27 Run 1 — Historical Schedule Scope Mechanical Repair V1

Status: **FROZEN MECHANICAL REPAIR BEFORE RERUN**  
Date: 2026-09-10 UTC

## Preserved failed execution

- workflow: `RB R27 Receiving Yard Mean Decomposition V1`
- run: `34420935737`
- job: `102695978724`
- head: `d6de4665fb520240d09225957d55b26a6c4cdcfc`
- conclusion: `failure`

This run produced **no R27 scientific result**.

Completed before failure:

1. frozen R27 plan / implementation / exact historical R25-R26 parent blob verification: PASS;
2. protected production-code boundary: PASS;
3. canonical leakage-safe R26 transition-state build: PASS.

Skipped because of the failure:

- exact R26 historical opportunity folds;
- R27 receiving-yard translation;
- all 27 R27 scientific/structural disposition gates;
- R27 evidence artifact upload.

## Exact failure

The 2019 historical bundle itself built and validated successfully for its intended 2019 REG Week 1-17 scope. The subsequent historical player-log construction requested seasons `2018,2019` and failed because the current schedule input exposed out-of-scope 2018 Week-18 rows as resolvable schedule candidates.

The player-log builder raised:

`RuntimeError: historical player logs could not resolve opponent for regular-season schedule rows`

with the reported offending 2018 Week-18 team rows including:

- LAC
- PHI
- BAL
- SEA
- IND
- DAL
- CHI
- HOU

The 2018 NFL regular-season contract used by this study is Weeks 1-17. Therefore 2018 Week 18 is outside the frozen regular-season scientific population.

## Prior project precedent

R25 Run 1 (`34347368040`, job `102452054367`) was preserved as a mechanical failure when Week-18-labelled rows appeared for a 17-week REG season. Its documented repair (`a74f8091edf12d3652345a945003509c58c9561c`) made exact regular-season schedule scope authoritative without changing any R25 formula, cohort, threshold, or production model.

R27 uses the same principle, but the current source drift now requires an additional staged-schedule guard because the schedule source itself exposes the invalid beyond-REG week.

## Authorized repair — scope only

Create a **staged schedule copy used only by `historical_player_logs.py`**.

For each schedule row whose season is requested by the historical player-log step:

- seasons `<= 2020`: retain only REG weeks `1..17`;
- seasons `>= 2021`: retain only REG weeks `1..18`.

The staging helper must:

1. preserve every retained row and every retained column value exactly;
2. remove only rows whose week exceeds the canonical regular-season maximum or is below Week 1 for the requested season;
3. emit an audit containing source rows, staged rows, removed rows, and removed `(season, week, team)` identities;
4. hard-fail if any in-scope row changes or disappears;
5. hard-fail if staged requested-season rows still contain an out-of-scope week;
6. make no fuzzy identity repair and no football-value transformation.

All other historical builders and validators continue consuming the original schedule artifact. Only the player-log schedule seam receives the staged copy.

## Explicitly forbidden changes

This repair does **not** authorize changes to:

- `RB_R27_RECEIVING_YARD_MEAN_DECOMPOSITION_V1_FROZEN_PLAN.md`;
- R27 evaluator or finalizer code;
- any of the 27 R27 gates or thresholds;
- the exact R26 vacancy/R8/R9 mechanism;
- R26/R27 cohorts or seasons;
- production YPT / catch-rate fallback logic;
- R22;
- protected production code;
- sportsbook separation;
- actual-outcome handling.

The locked R27 evaluator blob `a284592a0b2f948b8f12f973ca9eb7d9c08d6b23` and finalizer blob `4f8bf99b25210e8c88a6c8c4511ff9d565d47652` must remain byte-identical on the rerun.

## Classification

**`R27_RUN1_HISTORICAL_SCHEDULE_SCOPE_MECHANICAL_FAILURE_RERUN_AUTHORIZED`**

This is source/schedule compatibility plumbing only. It carries no information about whether the R27 receiving-yard candidate helps or hurts predictive accuracy.
