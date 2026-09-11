# NFL HANDOFF — 2026-09-11 — SEASON-LONG PRODUCTION READINESS CURRENT

## Status

This handoff supersedes the short-lived Week-2-only framing. The active objective is now **one production system that runs every week of the 2026 season without requiring a new research project each week**.

Branch: `production-season-long-readiness-2026`

Branch ancestry:
- current `main` checkpoint before readiness work: `3bce326a65328793d2ad830c9336518ee2db6346`
- Week-2 framing checkpoint: `c7433866cc3230c77edfb8d16a247f8f5c51a217`

GitHub is canonical over chat shorthand.

## Core operating principle

We need a **season-long, week-aware production engine**, not separate Week-1, Week-2, Week-3 models.

The stable object should be the football model / routing logic. Each weekly run refreshes only information that is legitimately available before that week's games.

The weekly production loop should look like:

`STABLE MODEL SCIENCE -> CURRENT-WEEK PREGAME STATE -> CURRENT-WEEK PROJECTIONS/DISTRIBUTIONS -> DOWNSTREAM SPORTSBOOK COMPARISON`

Current-week state can include, when source contracts are valid:

- schedule / opponent;
- current roster and team assignment;
- depth-chart hierarchy;
- injury / availability / inactive state;
- lagged snap share / route share / carry share / target share;
- role changes / vacancy / teammate return or absence;
- lagged team pace, pass/rush tendency and usage state;
- opponent defensive context;
- weather and venue where applicable;
- other timestamp-safe football information explicitly validated in the lineage.

The formula must not need to be reinvented because the calendar advances one week.

## Separate calibration lane

Completed 2026 games should be used as an **ongoing calibration and monitoring stream**, not silently folded into same-week projections.

This lane is distinct from weekly production.

Required cadence:

1. Run the frozen production model prospectively for a week.
2. Preserve the exact pregame projections and source snapshots.
3. After games complete, grade predictions against actual football outcomes.
4. Diagnose systematic 2026-season miscalibration by component (opportunity, efficiency, allocation, variance/tails, role-state transitions, etc.).
5. Only when evidence is sufficient, freeze a proposed calibration change prospectively.
6. Backtest it on historical data plus only already-completed 2026 games available at that decision point.
7. Promote only if predeclared gates pass.
8. The change then applies to **future** weeks; never rewrite or rescue already-completed weeks.

This gives us a model that works every week while still learning whether 2026 differs from the historical training regime.

## Anti-overfit rules for in-season calibration

- Never tune a formula just because one player or one week missed badly.
- Never use target-week outcomes in target-week features.
- Never tune against sportsbook lines upstream of football projections.
- Never retroactively change the frozen prediction used to grade a completed week.
- Prefer component-level evidence over aggregate betting ROI.
- Require repeated/systematic evidence before changing a stable coefficient or router.
- Record every promoted in-season calibration with effective week, frozen evidence set, prior authority, new authority, and rollback path.

## Current production science that remains frozen unless a future calibration earns promotion

- QB mean: M89/M90 / `QB_PASS_SYNTHESIS_V1`
- QB distribution: `C2_QB_MEAN_NEUTRAL_DISTRIBUTION_V1`
- WR: M38 hierarchy + `WR_R15_PRODUCTION_MODEL_V1`
- TE: `TE_R5P_PRODUCTION_MODEL_V1`
- RB rushing research authority: `RB_P3_SYNTHESIS_V1`
- RB receptions Week-1 refinement: R26
- RB receiving-yard mean: existing canonical production path
- RB receiving-yard distribution/tails Week-1 refinement: R22 using frozen R19 assets
- sportsbook data remains downstream only

Do not describe the Week-1 WR, TE, or qualified RB stack as unusable. The Week-1 production stack passed its documented production/mechanical gates. The current engineering problem is converting Week-1-specific production adapters into a legitimate **season-long routing contract** without discarding the research that produced them.

## RB season-long objective

### Rushing yards

P3 already contains a season-long research concept:

- Week 1: STACK1 full-stack rushing-yard projection unchanged;
- Weeks 2-18: `enriched RB carries × STACK1 implied YPC`.

Therefore the objective is **not to invent a Week-2 rushing model**. It is to make the W2-18 enriched-opportunity path reproducible every week with live, timestamp-safe pregame inputs.

The route should operate for any target week 2-18 using the same state-building logic:

- current role/depth state;
- lagged player/team rushing usage;
- current competing-RB strength and availability;
- current roster continuity/team-change state;
- current injury/vacancy state;
- team rushing-opportunity projection;
- conserved backfield allocation;
- STACK1 implied efficiency / P3 composition.

No week-number-specific hand tuning.

### Receptions

R26 was promoted only as a Week-1 refinement. The season-long task is to recover the stable football mechanism underneath that result and determine whether it can be expressed as a dynamic weekly entitlement state.

If yes, freeze one W2-18 router that updates from current-week role/availability/lagged receiving usage.

If no, explicitly route future weeks through the protected pre-R26 baseline until a season-long refinement earns promotion.

Do not reuse the Week-1 offseason-vacancy classifier as if it were a permanent season-long state.

### Receiving-yard distribution/tails

R22 is also Week-1 gated. Its receiving-yard mean is not the authority; it is a tail/distribution adapter.

The season-long task is to determine whether the R22/R19 mechanism can use stable, weekly-updating pregame state. If not, preserve the protected baseline receiving distribution for future weeks rather than forcing a Week-1-only state.

## WR and TE season-long requirement

The same standard applies to WR/TE:

- model authority must be valid across the season;
- weekly inputs update from prior information;
- role/depth/injury/team/opponent state can change weekly;
- no separate formula research merely because the week number changes.

Current Week-1 receiving-board UNDER concentration is a **sanity/calibration flag**, not a declaration that WR/TE science is unusable.

We should use the preserved Week-1 projections plus completed Week-1 outcomes, once games are final, to diagnose whether any component is seasonally miscalibrated. Any adjustment must be prospectively frozen for future weeks.

## Season-long production architecture to build/verify

Every target week should run the same orchestration with a target `season` and `week`:

1. Build authoritative schedule for target week.
2. Build authoritative current player/team universe.
3. Build current-week availability/injury state.
4. Build lagged-only player usage state from weeks `< target_week`.
5. Build lagged-only team/opponent state from games `< target_week`.
6. Build position-specific role/entitlement state.
7. Run stable position models.
8. Run joint simulation / distributions.
9. Freeze projections before sportsbook attachment.
10. Attach sportsbook markets downstream.
11. Produce identity, conservation, source-age and routing audits.
12. Preserve the exact pregame artifact for later grading.

Week 1 may require preseason/offseason priors because no current-season games exist. Weeks 2-18 should naturally transition to lagged 2026 information as it becomes available, with historical priors shrinking according to a frozen rule rather than ad hoc judgment.

## In-season learning framework

We need explicit monitoring for whether the 2026 season differs from historical calibration.

At minimum grade by week and cumulatively:

- QB attempts, YPA, passing yards;
- team pass/rush opportunity;
- RB carries, rushing yards, receptions, receiving yards;
- WR/TE targets, receptions, receiving yards;
- position/room allocation shares;
- mean error, MAE/RMSE, directional bias;
- calibration of over/under fair probabilities;
- variance/tail coverage;
- under/over portfolio concentration;
- error by role regime, injury/vacancy regime, depth rank, favorite/dog or game environment only when those variables are football-side and timestamp-safe;
- identity/source failures separately from model misses.

The grading system must distinguish:

- **MECHANICAL/DATA FAILURE**
- **ROLE/STATE FAILURE**
- **OPPORTUNITY MODEL MISS**
- **EFFICIENCY MODEL MISS**
- **DISTRIBUTION/CALIBRATION MISS**
- **NORMAL FOOTBALL VARIANCE**

This is how we improve through the season without turning every week into a new model-development exercise.

## Active work order

### Phase A — recover season-long RB state lineage

1. Recover STACK2 and STACK3 exact plans, workflows, run/result artifacts and metrics.
2. Recover ND2A/ND2B source audits that investigated role/availability/backfield as-of state.
3. Reconstruct precisely how the winning W2-18 `enriched_att` was produced.
4. Mark each feature as:
   - directly reproducible live every week;
   - reproducible from lagged NFL data;
   - requires current-week source acquisition;
   - historically unavailable / provenance-blocked;
   - outcome-leaky and therefore prohibited.
5. Freeze one season-long live-state contract for weeks 2-18.

### Phase B — season-long RB production routing

Implement/qualify one target-week RB router with explicit authorities for:

- rush_yards;
- receptions;
- rec_yards mean;
- rec_yards distribution/tails;
- rush+rec derived markets.

The router must accept any target week, fail closed when required state is missing, and never select a Week-1 adapter merely because no better code path exists.

### Phase C — season-long WR/TE/QB routing audit

Prove current authorities are not accidentally Week-1-specific and that all target-week state builders use lagged/current pregame information correctly.

### Phase D — generic no-odds weekly dry run

Replace the idea of a one-off "Week-2 dry run" with a **generic target-week no-paid-odds production test**.

At minimum validate target weeks 2 and another later-week fixture/synthetic target using the same orchestration to prove there is no hidden Week-2 special casing.

### Phase E — in-season grading/calibration harness

Build or consolidate a preserved-projection grader that can append each completed 2026 week without mutating prior records.

It should produce component-level diagnostics and explicitly separate monitoring from promotion.

## Weekly operating procedure after this is complete

The desired user experience each week is:

- choose target week;
- refresh current football inputs;
- run no-odds validation;
- if clean, acquire/attach current sportsbook markets;
- review projections/edges;
- after games, grade that frozen week;
- update cumulative calibration dashboard;
- only launch research when repeated evidence says a component needs it.

The user should **not** have to return every week to authorize a brand-new model formula.

## Parked lanes while season-long production readiness is being closed

- QB/WR public-intent V1B research;
- new game ML/spread/total science;
- new anytime-TD science;
- broad betting-card optimization unrelated to production sanity.

## Do not do

- Do not build a Week-2-only model.
- Do not create new Week-N formulas each week.
- Do not redo settled RB research generically.
- Do not use completed Week-1 outcomes to retroactively tune Week-1.
- Do not simply delete Week-1 gates without a replacement season-long state contract.
- Do not let sportsbook information define football projections.
- Do not spend paid odds calls to discover production bugs that target-week no-odds tests can catch.

## Exact next step

Recover STACK2/STACK3 plus ND2A/ND2B lineage and produce a **season-long feature/source matrix for RB opportunity and role state**, then freeze the W2-18 live-state contract before any production implementation change.

Update this handoff at every meaningful checkpoint.