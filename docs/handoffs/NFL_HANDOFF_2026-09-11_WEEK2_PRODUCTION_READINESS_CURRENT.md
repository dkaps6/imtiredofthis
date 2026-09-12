# NFL HANDOFF — 2026-09-11 — SEASON-LONG PLAYER-PROP PRODUCTION READINESS CURRENT

## Status

This handoff supersedes the merged Week-1 live-repair handoff as the active workstream on branch `production-week2-readiness-2026`.

Base `main` at branch creation: `3bce326a65328793d2ad830c9336518ee2db6346`.

GitHub is canonical. Do not rely on chat shorthand where it conflicts with committed lineage.

## Core objective correction

The active objective is **NOT to build a Week-2 model**.

The objective is to establish **one prospectively defined, production-safe player-prop model stack that can run every week of the 2026 regular season (Weeks 1-18)** without requiring fresh scientific redesign each week.

Week 2 is only the **first out-of-sample live continuity gate** for that season-long contract.

Normal in-season work after this bridge is closed should be:

1. run the same frozen weekly production architecture;
2. ingest only legitimate current-week pregame state (schedule, roster, depth, injury/availability, lagged usage/participation, matchup/environment inputs already authorized by the model contract);
3. preserve formula/routing/science unless a prospectively frozen recalibration study earns a change;
4. after games are complete, grade predictions against outcomes and maintain an expanding 2026 backtest/calibration ledger;
5. use accumulated completed-week evidence to detect systematic 2026 calibration errors, regime shifts, source drift, or role-allocation misspecification;
6. when a change is justified, freeze the hypothesis/gates first, test historically plus on completed 2026 weeks only, document it, and promote prospectively for future weeks;
7. never create ad-hoc week-specific formulas merely because one slate looks unusual.

The desired operating mode is therefore **stable season-long production + rolling postgame calibration**, not weekly model reinvention.

## Immediate correction / user-facing interpretation

Do **not** describe the Week-1 WR, TE, or qualified RB production models as unusable.

The Week-1 production stack passed its documented production/mechanical certification gates. The current concern is narrower:

- some sportsbook-facing priced edges, especially receiving UNDERS, show portfolio-level concentration that requires football sanity review before being promoted as wagers;
- this does not invalidate the Week-1 football model stack;
- several RB refinements were promoted with explicit Week-1-only production gates and therefore cannot simply be carried forward without first resolving their season-long source/state contracts.

## Priority override

Effective immediately, **Weeks 1-18 production continuity is the highest-priority project lane**.

Week 2 is the first live gate, but every implementation decision in this lane must be designed for repeated weekly use through Week 18.

Parked until season-long player-prop production continuity is closed:

- QB/WR shared-opportunity public-intent V1B research;
- broad Week-1 betting-card evaluation beyond what is needed for production sanity checks;
- new game ML/spread/total science;
- new anytime-TD science.

Those lanes remain important but must not preempt getting the existing player-prop stack into a durable all-season production state.

## Frozen production science that must not be casually redesigned

- QB mean: M89/M90 / `QB_PASS_SYNTHESIS_V1`
- QB distribution: `C2_QB_MEAN_NEUTRAL_DISTRIBUTION_V1`
- WR: M38 WR1 hierarchy + `WR_R15_PRODUCTION_MODEL_V1`
- TE: `TE_R5P_PRODUCTION_MODEL_V1`
- RB rushing research authority: `RB_P3_SYNTHESIS_V1`
- RB receptions Week-1 refinement: R26
- RB receiving-yard mean: existing canonical production path
- RB receiving-yard distribution/tails Week-1 refinement: R22 using frozen R19 assets
- sportsbook data remains downstream only

No future-week repair may use target-week outcomes as tuning evidence, weaken historical gates after seeing results, or allow sportsbook information upstream of football projections.

## Confirmed RB season-continuity facts

### P3 rushing is NOT a Week-1-only research model

Canonical P3 contract:

- Week 1: `STACK1` full-stack rushing-yard projection unchanged;
- Weeks 2-18: `enriched RB carries × STACK1 implied YPC`.

The production module `scripts/modeling/rb_rush_synthesis_v1.py` already implements the W2-18 route as `WEEKS2_18_ENRICHED_OPP_STACK_EFF`, and repository tests include a W2-18 composition test.

The unresolved problem is **production qualification of the live `enriched_att` source path**, not redoing months of RB rushing research.

The Week-1 promotion record says W2-18 promotion was withheld because the historical availability/injury source-timestamp contract remained unresolved. Production currently fails closed outside Week 1.

The season-long requirement is to make the W2-18 pregame state reproducible every week using the same source contract and routing rules, not to hand-build a new allocation model each week.

### STACK2 research lineage recovered

Historical branch: `research-rb-stack2-enriched-allocation-integration`.

Frozen STACK2 plan: `docs/migrations/RB_STACK2_ENRICHED_ALLOCATION_INTEGRATION_PLAN.md` on that branch.

The plan explicitly requires a timestamp-safe pregame allocation state using:

1. week-tagged depth-chart hierarchy;
2. lagged snap/participation share and trend;
3. lagged carry/touch share;
4. competing-RB strength and usage;
5. injury / availability / vacancy state;
6. roster/team-change state and continuity;
7. rookie/draft prior where pregame-available;
8. backfield concentration;
9. QB/non-RB rushing competition;
10. OL availability only when historically timestamp-safe.

It separates team rushing opportunity from player allocation, conserves team-level rushing opportunity, and evaluates Week 1 separately from Weeks 2-18.

### R26 receptions is presently Week-1 gated

Current production adapter is `RB_R26_WEEK1_RECEPTIONS_PRODUCTION_V1` and hard-gates `season=2026, week=1`.

R26's current production semantics use the frozen Week-1 offseason vacancy classification. That state cannot be silently reused as a season-long Week-2+ router.

The correct objective is to determine whether the underlying R26 identity/entitlement mechanism can be transported into a **dynamic weekly state contract valid for Weeks 2-18**. If not, the protected pre-R26 baseline remains the safe season-long fallback authority until a future refinement earns prospective promotion.

### R22 receiving-tail adapter is presently Week-1 gated

Current adapter is `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_V1` and hard-gates Week 1.

R22 changes receiving-yard tail shape while preserving the receiving-yard mean. It does not own the receiving-yard mean.

The correct objective is to determine whether its frozen strict-prior identity/tail mechanism is transportable under a **single weekly source/state contract valid for Weeks 2-18**. If not, production must use the protected baseline receiving distribution rather than force an unqualified Week-1 adapter.

## Season-long production contract we are trying to establish

For each future week, production should be able to run from the same architecture with only the week/season and current pregame data changing.

Required properties:

- one authoritative football projection per player/market;
- same formula/routing logic across future weeks unless a separately promoted recalibration supersedes it;
- no sportsbook data upstream;
- no target-game outcomes or PBP upstream;
- current-week schedule/current roster/depth/injury/availability inputs are timestamp-safe;
- lagged usage features use completed prior games only;
- rookies/new-team/return-from-injury cases follow frozen fallback priors and explicit routing, not manual week-specific invention;
- player identity/team/opponent integrity is certified every run;
- team/backfield opportunity is conserved where the model contract requires it;
- adapters either have a valid W2-18 contract or fail closed to a documented baseline;
- clean-checkout reproducibility;
- weekly outputs preserve version/route/source lineage so postgame grading is attributable.

## Active work order

### Phase A — recover exact W2-18 rushing winner and blocker

1. Recover STACK2 and STACK3 plans/results/artifacts/branch lineage.
2. Reconstruct exactly how `enriched_att` was produced in the winning evaluation.
3. Inventory every input by temporal/source semantics.
4. Identify the precise historical timestamp/provenance blocker that prevented W2-18 promotion.
5. Determine whether 2026 live/current sources already available in the repository can satisfy the same football semantics without leakage.
6. Freeze a **Weeks 2-18 production bridge plan** before implementation/results.

### Phase B — qualify P3 Weeks 2-18 live route

Required properties include:

- no sportsbook upstream;
- current weekly schedule/current roster authority;
- finite nonnegative `enriched_att` for all eligible RB/FBs;
- team/backfield carry conservation;
- exact P3 formula `enriched_att × STACK1 implied YPC`;
- no silent fallback to Week-1 route;
- explicit fallback semantics for sparse-history/rookie/role-change cases;
- identity/team/opponent integrity;
- clean-checkout reproducibility;
- no target-week outcome use.

### Phase C — qualify a Weeks 2-18 receptions authority

Recover the protected receptions baseline underneath R26 and freeze an explicit season-long decision:

- either qualify a legitimate dynamic R26 state mechanism for Weeks 2-18;
- or route Weeks 2-18 through the protected baseline until such a refinement earns promotion.

Do not invent a new receptions model merely to avoid a fallback.

### Phase D — qualify a Weeks 2-18 RB receiving-tail authority

Test transportability/operational semantics of the existing R22/R19 tail mechanism without moving the mean. If it cannot be prospectively qualified for repeated weekly use, preserve the protected baseline distribution.

### Phase E — season-long Full Slate dry-run gate

Before any paid future-week odds acquisition, prove the no-live-odds slate can be parameterized for an arbitrary regular-season week and that:

- schedule and current-player universe build correctly;
- QB/WR/TE authorities remain intact;
- explicit RB future-week routes are selected;
- no adapter is accidentally hard-coded to Week 1 or Week 2;
- all distribution/conservation/identity audits pass;
- pricing can be attached downstream afterward without changing football projections.

Week 2 is the first live invocation of this generic future-week contract, not a special-case model.

### Phase F — rolling 2026 postgame calibration ledger

Once a week is complete:

- snapshot actual outcomes and grade every frozen pregame projection;
- maintain error decomposition by market, position, player role, team, game environment, and projection route;
- track directional bias, calibration, MAE/RMSE, tail miss, coverage, and conservation diagnostics;
- distinguish random football variance from systematic 2026 calibration drift;
- accumulate evidence across completed 2026 weeks;
- only launch a recalibration experiment when there is a predeclared, testable hypothesis;
- any promoted change applies prospectively to future weeks and keeps prior outputs immutable.

This is the intended mechanism for learning during the season without turning every week into a new research project.

## Do not do

- Do not create week-specific models as the normal operating mode.
- Do not redo generic RB research already settled by M91-M96 / STACK1-STACK3.
- Do not use completed Week-1 outcomes to hand-tune Week-2 formulas after the fact; completed-week data may enter only a prospectively frozen calibration study or legitimate lagged feature already authorized by the model contract.
- Do not simply remove Week-1 fail-closed checks.
- Do not silently carry the R26 offseason vacancy set into future weeks.
- Do not silently disable R26/R22 without recording the authoritative W2-18 route.
- Do not spend another paid Full Slate call merely to discover code/identity problems that a no-odds dry run can expose.
- Do not start ML/spread/total or ATD model development until season-long player-prop production continuity is explicitly stable, unless the user changes priority.

## Exact next step

Recover STACK2 and STACK3 result lineage and determine the precise source/provenance gap between the historically validated W2-18 `enriched_att` calculation and a reproducible **every-week 2026 pregame source contract**. Update this handoff with every meaningful finding before model implementation changes.