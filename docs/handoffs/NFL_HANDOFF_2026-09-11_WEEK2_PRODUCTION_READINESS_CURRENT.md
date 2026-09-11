# NFL HANDOFF — 2026-09-11 — WEEK 2 PRODUCTION READINESS CURRENT

## Status

This handoff supersedes the merged Week-1 live-repair handoff as the active workstream on branch `production-week2-readiness-2026`.

Base `main` at branch creation: `3bce326a65328793d2ad830c9336518ee2db6346`.

GitHub is canonical. Do not rely on chat shorthand where it conflicts with committed lineage.

## Immediate correction / user-facing interpretation

Do **not** describe the Week-1 WR, TE, or qualified RB production models as unusable.

The Week-1 production stack passed its documented production/mechanical certification gates. The current concern is narrower:

- some sportsbook-facing priced edges, especially receiving UNDERS, show portfolio-level concentration that requires football sanity review before being promoted as wagers;
- this does not invalidate the Week-1 football model stack;
- the new urgent production-readiness issue is that several RB refinements were promoted with explicit Week-1-only production gates and therefore cannot simply be carried into Week 2 without a separate qualified transition.

## Priority override

Effective immediately, **Week-2 production readiness is the highest-priority project lane**.

Parked until Week-2 production readiness is closed:

- QB/WR shared-opportunity public-intent V1B research;
- broad Week-1 betting-card evaluation beyond what is needed for production sanity checks;
- new game ML/spread/total science;
- new anytime-TD science.

Those lanes remain important but must not preempt getting the existing player-prop stack safely through Week 2.

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

No Week-2 repair may use Week-1 outcomes as same-week tuning evidence, weaken historical gates after seeing results, or allow sportsbook information upstream of football projections.

## Confirmed Week-2 RB facts

### P3 rushing is NOT a Week-1-only research model

Canonical P3 contract:

- Week 1: `STACK1` full-stack rushing-yard projection unchanged;
- Weeks 2-18: `enriched RB carries × STACK1 implied YPC`.

The production module `scripts/modeling/rb_rush_synthesis_v1.py` already implements the W2-18 route as `WEEKS2_18_ENRICHED_OPP_STACK_EFF`, and repository tests include a W2-18 composition test.

The unresolved problem is **production qualification of the live `enriched_att` source path**, not redoing months of RB rushing research.

The Week-1 promotion record says W2-18 promotion was withheld because the historical availability/injury source-timestamp contract remained unresolved. Production currently fails closed outside Week 1.

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

Week-2 work must determine whether the underlying R26 identity/entitlement mechanism can be transported with a legitimate current-week state contract. If not, the protected pre-R26 baseline remains the safe fallback authority until a W2+ refinement earns promotion.

### R22 receiving-tail adapter is presently Week-1 gated

Current adapter is `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_V1` and hard-gates Week 1.

R22 changes receiving-yard tail shape while preserving the receiving-yard mean. It does not own the receiving-yard mean.

Week-2 work must determine whether its frozen strict-prior identity/tail mechanism is transportable with valid current-week features. If not, production must use the protected baseline receiving distribution rather than force an unqualified Week-1 adapter.

## Active work order

### Phase A — recover exact W2-18 rushing winner and blocker

1. Recover STACK2 and STACK3 plans/results/artifacts/branch lineage.
2. Reconstruct exactly how `enriched_att` was produced in the winning 2025 evaluation.
3. Inventory every input by temporal/source semantics.
4. Identify the precise historical timestamp/provenance blocker that prevented W2-18 promotion.
5. Determine whether 2026 live/current sources already available in the repository can satisfy the same football semantics without leakage.
6. Freeze a W2-18 production bridge plan **before implementation/results**.

### Phase B — qualify P3 W2-18 live route

Required properties include:

- no sportsbook upstream;
- current Week-2 schedule/current roster authority;
- finite nonnegative `enriched_att` for all eligible RB/FBs;
- team/backfield carry conservation;
- exact P3 formula `enriched_att × STACK1 implied YPC`;
- no silent fallback to Week-1 route;
- identity/team/opponent integrity;
- clean-checkout reproducibility;
- no Week-2 outcome use.

### Phase C — Week-2 receptions authority

Recover the protected receptions baseline underneath R26 and freeze an explicit W2+ decision:

- either qualify a legitimate dynamic R26 state mechanism for W2+;
- or route W2+ through the protected baseline until such a refinement earns promotion.

Do not invent a new receptions model merely to avoid a fallback.

### Phase D — Week-2 RB receiving-tail authority

Test transportability/operational semantics of the existing R22/R19 tail mechanism without moving the mean. If W2+ cannot be prospectively qualified, preserve the protected baseline distribution.

### Phase E — Week-2 Full Slate dry run

Before any paid Week-2 odds acquisition, run a no-live-odds Week-2 slate gate proving:

- schedule and current-player universe build correctly;
- QB/WR/TE authorities remain intact;
- explicit RB Week-2 routes are selected;
- no adapter is accidentally hard-coded to Week 1;
- all distribution/conservation/identity audits pass;
- pricing can be attached downstream afterward without changing football projections.

## Do not do

- Do not redo generic RB research already settled by M91-M96 / STACK1-STACK3.
- Do not use Week-1 outcomes to tune Week-2 formulas after the fact.
- Do not simply remove Week-1 fail-closed checks.
- Do not silently carry the R26 offseason vacancy set into future weeks.
- Do not silently disable R26/R22 without recording the authoritative W2+ route.
- Do not spend another paid Full Slate call merely to discover code/identity problems that a no-odds dry run can expose.
- Do not start ML/spread/total or ATD model development until Week-2 player-prop production readiness is explicitly stable, unless the user changes priority.

## Exact next step

Recover STACK2 and STACK3 result lineage and determine the precise source/provenance gap between the historically validated W2-18 `enriched_att` calculation and a reproducible 2026 Week-2 live pregame source contract. Update this handoff with every meaningful finding before model implementation changes.