# RB Rush + Receiving Conservation V2 — Production Promotion Plan

Date: 2026-09-24

Status: FROZEN BEFORE PRODUCTION CANDIDATE CERTIFICATION

## Scientific authority

Mean qualification:
- result: `RB_RUSH_REC_CONSERVATION_V2_MEAN_QUALIFIED`
- historical run: `36005675177`
- artifact: `10809812602`

Draw-level integration:
- result: `RB_RUSH_REC_CONSERVATION_V2_INTEGRATION_PASS`
- integration run: `36006624473`
- artifact: `10810946362`

The promoted formula is frozen and must not change:

`RB/FB rush_rec_yards draw = final-mean-aligned rush_yards draw + final-mean-aligned rec_yards draw`

`RB/FB rush_rec_yards model_proj = final standalone rush_yards model_proj + final standalone rec_yards model_proj`

## Scope

Production candidate applies only to:
- position RB or FB;
- `rush_rec_yards`;
- runtime Week != 1.

Week 1 remains exactly on the existing P3/R22/R26 authority. No V2 application is allowed in Week 1.

No QB, WR, TE, standalone rushing, standalone receiving, receptions, rush attempts, ATD, player universe, availability, or sportsbook logic may change.

## Implementation seam

Add a new production wrapper `run_pricing_with_full_roster_universe_v6_production.py`.

V6:
1. consumes the existing V5/R26 production stack unchanged;
2. enables the already-validated `RB_RUSH_REC_CONSERVATION_V2` football adapter before pricing;
3. emits an explicit production lineage/certification artifact after pricing;
4. does not copy or fork the football formula into a second implementation.

Update the stable `run_pricing_with_full_roster_universe_v3.py` public entrypoint to execute V6 while retaining V5 as its protected parent.

Do not modify the Week-1 P3/R22/R26 formulas.

## Certification gates

Using the exact preserved Week-2 input authority (run `35282021679`, artifact `10523345092`) and no new paid odds pull:

1. V6 completes successfully through the existing Full Slate production stack.
2. Existing QB C2, WR-R15, TE-R5P, availability/full-roster, R22/R26-not-applicable and market-lineage validators remain green.
3. Same priced offer identity set as V5 baseline.
4. Every non-`rush_rec_yards` `model_proj` remains identical within 1e-10.
5. Every non-`rush_rec_yards` `fair_prob` remains identical within 1e-10.
6. Standalone RB/FB rush and receiving means remain identical within 1e-10.
7. Every applied non-Week-1 RB/FB combo mean equals its priced standalone rush + receiving means within 1e-8.
8. Every V2 seam draw obeys pathwise conservation within 1e-10.
9. Week-1 application count = 0.
10. sportsbook inputs to V2 football = 0.
11. no duplicate priced offers.
12. no production code outside the explicit V6 activation/lineage seam changes football values.

## Regression test

Run focused unit/static tests for:
- flag OFF exact preservation;
- non-Week-1 RB/FB application;
- Week-1 no-op;
- non-RB no-op;
- missing component fail-closed;
- pathwise identity/conservation.

## Promotion rule

If all gates pass:
`RB_RUSH_REC_CONSERVATION_V2_PRODUCTION_CERTIFIED`

Then:
- update the stable production entrypoint to V6 on this branch;
- open one PR to main;
- do not merge until CI/review state is green.

If any gate fails:
`RB_RUSH_REC_CONSERVATION_V2_PRODUCTION_CERTIFICATION_FAIL`

Do not rescue by changing the scientific formula.

## Standing protections

- sportsbook remains downstream;
- no paid OddsAPI pull;
- no M96 reopening;
- no Week-2 outcome-derived tuning;
- no global SD rescale;
- QB pass yards stays frozen/prospective;
- TE-R5P / PR #627 unchanged.
