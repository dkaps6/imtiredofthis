# RB R26 Week-1 Receptions — Production Promotion Certification

**Decision:** QUALIFIED FOR PRODUCTION PROMOTION  
**Final qualification disposition:** `RB_R26_WEEK1_RECEPTIONS_PRODUCTION_INTEGRATION_PASS_READY_FOR_PROMOTION`

## Qualification authority

- branch: `production-rb-r26-receptions-v1`
- governing frozen plan: `docs/production/RB_R26_RECEPTIONS_PRODUCTION_INTEGRATION_V2_FROZEN_PLAN.md`
- frozen-plan commit: `ca5b641f077964fad58df5f6d1a9c5a4188fcbf1`
- implementation lock: `docs/production/RB_R26_RECEPTIONS_PRODUCTION_INTEGRATION_V1_IMPLEMENTATION_LOCK.md`
- lock commit: `46ede1c2a99c83aa88e36d70daaf9e0fbb402867`
- qualification head: `343372586bd4979c34487761d0af49b5986f68e8`
- qualification run: `34417740186`
- qualification job: `102686263562`
- artifact: `10129819192`
- artifact name: `rb-r26-week1-receptions-production-qualification-v1`
- artifact digest: `sha256:f8fbd74187e811fc765c6d236c4c005c19be6d007d1bec33800790e56eabdcde`
- gates: **35/35 PASS**

## Qualified production behavior

The Week-1 RB receptions path now has one authoritative model output.

For the 31 frozen R26 vacancy-active teams:

`existing protected football state -> R26 RB/FB target-entitlement redistribution -> R26-adjusted receptions MC distribution -> unchanged existing ML/state ensemble -> one final model_proj -> downstream sportsbook comparison`

For CIN, the sole frozen non-vacancy control, RB/FB receptions remain exact protected baseline.

R26 is not a second user-facing model and is not a manual choice against the old baseline. The old baseline is retained only in audit/lineage fields on R26-applied rows.

## Qualification evidence

The exact V4 control and V5 candidate used the same fixed Week-1 football state and same fixed downstream sportsbook offer board.

- V4 priced rows: `3374`
- V5 priced rows: `3374`
- current RB/FB rows: `107`
- R26-applied RB/FB players: `104`
- changed simulation arrays: `104`
- every changed array: vacancy-active RB/FB `receptions`
- forbidden changed arrays: `0`
- priced R26-applied side rows: `132`
- final simulation key universe: `2892`, exact V4 universe
- strict-prior identity max time: `202518`
- R9 refit: `false`
- Week-1 outcomes used: `0`
- sportsbook inputs to R26 football generation: `0`

Conservation / preservation:
- max RB+FB room-pool gap: `0.0`
- max team entitlement gap: `1.1102230246251565e-16`
- max non-RB entitlement delta: `0.0`
- max CIN entitlement delta: `0.0`
- RB receiving yards exact R22 at the R26 seam
- RB rush+receiving yards exact R22 at the R26 seam
- RB rushing arrays exact P3/V4 at the R26 seam
- QB arrays exact V4
- WR arrays exact V4
- TE arrays exact V4
- all non-RB arrays exact V4
- R22 max receiving-yard mean delta remains `3.552713678800501e-15`
- P3 final rush pricing remains exact

Pricing lineage:
- R26 adapter disposition: `RB_R26_WEEK1_RECEPTIONS_PRODUCTION_ADAPTER_PASS`
- R26 pricing disposition: `RB_R26_WEEK1_RECEPTIONS_PRICING_LINEAGE_PASS`
- max R26 trace mean vs priced `mc_proj` gap: `0.0`
- max final `model_proj` vs unchanged `ensemble_proj` gap: `0.0`
- ML/state inputs rewritten by R26: `false`
- ensemble method rewritten by R26: `false`
- one authoritative `model_proj`: `true`

## Scientific context preserved

This promotion does not erase historical exceptions. The parent R26 and R26E studies were 19/20 rather than universal sweeps; 2020 Week 1 and one 2023 all-season slice were preserved as failures. R26J/K investigated those failures without inventing an outcome-fit router. R26L then established 2026 as materially more modern-like than the anomalous 2020 source regime, and R26M/N/O/Q completed the leakage-safe prospective construction/integration/sealing chain.

The user explicitly authorized production integration after reviewing this evidence profile. Week-1 outcomes are not a prerequisite for the pregame promotion; R26S remains the postgame audit of the decision.

## Full Slate production entrypoint

The stable public entrypoint `scripts/run_pricing_with_full_roster_universe_v3.py` was moved at commit `ea67b0dce65d1448ff4910eb1f4cb09b50594dac` to route to `scripts/run_pricing_with_full_roster_universe_v5_production.py`.

The existing `.github/workflows/full-slate.yml` already invokes the stable public V3 entrypoint, so no broad workflow rewrite is required. When live player-prop markets are available and pricing executes, Full Slate therefore consumes the qualified V5/R26 stack. The standard artifact upload already includes `data/**` and `outputs/**`, which includes the R26 adapter/trace/array/pricing-lineage evidence.

## Promotion boundary

This certification authorizes only the Week-1 R26 RB/FB receptions refinement described above. It does not authorize unrelated changes to P3, R22, QB, WR, TE, ensemble weights, or sportsbook boundary semantics.
