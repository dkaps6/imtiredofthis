# RB R26O 2026 Week-1 Receptions Shadow Integration Compatibility V1 — Result

## Final canonical status

**Final corrected disposition:**

`R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_PASS_READY_FOR_PROSPECTIVE_SEAL`

The final corrected execution passed **38 / 38** frozen structural gates. This authorizes only immutable research-shadow sealing and downstream prospective observation. It does **not** authorize live production activation or production promotion.

## Final corrected canonical execution

- Branch: `research-rb-r26o-2026-week1-receptions-shadow-integration-compatibility-v1`
- Frozen plan: `96e4a6977458b55ec00cb29fbc9de05c56c89b88`
- Frozen evaluator: `6a171cb32e77c7d6691ceb75da9861a5d6dd7bc4`
- Run-1 selector-staging repair note: `e5ac65c8dcb2b73f59706880e893c9f72f5d115d`
- Run-2 dtype-repair note: `2c5fc54c0c88aff28e581a901331d175adc87adf`
- Dtype compatibility wrapper: `960fbb0bc6a29724b3d0b7901c92ac905bd69f0e`
- Gate-15 repair note: `c40de155c3f19785f8398cbcb9aaa2ab2445c2f4`
- Gate-15 audit-payload wrapper: `85c3f2ca6bc73c302a3a33f3b6e75f4df42ae78a`
- R26P authorization run: `34399525657`
- R26P artifact: `10122862934`
- R26P digest: `sha256:2e596f3a3bcbef156983664d332ded94ef41dae1402d1a96ab0b61d185c68d47`
- Final corrected head: `e7014a6e365cbb776e48085dcef12dfece744ca4`
- Workflow run: `34399750746`
- Job: `102628405629`
- Artifact: `10123070453`
- Artifact name: `rb-r26o-2026-week1-receptions-shadow-integration-compatibility-v1`
- Artifact digest: `sha256:27307ad84c232935aed3b25c0e0c9bff4da70cd2915376f41dfa3f85c25f14d0`
- Workflow conclusion: `SUCCESS`

## Final frozen-gate result

All **38 / 38** gates passed.

Key evidence:

- 468 current production players / 32 teams / 16 games
- 107 RB/FB rows
- 104 R26N-changed vacancy-active RB/FB rows
- exactly 104 reception arrays changed
- forbidden changed arrays: `0`
- CIN non-vacancy RB/FB receptions exact baseline
- non-RB/FB receptions exact baseline
- all non-reception arrays exact baseline
- all RB/FB `rec_yards` exact R22 baseline
- all RB/FB `rush_rec_yards` exact R22 baseline
- all RB/FB rushing arrays exact baseline
- all QB pass-yard arrays exact baseline
- full simulation key universe exact at 2,892 keys
- candidate replay deterministic
- candidate reception arrays finite, nonnegative, integer-valued
- max baseline MC mean gap vs sealed R26N: `0.017563893432533284`
- max candidate MC mean gap vs sealed R26N: `0.018766081922851008`
- max candidate-vs-baseline delta gap vs sealed R26N: `0.023037323394851983`
- frozen tolerance for each MC compatibility gate: `0.05 receptions`
- corrected Gate 15 R22 mean-parity evidence: `mean_parity=true`, `max_mean_delta=3.552713678800501e-15`
- 2026 outcomes used: `0`
- sportsbook football inputs used: `0`
- same-week depth used: `false`
- R9 refit: `false`
- R22 changed by R26O shadow splice: `false`
- receiving-yard means changed: `false`
- production parameters changed: `false`
- live shadow production activation authorized: `false`
- production promotion authorized: `false`
- prospective seal design authorized: `true`

## What R26O changes

R26O changes only the 2026 Week-1 **RB/FB receptions distributions** for the 104 vacancy-active players whose R26N entitlement changed.

It does not change the team RB target pool itself. It redistributes that finite backfield receiving opportunity among the RB/FB players according to the already-qualified R26N/R9 role-transition entitlement.

Examples from the final 25,000-draw shadow manifest:

- Omarion Hampton: `2.12164 -> 3.23328` receptions
- Jonathan Taylor: `1.57412 -> 2.56996`
- Jaylen Warren: `1.54380 -> 2.52768`
- Woody Marks: `1.35000 -> 2.18748`
- D'Andre Swift: `1.67208 -> 2.38604`
- Bijan Robinson: `2.89060 -> 3.42608`
- Bam Knight: `1.11328 -> 1.62536`
- Brian Robinson: `0.68560 -> 0.19776`
- Jeremiyah Love: `1.15676 -> 0.81148`

These are prospective research-shadow receptions only. Production remains unchanged.

---

# Preserved prior scientific execution history

## First true R26O scientific execution — preserved FAIL

Before the Gate-15 contract defect was independently proven, R26O reached a scientific disposition at:

- head `550d3d532e9f27c34b90ecd027f3811292674862`
- run `34398759284`
- job `102625073624`
- artifact `10122672501`
- digest `sha256:6d72de1eb8902fd8d956feb01b4d1a3cb5847aa58417ad4f416f3e1102e96e82`

Executed disposition:

`R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_FAIL_NO_SHADOW`

That execution passed 37 / 38 gates. The sole failed gate was:

`15_r22_baseline_mean_parity`

with recorded evidence:

`{"mean_parity": null, "max_mean_delta": null}`

The FAIL remains part of the scientific paper trail and is not deleted or rewritten.

## R26P Gate-15 forensic

A separately frozen, read-only forensic was required before any correction.

Canonical R26P:

- branch `research-rb-r26p-r26o-gate15-contract-forensic-v1`
- frozen plan `51eac07260752809bf9426e38a1b5c46abafe771`
- evaluator `8c787208a70615bd4c5f596a3457f7fa08f5447a`
- run `34399525657`
- job `102627643995`
- artifact `10122862934`
- digest `sha256:2e596f3a3bcbef156983664d332ded94ef41dae1402d1a96ab0b61d185c68d47`
- disposition `R26P_GATE15_EVIDENCE_WIRING_DEFECT_CONFIRMED_MECHANICAL_RERUN_AUTHORIZED`
- 17 / 17 frozen forensic tests passed

R26P proved under immutable hashes that protected R22 returns:

`(adapted_result, trace_dataframe, audit_payload)`

while the frozen R26O evaluator had bound its second return value to the variable queried as the R22 audit payload. Thus the earlier `null/null` Gate-15 evidence came from querying the trace DataFrame, not from a measured R22 parity failure.

Canonical R22 independently had:

- `gates.mean_parity = true`
- `max_mean_delta = 5.329070518200751e-15`
- `gates.receptions_exact = true`

R26P did not convert the earlier FAIL to PASS. It authorized only the hash-tracked evidence-wiring correction and exact rerun that produced the final canonical 38/38 result above.

## Authority / next step

R26O PASS authorizes only a separately frozen **pre-outcome prospective seal** of the exact corrected artifact and reception arrays for later Week-1 observation.

The next study must:

1. pin run `34399750746`, artifact `10123070453`, digest `sha256:27307ad84c232935aed3b25c0e0c9bff4da70cd2915376f41dfa3f85c25f14d0`;
2. seal the exact R26O disposition, 38-gate matrix, RB receptions manifest, and draw arrays without recomputation;
3. freeze the future production-vs-R26O Week-1 receptions scoring rules before outcomes are observed;
4. use zero Week-1 outcomes during the seal;
5. leave production and R22 unchanged.
