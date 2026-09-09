# RB Week 1 2026 — Pregame Operational Readiness V1 — FROZEN PLAN

Status: **FROZEN BEFORE IMPLEMENTATION / EXECUTION**

## 1. Purpose

This is an operational readiness audit, not a model-fitting study and not a production-promotion study.

It answers one pregame question:

> Can the 2026 Week 1 running-back stack be run today with complete, internally consistent player coverage for rushing, receiving yards, and receptions, while exposing the already-sealed R26 receptions candidate side-by-side without changing protected production?

The audit must use zero 2026 Week 1 outcomes. It may not fit, refit, tune, regenerate, redistribute, or promote any football value.

## 2. Exact protected authorities

### Current production Full Slate
- production head: `f8417f55b04ce0e19baf260e9d532765034c47f1`
- run: `34317211395`
- artifact: `10090547415`
- artifact name: `run_34317211395`
- artifact digest: `sha256:7eab77e41c5879d4f54d87497eee0d1186010784cfc4ef966178930e16fb7c3b`
- expected Week 1 RB rushing authority: `RB_P3_SYNTHESIS_V1`
- expected route: `WEEK1_STACK_OVERRIDE`
- expected simulation iterations: `25000`
- sportsbook inputs to RB football synthesis: `0`

### R22 Week 1 RB receiving-yard production authority
- run: `34298516960`
- artifact: `10084118525`
- artifact name: `rb-r22-week1-production-integration-v2`
- artifact digest: `sha256:2391bd9914e9d0029c63529829496fb9b462d008bdcc3f8c37bb3e1079580bd1`
- expected integration disposition: `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_PASS`
- expected adapter disposition: `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_ADAPTER_PASS`
- expected pricing-lineage disposition: `RB_R22_WEEK1_RECEIVING_TAIL_PRICING_LINEAGE_PASS`
- expected adapted RB keys: `94`
- FB rows remain exact/unadapted by R22.

### R26Q immutable receptions candidate seal
- run: `34400524030`
- artifact: `10123251043`
- artifact name: `rb-r26q-2026-week1-receptions-prospective-seal-v1`
- artifact digest: `sha256:dd3ec0e8e3831ab7f2255c2e5abf343cda8a7943d33a1d4863e52372d6f858a1`
- head: `68661da94f03cab2f96182d47636cf55e088b5de`
- disposition: `R26Q_2026_WEEK1_RECEPTIONS_PROSPECTIVE_SEAL_PASS_READY_FOR_OBSERVATION`
- sealed RB/FB rows: `107`
- changed candidate rows: `104`
- unchanged CIN controls: `3`
- draws per array: `25000`
- seed: `42`

### R26R immutable pregame observation authority
- run: `34401814588`
- artifact: `10124274040`
- artifact name: `rb-r26r-2026-week1-prospective-observation-snapshot-v1`
- artifact digest: `sha256:b4d3e573909803e892d57858c9b0c4bdac089bcf078c1d6f6e53366e4622303e`
- head: `469aa40c90c738e12a82ee32ccca70c9cdbbc29f`
- disposition: `R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_PASS_MARKET_CAPTURED`
- frozen gates: `30/30 PASS`
- matched RB/FB reception book lines: `70`
- matched sealed RB/FB players: `35`
- Week 1 outcomes used: `0`
- sportsbook football inputs used: `0`

## 3. What is being certified

### Production-active Week 1 RB components
1. **Rushing attempts / carries expectation** from protected RB-P3 (`stack_att`).
2. **Rushing yards expectation** from protected RB-P3 (`stack_yards`).
3. **RB receiving-yard distribution** from protected R22 for the 94 RB rows, with canonical mean preserved exactly.
4. **Current production receptions mean** from the sealed baseline already carried into the R26Q/R26R lineage.

### Research sidecar only
5. **R26 receptions candidate mean/distribution** from the immutable R26Q seal.

R26 is explicitly **not production-active** in this audit. The audit only makes the sealed candidate visible beside current production so the Week 1 slate can be inspected before kickoff.

## 4. Player-universe contract

The audit begins from the protected production Week 1 RB rushing context.

Expected exact universe:
- 107 unique RB/FB player keys;
- 32 teams;
- 94 RB;
- 13 FB;
- season 2026;
- week 1.

The exact `(team, player_clean_key)` set must equal the exact 107-row sealed R26Q manifest set.

The exact 94 production RB keys must equal the exact 94 R22 adapted RB keys.

No fuzzy matching is allowed. Any key mismatch fails readiness.

## 5. Frozen readiness checks

A PASS requires all of the following:

1. exact Full Slate artifact digest/head verified by workflow;
2. exact R22 artifact digest verified by workflow;
3. exact R26Q artifact digest/head verified by workflow;
4. exact R26R artifact digest/head verified by workflow;
5. production RB rushing context contains exactly 107 unique RB/FB players across 32 teams;
6. production RB position split is exactly 94 RB / 13 FB;
7. every production RB/FB row is season 2026 / week 1;
8. every production RB/FB row uses route `WEEK1_STACK_OVERRIDE` and version `RB_P3_SYNTHESIS_V1`;
9. every production RB/FB row is marked synthesis-applied and football-only/no-odds;
10. every production RB/FB row reports sportsbook football inputs `0`;
11. every production RB/FB row reports `25000` simulation iterations;
12. all protected `stack_att` and `stack_yards` values are finite and nonnegative;
13. R22 adapter disposition PASS;
14. R22 production-integration disposition PASS;
15. R22 pricing-lineage disposition PASS;
16. R22 has exactly 94 adapted RB keys;
17. R22 trace contains exactly the same 94 `(team, player_clean_key)` RB keys as protected production;
18. R22 maximum receiving-yard mean delta remains within `1e-10` yards;
19. R22 receptions-exact, FB-exact, non-RB-exact, RB-other-markets-exact, and rush+receiving identity gates all remain true;
20. R22 reports current/future outcomes used `0` and sportsbook inputs added `0`;
21. R22 reports production mean parameters changed `0`;
22. R26Q exact PASS disposition and 28/28 gate matrix;
23. R26Q sealed manifest contains exactly 107 RB/FB rows with 104 vacancy-active changes and 3 unchanged CIN controls;
24. R26Q contains exactly 107 arrays, each exactly 25,000 finite, nonnegative, integer-valued reception draws whose SHA-256 matches the manifest;
25. exact 107-key equality between protected production RB universe and R26Q sealed manifest;
26. R26Q confirms zero Week 1 outcomes, zero sportsbook football inputs, no R9 refit, no same-week depth, no production parameter change, no production promotion;
27. R26R exact PASS disposition and 30/30 gate matrix;
28. R26R confirms zero Week 1 outcomes, zero sportsbook football inputs, no candidate regeneration, no production parameter change, no production promotion;
29. current pregame role snapshot can exact-join by `(team, player_clean_key)` for all 107 sealed RB/FB rows; unresolved role count must be zero for readiness PASS;
30. the materialized 107-row player readiness manifest contains finite current production rushing values, finite current production baseline receptions means, finite R26 candidate receptions means, and an explicit R22 receiving-yard status for every player;
31. no Week 1 outcomes are ingested by this audit;
32. no football values are regenerated or changed;
33. production parameters changed = false;
34. production promotion performed = false;
35. R26 live-production activation performed = false.

## 6. Player readiness manifest

Persist one row for each of the 107 protected RB/FB players with at least:

- season
- week
- event_id
- team
- opponent
- player
- player_clean_key
- position
- current pregame role / model role
- `production_rush_attempts_mean`
- `production_rush_yards_mean`
- `production_rush_implied_ypc`
- `production_receptions_mean`
- `r26_candidate_receptions_mean`
- `r26_receptions_delta`
- R26 candidate p10 / p25 / p50 / p75 / p90
- vacancy-active flag
- `r22_receiving_yards_mean` when R22-adapted
- `r22_receiving_yards_distribution_status`
- R22 p30/p50/state-probability where applicable
- R26R market-covered flag
- captured reception market line median where available
- component readiness flags for rushing, receiving yards, production receptions, and R26 research sidecar.

For the 13 FB rows, R22 status must explicitly state that R22 did not alter the row and the canonical production receiving distribution remains exact. Do not invent an R22-adapted mean for FBs if that value is not present in the protected R22 trace.

## 7. Frozen disposition

### PASS
`RB_WEEK1_2026_PREGAME_OPERATIONAL_READINESS_PASS_PRODUCTION_STACK_READY_R26_SIDECAR_READY`

Meaning:
- protected Week 1 RB production is operationally ready for rushing and receiving-yard execution;
- the current production receptions baseline is present and internally aligned to the same 107-player universe;
- the immutable R26 receptions candidate is available as a fully sealed pregame research sidecar for all 107 players;
- production remains unchanged.

### FAIL
`RB_WEEK1_2026_PREGAME_OPERATIONAL_READINESS_FAIL_NOT_READY`

Meaning: one or more frozen operational contracts failed. Do not silently substitute, fuzzy-match, regenerate, or promote anything.

## 8. Authority ceiling

This audit may not:
- promote R26 into production;
- change any RB-P3 rushing value;
- change any R22 receiving-yard value/distribution;
- change any production receptions value;
- regenerate R26 arrays;
- refit R9;
- use same-week depth to modify the sealed R26 candidate;
- use sportsbook values as football inputs;
- use 2026 Week 1 outcomes;
- change QB/WR/TE models.

The separate R26S postgame prospective evaluation remains untouched and will be rerun unchanged only when authoritative Week 1 outcomes and snap counts exist.

---

**Freeze rule:** This readiness contract is frozen before its evaluator is implemented or executed. Operational failures may be repaired only through documented value-neutral compatibility fixes; no readiness threshold may be changed to manufacture a PASS.
