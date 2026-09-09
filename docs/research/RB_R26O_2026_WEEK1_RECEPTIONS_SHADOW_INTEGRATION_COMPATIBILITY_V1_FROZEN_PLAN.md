# RB R26O 2026 Week-1 Receptions Shadow Integration Compatibility V1 — Frozen Plan

Status: FROZEN BEFORE IMPLEMENTATION / EXECUTION
Date: 2026-09-09

## Scientific question

Can the already-sealed R26N 2026 Week-1 RB/FB opportunity/reception candidate be exposed as a **receptions-only shadow distribution** inside the exact current Full Slate V4/R22 stack while every protected receiving-yard, rush+receiving, rushing, QB, TD, non-RB, and non-reception production array remains element-for-element unchanged?

This is an integration/compatibility study, not a new R9 fit and not an outcome-performance study.

## Candidate

`RB_R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_COMPATIBILITY_V1`

The candidate is deliberately narrow:

1. Reconstruct the exact current 468-player 2026 Week-1 promoted football metrics from the immutable current Full Slate artifact using protected production code.
2. Verify the reconstructed baseline entitlement matches the sealed R26N baseline entitlement for all 468 players.
3. Run the canonical promoted V3 simulator at the production Monte Carlo contract (`25,000` iterations, seed `42`) on the exact current baseline metrics.
4. Apply the protected R22 receiving-yard adapter to that baseline V3 result to obtain the exact baseline V4/R22 result.
5. Create a candidate metrics copy whose **only football-value change** is replacing `entitlement_tgt_share` with the sealed R26N `candidate_entitlement_tgt_share` for the same exact 468 player keys.
6. Run the canonical promoted V3 simulator on that candidate metrics copy with the same `25,000` iterations and seed `42`.
7. Construct the R26O shadow result as a deep copy of baseline V4/R22 and replace **only** the `receptions` arrays for the 104 R26N-changed vacancy-active RB/FB player rows with the corresponding candidate-V3 reception arrays.
8. Leave the three non-vacancy CIN RB/FB reception arrays exact baseline.
9. Leave every other array in the full result exact baseline.
10. Persist only research/shadow evidence; do not write to production paths or activate production pricing.

## Why this seam is legitimate

The current protected R22 production adapter already changes only RB receiving-yard/rush+receiving distribution authority while explicitly preserving RB receptions. R26O tests the mirror-image compatibility seam: R26N may supply RB/FB receptions shadow distributions while R22 yardage arrays remain exact.

No new receiving-yard model, catch-rate model, R9 model, target model, or sportsbook feature is introduced.

## Immutable parents

### R26N structural candidate authority

- run `34396075045`
- artifact `10121598376`
- artifact name `rb-r26n-2026-week1-unmodified-r26-structural-candidate-v1`
- digest `sha256:887929203053cb62904aaaeda9d995c9645163814da181972799f08fe4465c62`
- exact head `3b7a00e282cb925bd7a33175bc0b7d08d1467b2f`
- required disposition `R26N_2026_WEEK1_STRUCTURAL_CANDIDATE_PASS_READY_FOR_SHADOW_INTEGRATION_DESIGN`
- requires `shadow_integration_design_authorized == true`
- expected rows `468`
- expected RB/FB rows `107`
- expected changed RB/FB rows `104`
- expected vacancy teams `31`
- expected non-vacancy teams `[CIN]`

R26O consumes the sealed `r26n_candidate_entitlement_overlay.csv`; it does not rerun/refit R26N.

### Current Full Slate production-state authority

- run `34317211395`
- artifact `10090547415`
- artifact name `run_34317211395`
- digest `sha256:7eab77e41c5879d4f54d87497eee0d1186010784cfc4ef966178930e16fb7c3b`
- head `f8417f55b04ce0e19baf260e9d532765034c47f1`
- expected PlayerForm rows `468`
- expected model-context rows `468`
- expected teams `32`
- expected games `16`

### R22 production-integration authority

This is qualification authority only; it is not used as the current 469-player data baseline.

- run `34298516960`
- artifact `10084118525`
- artifact name `rb-r22-week1-production-integration-v2`
- digest `sha256:2391bd9914e9d0029c63529829496fb9b462d008bdcc3f8c37bb3e1079580bd1`
- required integration disposition `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_PASS`
- required adapter disposition `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_ADAPTER_PASS`
- required R22 gates include `receptions_exact == true`, `mean_parity == true`, `non_rb_exact == true`, and `rb_nonreceiving_markets_exact == true`.

### Protected production-code authority

All production/model code must remain byte-clean versus:
`f8417f55b04ce0e19baf260e9d532765034c47f1`

R22 committed R19 asset hashes must remain:
- model SHA-256 `9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba`
- residual-pool file SHA-256 `c69a268a5a1683e846bcb5f59fe55bcae20d70c679792449ea77e6548b37a362`

## Exact current-identity compatibility

R26O may reuse the already-documented R26N mechanical staging seam:
- immutable current production parent remains untouched;
- staged model-context `player_clean_key` may be derived only by exact one-to-one `(team, player display name)` mapping from immutable PlayerForm;
- no fuzzy matching;
- no name normalization heuristic;
- no players added/removed;
- no football-value changes.

R26O must verify the staged current population is exactly 468 players before simulation.

## Monte Carlo contract

- iterations: exactly `25,000`
- baseline seed: exactly `42`
- candidate seed: exactly `42`
- deterministic replay: candidate V3 must reproduce byte-identical RB/FB reception arrays on a second same-seed replay.

The only allowed difference between baseline metrics and candidate metrics is `entitlement_tgt_share`, and that column must equal the sealed R26N candidate entitlement exactly by `(event_id, team, player_clean_key)`.

## Frozen Monte Carlo mean-parity tolerance

Because R26N stores analytical prospective means while R26O materializes finite Monte Carlo arrays, the frozen compatibility tolerance is:

- max absolute baseline V3 RB/FB reception-mean gap vs R26N `baseline_receptions`: `<= 0.05 receptions`;
- max absolute candidate V3 RB/FB reception-mean gap vs R26N `candidate_receptions`: `<= 0.05 receptions`;
- max absolute simulated reception-mean **delta** gap vs R26N analytical reception delta: `<= 0.05 receptions`.

These are Monte Carlo/reconstruction compatibility gates, not predictive-performance thresholds.

No 2026 outcomes may be used to set or evaluate them.

## Frozen structural gates

R26O passes only if **all** gates below pass:

1. exact R26N artifact digest and required PASS disposition;
2. R26N `shadow_integration_design_authorized == true`;
3. exact current Full Slate artifact digest/head;
4. exact R22 integration artifact digest and required R22 dispositions;
5. protected production-code boundary clean;
6. committed R19 model/pool hashes exact;
7. current staged production population exactly 468 rows / 32 teams / 16 games;
8. R26N overlay exactly 468 rows / 107 RB+FB / 104 changed RB+FB;
9. exact player-key equality between current production reconstruction and R26N overlay;
10. reconstructed baseline entitlement equals sealed R26N baseline entitlement, max absolute delta `<= 1e-12`;
11. candidate metrics differ from baseline metrics only in `entitlement_tgt_share`;
12. candidate entitlement equals sealed R26N candidate entitlement, max absolute delta `<= 1e-12`;
13. baseline V3 simulation has exactly 25,000 draws per modeled array;
14. protected R22 adapter leaves baseline V3 RB/FB receptions element-exact in baseline V4;
15. baseline R22 receiving-yard mean-parity gate passes;
16. candidate V3 same-seed replay is deterministic for all RB/FB receptions;
17. all candidate RB/FB reception arrays finite, nonnegative, integer-valued;
18. baseline V3 RB/FB reception means match R26N analytical baseline within `0.05`;
19. candidate V3 RB/FB reception means match R26N analytical candidate within `0.05`;
20. simulated RB/FB reception-mean deltas match R26N analytical deltas within `0.05`;
21. shadow result changes exactly the 104 R26N-changed vacancy-active RB/FB `receptions` arrays;
22. the three CIN non-vacancy RB/FB reception arrays are element-exact baseline;
23. all non-RB/FB reception arrays are element-exact baseline;
24. every non-`receptions` array in the full shadow result is element-exact baseline;
25. every RB/FB `rec_yards` array is element-exact baseline V4/R22;
26. every RB/FB `rush_rec_yards` array is element-exact baseline V4/R22;
27. every RB/FB `rush_yards` and `rush_att` array is element-exact baseline V4/R22;
28. every QB `pass_yards` array is element-exact baseline V4/R22;
29. full simulation key universe is exact baseline before/after splice;
30. no current/future 2026 outcomes used;
31. no sportsbook football inputs used;
32. no same-week depth used;
33. no R8/R9 fit/refit;
34. no receiving-yard mean changed;
35. no receiving-yard/R22 distribution regenerated by the R26O splice;
36. no production parameter/file changed;
37. no live shadow production activation;
38. no production promotion authority.

## Required outputs

R26O must persist:
- `r26o_disposition.json`;
- `r26o_gate_matrix.csv`;
- `r26o_rb_receptions_shadow_manifest.csv` with baseline/candidate means, R26N analytical means, deltas, quantiles, vacancy state, and per-array SHA-256;
- compressed `r26o_rb_receptions_shadow_arrays.npz` containing the 107 sealed RB/FB shadow reception arrays;
- `r26o_full_result_exactness_audit.csv` summarizing allowed and forbidden array changes;
- reconstruction/identity audit evidence;
- plan / implementation / code hashes through the workflow artifact.

The artifact must be sufficient to reproduce later prospective Week-1 evaluation without rerunning R26N or changing the pregame candidate.

## Dispositions

PASS:
`R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_PASS_READY_FOR_PROSPECTIVE_SEAL`

FAIL:
`R26O_2026_WEEK1_RECEPTIONS_SHADOW_INTEGRATION_FAIL_NO_SHADOW`

## Authority ceiling

A PASS means only:
- the R26N reception mechanism can coexist as a sealed **research shadow reception distribution** with the exact current V4/R22 stack;
- a separately governed prospective-seal / downstream market-observation or postgame evaluation step may be designed using the immutable R26O artifact.

A PASS does **not** authorize:
- replacing production receptions;
- changing R22 receiving yards or rush+receiving distributions;
- changing production pricing;
- changing any production model parameter;
- using 2026 outcomes to alter the sealed candidate;
- production promotion.

The candidate, all 38 gates, the Monte Carlo contract, and the `0.05` compatibility tolerances are frozen before implementation or execution.