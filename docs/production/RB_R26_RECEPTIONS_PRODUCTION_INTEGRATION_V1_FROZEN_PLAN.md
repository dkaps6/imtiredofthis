# RB R26 Receptions Production Integration V1 — Frozen Plan

**Frozen before implementation and qualification results.**

## Decision

The project requires one authoritative pregame RB receptions output. The existing Full Slate RB receptions baseline remains the upstream baseline component, but the qualified R26 receiving-entitlement mechanism will become the final RB/FB receptions refinement for its frozen Week-1 scope if this integration passes every gate below.

This is not a blend and not a second user-facing model. The final pricing layer must consume one receptions distribution and one `model_proj` per player/market.

## Protected parent authority

Production/Full Slate parent:
- commit `f8417f55b04ce0e19baf260e9d532765034c47f1`
- Full Slate run `34317211395`
- artifact `10090547415`
- digest `sha256:7eab77e41c5879d4f54d87497eee0d1186010784cfc4ef966178930e16fb7c3b`

Current `main` at plan freeze is `eea168a510672b4cf10fed16d763521eafe530e0`; `main` is four commits ahead of the protected parent and the only changed file since `f8417f...` is `CURRENT_NFL_RESEARCH_HANDOFF.md`. Production code is therefore still byte-identical to the protected authority at plan freeze.

Protected RB stack entering this change:
- rushing: RB-P3
- receiving-yard distributions/tails: R22 using pinned R19 assets
- receptions: current protected baseline
- QB: M89/M90 + mean-neutral C2
- WR: M38 WR1 + WR-R15 WR2+
- TE: TE-R5P

## Scientific parent evidence

R26 historical/prospective lineage is preserved, including failures:
- R26 parent: run `34356222339`, artifact `10106271075`, digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`, 19/20, disposition `RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW`.
- R26E Week-1 qualification: run `34368268224`, artifact `10110785184`, digest `sha256:2b64fe25a1024136f2bb2cdde42bc74de290b6b9df63b4bb6cd675f6095b8bb7`, 19/20; 2021-2025 Week 1 improved, 2020 Week 1 was harmful.
- R26J/R26K established that 2020 was distinct but did not justify an outcome-fit router or a special 2020 exclusion rule.
- R26L: run `34389455694`, artifact `10119058769`, digest `sha256:3351dfb5bbf6b571174a94ddf0a03179d70786edbb91d316e4fdf2c3cf005c46`, disposition `2026_SOURCE_REGIME_MODERN_LIKE_FOR_PROSPECTIVE_QUALIFICATION`.
- R26M: run `34390505549`, artifact `10119429741`, digest `sha256:1306a3a2e58a0b129ac7e9fe34ad6407d87c491494e8fc27dd0284ba96996b76`, disposition `2026_WEEK1_UNMODIFIED_R26_SHADOW_CANDIDATE_DESIGN_QUALIFIED`.
- R26N: run `34396075045`, artifact `10121598376`, digest `sha256:887929203053cb62904aaaeda9d995c9645163814da181972799f08fe4465c62`, 28/28.
- R26O: run `34399750746`, artifact `10123070453`, digest `sha256:27307ad84c232935aed3b25c0e0c9bff4da70cd2915376f41dfa3f85c25f14d0`, 38/38.
- R26Q immutable prospective seal: run `34400524030`, artifact `10123251043`, digest `sha256:dd3ec0e8e3831ab7f2255c2e5abf343cda8a7943d33a1d4863e52372d6f858a1`, 28/28.
- Week-1 pregame operational-readiness run `34414256467`, artifact `10128488192`, digest `sha256:84001f73ecd9c1169c87b6109c705c06a55660ace9d69fd4894aecd3c48e8073`, 22/22, proved the 107-player P3/R22/baseline/R26 stack can be assembled pregame with zero Week-1 outcomes and zero sportsbook football inputs.

The promotion decision is based on this historical + leakage-safe current-regime evidence. It does **not** claim that 2026 Week-1 outcomes are already known.

## Frozen production mechanism

### 1. Population and current roster

The production adapter consumes the current Full Slate football universe built from the current Ourlads roster/depth snapshot. It must not consume the sealed R26Q player list as the live roster authority.

This deliberately resolves current-roster substitutions such as RB3-for-RB3 changes without importing a stale sealed player into live production.

### 2. Week-1 vacancy scope

R26 remains a Week-1 receiving-entitlement refinement. Team-level offseason room-vacancy state is frozen from the exact R26L authority. The 31 vacancy-active teams are:

`ARI, ATL, BAL, BUF, CAR, CHI, CLE, DAL, DEN, DET, GB, HOU, IND, JAX, KC, LAC, LAR, LV, MIA, MIN, NE, NO, NYG, NYJ, PHI, PIT, SEA, SF, TB, TEN, WAS`

`CIN` is the sole non-vacancy control and must remain exact baseline for RB/FB receptions.

Same-week roster/depth changes can change which current players occupy a room, but they do not rewrite the frozen team-level offseason vacancy classification.

### 3. Strict-prior player identity

Use the repo-pinned R19 serialized R8/R9 identity scorer only:
- model path `data/models/rb_r19_production_v1/rb_r19_tail_scorer_model_v1.json`
- exact inner-model SHA-256 `9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba`
- residual-pool SHA-256 `c69a268a5a1683e846bcb5f59fe55bcae20d70c679792449ea77e6548b37a362`
- history only through 2025
- no R9 fit/refit
- `StandardScaler + Ridge`, alpha 20, prediction clip 1.0, reliability 1.0
- feature order must equal `scripts.modeling.rb_receiving_identity_runtime_v1.FEATURES`.

Current players with no strict-prior history use the existing identity-runtime zero/default behavior; no fuzzy identity or ad hoc manual player adjustment is permitted.

### 4. Entitlement transform

For current RB/FBs only:
1. start from the existing post-TE-R5P/post-WR-R15 explicit target entitlement produced by the protected Full Slate stack;
2. preserve each `(event_id, team)` RB+FB entitlement pool exactly;
3. calculate baseline within-room share;
4. score each current RB/FB with `log(baseline_within_share + EPS) + calibrated_R9_residual`;
5. on the 31 vacancy-active teams, softmax those scores back into the exact same RB+FB pool;
6. on CIN, keep exact baseline entitlement.

No non-RB entitlement may change.

### 5. Receptions-only final overlay

The R26 transform may change **only** RB/FB `receptions` simulation arrays in the final Full Slate result.

Implementation order:

`protected V3 stack -> QB C2 -> R22 rec-yard adapter -> R26 receptions adapter -> pricing`

The adapter may construct a candidate simulation internally from current metrics, but only the candidate RB/FB `receptions` arrays are copied into the final result. Every other final array must remain byte/element exact to the R22-protected V4 result.

Specifically unchanged:
- RB rush yards
- RB rush attempts/carries
- RB receiving yards
- RB rush+receiving yards
- RB touchdowns
- QB markets
- WR markets
- TE markets
- all non-RB markets.

### 6. One authoritative output

For eligible current Week-1 RB/FB reception rows, the final production simulation result contains the R26-adapted `receptions` array. Pricing therefore emits one authoritative `model_proj` from that final array.

The old baseline reception mean/distribution may be written to audit/lineage files, but it is not a competing user-facing betting choice and must not be used by pricing once R26 is applied.

CIN remains baseline by design, but still has one final authoritative output.

### 7. Sportsbook boundary

Sportsbook offers/lines remain downstream only:
- 0 sportsbook rows used to build player universe
- 0 sportsbook inputs to R26 identity, vacancy, entitlement, catch-rate, or simulation generation
- sportsbook may be joined only after final football distributions exist for pricing/edge comparison.

## Frozen qualification gates

PASS requires all gates below. No threshold may be weakened after results.

1. plan predates implementation/results;
2. protected parent production files are byte-identical to `f8417f...` before R26 additions;
3. exact R19 model hash and feature contract;
4. exact R26L 31-team vacancy set and CIN control;
5. current Full Slate universe is 468 players / 32 teams / 16 games, unless a documented live-roster count change occurs; regardless, all 32 teams and 16 games are required;
6. current RB/FB universe covers all 32 teams and has unique `(event_id, team, player_clean_key)` keys;
7. strict-prior identity maximum time is before 2026 Week 1;
8. no R9 refit;
9. RB+FB room entitlement conserved to <= 1e-12 on every team;
10. all-team target entitlement conserved to <= 1e-12;
11. non-RB entitlement delta <= 1e-12;
12. CIN RB/FB entitlement exact baseline;
13. R26 candidate reception arrays are finite, nonnegative integer draws;
14. final result key universe is exact V4 key universe;
15. only RB/FB `receptions` arrays on vacancy-active teams may change;
16. forbidden changed arrays = 0;
17. CIN final RB/FB receptions arrays exact V4 baseline;
18. all RB/FB receiving-yard arrays exact V4/R22 baseline;
19. all RB/FB rush+receiving-yard arrays exact V4/R22 baseline;
20. all RB/FB rushing arrays exact V4/P3 baseline;
21. all QB arrays exact V4 baseline;
22. all WR arrays exact V4 baseline;
23. all TE arrays exact V4 baseline;
24. all non-RB arrays exact V4 baseline;
25. R22 integration audit still passes with receiving-yard mean preservation;
26. P3 final rush pricing remains exact P3 synthesis mean;
27. final priced RB/FB receptions `model_proj` equals the R26-adapted final MC mean for every applied row;
28. exactly one final `model_proj` is emitted per priced reception offer; no baseline-vs-R26 choice enters pricing;
29. R26 pricing lineage columns/audit identify applied rows and preserve baseline mean only as audit evidence;
30. Week-1 outcomes used = 0;
31. sportsbook football inputs used = 0;
32. no outcome-derived tuning/router/blend added;
33. no same-week outcome data used;
34. workflow can run from a clean checkout with repo-pinned production assets;
35. final authority disposition is exactly `RB_R26_WEEK1_RECEPTIONS_PRODUCTION_INTEGRATION_PASS_READY_FOR_PROMOTION`.

## Failure disposition

Any failed gate => `RB_R26_WEEK1_RECEPTIONS_PRODUCTION_INTEGRATION_FAIL_NO_PROMOTION`.

A failure is preserved. Mechanical failures may receive only a separately documented value-neutral repair; scientific/contract failures may not be relabeled.

## Promotion authority

The user explicitly authorized moving R26 into the single authoritative RB receptions pathway. Therefore, if and only if the frozen qualification run passes all 35 gates, the integration may be promoted to `main` as the Week-1 production receptions refinement without waiting for 2026 Week-1 outcomes.

R26S remains the postgame audit of this pregame production decision; it is not a prerequisite for this promotion.
