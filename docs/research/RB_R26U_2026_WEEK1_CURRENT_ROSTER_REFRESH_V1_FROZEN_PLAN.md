# RB R26U — 2026 Week 1 Current-Roster Pregame Refresh V1 — FROZEN PLAN

Status: **FROZEN BEFORE IMPLEMENTATION / EXECUTION**

## 1. Purpose

R26U is a pregame operational refresh of the already-qualified R26 RB receiving-opportunity mechanism onto the most recent pregame Week-1 roster snapshot.

It answers:

> Can the exact frozen R26 mechanism be rerun on the current 2026 Week-1 RB/FB roster, using each current player’s own strict-prior identity/history and current protected football context, without using Week-1 outcomes or sportsbook values as football inputs?

R26U is **not** a refit, a new R26 model, a postgame study, or a production promotion.

The original immutable R26Q candidate remains untouched and remains the authority for the later R26S prospective postgame evaluation. R26U creates a separate current-roster pregame sidecar candidate for operational Week-1 use/review.

## 2. Why this refresh is necessary

Two complementary pregame readiness audits are already complete:

### Static 35-gate readiness authority
- run `34412854521`
- artifact `10127920603`
- digest `sha256:b619429022a5bfc50257ebbb55600c1ff1299e4b706997d278545f66a469dd5e`
- head `ee6221bb484efb993ba6aa8149085daef0c04db1`
- disposition `RB_WEEK1_2026_PREGAME_OPERATIONAL_READINESS_PASS_PRODUCTION_STACK_READY_R26_SIDECAR_READY`
- 35/35 gates PASS

### Fresh-roster 22-gate readiness authority
- run `34414256467`
- artifact `10128488192`
- digest `sha256:84001f73ecd9c1169c87b6109c705c06a55660ace9d69fd4894aecd3c48e8073`
- head `e9f98e98b0a2811b2409b217a88141bc984ba83a`
- disposition `RB_WEEK1_PREGAME_OPERATIONAL_READINESS_PASS_WITH_ROSTER_DRIFT_WARNING`
- 22/22 gates PASS
- reconstructed protected simulation: 2,892/2,892 baseline arrays exact
- fresh RB/FB rows: 107
- shared sealed/current RB/FB identities: 106
- current-only identity: `NE | Lan Larison`
- sealed-only identity: `NE | Corey Kiner`
- role changes among the 106 shared players: 0

The current player must be modeled from his own identity. R26U must **not** copy, transfer, inherit, or relabel Corey Kiner’s football values onto Lan Larison.

## 3. Immutable scientific authorities

### Protected production code / baseline
- commit `f8417f55b04ce0e19baf260e9d532765034c47f1`
- Full Slate run `34317211395`
- artifact `10090547415`
- digest `sha256:7eab77e41c5879d4f54d87497eee0d1186010784cfc4ef966178930e16fb7c3b`

### Frozen R26 mechanism implementation
R26U must reuse byte-identically:
- R26N builder: `scripts/backtest/build_rb_r26n_2026_week1_unmodified_r26_structural_candidate_v1.py`
- R26N builder commit: `5299ce54575ffcfe33ad203db0ee00285181291f`
- identity-key staging helper: `scripts/backtest/stage_r26n_production_identity_key_repair_v1.py`
- helper commit: `0b4c2df7c14b16bd0e945b425aac7f35e015fa3d`
- dtype compatibility wrapper: `scripts/backtest/run_rb_r26n_with_identity_dtype_compat_v1.py`
- wrapper commit: `9abdaa9a6262aa87ac601d7bc665209b94036b20`

No R26 coefficient, feature order, clipping value, softmax/allocation logic, reliability value, catch-rate bound, vacancy rule, conservation rule, or scientific gate may change.

### R26 qualification parents
R26M:
- run `34390505549`
- artifact `10119429741`
- name `rb-r26m-2026-week1-prospective-qualification-synthesis-v1`
- digest `sha256:1306a3a2e58a0b129ac7e9fe34ad6407d87c491494e8fc27dd0284ba96996b76`
- required disposition `2026_WEEK1_UNMODIFIED_R26_SHADOW_CANDIDATE_DESIGN_QUALIFIED`

R26L:
- run `34389455694`
- artifact `10119058769`
- name `rb-r26l-2026-week1-regime-transportability-v1`
- digest `sha256:3351dfb5bbf6b571174a94ddf0a03179d70786edbb91d316e4fdf2c3cf005c46`
- required disposition `2026_SOURCE_REGIME_MODERN_LIKE_FOR_PROSPECTIVE_QUALIFICATION`
- frozen Week-1 vacancy rooms: 31 teams; CIN is the sole non-vacancy control room.

R19 serialized R9:
- run `34288244770`
- artifact `10080377483`
- name `rb-r19-deployable-tail-scorer-refit-v1`
- digest `sha256:11432b9d7b7f2367935a862b63c30df9f40955e479806ab67d90921b63a907c7`
- inner model SHA `9ed6a98b0022e86992fb468df40a9fd79a54bc87885777ac5955a898b5c292ba`
- training season 2025; reliability 1.0; no R26U refit.

## 4. Pinned current-roster snapshot

Use the exact `current_roles_ourlads.csv` captured inside fresh-readiness artifact `10128488192`.

Do not refetch a different roster inside this R26U execution. This pins a pregame roster snapshot that was already captured before Week-1 outcomes and passed the 22-gate readiness audit.

The pinned snapshot must contain:
- 468 total current Ourlads role rows;
- 32 teams;
- exactly 107 RB/FB rows;
- `Lan Larison` present for NE;
- `Corey Kiner` absent from the current RB/FB set.

A later Week-1 refresh may use a newer snapshot under a new frozen execution, but this run’s roster input is immutable.

## 5. Current pregame football-root construction

Start from a copy of exact protected production artifact `10090547415` and replace **only the roster input** with the pinned current Ourlads snapshot before refreshing player-level pregame context.

Allowed current-root refresh steps using protected production code:
1. replace `data/roles_ourlads.csv` in an isolated staged copy with the pinned current snapshot;
2. keep protected Week-1 schedule/team context and all static promoted model assets exact;
3. run protected PlayerForm v2 loader for season 2026 / prior season 2025;
4. run protected player-scoring enrichment;
5. run protected model-context bridge needed by the promoted entitlement stack;
6. run protected Week-1 P3 no-odds RB context on the refreshed player universe.

No sportsbook fetch is allowed.

No protected production file in `main` is modified. All refresh outputs exist only in the research run workspace/artifact.

## 6. Pregame outcome-leakage block

Before R26 scoring, R26U must prove that the refreshed current root contains **zero 2026 Week-1 observed player-game rows**.

Specifically:
- `data/player_game_logs.csv` may contain historical/prior-season observations;
- it must contain zero rows with `season == 2026` and `week == 1` representing completed/observed player statistics;
- R26 identity history itself remains capped at 2025 / max strict-prior time key < 202601.

If any Week-1 observed player statistics are present, R26U fails closed and must not score a current candidate.

Live/pregame roster identity is allowed. Same-week game outcomes are not.

## 7. Current-player identity rule

For every current RB/FB, including Lan Larison:
- use that player’s own `player_clean_key` and current team identity;
- attach the frozen R26 strict-prior identity atlas through 2025;
- require all 19 frozen R9 features to be finite;
- score the exact serialized R9 model;
- use the current player’s refreshed baseline entitlement/catch-rate context;
- allocate only through the unchanged frozen R26 room mechanism.

Forbidden:
- mapping Larison to Kiner;
- copying Kiner’s historical features, baseline entitlement, target expectation, catch rate, residual, or reception projection;
- fuzzy identity substitution;
- manual value edits;
- outcome-driven backfill.

If Larison cannot be scored from the existing frozen identity/runtime contract, R26U fails. Cold-start methodology would require a separate frozen study.

## 8. Frozen structural gates

R26U PASS requires all of the following:

1. exact fresh-readiness artifact `10128488192` digest/head/disposition verified;
2. exact protected Full Slate artifact `10090547415` digest/head verified;
3. exact R26M/R26L/R19 artifact digests verified;
4. frozen R26N builder/helper/dtype-wrapper remain byte-identical to their pinned commits;
5. all protected production runtime files used by roster refresh are byte-clean against `f8417f55...`;
6. pinned current Ourlads snapshot is 468 rows / 32 teams / 107 RB/FB and contains Larison not Kiner in the current RB/FB set;
7. refreshed PlayerForm current universe is 468 players / 32 teams / 16 Week-1 games;
8. refreshed PlayerForm RB/FB set is exactly 107 unique team-player keys and equals the pinned current Ourlads RB/FB set;
9. refreshed `player_game_logs.csv` contains zero observed 2026 Week-1 player rows;
10. refreshed model-context contains 468 exact current player keys and no unresolved player identity;
11. refreshed P3 context contains 107 unique current RB/FB keys across 32 teams;
12. refreshed P3 RB/FB key set equals refreshed PlayerForm current RB/FB key set exactly;
13. P3 reports Week-1 route, 25,000 iterations, and zero sportsbook inputs on all 107 rows;
14. R26M exact qualification remains valid;
15. R26L exact modern-like parent remains valid;
16. R19 serialized R9 hash/contract remains exact and no refit occurs;
17. frozen R26N structural evaluator executes unchanged and all 28 original structural gates pass on the refreshed root;
18. refreshed R26 candidate contains exactly 107 RB/FB rows / 32 teams;
19. refreshed R26 candidate key set equals the refreshed current RB/FB key set exactly;
20. refreshed candidate contains Lan Larison and does not contain Corey Kiner;
21. Lan Larison’s 19 frozen strict-prior identity features are finite and his R9 residual/candidate entitlement/targets/receptions are finite;
22. 31 R26L vacancy teams still map; CIN remains the sole non-vacancy room;
23. RB-room pool conservation remains <= 1e-12;
24. all-player team entitlement conservation remains <= 1e-12;
25. non-RB/FB entitlement remains exact;
26. CIN non-vacancy RB/FB entitlement remains exact;
27. candidate entitlements/reception expectations are finite and nonnegative;
28. player universe is conserved through R26 overlay;
29. Week-1 outcomes used = 0;
30. sportsbook football inputs used = 0;
31. same-week depth used as a **model feature** = false; the pinned roster is universe identity only;
32. no R9 refit/tuning/model-parameter change;
33. no R22/receiving-yard/rushing/QB/WR/TE production science change;
34. no production promotion or live-shadow activation;
35. original R26Q files/arrays remain untouched and are not overwritten by R26U.

## 9. Descriptive comparison to original sealed R26Q

For the 106 RB/FB identities shared between original R26Q and R26U, report:
- old sealed baseline receptions mean;
- old sealed R26 receptions mean;
- refreshed baseline receptions mean;
- refreshed R26 receptions mean;
- current-minus-sealed deltas.

For NE:
- report Corey Kiner only as `SEALED_ONLY_OLD_SNAPSHOT`;
- report Lan Larison only as `CURRENT_ONLY_REFRESHED_SNAPSHOT`.

These comparisons are descriptive; no old value may be used to create a new player value.

## 10. Frozen dispositions

### PASS
`R26U_2026_WEEK1_CURRENT_ROSTER_REFRESH_PASS_READY_FOR_CURRENT_SIDECAR_INTEGRATION`

Meaning: the exact frozen R26 mechanism successfully generated a current-roster Week-1 structural reception candidate from the pinned pregame roster, including any new player from that player’s own identity/runtime contract. This authorizes only a separately frozen current-roster distribution/operational sidecar integration step.

### FAIL
`R26U_2026_WEEK1_CURRENT_ROSTER_REFRESH_FAIL_NO_CURRENT_SIDECAR`

Meaning: any roster, leakage, identity, strict-prior, frozen-model, conservation, or structural gate failed. Do not substitute values or tune the candidate.

## 11. Required artifacts

At minimum:
- `r26u_disposition.json`
- `r26u_gate_matrix.csv`
- `r26u_current_roster_audit.csv`
- `r26u_current_playerform_audit.csv`
- `r26u_current_p3_context.csv`
- exact unchanged R26N structural outputs on the refreshed root;
- `r26u_shared_player_comparison_vs_r26q.csv`
- exact source/provenance hashes for the pinned Ourlads snapshot, protected production parent, R26M/R26L/R19, and frozen R26 implementation files.

## 12. Authority ceiling

Even on PASS, R26U is research/pregame sidecar only. It does not:
- promote R26 into production;
- replace production receptions automatically;
- modify R22 receiving yards;
- modify P3 rushing science;
- activate a live production shadow;
- alter original R26Q/R26S prospective lineage.

---

**Freeze rule:** the scientific/modeling mechanism and 35 gates above may not change after execution begins. Mechanical representation failures may be repaired only through separately frozen value-neutral compatibility fixes. If current-player identity cannot be scored under the existing frozen contract, preserve FAIL and open a separate cold-start study.