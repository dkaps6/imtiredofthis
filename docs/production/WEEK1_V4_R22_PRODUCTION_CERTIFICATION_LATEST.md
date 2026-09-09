# Week 1 V4 / R22 Production Certification — Latest Handoff

Date: 2026-09-08/09 UTC
Branch: `production-week1-v4-r22`
Base production authority at branch cut: `main` (`69b96aa0a0180a04107eabea33e08e26e0325eaa`)
Current status: CLEAN PRODUCTION CERTIFICATION PASS; PROMOTION PR #512 OPEN; EXACT-HEAD PR CI IN PROGRESS

## Purpose

Promote the already-certified football-first V3 stack plus RB-R22 Week-1 receiving-tail distribution into a clean production branch without merging the large research-history branch into `main`.

The clean branch intentionally carries only production runtime, frozen/promoted model assets, canonical Full Slate wiring, validators, provenance, and a minimal compatibility shim required for Week 1.

## Certified football stack represented by this branch

- 469-player sportsbook-independent football universe across 32 teams / 16 games.
- M89/M90 QB passing-yard mean authority.
- QB C2 mean-neutral passing-yard distribution selector.
- M38 WR1 anchor + WR-R15 WR2+ entitlement.
- TE-R5P production entitlement.
- RB-P3 Week-1 rushing authority.
- RB-R22 Week-1 receiving-yard tail distribution, using exact hash-pinned R19 assets and preserving receiving-yard means/receptions.
- Sportsbook rows remain downstream lookup/pricing only and do not define the football universe or generate football distributions.

## Research/integration parent evidence

R22 production-integration result:
- run `34298516960`
- job `102300245980`
- artifact `10084118525`
- digest `sha256:2391bd9914e9d0029c63529829496fb9b462d008bdcc3f8c37bb3e1079580bd1`
- disposition `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_PASS`

Key R22 certified facts:
- 94 RBs adapted.
- maximum receiving-yard mean delta `5.329070518200751e-15` yards.
- minimum rank Spearman `0.9999999999999999`.
- receptions exact.
- non-RB outputs exact.
- same-checkout V3 vs V4 priced rows `3244` vs `3244`.
- intended probability-change rows observed `202`.
- unexpected probability-change rows `0`.
- sportsbook inputs to R22 football adapter `0`.
- current/future 2026 outcomes used `0`.

Canonical research-branch activation also passed before the clean production transplant.

## Clean production branch lineage

Important commits:

- `3d0d76406edf25f6edda6d1005816fdeaf69b378` — transplant certified V4 runtime/assets into clean branch from `main`.
- `c9c8342cefd13fbb30427ffb988e9a7f39fb0daa` — remove training-only fit scripts from Week-1 production candidate.
- `b419c2ecdf19ff16505dee482fd8d945fa118032` — remove QB production runtime dependency on research/backtest tree.
- `a8785fb7c9a220c418b045151d85fb47b9e7301e` — extract frozen RB receiving identity runtime from research tree.
- `9f21792dc841f08a0e4143aa6a32f56395e11cd3` — add production-safe compatibility shim for the exact R8 identity runtime objects required by R22.
- `7a4fe1e3b832006740dbd244728dc016ca0d9d2f` — document clean-production certification lineage.
- `cc7aa770cf43852dfe08e585467d3a44e190ddf3` — update static repo audit to validate the promoted V4 Full Slate chain.
- `64ee5b4b02e9676766b20ea0498dda71bab80ee6` — update 2026 production-readiness audit to validate the promoted V4 chain rather than requiring direct `run_pricing_v2.py` workflow invocation.

The runtime packaging/audit changes above do not alter any R22 model coefficient, feature ordering, probability model, residual pool, seed, target entitlement, mean, or frozen scientific gate.

The only file retained under `scripts/backtest/` on this clean branch is `evaluate_rb_r8_receiving_identity_v1.py`, a 15-line compatibility shim that re-exports `FEATURES`, `EPS`, `_identity_atlas`, and `_attach_identity` from the production-safe runtime module. It is not the historical R8 backtest implementation.

## Clean-production certification attempts

### Earlier clean attempt — QB runtime dependency

A clean certification attempt failed before V3/V4 because `run_qb_distribution_state_context.py` still imported a helper from `scripts/backtest/`.

Repair: copied the exact strict-prior PBP aggregation behavior into a production-safe module/runtime path. No QB model science changed.

Resulting repair commit: `b419c2ecdf19ff16505dee482fd8d945fa118032`.

### Run `34302225846` — R22 runtime dependency

Head: `b419c2ecdf19ff16505dee482fd8d945fa118032`
Conclusion: FAIL (mechanical packaging)

Passed before failure:
- exact R19 asset hashes;
- immutable no-credit Week-1 source replay;
- sportsbook boundary hardening/quarantine;
- 469-player identity/player evidence;
- all certified football components;
- preserved V3 control.

Failure:
- public V4 entrypoint could not import historical R8 identity helpers because the clean branch intentionally did not contain the research backtest tree.

Repair:
- extracted only the frozen strict-prior identity state/snapshot logic into production-safe `scripts/modeling/rb_receiving_identity_runtime_v1.py`;
- added the minimal compatibility shim described above.

No football/science gate changed.

### Run `34303368756` — V4/R22 + governance pass; stale repo-audit contract

Head: `9f21792dc841f08a0e4143aa6a32f56395e11cd3`
Job: `102314854104`
Artifact: `10085814044`
Artifact digest: `sha256:88565626e1f9101b909b52b88539230c5cc873e52fadf9c9ab55cec43bc14413`
Conclusion: FAIL only at final static repo audit.

Confirmed V4/R22 outputs:
- `RB_R22_WEEK1_RECEIVING_TAIL_PRICING_LINEAGE_PASS`
- 94 adapted RB keys
- 138 priced receiving-yard side rows stamped
- 66 rush+receiving side rows stamped
- 130 RB reception rows present and explicitly unadapted
- max mean delta `5.329070518200751e-15`
- zero sportsbook inputs to the adapter
- production mean parameters changed `0`
- `FULL_SLATE_CERTIFIED_STACK_READY_M38_R15_TE_R5P_C2_R22_WEEK1_WITH_DECLARED_SCIENCE_LIMITATIONS`
- same-checkout R22 integration audit PASS
- 202 allowed probability changes
- 0 unexpected probability changes

Failure was the stale `scripts/utils/audit_repo.py` requirement that Full Slate invoke `scripts/run_pricing_v2.py` directly. Repair commit: `cc7aa770cf43852dfe08e585467d3a44e190ddf3`.

### Run `34304159567` — first repo audit fixed; second readiness audit stale

Head: `cc7aa770cf43852dfe08e585467d3a44e190ddf3`
Artifact: `10086090067`
Artifact digest: `sha256:728a545d4c2c13faf8f8e3a1ef5bffc4c4dcd4aa4f0c1b57a6e29448b0bbb4cd`
Conclusion: FAIL only at `scripts/audit_2026_production_readiness.py --strict`.

All football, V4/R22, governance, same-checkout differential, and the repaired repo audit passed. The remaining P0 was the second stale direct-`run_pricing_v2.py` workflow-token expectation. Repair commit: `64ee5b4b02e9676766b20ea0498dda71bab80ee6`.

### Run `34305881612` — AUTHORITATIVE CLEAN PRODUCTION CERTIFICATION PASS

Head: `64ee5b4b02e9676766b20ea0498dda71bab80ee6`
Job: `102322344207`
Artifact: `10086673577`
Artifact digest: `sha256:2fc1827e1b67f19342005adacf95fc3b68f5a88b03b29bd13b858b72749819d3`
Conclusion: SUCCESS

Every certification step passed, including:
1. setup/dependencies
2. exact R19 production assets
3. immutable already-paid Week-1 replay
4. sportsbook boundary hardening
5. player evidence + stable identity
6. certified football components
7. preserved V3 control
8. canonical V4 candidate through the public Full Slate entrypoint
9. production governance + stack certification
10. strict repository audit
11. strict 2026 production-readiness audit
12. final Week-1 V4 contract
13. evidence upload

This is the authoritative clean-branch certification for promotion review.

## Current canonical Week-1 pricing chain

`.github/workflows/full-slate.yml`
→ `scripts/run_pricing_with_full_roster_universe_v3.py` (public compatibility entrypoint)
→ `scripts/run_pricing_with_full_roster_universe_v4_production.py`
→ preserved `scripts/run_pricing_with_full_roster_universe_v3_core.py`
→ exact R22 receiving-tail adapter
→ downstream sportsbook pricing/lineage.

`run_pricing_v2.py` remains a subordinate audited dependency used by the full-roster runtime; it is no longer the direct Full Slate workflow authority.

## Promotion status

Promotion PR: `#512` — `Promote Week 1 V4 / R22 production stack`

At PR creation:
- head `64ee5b4b02e9676766b20ea0498dda71bab80ee6`
- base `main` `69b96aa0a0180a04107eabea33e08e26e0325eaa`
- branch was 10 commits ahead / 0 behind `main`
- GitHub mergeability resolved to `true`
- exact-head `Repo CI` and `Backtest Historical Input Validation` were in progress

Do not merge until exact-head PR checks are green and no new integrity/governance concern appears.

## Remaining production steps

1. Require exact-head PR `Repo CI` and `Backtest Historical Input Validation` to pass.
2. If green and head is unchanged, merge PR #512 to `main` using exact-head protection.
3. Run canonical `.github/workflows/full-slate.yml` from `main` as actual production authority.
4. Verify final `main` Full Slate outputs/lineage and lock the exact production baseline.
5. Only then label V4/R22 production-active on `main` and begin the next research lane.

## Science intentionally still open after Week-1 V4 promotion

- RB receiving entitlement/receptions and receiving-yard mean refinement (R22 is distribution-shape only).
- Targeted QB attempts/dropbacks/pass-rate/YPA/sack-scramble decomposition sanity while preserving M89/M90 authority absent a separately frozen replacement.
- WR/TE receiving efficiency/distribution calibration around the promoted entitlement stack.
- Fully shared QB↔receiver conservation / coherent game-state simulation.
- Dedicated anytime-TD probability model.
- Game-level moneyline/spread/total model built from the certified joint football state.

Do not conflate these open science lanes with the clean-production certification above.
