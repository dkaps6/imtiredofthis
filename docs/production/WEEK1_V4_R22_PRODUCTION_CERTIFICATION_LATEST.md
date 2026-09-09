# Week 1 V4 / R22 Production Certification — Latest Handoff

Date: 2026-09-08/09 UTC
Branch: `production-week1-v4-r22`
Base production authority: current `main` at branch cut (`69b96aa0a0180a04107eabea33e08e26e0325eaa`)
Status: CLEAN PRODUCTION CANDIDATE; V4/R22 EXECUTION + GOVERNANCE PASS; FINAL STATIC REPO AUDIT REPAIR IN PROGRESS

## Purpose

Promote the already-certified football-first V3 stack plus RB-R22 Week-1 receiving-tail distribution into a clean production branch without merging the large research-history branch into `main`.

The clean branch intentionally carries only production runtime, frozen/promoted model assets, canonical Full Slate wiring, validators, and provenance required for Week 1.

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

The latter two changes are mechanical runtime packaging only. They do not alter any R22 model coefficient, feature ordering, probability model, residual pool, seed, target entitlement, mean, or frozen scientific gate.

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
- added a minimal compatibility shim at the historical import path exposing only `FEATURES`, `EPS`, `_identity_atlas`, `_attach_identity`.

No football/science gate changed.

### Run `34303368756` — V4/R22 + governance pass; stale repo-audit contract

Head: `9f21792dc841f08a0e4143aa6a32f56395e11cd3`
Job: `102314854104`
Artifact: `10085814044`
Artifact digest: `sha256:88565626e1f9101b909b52b88539230c5cc873e52fadf9c9ab55cec43bc14413`
Conclusion: FAIL only at final static repo audit.

Successful steps:
1. setup/dependencies
2. exact R19 production assets
3. immutable already-paid Week-1 source artifact
4. sportsbook boundary hardening
5. player evidence + stable identity
6. certified football components
7. preserved V3 control
8. canonical V4 candidate through the public Full Slate pricing entrypoint
9. production governance + stack certification

Confirmed V4/R22 outputs in this run:
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

Only failing item:

`python scripts/utils/audit_repo.py --strict`

reported:

`full-slate workflow does not invoke scripts/run_pricing_v2.py`

This is a stale static-audit wiring expectation. The promoted Full Slate intentionally invokes `scripts/run_pricing_with_full_roster_universe_v3.py`, whose public Week-1 entrypoint routes to `run_pricing_with_full_roster_universe_v4_production.py`; V4 preserves the certified V3 core and applies R22 afterward.

The audit must be updated to verify this exact promoted chain rather than requiring the old direct pricing entrypoint. This is a governance-contract repair, not a football-model change.

## Current canonical Week-1 pricing chain on this branch

`.github/workflows/full-slate.yml`
→ `scripts/run_pricing_with_full_roster_universe_v3.py` (public compatibility entrypoint)
→ `scripts/run_pricing_with_full_roster_universe_v4_production.py`
→ preserved `scripts/run_pricing_with_full_roster_universe_v3_core.py`
→ exact R22 receiving-tail adapter
→ downstream sportsbook pricing/lineage.

## Remaining production steps

1. Repair the static repository audit to validate the promoted V4 chain strictly rather than requiring direct `run_pricing_v2.py` workflow invocation.
2. Rerun the clean no-credit production certification.
3. Require the final Week-1 V4 contract step to pass.
4. If green, promote the clean branch to `main`.
5. Run canonical `.github/workflows/full-slate.yml` from `main` once more as the actual production authority.
6. Only then label V4/R22 production-active on `main`.

## Science intentionally still open after Week-1 V4 promotion

- RB receiving entitlement/receptions and receiving-yard mean refinement (R22 is distribution-shape only).
- WR/TE receiving efficiency/distribution calibration.
- fully shared QB↔receiver conservation / coherent game-state simulation.
- dedicated anytime-TD probability model.
- game-level moneyline/spread/total model built from the certified joint football state.

Do not conflate these open science lanes with the clean-production mechanical certification described above.
