# Week 1 V4 / R22 Production Certification — Latest Handoff

Date: 2026-09-09 UTC
Production authority: `main`
Production commit: `f8417f55b04ce0e19baf260e9d532765034c47f1`
Status: **PRODUCTION ACTIVE — CLEAN PROMOTION + CANONICAL MAIN FULL SLATE VERIFIED**

## Final production disposition

Week-1 V4/R22 has completed the full clean-production path:

1. Research/integration science passed on the dedicated research lineage.
2. A clean production-only branch was cut from `main` and carried only runtime, frozen model assets, validators, provenance, and the minimal compatibility shim required by the certified R22 runtime.
3. Clean production certification passed every football, integrity, governance, and final-contract gate.
4. Promotion PR #512 passed exact-head CI and historical-input validation and was merged to `main`.
5. Canonical `.github/workflows/full-slate.yml` then ran from the merge commit on `main` and completed successfully.
6. No new OddsAPI credits were spent during this final `main` verification; live sportsbook-fetch/pricing steps were skipped while the football stack and repository contracts were exercised.

This document therefore marks `f8417f55b04ce0e19baf260e9d532765034c47f1` as the frozen Week-1 V4/R22 production baseline. New research must branch from this state and must beat or extend it through separately frozen gates; it must not mutate this baseline in place.

## Final main verification

Canonical Full Slate run: `34317211395`

- branch: `main`
- head SHA: `f8417f55b04ce0e19baf260e9d532765034c47f1`
- job: `102355832124`
- conclusion: `SUCCESS`
- artifact: `10090547415`
- artifact digest: `sha256:7eab77e41c5879d4f54d87497eee0d1186010784cfc4ef966178930e16fb7c3b`

Passed main-run stages included:

- season/runtime context
- Ourlads roster
- authoritative team-week map
- TeamForm preseason priors
- promoted QB M89/M90 context
- weather and injuries
- Coverage v2
- PlayerForm v2 / stable identity
- canonical Team Context v3
- Bayesian baseline
- ML v2
- State v2
- empirical rules
- ensemble framework
- promoted RB P3 context
- QB C2 football-only distribution state context
- strict repository audits
- artifact upload

Live sportsbook fetching, live-offer assembly, and pricing were intentionally skipped in this no-credit main verification. The purpose was to verify that the promoted football/runtime authority on `main` executes cleanly without spending additional odds credits.

## Certified Week-1 football stack

The production stack represented by `main@f8417f55...` is:

- 469-player sportsbook-independent football universe across 32 teams / 16 games.
- M89/M90 QB passing-yard mean authority.
- QB C2 mean-neutral passing-yard distribution selector.
- M38 WR1 anchor + WR-R15 WR2+ entitlement.
- TE-R5P production entitlement.
- RB-P3 Week-1 rushing authority.
- RB-R22 Week-1 receiving-yard tail distribution.
- Exact hash-pinned R19 scorer/residual-pool assets beneath R22.
- Receiving-yard means and receptions preserved by the R22 tail adapter.
- Sportsbook rows remain downstream lookup/pricing only and never define the football universe or generate football distributions.

Canonical production pricing chain:

`.github/workflows/full-slate.yml`
→ `scripts/run_pricing_with_full_roster_universe_v3.py`
→ `scripts/run_pricing_with_full_roster_universe_v4_production.py`
→ preserved `scripts/run_pricing_with_full_roster_universe_v3_core.py`
→ exact R22 receiving-tail adapter
→ downstream sportsbook pricing/lineage.

`run_pricing_v2.py` remains a subordinate audited dependency used by the full-roster runtime; it is not the direct Full Slate production authority.

## Authoritative clean-production certification

Run `34305881612`

- head: `64ee5b4b02e9676766b20ea0498dda71bab80ee6`
- job: `102322344207`
- artifact: `10086673577`
- digest: `sha256:2fc1827e1b67f19342005adacf95fc3b68f5a88b03b29bd13b858b72749819d3`
- conclusion: `SUCCESS`

Every certification step passed, including exact R19 assets, immutable paid replay, sportsbook-boundary hardening, player evidence/identity, certified football components, preserved V3 control, canonical V4 public entrypoint, production governance, both strict audits, final Week-1 V4 contract, and evidence upload.

The final two pre-pass failures were mechanical governance-contract issues only:

- run `34303368756`: stale repo audit still required direct `run_pricing_v2.py` Full Slate invocation; repaired by `cc7aa770cf43852dfe08e585467d3a44e190ddf3`.
- run `34304159567`: second stale 2026 readiness audit had the same superseded expectation; repaired by `64ee5b4b02e9676766b20ea0498dda71bab80ee6`.

Neither repair changed model coefficients, seeds, thresholds, gates, entitlement, probability science, or football assumptions.

## R22 research/integration evidence

Authoritative R22 production-integration result:

- run `34298516960`
- job `102300245980`
- artifact `10084118525`
- digest `sha256:2391bd9914e9d0029c63529829496fb9b462d008bdcc3f8c37bb3e1079580bd1`
- disposition `RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_PASS`

Certified R22 facts:

- 94 RBs adapted.
- maximum receiving-yard mean delta `5.329070518200751e-15` yards.
- minimum rank Spearman `0.9999999999999999`.
- receptions exact.
- non-RB outputs exact.
- same-checkout V3 vs V4 priced rows `3244` vs `3244`.
- intended probability-change rows `202`.
- unexpected probability-change rows `0`.
- sportsbook inputs to R22 adapter `0`.
- current/future 2026 outcomes used `0`.

R22 therefore owns **Week-1 RB receiving-yard distribution shape/tails only**. It does not claim to solve target entitlement, receptions, or receiving-yard point mean.

## Production branch lineage retained for audit

Important clean-production commits include:

- `3d0d76406edf25f6edda6d1005816fdeaf69b378` — transplant certified V4 runtime/assets.
- `c9c8342cefd13fbb30427ffb988e9a7f39fb0daa` — remove training-only fit scripts.
- `b419c2ecdf19ff16505dee482fd8d945fa118032` — remove QB runtime dependency on research/backtest tree.
- `a8785fb7c9a220c418b045151d85fb47b9e7301e` — extract frozen RB receiving identity runtime.
- `9f21792dc841f08a0e4143aa6a32f56395e11cd3` — production-safe R8 identity compatibility shim.
- `cc7aa770cf43852dfe08e585467d3a44e190ddf3` — promoted V4 repo-audit contract.
- `64ee5b4b02e9676766b20ea0498dda71bab80ee6` — promoted V4 2026 readiness contract.
- `edb41c72c61ee0c699913876cde7ff7d15623366` — final mechanical Repo-CI TD-prior fixture repair.
- `f8417f55b04ce0e19baf260e9d532765034c47f1` — PR #512 merge / production baseline.

## Next authorized research sequence

Production promotion is complete. The next work is research, isolated from `main`.

1. **RB receiving entitlement / targets / receptions / receiving-yard mean** — next active lane. Preserve R22 tail science and P3 rushing.
2. Targeted QB attempts/dropbacks/pass-rate/YPA/sack-scramble decomposition; preserve M89/M90 unless a separately frozen replacement wins.
3. Selective WR/TE receiving efficiency/distribution calibration around M38+R15 and TE-R5P.
4. Unified coherent joint-game simulation with finite plays/carries/targets, QB↔receiver reconciliation, possessions, TDs, and score-state feedback.
5. Dedicated football-only anytime-TD opportunity/entitlement/probability calibration.
6. Game-level final-score, moneyline, spread, and total distributions from the same certified football state.
7. Week-1 player/game output package only from certified components.

## Research governance after production lock

Every new lane must:

- begin with a frozen plan before outcomes are inspected;
- use strict-prior/leakage-safe football-only inputs upstream;
- keep sportsbook data downstream for comparison only;
- preserve all failed/null experiments;
- avoid threshold shopping or post-result gate changes;
- score player-level MAE/RMSE/bias/correlation and error tails, plus calibration metrics where appropriate;
- preserve promoted authorities unless a separately frozen candidate passes stronger gates;
- retain exact branch, commit, workflow-run, artifact, and disposition lineage.

The current research branch is cut directly from the frozen production SHA so later research cannot be confused with the production-active baseline.