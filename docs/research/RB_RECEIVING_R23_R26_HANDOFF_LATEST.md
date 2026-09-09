# RB Receiving Research Handoff — R23 through R26D

Status: ACTIVE
Last updated: 2026-09-09
Production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Current research branch: `research-rb-r26d-exit-significance-effect-atlas-v1`

## Standing governance

This lane follows `docs/research/RESEARCH_COMPONENT_PRESERVATION_DOCTRINE.md`.

A failed full candidate does not imply all of its mechanics are discarded. Supported components are inherited into the next candidate whenever their contribution can be isolated, while the original failed run and failed gate remain preserved exactly.

## Production authority that must not be disturbed

- RB rushing: `RB_P3_SYNTHESIS_V1` Week-1 production authority.
- RB receiving distribution/tail: R22 production-certified and main-active.
- R22 preserves receiving-yard means and receptions; it changes only the certified RB receiving-yard tail/distribution behavior.
- Current RB receiving entitlement/receptions/receiving-yard mean remain unresolved research lanes.

## R23/R24/R25 summary

- R23 found useful target/reception signal but did not produce robust receiving-yard mean/tail behavior. No promotion.
- R24 preserved production YPT while testing the improved entitlement. Targets/receptions improved, but receiving-yard gates still failed. No promotion.
- R25 isolated a receptions specialist on 2020-2022. Full scientific disposition failed. Low-history MAE improved but RMSE/p90 and bias showed harmful compression. No promotion.

The correct cumulative inference was to stop forcing one model to solve targets, receptions, receiving-yard mean, and tail simultaneously.

## R26 V1 — parent mechanism to preserve, full candidate not promoted

Canonical result:
- workflow run: `34356222339`
- artifact: `10106271075`
- artifact digest: `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`
- full-candidate disposition: `RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW`
- frozen gates passed: `19 / 20`

### Supported components to preserve

1. Non-vacancy RB rooms stay exactly on the production baseline.
2. Vacancy is a real pregame state worth modeling.
3. R9 receiving identity is generally useful for redistributing a fixed RB receiving pool.
4. RB-room target mass conservation remained exact.
5. Team entitlement remained exact.
6. Non-RB entitlement remained exact.
7. Receiving-yard production means remained exact.
8. R22 authority remained exact.
9. Sportsbook inputs upstream remained zero.
10. Future/target-game outcome leakage remained zero.
11. Global receptions MAE improved modestly and RMSE/bias/p90 were non-worse/improved.
12. Week-1 receptions MAE improved about 5.9%.
13. Vacancy-incumbent targets/receptions improved pooled.
14. Vacancy RB1 underprojection was materially repaired pooled.
15. The mechanism improved vacancy-incumbent reception MAE in 5 of 6 seasons.

### Failed/unsupported component

The activation gate was too coarse: **any ACT/INA RB/FB roster exit** triggered the R9 redistribution as though all exits vacated meaningful receiving work.

The decisive frozen failure was 2023 vacancy-incumbent receptions MAE:
- baseline `1.174518`
- candidate `1.227870`
- about 4.54% worse

This failed the predeclared no-season-worsens-more-than-2% gate. No shadow was authorized.

### Do-not-change list for the next child candidate

Unless a separately frozen study later disproves them, preserve:
- R9 identity formula/mechanics;
- finite RB target pool;
- non-vacancy production-exact behavior;
- non-RB exactness;
- receiving-yard production means;
- R22;
- strict-prior/leakage contract;
- sportsbook separation;
- the same 20 scientific protections or stronger.

Do not rescue R26 by shrinking the R9 coefficient, changing the 2023 gate, cherry-picking seasons, or redefining cohorts after seeing outcomes.

## R26B — 2023 forensic atlas

Canonical run: `34360534200`
Scientific purpose: explain the sole bad season without refitting R26.

Key result:
- 2023 Week 1 was not the failure; R26 improved the Week-1 regime.
- damage was concentrated in Weeks 2+, especially later-season windows.
- the failure was primarily **within-room player allocation/order**, not total RB-room receiving volume.
- generic vacancy remained a useful concept, but later-season roster churn could falsely activate the mechanism.

Disposition: `FORENSIC_MECHANISM_IDENTIFIED`.

## R26C — exited-player significance source audit

Corrected canonical run: `34361409319`
Head: `cb051095a28a975c249230ed493a5fde76101424`
Artifact: `10108036449`
Artifact digest: `sha256:1c7ee705179d91b7e3e53c39b57783314bd87876df6c784d5d1930ee5cf79156`
Disposition: `EXIT_SIGNIFICANCE_SOURCE_READY`

Source integrity:
- 685 / 685 parent vacancy team-weeks reconstructed
- 910 exited RB rows
- prior receiving-history coverage: 89.23%
- vacancy team-weeks with at least one exited-player history signal: 89.34%
- prior lagged-depth coverage: 54.51% diagnostic-only
- target-game outcomes used: 0
- target-game participation used: 0
- sportsbook inputs used: 0
- same-week historical depth used: false
- production parameters changed: false

R26C proves that the significance of the departed RB can be measured with strict-prior football data before constructing a new scientific candidate.

## Current step — R26D Exit Significance × R26 Effect Atlas

R26D is a diagnostic, not a fitted model.

Goal:
Determine whether the effect of the frozen R26 V1 mechanism depends reproducibly on how much receiving responsibility the departed RB carried before exiting.

Inputs must be immutable:
- R26 V1 prediction/evaluation artifact `10106271075`
- R26C source artifact `10108036449`

No R26 coefficient or prediction may be changed in R26D.

R26D will predeclare football-natural source buckets before grading outcomes, using only strict-prior metrics already emitted by R26C. Primary departed-player dimensions:
- prior targets/game: `0`, `0-1`, `1-2`, `>2`, `NO_PRIOR_HISTORY`
- prior RB-room target share: `<0.10`, `0.10-0.25`, `0.25-0.50`, `>=0.50`, `NO_PRIOR_HISTORY`
- last-8 targets/game analogues
- Week 1 versus Week 2+ as an already-known phase distinction, not a tunable threshold

Questions:
1. Does R26 help more when at least one departed RB had meaningful strict-prior receiving usage?
2. Does R26 hurt or become neutral when all departures were low/no receiving usage?
3. Is the direction replicated outside 2023?
4. Is a source-defined significance gate supported across seasons rather than selected to rescue 2023?
5. Does the signal hold for RB1 and RB2+ incumbents separately?

Only if those relationships replicate may a later child candidate replace the coarse `any exit` activation gate with a predeclared significance-aware activation rule while preserving the successful R26 parent mechanics.

## Parked items

- 2026 ESPN/nflverse current hierarchy audit: intentionally parked until the RB receiving lane is resolved/bounded.
- RotoBaller Week-1 WR/CB article parser: pinned for later CB-matchup work; not part of current RB research.
- Prior-position component-salvage audit: after RB is resolved/bounded, review older failed/mixed QB/WR/TE experiments for independently supported subcomponents that may have been discarded before the component-preservation doctrine was adopted.
