# RB Receiving Research Handoff — R23 through R26J

Status: ACTIVE
Last updated: 2026-09-09
Production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Current research branch: `research-rb-r26j-2020-week1-comparability-source-audit-v1`
Current frontier: freeze an R26K-style outcome/mechanism atlas using only R26J-predeclared structural dimensions; no child prediction yet.

## Standing governance

This lane follows `docs/research/RESEARCH_COMPONENT_PRESERVATION_DOCTRINE.md`.

A failed full candidate does not imply all mechanics are discarded. Preserve exact failed dispositions/gates, inherit independently supported mechanics, and change only the unsupported component. Do not move thresholds, redefine cohorts, weaken gates, or create season exemptions after outcomes are visible.

After RB is resolved/bounded, perform the parked component-salvage audit of older failed/mixed QB/WR/TE research.

## Production authority that must not be disturbed

- Production base: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`.
- RB rushing: `RB_P3_SYNTHESIS_V1`, Week-1 production authority.
- RB receiving tail/distribution: R22 production-certified/main-active.
- R22 preserves receptions and receiving-yard means.
- RB receiving entitlement/receptions/receiving-yard mean remain unresolved research lanes.
- No R23-R26J research result has changed production.

## R23-R25

- R23: useful target/reception signal; receiving-yard behavior not robust. No promotion.
- R24: preserved production YPT; targets/receptions improved; receiving-yard gates failed. No promotion.
- R25: receptions specialist; low-history MAE gain came with harmful RMSE/p90/bias compression. No promotion.

Cumulative inference: do not force targets, receptions, receiving-yard mean, and tail into one model. R22 already owns tail shape.

## R26 V1 — parent mechanism to preserve

Canonical:
- run `34356222339`
- artifact `10106271075`
- digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`
- disposition `RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW`
- 19/20 gates passed.

Mechanism:
- non-vacancy room -> baseline exact;
- vacancy -> R9 receiving identity redistributes the fixed RB receiving pool;
- RB-room/team mass conserved;
- non-RB exact;
- receiving-yard mean exact;
- R22 exact;
- sportsbook/future inputs zero.

Supported:
- vacancy is useful state;
- R9 identity is useful;
- pooled targets/receptions improve;
- Week 1 is strong;
- RB1 underprojection improves;
- 5/6 seasons improve.

Only full-candidate failure: 2023 all-season vacancy-incumbent reception MAE `1.174518 -> 1.227870`, ~4.54% worse vs <=2% frozen season cap.

## R26B

Run `34360534200`, disposition `FORENSIC_MECHANISM_IDENTIFIED`.
2023 failure was Weeks 2+, especially late-season churn; Week 1 was not the problem. Failure was player allocation/order, not room-total receiving opportunity.

## R26C

Run `34361409319`; artifact `10108036449`; digest `sha256:1c7ee705179d91b7e3e53c39b57783314bd87876df6c784d5d1930ee5cf79156`.
Disposition `EXIT_SIGNIFICANCE_SOURCE_READY`.
Strict-prior exited-player receiving significance is safely measurable; 685/685 vacancy team-weeks reconstructed, 910 exits, ~89% history coverage, no target-game/sportsbook leakage.

## R26D

Run `34364089085`; artifact `10109078127`; digest `sha256:827d45611a0ff588093f2ab6a5b74eb739dd5602afa9461a713083a8c44e8756`.
Disposition `EXIT_SIGNIFICANCE_EFFECT_MIXED_NO_ROUTER`, 7/8 gates.

Frozen meaningful exit definition:
`max prior targets/game > 1 OR max prior RB-room target share >= 0.25`.
Do not retune.

Meaningful exits pooled: receptions MAE ~1.81% better; low exits ~0.91% worse. All-season router failed temporal replication, but Week-1 R26 improved low/meaningful/unknown exit classes by ~7.22% / 7.62% / 11.67% respectively. Weeks 2+ behave differently.

## R26E — Week-1 qualification

Run `34368268224`; artifact `10110785184`; digest `sha256:2b64fe25a1024136f2bb2cdde42bc74de290b6b9df63b4bb6cd675f6095b8bb7`.
Disposition `WEEK1_COMPONENT_NOT_QUALIFIED_NO_SHADOW`, 19/20 gates.

Pooled Week-1 vacancy incumbents:
- receptions MAE `1.424485 -> 1.312299` (~7.88% better)
- RMSE/bias/p90 improve
- target MAE `1.687842 -> 1.588760`
- RB1 ~11.9% better; RB2+ ~4.0% better
- all-RB Week-1 rec MAE ~5.89% better.

Five of six seasons improve. Sole failure: 2020 `1.303640 -> 1.420534`, +8.97% harmful.

## R26F — 2020 forensic atlas

Run `34365496225`; artifact `10109658591`; digest `sha256:bce3f1a81ff516f962109dcadc9ad1e6efa2273c3a96851e94abbe9e4c332022`.
Disposition `WEEK1_FAILURE_MECHANISM_IDENTIFIED_REPLICATED`.

2020 room-total receiving error improved while summed player allocation error worsened. Balanced turnover (`exits == entrants`) identified as a replicated risk state, but later work showed it is not sufficient by itself.

## R26G — blanket balanced-turnover guard

Run `34365957361`; artifact `10109840313`; digest `sha256:0523055f7b39500d5d6b9fbc7ed85185cae41bf4518889e7f6c58104021c15ca`.
Disposition `WEEK1_BALANCED_TURNOVER_GUARD_MIXED_NO_SHADOW`.

Balanced rooms fell back to baseline. This partially repaired 2020 but was too broad and erased genuine 2024/2025 R26 gains.

## R26H — balanced-turnover role-state atlas

Run `34369024680`; job `102525128622`; artifact `10111095834`; digest `sha256:c3fa00de4262bb98b1f73d6931005f963d0ded4cc9a9fe53734ae421e34ef18f`.
Disposition `BALANCED_TURNOVER_ROLE_STATE_CHILD_DESIGN_SIGNAL`.
Result: `docs/research/RB_R26H_WEEK1_BALANCED_TURNOVER_ROLE_STATE_ATLAS_V1_RESULT.md`.

Supported balanced states:
1. meaningful exit + veteran entrant (`n=37`): rec MAE ~8.22% better;
2. meaningful exit + no-prior-NFL entrant (`n=77`): rec MAE ~4.76% better.

These replicated outside 2020 and authorized R26I design only.

## R26I — Week-1 selective restoration

Frozen plan: `docs/research/RB_R26I_WEEK1_SELECTIVE_RESTORATION_V1_FROZEN_PLAN.md`.
Result: `docs/research/RB_R26I_WEEK1_SELECTIVE_RESTORATION_V1_RESULT.md`.

Valid run:
- head `5d28fa3b3dbe19a5314524e15c4ae85999c02535`
- run `34370018586`
- job `102528540110`
- artifact `10111502087`
- digest `sha256:5389c77d747cf1862e4423ea5d65519b7a1393897a89cce21b5615b38ab0b4b6`
- disposition `WEEK1_SELECTIVE_RESTORATION_MIXED_NO_SHADOW`
- 22/27 gates passed.

First run `34369824459` failed mechanically before science due pandas mask/index alignment. Commit `5d28fa3b...` fixed only mask placement after merge; logic/gates were unchanged.

R26I endpoint logic:
- non-vacancy -> baseline exact;
- unbalanced vacancy -> R26 exact;
- balanced meaningful exit + supported entrant state -> R26 exact;
- unsupported balanced -> baseline exact.

Structural integrity all passed. Max child RB-room target-mass gap `5.329070518200751e-15`; R22/means/production exact; no fit/refit/sportsbook.

Pooled vacancy incumbents:
- baseline rec MAE `1.424485`
- R26 `1.312299`
- R26I `1.328356`.

Roles:
- RB1 `1.803491 -> 1.637160`
- RB2+ `1.186037 -> 1.134076`.

Global Week 1:
- baseline `1.312603`
- R26 `1.235284`
- R26I `1.242177`.

Season R26I vs baseline:
- 2020: `1.303640 -> 1.434717`, **+10.05% harmful**
- 2021: -9.39% better
- 2022: -3.34% better
- 2023: -8.03% better
- 2024: -13.68% better
- 2025: -23.91% better.

Failed gates:
- 2020 <=2% season safety;
- four R26 preservation gates (pooled rec, pooled targets, 2021-25 rec, global W1 rec).

Interpretation:
- do not discard R26 Week-1 logic; it remains broadly strong;
- R26H state distinctions are useful pooled but insufficient to explain 2020;
- conservative fallback costs too much later-season signal;
- audit 2020's pregame regime before another child.

## R26J — 2020 Week-1 comparability source audit

Frozen plan: `docs/research/RB_R26J_2020_WEEK1_COMPARABILITY_SOURCE_AUDIT_V1_FROZEN_PLAN.md`.
Result: `docs/research/RB_R26J_2020_WEEK1_COMPARABILITY_SOURCE_AUDIT_V1_RESULT.md`.

Canonical valid execution:
- branch `research-rb-r26j-2020-week1-comparability-source-audit-v1`
- frozen-plan commit `ff7e2c535a2283c02e0bb1c4db19becb721e3de7`
- valid head `1dc136253bc0b7b27acb72014d7fbffb8f65900c`
- run `34374987828`
- job `102545404859`
- artifact `10113466373`
- digest `sha256:7d886f228487e052ee1617cc5f42b974cd9da6a9d448f3a86b1de82cb4fc4f46`
- disposition `2020_SOURCE_REGIME_DISTINCT_FOR_MECHANISM_FOLLOWUP`.

Initial run `34370743209` failed mechanically before science because R26C's exited-player source state did not guarantee a `player_key` column after feature-frame materialization. Repair commit `1dc136253b...` changed only exited-row counting to `groupby(...).size()`; all frozen science stayed unchanged.

R26J was explicitly **source-only/pregame-only**:
- `actual_*` selected fields: `[]`
- target-game outcome/participation features: `0`
- sportsbook: `0`
- same-week historical depth: false
- production changes: false
- predictions regenerated: false
- R9 refit: false
- R26C reconstruction: `1.0`
- all 10 integrity gates passed.

Population: 170 Week-1 vacancy rooms across 2020-2025.

Frozen distinctness requirement: at least 3 independent structurally distinct dimensions spanning at least 2 of A-D.
Actual result: **10 dimensions spanning all 4 A-D sections**.

Distinct 2020 dimensions:

### A — room continuity / turnover
- `current_room_n`: 2020 `4.4643` vs 2021-25 `3.9987`; above the full later-season range.
- `continuing_n`: 2020 `2.25` vs `2.0026`; above the later-season range.
- `entrants_n`: 2020 `2.2143` vs `1.9960`; above the later-season range.

### B — entrant composition
- `veteran_entry_n`: 2020 `0.8571` vs `0.5598`; ~53.1% higher and above the later-season range.
- `veteran_entry_share`: 2020 `0.1870` vs `0.1424`; ~31.3% higher and above the later-season range.

### C — exited receiving state
- `exit_history_coverage`: 2020 `0.9524` vs `0.9119`; above the later-season range.
- `sum_exit_last8_targets_pg`: 2020 `3.5864` vs `3.3128`; above the later-season range.

### D — baseline returning-room state
- `baseline_room_hhi`: 2020 `0.25247` vs `0.28348`; below the entire later-season range.
- `baseline_top_room_share`: 2020 `0.32547` vs `0.35501`; below the later-season range.
- `incumbent_n`: 2020 `2.25` vs `2.0026`; above the later-season range.

Coherent football interpretation:
**2020 Week-1 vacancy rooms were larger, carried more continuing backs and more entrants (especially established veterans), had more recently vacated receiving work, and entered with a flatter baseline receiving allocation across incumbents.**

That is a plausible football mechanism for why R26 could improve room totals while misallocating individual receptions in 2020.

Important non-distinctions:
- R9 reliability was exactly 1.0 every season;
- R26 allocation-shift magnitude was not unusual;
- balanced-turnover frequency alone did not satisfy the frozen distinctness rule;
- meaningful-exit count was not unusual;
- max exited-player prior share/target rate was not unusual;
- source quality was not degraded in 2020.

Governance consequence:
- R26J **does authorize** a frozen outcome/mechanism follow-up.
- R26J **does not authorize** excluding 2020, weakening its safety gate, shadowing R26, production promotion, R9 refit, or another router directly.

## Current frontier — R26K-style outcome/mechanism atlas

Next action must be frozen before error slicing.

Use only R26J-predeclared structural dimensions/states and the immutable R26/R26J evidence. Then evaluate how those states relate to R26-vs-baseline individual allocation error and whether the harmful mechanism replicates outside 2020.

Primary hypotheses to test without inventing new thresholds after seeing outcomes:
1. larger vacancy rooms with more incumbents + entrants create harder within-room identity allocation;
2. high veteran-entry presence interacts with flatter baseline concentration;
3. high recently-vacated receiving volume plus flat incumbent baseline causes R26 to redistribute in the wrong player order;
4. the mechanism should be assessed for replication in comparable 2021-2025 room states, not treated as a 2020-only exception.

R26K should be mechanism/diagnostic only. A later child design is authorized only if a football-coherent state replicates under a frozen rule.

## After Week-1 component is resolved/bounded

Study Weeks 2+ separately with current-season recency/role/availability; do not force offseason Week-1 logic onto in-season churn.

Then return to remaining major lanes:
1. RB receiving efficiency/yard mean after opportunity is resolved/bounded;
2. QB attempts/dropbacks/pass-rate/YPA/sack-scramble decomposition;
3. selective WR/TE efficiency/distribution;
4. unified coherent game simulation;
5. ATD;
6. game ML/spread/total;
7. final Week-1 prediction package.

## Parked items

- 2026 ESPN/nflverse current hierarchy audit: parked until RB receiving is resolved/bounded.
- RotoBaller Week-1 WR/CB article parser: pinned for later CB work.
- Post-RB component-salvage audit: old failed/mixed QB/WR/TE research.
