# RB Receiving Research Handoff — R23 through R26I

Status: ACTIVE
Last updated: 2026-09-09
Production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Current research branch: `research-rb-r26i-week1-selective-restoration-v1`
Current frontier: R26J 2020 Week-1 comparability/offseason-regime source audit.

## Standing governance

This lane follows `docs/research/RESEARCH_COMPONENT_PRESERVATION_DOCTRINE.md`.

A failed full candidate does not imply all mechanics are discarded. Preserve exact failed dispositions/gates, inherit independently supported mechanics, and change only the unsupported component. Do not move thresholds, redefine cohorts, or weaken gates after outcomes are visible.

After RB is resolved/bounded, perform the parked component-salvage audit of older failed/mixed QB/WR/TE research.

## Production authority that must not be disturbed

- RB rushing: `RB_P3_SYNTHESIS_V1`, Week-1 production authority.
- RB receiving tail/distribution: R22 production-certified/main-active.
- R22 preserves receptions and receiving-yard means.
- RB receiving entitlement/receptions/receiving-yard mean remain unresolved research lanes.

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

2020 room-total receiving error improved while summed player allocation error worsened. Balanced turnover (`exits == entrants`) identified as replicated risk state.

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
- R26I `1.328356`

R26I remained materially better than baseline, but surrendered part of R26's gain.

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

Interpretation under component preservation:
- do NOT discard R26 Week-1 logic; it remains broadly strong;
- R26H state distinctions are useful pooled but insufficient to explain 2020;
- conservative fallback costs too much later-season signal;
- 2020 likely represents a deeper offseason/roster/source regime difference not captured by balanced turnover, exit significance, or entrant-history class.

## Current frontier — R26J 2020 comparability/offseason-regime audit

Next step is diagnostic/source-first, not another child prediction.

Freeze before outcome slicing. Use strict-prior/pregame-only state to compare 2020 Week 1 against 2021-2025 on:
- current/prior RB-room size and continuity;
- number/proportion of entrants;
- veteran vs no-prior-NFL entrant composition;
- number of exits and multiple meaningful exits;
- max and summed vacated prior target volume/share;
- returning room receiving-identity concentration where safely available;
- role-replacement symmetry if measurable from strict-prior identity;
- any 2020 COVID-era roster/source/timing semantic difference.

Do not exclude 2020 merely because it is inconvenient. A later comparability decision requires a separately frozen contract and evidence.

## After Week-1 component is resolved/bounded

Study Weeks 2+ separately with current-season recency/role/availability; do not force offseason Week-1 logic onto in-season churn.

## Parked items

- 2026 ESPN/nflverse current hierarchy audit: parked until RB receiving is resolved/bounded.
- RotoBaller Week-1 WR/CB article parser: pinned for later CB work.
- Post-RB component-salvage audit: old failed/mixed QB/WR/TE research.
