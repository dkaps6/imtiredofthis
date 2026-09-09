# RB R25 — Receptions Specialist V1

Status: **FROZEN BEFORE R25 OUTCOME INSPECTION**
Date: 2026-09-09
Base production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Research branch: `research-rb-r25-receptions-specialist-v1`

## Why R25 exists

R23 and R24 found a repeatable mechanistic pattern on 2023-2025: the fixed strict-prior RB-room entitlement/catch-conversion candidate improved **targets and receptions**, while the corresponding receiving-yard mean failed the frozen yardage/tail/role gates. Those failures remain failures and are not being retuned.

R25 therefore asks a narrower question: can the already-defined R23 opportunity/conversion mechanism stand on its own as a **targets/receptions specialist**, with receiving-yard point mean and R22 receiving-yard tail authority held completely unchanged?

R25 is not allowed to use the known 2023-2025 R23/R24 outcomes as its confirmation sample. To avoid post-hoc threshold shopping, the confirmation block is frozen as **2020, 2021, 2022**, using the exact same fixed opportunity/conversion mechanism and no parameter retuning.

## Frozen scientific question

Does the fixed R23 opportunity/conversion mechanism improve individual RB targets and receptions out of sample on 2020-2022 while preserving the finite RB receiving room and every currently certified Week-1 production authority?

The tested chain is only:

`finite RB-room target mass -> individual RB target entitlement -> shrunk catch conversion -> receptions`

R25 does **not** alter receiving-yard mean or distribution.

## Fixed candidate inherited from R23

R25 must use the R23 opportunity/conversion formula unchanged:

- recent window: previous **6 available player games**;
- stabilizing history: previous **16 available player games**;
- RB-room entitlement score:
  - recent player targets
  - plus one stabilizing RB-room pseudo-game weighted by stable RB-room share
  - plus one role-prior RB-room pseudo-game weighted by the existing production room share;
- candidate RB-room shares are normalized to preserve the existing finite RB-room target mass exactly;
- catch conversion is the same empirical-Bayes shrunk prior calculation used in R23;
- catch rate remains clipped to the same R23 bounds `[0.35, 0.95]`;
- no coefficient, window, pseudo-count, clip, role split, or threshold may be changed after R25 results are observed.

The R23 receiving-efficiency/YPR branch is explicitly excluded.

## Protected production authorities

R25 may not change or replace:

1. M89/M90 QB passing-yard point-mean authority.
2. C2 mean-neutral QB passing-yard distribution selector.
3. M38 WR1 + WR-R15 WR2+ entitlement.
4. TE-R5P entitlement.
5. RB P3 Week-1 rushing authority.
6. Current production RB receiving-yard point mean.
7. RB-R22 Week-1 receiving-yard tail transformation and pinned R19 assets.
8. Canonical Full Slate production workflow.
9. Sportsbook independence upstream.

Receiving-yard projections must be byte/value-identical between baseline and R25 candidate in the research outputs.

## Confirmation block

Primary untouched temporal confirmation seasons:

- **2020**
- **2021**
- **2022**

Each season uses only information strictly prior to the player-game kickoff. The immediately prior season may be used by historical builders where timing-safe. 2023-2025 are not scored for R25 promotion because their R23/R24 outcomes are already known. 2026 outcomes are forbidden.

## Primary population

All RB/FB player-games with valid team-game identity and reception labels, reported as:

- ALL RB/FB
- RB1 / lead-back role
- RB2+ / complementary role
- low-history rows (`<3` prior player games)
- established-history rows (`>=3` prior player games)

No cohort can be removed after outcomes are observed.

## Frozen metrics

For **targets** and **receptions** separately:

- MAE
- RMSE
- bias
- Pearson correlation
- Spearman correlation
- median absolute error
- p75 absolute error
- p90 absolute error

Structural diagnostics:

- RB-room mass conservation gap
- negative/non-finite candidate count
- sportsbook input count
- future-outcome usage count
- receiving-yard parity baseline vs candidate (must be exact)
- protected production-authority diff audit against `main@f8417f55b04ce0e19baf260e9d532765034c47f1`

## Frozen promotion gates

All gates must pass.

### Integrity

1. Sportsbook inputs used upstream = **0**.
2. 2026/future outcomes used = **0**.
3. Strict-prior timing contract holds.
4. RB-room target mass conservation max absolute gap `< 1e-10`.
5. Candidate targets/receptions are finite and non-negative.
6. Receiving-yard point means are exactly unchanged between baseline and candidate.
7. Protected production authority files are unchanged from the production base.

### Scientific — pooled 2020-2022

1. **Targets MAE improves** versus production baseline.
2. Targets RMSE is non-worse.
3. **Receptions MAE improves by at least 0.50%** versus production baseline.
4. Receptions RMSE is non-worse.
5. Absolute receptions bias may not worsen by more than **0.05 receptions**.
6. Receptions p90 absolute error may not worsen by more than **2.0%**.
7. Receptions Spearman correlation may not decline by more than **0.01 absolute**.

### Temporal replication

8. Receptions MAE must improve in at least **2 of 3** confirmation seasons.
9. No single season may worsen receptions MAE by more than **1.0%**.

### Role robustness

10. Neither RB1 nor RB2+ pooled receptions MAE may worsen by more than **0.75%**.
11. At least one of RB1 or RB2+ must improve receptions MAE.

## Disposition

- **PASS_RECEPTIONS_SPECIALIST:** every integrity and scientific gate passes. This only authorizes a separate Week-1 production integration/certification study for targets/receptions. It does not authorize direct merge to `main`.
- **MIXED_OR_FAIL_NO_PROMOTION:** any frozen gate fails. Preserve the result. Do not retune R25 or redefine cohorts/gates.

## Required artifacts

- frozen plan + SHA256
- exact implementation SHA256
- 2020/2021/2022 historical validation reports
- per-player predictions
- season/role metrics
- conservation/parity audit
- pooled frozen-gate disposition JSON
- exact run/job/commit/artifact lineage

## What R25 cannot claim

Even if R25 passes, it does not solve RB receiving-yard efficiency/mean, R22 tail science, WR/TE receiving efficiency, shared QB-receiver conservation, anytime-TD calibration, or game-level score/ML/spread/total modeling.

This plan was committed before any R25 2020-2022 candidate outcomes were inspected.