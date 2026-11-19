# Allocative prior for ALL position blocks, pregame universe, production carry chain: PREREGISTRATION V1

**Date:** 2026-11-19, frozen before any arm runs. **Research only. No production change.**

**Parent:** `RB_ALLOCATIVE_PRIOR_PREGAME_RESULTS_V1.md`.
- The RB-only allocative prior cut RB carry error by 14-16% (CIs far from 0).
- **K3 fired:** the RB block became coherent while the QB block stayed marginal (0.14 per
  rostered QB, block posterior 0.239 vs actual about 0.135), so under the shared 0.95 cap,
  carries moved from backs to quarterbacks.

This design applies the **same** structure to every block. It adds **no new free parameters
or functional forms**: the same rank-table definition and block-mass definition, fitted the
same way.

**Prior-work check:** the same as the parent. Additionally, no study has applied an
allocative QB, WR or TE rush prior. The closed families (pool selection, precision weights,
prior strengths, cap rescue) are untouched. `GROUP_STRENGTH`, `PRIOR_PLAYER_CAP`, top-5 and
the 0.95 cap are unchanged.

## Design

**Universe, chain, holdouts, training and scored rows:** identical to the parent.
- Pregame roster ACT minus Out/Doubtful;
- Bayes → top-5 → 0.95 cap → x actual team rushes;
- tables from 2019-2024 applied to 2025, and from 2019-2025 applied to 2026 weeks 4-10;
- weeks 4+;
- RBs who took a snap for the primary metric.

**Arms:**
- **C0:** production.
- **C1 (RB block only):** the parent's arm, re-run for reference.
- **C3 (all blocks, the candidate):** for each position block P in {QB, RB, WR, TE}, each
  player's shrinkage target is `alloc_P[K_P][k] x block_P`.
  - **K_P:** the number of block-P players in the team's pregame universe, pooled at 5.
  - **k:** the rank by the same pregame proxy as the parent (current share if current_games
    > 0, else prior share, else 0).
  - **`alloc_P`:** the mean within-block carry share by (K, k), with the same 20-observation
    cell floor. The fallback is C0's group target for that block.
  - **`block_P`:** the mean of block-P universe carries / team carries.
- **C4 = C3 + recency (a = 0.85):** reported, not gating.

## Metrics, kill and pass: the same as the parent, plus one QB guard

- **Primary:** the RB expected-carries MAE gain, C0 minus C3, with a team-game clustered CI.
- **K1:** the primary CI includes 0 in either holdout → null.
- **K2:** committee RBs (K ≥ 3, k ≥ 2) are worse with the CI excluding 0 → rejected.
- **K3:** team RB mass |projected - actual| worsens in either holdout → mis-specified.
- **K4 (new, guarding the block that broke the parent):** team **QB** mass |projected -
  actual| worsens in either holdout → mis-specified.
- **Reported:**
  - QB player expected-carries MAE, C0 vs C3;
  - lead-back bias;
  - the fitted tables and block masses.
- **PASS:** K1-K4 all clear. A pass is a candidate only. Promotion requires:
  - a production replay on the Ourlads universe;
  - a check of its interaction with the RB P3 Week-1 route and the INERT concentration
    transform;
  - prospective grading.

## Prediction (scored after)

- The C3 primary gain is similar to C1's (14-16%).
- The QB block mass falls toward actual.
- The RB team mass moves toward actual, so K3 clears.

**Named risk:** QB rushing is dominated by one player per team. A QB1 with little rushing
history whose backup is a runner, or a starting-QB change, can give the QB rank proxy the
wrong order. The QB table may then misplace mass, and K4 would fire.
