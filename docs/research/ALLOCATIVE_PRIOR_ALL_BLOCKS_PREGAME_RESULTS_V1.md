# Allocative prior for all position blocks, pregame universe: RESULTS V1

**Date:** 2026-11-19.

**Preregistration:** `ALLOCATIVE_PRIOR_ALL_BLOCKS_PREGAME_PREREGISTRATION_V1.md`, frozen at
482c003.

**Script:** `scripts/research/evaluate_allocative_prior_all_blocks_pregame_v1.py`.

**Rows:** `data/research/allocative_prior_all_blocks_pregame_v1{,_tables,_rows}.csv`.

**Verdict: K3 FIRED AGAIN, smaller. Not a pass. Nothing promoted.**
- K1, K2 and K4 clear.
- The per-player gain is the largest measured in this lane.
- The remaining failure lives in the 0.95 cap/residual stage, which is a **closed research
  family**. Reopening it is an operator decision.

## Results (C3 = all blocks; C1 = RB only, from the parent; C4 = C3 + recency, reported)

| | 2025 | 2026 |
|---|---|---|
| **RB carries MAE, C0 → C3** | 3.256 → **2.761** | 3.457 → **2.854** |
| **gain [95% CI]** | **+0.495 [+0.419, +0.573]** | **+0.603 [+0.460, +0.748]** |
| C4 (+ recency) gain | +0.571 [+0.482, +0.659] | +0.676 [+0.537, +0.826] |
| lead back MAE, C0 → C3 | 4.78 → 3.82 | 5.23 → 4.18 |
| lead back bias (actual - projection), C0 → C3 | +3.74 → +1.75 | +4.12 → +1.91 |
| committee (K≥3, k≥2) MAE, C0 → C3 | 2.48 → 2.20 | 2.55 → 2.17 |
| QBs who played, MAE C0 → C3 | 1.71 → 1.66 (CI crosses 0) | 1.67 → 1.65 (CI crosses 0) |
| **K3: team RB mass \|proj - actual\|, C0 → C3** | 2.547 → **2.684** | 2.753 → **2.959** |
| K4: team QB mass \|proj - actual\|, C0 → C3 | 2.000 → 1.839 | 2.202 → 1.835 |

**Gates:**
- **K1, clear.**
- **K2, clear:** committee backs improve.
- **K4, clear:** the QB block that broke the parent is fixed. QB over-projection falls from
  +1.0 to +0.5 carries per team in 2025, and from +1.3 to +0.9 in 2026.
- **K3, FIRED:** team RB mass error is worse by 0.14 (2025) and 0.21 (2026) carries per
  team-week. That is a third of the parent's 0.54 / 0.66, but it is still a worsening, and the
  gate says no worsening.

## Why K3 still fires: measured (2026 wk 4-10)

| | C0 | C3 |
|---|---|---|
| mean team share sum after top-5 | 1.121 | **1.001** (coherent) |
| teams over the 0.95 cap | 88% | 66% |
| residual after the cap, mean | 0.058 | **0.084** |
| RB posterior held by backs who did not play | 0.209 per team | 0.111 per team |
| fitted block masses | | QB 0.152, RB 0.814, WR 0.030, TE 0.003 (sum 0.999) |

The prior is now coherent: it sums to about 1.0. The chain then loses RB mass in two places:
1. **The fixed 0.95 cap reserves at least 5% for "unmodeled rushers"**, and the residual
   bucket grows when a coherent team sums under 0.95. `PRISM_RB_RUSH_SHARE_MASS_BUDGET_V1.md`
   measured the real unmodeled-position carry mass at **0.00 on 27/27 teams**.
2. **Backs who are active pregame and do not play** still hold 0.111 of share. That is half
   of C0's 0.209, and it is the roster-coverage and time-dependent residual problem in the
   same study.

**The prior cannot fix either.** Both sit downstream of the shrinkage target.

## Prediction scored

- "Primary similar to C1": **right**. It was slightly larger, +0.50 / +0.60 vs +0.44 / +0.55.
- "QB mass falls toward actual": **right.**
- "RB team mass moves toward actual, K3 clears": **wrong.** It moved toward actual compared
  with C1, but not compared with C0.
- **Named risk** (QB ordering misplaces mass, K4): **did not happen.**

## Disposition

**Not passed under the frozen gates. Nothing promoted. No production code changed.**

What is established, with CIs far from 0 on two blind holdouts and a pregame universe:
**allocative shrinkage targets for every block cut RB carry error by 15-17%, and lead-back
under-projection by about 54%.** They also fix the QB over-allocation.

**The next step touches a CLOSED family.** The remaining RB mass loss sits in the 0.95 cap and
residual stage. "Residual/top-five normalization rescue" was **closed** on 2026-09-26/29
(`NFL_HANDOFF_2026-09-26_WEEK3_FULL_SLATE_REPAIRED_SYSTEMS_AUDIT_CURRENT.md`: "current Week-3
residual state does not support the historical seam as a live normalization target").

Since then, the mass-budget study measured the empirical budget at about 1.0, which is new
evidence. Whether that justifies reopening is a **governance decision for the operator**. This
study does not reopen it on its own authority.
