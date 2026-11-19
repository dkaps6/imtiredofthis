# Receiving: allocative target prior + team budget 1.0: RESULTS V1

**Date:** 2026-11-19.

**Preregistration:** `TARGET_ALLOCATIVE_PRIOR_BUDGET_PREREGISTRATION_V1.md`, frozen at ebf3ca5.

**Script:** `scripts/research/evaluate_target_allocative_prior_budget_v1.py`.

**Rows:** `data/research/target_allocative_prior_budget_v1{,_tables,_rows}.csv`.

**Verdict: PASS on the gates as written**, primary, NI-1, NI-2 and M1, with one gate flagged.
**NI-2 was mis-specified by me: it cannot fail under renormalization**, so team-mass coherence is
**not** established by it. A block-level diagnostic replaces it below.

Candidate only. A prospective test was frozen before week 11:
`TARGET_ALLOCATIVE_PRIOR_BUDGET_PROSPECTIVE_PREREGISTRATION_V1.md`.

## Results (expected targets vs actual, players who played)

| | 2025 (wk 4-18) | 2026 (wk 4-10) |
|---|---|---|
| **WR+TE MAE, C0 → TD1** | 1.594 → **1.488** | 1.631 → **1.503** |
| **gain [95% CI, team-game clustered]** | **+0.106 [+0.083, +0.130]** | **+0.129 [+0.087, +0.168]** |
| T3 (allocative, cap 0.95) gain | +0.112 [+0.089, +0.134] | +0.134 [+0.100, +0.172] |
| TN0 (budget only) gain | +0.007 | +0.010 |
| depth WR/TE (rank ≥ 3), TD1 gain | +0.155 [+0.127, +0.183] | +0.182 [+0.144, +0.221] |
| **WR1 bias (actual - proj), C0 → TD1 / T3** | +0.81 → -0.48 / -0.04 | +1.26 → -0.36 / +0.07 |
| TE bias, C0 → TD1 | +0.05 → -0.16 | -0.22 → -0.33 |
| RB targets gain, TD1 | +0.047 | +0.070 |

**A 7-8% cut in WR+TE target error.** That is about a third of the rushing gain (19-21%), as
predicted from the target-share noise floor.

**WR1 under-projection is removed:**
- T3 lands it almost exactly (-0.04 / +0.07);
- TD1 slightly over-corrects (-0.48 / -0.36).

**The allocative prior is the entire gain.** The budget-only control adds about 1%.

## NI-2 was not an informative gate

With the team stage renormalized to 1.0 and multiplied by **actual** team targets, the projected
team total equals the actual team total, less the targets to players outside the universe. NI-2
(team target mass) therefore passes by construction: 1.54 → 0.06. **It tests nothing.**

The informative check is **block** mass, below. The rushing K3 gate was a block gate (RB mass)
and remains informative.

## Block mass, per team-game, players who played

| | 2025 actual | C0 | T3 | TD1 | 2026 actual | C0 | T3 | TD1 |
|---|---|---|---|---|---|---|---|---|
| **WR** | **17.45** | 15.03 | 15.00 | **15.90** | **18.20** | 14.89 | 15.41 | **16.28** |
| TE | 7.35 | 7.20 | 7.41 | 7.85 | 7.59 | 8.31 | 8.21 | 8.68 |
| RB | 5.43 | 5.26 | 5.51 | 5.85 | 5.77 | 5.33 | 5.64 | 5.97 |

- **WRs are still short by about 1.5-2 targets per team-game** in every arm.
- TD1 closes about a third of that.
- **TEs are over-projected** under TD1.

The likely mechanism is the rules layer's measured target multipliers:
- **TE x1.265** and **RB x1.20** inflate those blocks before the team stage;
- under the cap (C0, T3) or renormalization (TD1), that mass comes out of WRs.

**Not tested here.** Next: `RECEIVING_RULE_SEMANTICS_INTEGRITY_V1_STAGE1_RESULT.md` found
receiving-rule semantic defects, which must be checked before any study of those multipliers.

## Prediction scored

- "TD1 passes, smaller gain than rushing": **right** (7-8% vs 19-21%).
- "WR1 bias shrinks by more than half": **right** (0.81 → 0.48 abs, 1.26 → 0.36).
- "Team target mass improves": **true but uninformative** (NI-2, by construction).
- **Named risk** (the noise floor makes the primary null): **did not happen.** CIs are clear
  of 0.
- **Second named risk** (M37 sharpening already captures the WR shape): **did not happen.** The
  allocative prior still removes the WR1 under-projection on top of M37.

## Disposition

**Candidate, not promoted.** Promotion requires:
- the prospective test;
- an end-to-end rec_yards and receptions test, with side accuracy scored **separately** from
  projection accuracy (`PRISM_ALPHA_COMPRESSION_BOARD_CHECK_V1.md` §3);
- the rules-multiplier question resolved;
- an Ourlads replay.

No sportsbook variable was used.
