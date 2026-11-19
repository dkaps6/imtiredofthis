# Allocative prior + corrected team budget (cap/residual): PREREGISTRATION V1

**Date:** 2026-11-19, frozen before any arm runs. **Research only. No production change.**

**Operator authorization (2026-11-19):** "Yes you should reopen it." This reopens the CLOSED
family "residual/top-five normalization rescue".

## Why the closure is reopened: the closure dissected

**The closure:** `NFL_HANDOFF_2026-09-26_WEEK3_FULL_SLATE_REPAIRED_SYSTEMS_AUDIT_CURRENT.md`
§7 says: *"current Week-3 residual state does not support the historical seam as a live
normalization target."*

Its evidence (§A, same doc) was that vacated opportunity does **not** create a large
residual, because affected teams renormalize to 0.95 modeled and 0.05 residual. *"The
unresolved problem is successor identity/concentration, not missing team mass."*

**Validity:**
- **Valid for its question:** does an unavailable player leave unallocated mass? On Week 3 he
  did not.
- **Its premise was later contradicted.** `PRISM_RB_RUSH_SHARE_MASS_BUDGET_V1.md` ran the real
  `simulate()` on the W7 production universe and measured:
  - team RB carries **19.77 vs 23.15 actual, bias -3.38 [-6.36, -0.53]**;
  - carries by genuinely unmodeled positions = **0.00 on 27/27 teams**, so the empirical
    budget is 1.0.
- "Not missing team mass" was **false**, and the 0.05 reserve has **no empirical basis**. That
  is new evidence, which is the closure's own stated condition for reopening.

**The parent result motivating this test:** `ALLOCATIVE_PRIOR_ALL_BLOCKS_PREGAME_RESULTS_V1.md`.
- The all-block allocative prior cut RB carry MAE by 15-17%.
- K3 fired because the coherent prior (team sum 1.001) still passed through the 0.95 cap,
  which trimmed 66% of teams, and a residual that grew from 0.058 to 0.084.

## Scope

**Changed:** rush-share allocation only, at the cap/residual stage:
- **unchanged:** `_top_n_shares(5)` and the per-player 0.95 clip;
- **changed:** the **team** cap/residual stage only.

Receiving allocation, which uses the same `_allocate_counts`, is **out of scope**. A production
change would need a separate receiving test.

**Unchanged:** the universe, Bayes weights, tables, holdouts and scored rows, all identical to
the parent.

## Arms

- **C0:** production (marginal priors, team cap 0.95).
- **C3:** all-block allocative prior, team cap 0.95 (the parent's candidate).
- **N0 (control):** production priors, with the team stage **renormalized to exactly 1.0**
  (residual 0).
- **D1 (candidate):** all-block allocative prior, with the team stage **renormalized to
  exactly 1.0** (residual 0). This is the conservation law in `PRISM_RB_RUSH_SHARE_MASS_BUDGET_V1.md`
  §5, with the target set to the measured budget of 1.0.
- **D2 (reported):** all-block allocative prior, with the team cap **raised to 1.0** (scale
  down only if the sum exceeds 1.0; otherwise a residual of 1 - sum).

## Gates (vs C0, both holdouts: 2025 and 2026 wk 4-10)

- **K1:** the D1 primary (RB expected-carries MAE gain) CI includes 0 → null.
- **K2:** committee RBs (K ≥ 3, k ≥ 2) are worse with the CI excluding 0 → rejected.
- **K3:** team RB mass |projected - actual| worsens → mis-specified.
- **K4:** team QB mass |projected - actual| worsens → mis-specified.
- **K5 (control):** if N0 matches D1 within CI on the primary, the gain is "the budget
  alone", and the allocative prior is not needed. Recorded, not a kill on D1.
- **PASS:** K1-K4 all clear for D1.

**Promotion (separately) also needs:**
- a receiving-side test of the same cap change;
- a production replay on the Ourlads universe;
- a check of its interaction with RB P3 Week-1 and the INERT concentration transform;
- prospective grading.

## Prediction (scored after)

- D1 passes.
- Team RB mass bias falls near 0.
- The primary gain exceeds C3's (more than 17%).
- N0 alone improves mass but not shape, so it is well below D1 (K5 not fired).

**Named risk:** forcing every team to sum to 1.0 also scales up backs who will not play and
committee backs on teams whose posterior sum is under 1.0. Either committee backs get worse
(K2) or the QB block overshoots (K4).
