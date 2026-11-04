# PRISM — ALLOCATOR SHAPE FIX, PREREGISTRATION **V2** (MASS-PRESERVING)

**Frozen:** 2026-11-03, **before any V2 arm is computed.**
**Status:** PREREGISTRATION. No result here. Nothing promoted.
**Supersedes:** `PRISM_ALLOCATOR_SHAPE_PREREGISTRATION_V1` — whose arms were mis-specified. See
`PRISM_ALLOCATOR_SHAPE_RESULTS_V1_MISSPECIFIED.md`. The V1 gate failures are **not** evidence
about concentration and are not carried forward as priors here.

---

## 0. Why V2 exists

V1 applied `share^γ × A` with no renormalisation. Because the production normaliser **only
scales down**, a power transform on values below 1 shrank every team vector under the cap and
**deflated every player** — a team retained 61% of its opportunity mass at γ=1.25. V1 therefore
tested *concentration plus deflation*, and its fitted γ≈1.00 was the optimiser avoiding the
deflation rather than rejecting concentration.

V1 §2 cited `_sharpen_wr_target_shares` as precedent **and then did not follow it.** That
function's contract is explicit:

```python
sharpened = wr * mult
sharpened *= total / float(sharpened.sum())   # restore the original WR mass
```

V2 follows it.

## 1. Research question

**Holding allocated mass fixed at E1's level, does concentrating the shape correct the
allocator's tier bias without harming any tier?**

This is now a **pure shape question.** V1 confounded shape with mass; V2 cannot, by construction.

## 2. Arms

`A` = fraction of prior weeks with ≥1 target. `e = share × A` is the E1 vector.

- **C1** production: `share`. Reference only.
- **E1** the incumbent winner and **the comparator**: `e`. Beating production is not the bar.
- **S1′** `w = e^γ`, then **`w ← w · (Σe / Σw)` per team** (mass restored to E1's), then the
  production normaliser. **γ fitted leave-one-week-out** (fit W7 → apply W8; fit W8 → apply W7),
  minimising mean |per-player bias| on the fitting week. Grid γ ∈ [1.0, 2.5] step 0.05.
- **S2′** the same transform with **γ = 1.25 fixed a priori**, robustness arm. Basis unchanged
  from V1: between no concentration (1.0) and the 1.75 of
  `PRISM_RB_CARRY_CONCENTRATION_RESULTS_V1`; **not derived from any outcome in this sample.**

**Nesting check (mechanical, must hold):** at γ = 1.00, S1′/S2′ must reproduce E1 to within
1e-12 per row. If it does not, the implementation is wrong and the run is void. This is the
check V1 lacked.

**Mass check (mechanical, must hold):** for every team-week and every γ on the grid, the team
sum of the arm must equal the team sum of E1 to within 1e-9 **before** the production
normaliser. The V1 defect must be impossible to reproduce silently.

## 3. Sample and metric

Unchanged from V1 so the comparison is like-for-like: W7 + W8 committed production universes,
769 player-weeks, allocation rule scored **in expectation** — no Monte Carlo, no production code
modified. Minimum **100 players with ≥2 evaluated weeks**.

- **M1** mean over players of `|bias_p|`, **excluding never-targeted players** as the decision
  figure (E1 qualification §7a: including them dilutes the metric by 40% of the panel).
- **M2** breadth versus E1.
- **M3** mandatory tier table.

Bootstrap resampled **over players**, 10,000 iterations.

## 4. Gates — all must pass

Unchanged from V1. These were the right gates applied to the wrong arms.

- **G1** arm beats **E1** on M1 (excluding never-targeted), 95% CI excluding 0.
- **G2** breadth ≥**60%**, CI lower bound above 50%.
- **G3** ≤5% of players who drew ≥1 target receive zero allocation.
- **G4 — NO TIER HARMED versus E1:** every tier in M3 has mean gain ≥ **−0.002**.
  **This is the gate E1 itself failed. It is the point of the study.**
- **G5** same sign in W7 and W8 scored separately.
- **G6 — direction:** the 20%+ tier bias must move **toward zero** without overshooting past it.

## 5. Kill conditions

**K1** no arm passes G1 · **K2** G2 fails · **K3** G3 breached · **K4** G4 breached — name the
harmed tier, promote nothing · **K5** weeks disagree · **K6** G6 overshoot · **K7** S2′ matches
S1′ within CI → the fitted parameter is unnecessary, carry the simpler arm · **K8** either
mechanical check in §2 fails → **the run is void and reported as void**, not as a finding.

## 6. Prediction (recorded before running, per repo discipline)

**S1′ and S2′ will both improve the tier table over E1 and the fitted γ will land well above
1.0** — because with mass held fixed, the E1 tier table is structurally a pure shape error: the
low tiers are over-allocated (<5% +0.0145, 5–12% +0.0046) by roughly what the high tiers are
under-allocated (12–20% −0.0231, 20%+ −0.0923), and E1's team sum (0.9573) already sits near the
truth (0.9436). Concentration moves all four tiers toward zero simultaneously.

**My V1 prediction was wrong and the reason was my own specification**, so this prediction
carries no weight as evidence. It is recorded to be scored, not to be relied on.

**The risk I can name in advance:** γ may be unable to fix the 20%+ tier without pushing the
5–12% tier negative, since those two are adjacent in rank and a monotone power transform cannot
move them independently. If so, G4 fails on 5–12% and the honest conclusion is that a one-
parameter monotone transform is the wrong instrument — the shape error is not a single
concentration deficit.

## 7. Scope

A pass licenses **one** follow-up: implementation in `scripts/simulation_v2.py` behind a version
flag, default off; the full production validation suite; **prospective grading before any live
use**; and a separate downstream test for receptions and `rec_yards`, since the allocation rule
is scored in expectation only.

Two weeks remains a thin basis for a shape parameter. **If S1′'s fitted γ differs materially
between W7 and W8, that instability is itself the finding** and must be reported, not averaged.

`rush_shares` out of scope. No sportsbook data in any arm.
