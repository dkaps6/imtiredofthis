# PRISM — ALLOCATOR SHAPE FIX, PREREGISTRATION V1

**Frozen:** 2026-11-03, **before any arm is computed.**
**Status:** PREREGISTRATION. No result here. Nothing promoted.

---

## 0. Where this stands

`PRISM_MC_LEAK_FIX_RESULTS_V1` found **E1** (availability-weighted allocation) passes its five
gates, and §7 then qualified it: E1 **harms the 5–12% share tier** (31% improved), and the
flagged implementation was blocked until that is addressed. This study addresses it.

**Diagnosis (descriptive, pre-registration):**

| | team sum before normalisation | teams over the 0.95 cap |
|---|---|---|
| C1 raw share | **1.5110** | 54/54 |
| **E1 share × A** | **0.9573** | 29/54 |

**E1 fixes the mass.** The availability weighting alone brings the team sum from 1.51 to 0.957 —
essentially the cap. What it does not fix is the **shape**, because the freed mass is
redistributed **proportional to share**, giving every tier the same multiplicative lift:

| tier | actual | C1 bias | E1 bias |
|---|---|---|---|
| <5% | 0.0349 | +0.0310 | **+0.0145** still over |
| **5–12%** | 0.0817 | −0.0076 | **+0.0046** flipped to over |
| 12–20% | 0.1578 | −0.0556 | −0.0231 |
| **20%+** | 0.2693 | −0.1403 | **−0.0923** still −34% |

Low tiers remain over-allocated, high tiers remain badly under-allocated. **That is compression**
— the same defect `PRISM_ALPHA_TIER_RESULTS_V1` found B1 correcting in the share *estimator*,
now visible in the *allocator*.

---

## 1. Research question

**Does a concentration transform applied after availability weighting correct the allocator's
shape without harming any tier?**

## 2. Arms

All arms normalise with the production rule (clip to [0,0.95]; scale the team sum **down** to
0.95 only if it exceeds it). `A` = fraction of prior weeks with ≥1 target.

- **C1** production: `share`.
- **E1** the incumbent winner: `share × A`. **This is the comparator — beating production is no
  longer the bar.**
- **S1** `share^γ × A`, **γ fitted leave-one-week-out** (fit on W7 → apply to W8; fit on W8 →
  apply to W7), minimising mean |per-player bias| on the fitting week. Grid γ ∈ [1.0, 2.0] step
  0.05.
- **S2** `share^γ × A` with **γ = 1.25 fixed a priori**, reported as a robustness arm. Basis:
  strictly between no concentration (1.0) and the 1.75 used in
  `PRISM_RB_CARRY_CONCENTRATION_RESULTS_V1`; **not derived from any outcome in this sample.**

Precedent: `_sharpen_wr_target_shares` and `_sharpen_rb_rush_shares` already exist in
`simulation_v2.py`, so a concentration transform is an established shape of change here, not a
novel mechanism.

## 3. Sample and metric

Identical to `PRISM_MC_LEAK_FIX_PREREGISTRATION_V1`: W7 + W8 committed production universes,
769 player-weeks, allocation rule scored **in expectation** — no Monte Carlo, no production code
modified. Minimum **100 players with ≥2 evaluated weeks**.

- **M1** mean over players of `|bias_p|`, reported **excluding never-targeted players** as the
  decision figure (§7a of the E1 qualification established that including them dilutes the
  metric: 40% of the panel).
- **M2** breadth versus E1.
- **M3** mandatory tier table.

Bootstrap resampled **over players**, 10,000 iterations.

## 4. Gates — all must pass

- **G1** arm beats **E1** on M1 (excluding never-targeted), 95% CI excluding 0.
- **G2** breadth ≥**60%** on that population, CI lower bound above 50%.
- **G3 — no starved starters:** ≤5% of players who drew ≥1 target receive zero allocation.
- **G4 — NO TIER HARMED versus E1:** every tier in M3 has mean gain ≥ **−0.002**.
  **This is the gate E1 itself failed. It is the point of the study.**
- **G5** same sign in W7 and W8 scored separately.
- **G6 — direction:** the 20%+ tier bias must move **toward zero** and must not overshoot past
  zero. *A transform that cures under-allocation by creating over-allocation is not a fix.*

## 5. Kill conditions

**K1** no arm passes G1 · **K2** G2 fails · **K3** G3 breached · **K4** G4 breached — name the
harmed tier, promote nothing · **K5** weeks disagree · **K6** G6 breached (overshoot) ·
**K7** S2 (fixed γ) matches S1 (fitted γ) within CI — then the fitted parameter is unnecessary
and the simpler arm is what gets carried forward.

## 6. Scope

A pass licenses **one** follow-up: implementing in `scripts/simulation_v2.py` behind a version
flag, default off; the full production validation suite; and **prospective grading before any
live use**, plus a separate downstream test for receptions and `rec_yards` — the allocation rule
is scored in expectation only.

Two weeks is a thin basis for a shape parameter. **If S1's fitted γ differs materially between
W7 and W8, that instability is itself a finding** and must be reported rather than averaged away.

`rush_shares` is out of scope. No sportsbook data in any arm.
