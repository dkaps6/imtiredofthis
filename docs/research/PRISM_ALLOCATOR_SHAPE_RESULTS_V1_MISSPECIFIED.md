# PRISM — ALLOCATOR SHAPE V1 — **MIS-SPECIFIED ARMS; THE QUESTION IS UNANSWERED**

**Date:** 2026-11-03
**Preregistration:** `PRISM_ALLOCATOR_SHAPE_PREREGISTRATION_V1.md`.
**Status:** Arms ran and failed their gates, but **the arms do not test the hypothesis.**
**No conclusion may be drawn about concentration.** Nothing promoted.

---

## 1. What the gates said

| arm | G1 vs E1 | G2 | G3 | G4 worst tier | G5 |
|---|---|---|---|---|---|
| S1 (γ fitted LOWO) | FAIL −0.00056 | FAIL 42% | PASS | FAIL 20%+ −0.00311 | FAIL |
| S2 (γ = 1.25) | FAIL −0.01306 | FAIL 32% | PASS | FAIL 20%+ −0.04703 | FAIL |

Fitted γ came out at **1.00 (apply-to-W7)** and **1.05 (apply-to-W8)** — essentially no
concentration.

## 2. Why those numbers are not an answer

S2 made the **20%+ tier worse** (−0.1486 against E1's −0.1009). A concentration transform that
*hurts* the top tier is incoherent, and inspecting it found the cause:

| γ | team sum of `share^γ × A` | teams above the 0.95 cap |
|---|---|---|
| **1.00** | **0.9573** | 29/54 |
| 1.10 | 0.7967 | 4/54 |
| **1.25** | **0.6067** | **0/54** |
| 1.50 | 0.3880 | 0/54 |

**The production normaliser only ever scales DOWN** (`if total_share > 0.95`). A raw power
transform on values below 1 shrinks the whole vector; the team sum falls under the cap; no
upward scaling occurs; **every player is deflated**. At γ=1.25 the team retains 61% of its
opportunity mass.

So S1 and S2 tested **"concentration *plus* deflation"**. The optimiser choosing γ≈1.00 was
correctly avoiding the deflation, **not rejecting concentration**. The gate failures are
artefacts of the specification.

## 3. The error was mine, and the codebase already had the answer

`_sharpen_wr_target_shares` in `scripts/simulation_v2.py` states its own contract in its
docstring: *"Apply the Migration 37 winner **while preserving total team WR target mass**."*
`_sharpen_rb_rush_shares` is likewise described as **mass-preserving**.

**Both existing concentration transforms in this repository are mass-preserving. Mine was not.**
The precedent was there and I did not follow it.

This was caught by refusing to accept a counterintuitive result — a concentration arm hurting
the tier concentration is supposed to help — rather than recording it as a finding. Had S2
merely underperformed without that inversion, the mis-specification would plausibly have been
written up as "concentration does not help."

## 4. Status

- **No conclusion about concentration.** The hypothesis in
  `PRISM_ALLOCATOR_SHAPE_PREREGISTRATION_V1` §1 is **unanswered**.
- E1 and its §7 qualification are untouched and remain the state of that lane.
- The V1 gate results above must **not** be cited as evidence against a concentration fix.

## 5. Next

V2 preregisters the **mass-preserving** form, matching the existing codebase contract:

```
w = share^γ
w = w · (Σ share / Σ w)      per team — restores the original mass
w = w · A                     availability weighting
then the production normaliser (clip, scale down only if the sum exceeds 0.95)
```

Written before any V2 arm is computed, so the redesign is driven by the specification defect
alone and by no observed V2 outcome.
