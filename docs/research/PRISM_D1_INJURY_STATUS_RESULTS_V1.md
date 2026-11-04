# PRISM — D1 RESULTS: OUT RETENTION IS **EXACTLY ZERO**; REDISTRIBUTION HURTS

**Date:** 2026-11-03
**Preregistration:** `PRISM_D1_INJURY_STATUS_PREREGISTRATION_V1.md`, frozen before any arm ran.
**Verdict: NOTHING PROMOTED — K1.** G2 was stated too broadly and **contradicts A3's own
definition**; per my standing rule a gate found mis-specified is recorded as such and blocks
promotion. **A1 and A2 independently satisfy G2 exactly (0.0e+00)** and their results stand as
measured.
**Artifacts:** `data/research/d1_injury_status_v1.csv`,
`scripts/research/evaluate_d1_injury_status_v1.py`.

---

## 1. Part 1 — the calibration, and it is unambiguous

Empirical opportunity retention, 2026 W1–8, skill positions, baseline from **prior weeks only**:

| designation | n | % who played | **target retention** | rush retention |
|---|---|---|---|---|
| **OUT** | 124 | **0.0%** | **0.0000** | **0.0000** |
| **DOUBTFUL** | 5 | **0.0%** | **0.0000** | **0.0000** |
| QUESTIONABLE | 112 | 64.3% | **0.8535** | **0.5718** |

| | OUT | DOUBTFUL | QUESTIONABLE |
|---|---|---|---|
| **truth** | **0.00** | **0.00** | 0.85 / 0.57 |
| **production's implied** | **1.00** | **1.00** | 1.00 |
| the *intended* rule | 0.50 | 0.50 | 1.00 |

**Not one of 124 OUT skill players took a snap.** Retention is exactly zero — **K6 does not
fire**, the premise holds. Production's implied 1.00 is maximally wrong; the intended 0.50 is
half as wrong but still wrong.

**A second, unpreregistered observation worth recording:** QUESTIONABLE **rush** retention is
**0.5718** against **0.8535** for targets. A questionable RB loses far more of his carries than a
questionable receiver loses targets — plausibly because a hobbled back is spelled while a hobbled
receiver still runs routes. **Production applies 1.00 to both.** Descriptive only, n=50, and it
needs its own study.

## 2. Part 2 — A2 is right, A1 is half-right, A3 is wrong

| arm | mean \|bias\| | gain vs C0 | 95% CI | G3 |
|---|---|---|---|---|
| C0 production | 0.05539 | — | — | — |
| A1 intended ×0.50 | 0.05276 | +0.00264 | [+0.00150,+0.00395] | PASS |
| **A2 calibrated, mass discarded** | **0.05019** | **+0.00521** | **[+0.00290,+0.00779]** | **PASS** |
| A3 calibrated + redistribution | 0.05362 | +0.00177 | [−0.00074,+0.00460] | **fail** |

**On the 58 designated players — where the arms actually act:**

| arm | mean share given | mean \|bias\| |
|---|---|---|
| C0 | **0.1284** | **0.1078** |
| A1 | 0.0936 | 0.0730 |
| **A2** | **0.0502** | **0.0351** |

**A2 cuts the error on designated players by 67%.**

**K5 does not fire.** A2 − A1 = **+0.00257 [+0.00139,+0.00392]**, CI excluding zero. **Both
defects are real and both need fixing**: the artifact ships without status *and* the intended
×0.50 rule was itself badly calibrated.

**G4:** every arm improves every tier; **A2 is best at the top** (Q4 +0.0132). No tier harmed.

## 3. Redistribution is harmful — and my prediction was wrong

I predicted **A3 > A2**. The opposite: **A3 loses half of A2's gain and its CI crosses zero.**

The reason is visible in the mass table (G6, realized **0.9536**):

| arm | mean team sum | distance from truth |
|---|---|---|
| C0 | 1.5110 | 0.5574 |
| A1 | 1.4736 | 0.5200 |
| **A2** | **1.4270** | **0.4734** |
| A3 | **1.5110** | 0.5574 |

**A3 is mass-preserving, so it holds the team at 1.5110 — already 58% above truth.** Handing an
inactive alpha's 0.14–0.25 to his teammates keeps the team inflated; **discarding it moves toward
reality.**

**The general lesson, which contradicts my instinct in both this study and the allocator work:**
mass preservation is only a virtue when the mass is *correct*. With team sums 58% too high, every
mass-preserving transform is locked to a wrong total, and **discarding freed opportunity is
strictly better than redistributing it until the level itself is fixed.** That also retroactively
explains D2's result, where the mass-preserving M1 lost to the targeted M2.

## 4. Why nothing is promoted

**G2 as written — "every arm must leave undesignated players bit-identical to C0" — is
incompatible with A3 by construction**, since redistributing *to* undesignated players is what A3
*is*. Evaluated per arm:

| arm | G2 |
|---|---|
| A1 | **0.0e+00 PASS** |
| A2 | **0.0e+00 PASS** |
| A3 | 1.158e-01 — cannot pass, and never could |

I am recording G2 as **mis-specified**, which blocks promotion, rather than quietly rescoping it
to A1/A2 and declaring a pass. G1 (A3 mass preservation, 2.2e-16) and G5 (no lookahead) both held.

**This is the sixth specification error of this family in one session** — the allocator's
non-mass-preserving transform, the board check's lookahead tiering, V1's copied G7, V1's
unexercised G5, V2's infeasible G7 threshold, and now a gate whose scope contradicts one of its
own arms. **Six for six caught before publication, and still six.** The arms have been right
almost every time; the scaffolding keeps being wrong, and at this count that is a process problem
rather than a run of bad luck.

**Sample limitation:** Part 2 rests on **58 designated rows in two weeks**, and Part 1's n=5 for
DOUBTFUL cannot support a calibrated value on its own (the study used 0.0000 from that cell;
pooling it with OUT is defensible since both measured exactly zero, but it is an assumption, not
a measurement). Part 1's OUT and QUESTIONABLE cells (n=124, n=112) are solid.

**Timing caveat, named in the preregistration and still unmeasured:** nflverse designations are
the **final** report; production consumes an earlier snapshot. Part 1's retention is therefore an
**upper bound** on what a live pipeline could act on, and this study cannot measure that loss.

## 5. Prediction scored

- **OUT retention ≈ 0.00–0.02 — RIGHT**, exactly 0.0000.
- **QUESTIONABLE high, 0.85–0.95 — RIGHT**, 0.8535.
- **A2 and A3 beat C0 decisively — HALF RIGHT.** A2 did; A3's CI crosses zero.
- **A1 beats C0 but loses to A2 — RIGHT**, and K5 confirms both defects.
- **A3 > A2 — WRONG**, and the reason (§3) is the more valuable finding than the prediction would
  have been.

**Four of five.** The named risks were both sound: n=29 (realised as 58) is thin, and the timing
loss remains unmeasured.

## 6. The production fix, which is not a research result

Separate from the arms, and needing no preregistration:

1. **`build_injuries_weekly` must emit its declared SCHEMA.** The shipped artifact has 5 of 11
   columns and is missing every status field.
2. **`scripts/audit_2026_production_readiness.py` must assert the injuries schema** so a file
   without `status` **fails closed** instead of shipping — AGENTS.md §4 already requires this and
   the gap let a dead rule run all season.
3. Once status arrives, the rule's constants should be **0.00 for OUT/DOUBTFUL** and **0.85
   targets / 0.57 rush for QUESTIONABLE**, not a flat 0.50 — and **the freed mass should be
   discarded, not redistributed**, until the level defect is fixed.

## 7. Next

**D3** (phantom yards at zero opportunity: 2.40 receiving, 1.19 rushing) per the agreed sequence.

The interaction warning now has evidence behind it: **C0 1.5110, D2's best arm 1.3772, D1's best
arm 1.4270, truth 0.9536.** Four mechanisms are competing to remove the same **+0.5574**, and
§3 shows the *level* is what governs whether any individual fix helps. **A joint D1+D2+D3 study is
now mandatory before anything ships**, and it should be designed around the total, not the parts.

No sportsbook variable was read anywhere in this study. No production code modified.
