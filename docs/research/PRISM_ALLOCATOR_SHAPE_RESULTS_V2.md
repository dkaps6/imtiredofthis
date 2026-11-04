# PRISM — ALLOCATOR SHAPE V2 — **CONCENTRATION REJECTED ON A CLEAN TEST; HEADROOM LOCATED**

**Date:** 2026-11-03
**Preregistration:** `PRISM_ALLOCATOR_SHAPE_PREREGISTRATION_V2.md`, frozen before any arm ran.
**Verdict:** **K1 · K2 · K4.** Nothing promoted. E1 and its §7 qualification stand unchanged.
**But the failure is informative, and the diagnostic that followed it relocates the lane.**

---

## 1. The test is clean this time

Both mechanical preconditions from §2 of the preregistration passed:

| check | requirement | observed |
|---|---|---|
| mass preservation, all 31 γ on the grid | ≤ 1e-9 | **2.2e-16** |
| nesting: γ=1.00 reproduces E1 per row | ≤ 1e-12 | **0.0e+00** |

So unlike V1, this measures concentration and nothing else.

## 2. Gates

| arm | γ | M1 | gain vs E1 | G1 | G2 breadth | G3 | G4 | G6 |
|---|---|---|---|---|---|---|---|---|
| C1 production | — | 0.05436 | −0.01233 | — | — | PASS | — | — |
| **E1 comparator** | — | **0.04203** | — | — | — | PASS | — | — |
| **S1′** fitted LOWO | 1.05 / 1.20 | 0.04140 | **+0.00063** `[−0.00040,+0.00163]` | **FAIL** | 50.5% | PASS | **FAIL** | PASS |
| **S2′** fixed | 1.25 | 0.04124 | **+0.00079** `[−0.00120,+0.00272]` | **FAIL** | 48.9% | PASS | **FAIL** | PASS |

**K1** neither arm's CI excludes zero. **K2** breadth is a coin flip. **K4** G4 breached:

| tier | n | actual | E1 bias | S1′ gain | S2′ gain |
|---|---|---|---|---|---|
| <5% | 463 | 0.0059 | +0.0195 | +0.0011 | +0.0023 |
| **5–12%** | 121 | 0.0817 | +0.0021 | **−0.0025** | **−0.0044** |
| **12–20%** | 98 | 0.1578 | −0.0306 | **−0.0032** | **−0.0070** |
| 20%+ | 87 | 0.2693 | −0.1009 | +0.0054 | +0.0104 |

**γ instability, reportable per §7:** the fitted γ was **1.05** on the W7 fit and **1.20** on the
W8 fit, on a grid running to 2.50. For a one-parameter shape fit on two weeks that spread is
material, and it is a second reason not to carry S1′ forward.

## 3. What the failure actually says

**G6 passed.** Concentration moves the 20%+ tier toward zero (−0.1009 → −0.0849 at γ=1.25). The
direction of the hypothesis was right. It fails on magnitude and on cost:

- it recovers **+0.0104 of the −0.1009** needed — **10% of the problem**;
- it pays **−0.0070** at 12–20% and **−0.0044** at 5–12% to get it.

The 12–20% tier was **already under-allocated** at −0.0306. A monotone transform cannot take
from the over-allocated bottom without also taking from an under-allocated middle, because the
two are interleaved in predicted rank. **This is the risk named in §6 of the preregistration
before the run**, and it is the honest conclusion: *a one-parameter monotone transform is the
wrong instrument.*

## 4. Diagnostic: where the headroom is

Three oracles on the same panel, each granted one piece of future knowledge, E1's mass held
fixed throughout:

| arm | M1 | gain vs E1 | what it is granted |
|---|---|---|---|
| S2′ shape | 0.04124 | +0.00079 | a monotone power transform |
| **ORACLE_ZERO** | 0.03676 | **+0.00527** | knows exactly who draws 0 targets |
| **ORACLE_BOTH** | 0.02513 | **+0.01690** | that **+** perfect within-team ranking |

The two effects are **almost exactly additive** (0.00527 + 0.01165 = 0.01692 against a measured
0.01690), so they are **separable defects**, not one defect seen twice.

**The ranking is not broken.** Within-team Spearman of E1 against actual: **mean +0.762, median
+0.796, only 6% of team-weeks below 0.5.**

**The universe is.** Mean **14.2** players per team-week in the production universe; mean **7.1**
of them draw a target. **Half the universe is fictional**, and E1 still hands never-targeted
players **16.5%** of all allocated mass (8.07 of 49.02 share-units). The bottom tier's excess
(**+9.04** share-units) and the top two tiers' deficit (**−11.78**) are the same mass in the
wrong hands.

And that is why the top of the distribution cannot be reached by sharpening:

| | mean | p90 | max |
|---|---|---|---|
| E1's top-ranked player, predicted | 0.2096 | 0.2614 | 0.3183 |
| **the actual team leader, actual** | **0.2898** | **0.3911** | **0.4615** |
| ORACLE_ZERO's top player, predicted | 0.2551 | 0.3264 | 0.5666 |

E1's single largest producible share is 0.3183 against a realized max of 0.4615. Gating
participation alone lifts the mean top share 0.2096 → 0.2551 and raises the reachable ceiling to
0.5666. **The mass needed for the alpha is already in the team's budget; it is being spent on
players who catch nothing.**

## 5. Caveats that limit §4 — these are ceilings, not available gains

Both oracles use **post-hoc knowledge of a single realized game**, so each includes
unforecastable within-game noise: a WR3 who happened to catch zero on 18 routes is "a zero" to
ORACLE_ZERO, and ranking the realized order means ranking its coin flips. **+0.00527 and
+0.01690 are therefore inflated upper bounds**, and a real participation model captures some
fraction of them.

This is the same caution as `PRISM_SNAP_SIGNAL_AND_NOISE_FLOOR_V1`, where an apparent headroom of
0.039 became **0.00768** once computed on identical rows. **Do not stack these figures against
that one** — different population, different metric, different estimand.

No gate here was renegotiated after seeing a number. No arm is promoted.

## 6. Prediction scored

§6 of the preregistration predicted **both arms would improve the tier table and the fitted γ
would land well above 1.0.** **Wrong on both counts** — γ came in at 1.05/1.20 and two tiers were
harmed. The **named risk** in the same section — that a monotone transform cannot move the
adjacent middle and top tiers independently, so G4 would fail on 5–12% — **was right, and it is
what the study found.**

That is **two consecutive studies where my stated prediction failed** (`PRISM_ALPHA_TIER` picked
B3, B1 won; here, concentration). The useful half of this preregistration was the falsifiable
risk, not the prediction. Recorded so the pattern is visible rather than flattering.

## 7. Where this lane goes

**Shape is closed.** `PRISM_ALLOCATOR_SHAPE_*` V1 (void) and V2 (rejected) together retire the
concentration hypothesis for target allocation. Do not re-run it with another γ parameterisation.

The next instrument is a **participation / activation gate** — predicting *who is in the route
pool at all this week* — and then **top-end ordering**, which is the named-player question of
which of a team's two or three candidates is the alpha in this specific week. Both are
player-identity questions in the sense of AGENTS.md's first principle, not position-aggregate
questions, and the second is where the larger half of the headroom sits.

This connects to the one mechanism already qualified and **not reproducible (K6)** — active-role
participation, which has no script, no workflow and no committed data artifact. **Rebuilding it
reproducibly is now the highest-value open item in this lane**, because §4 says participation is
where a real gain starts, and §5 of `PRISM_AVAILABILITY_SIGNAL_RESULTS_V1_K6` says we cannot
currently reproduce the one result we have about it.

`rush_shares` remains out of scope. No sportsbook data entered any arm.
