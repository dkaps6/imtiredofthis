# PRISM — D2 RESULTS: THE MULTIPLIERS CARRY NO INFORMATION AND SHOULD BE DELETED

**Date:** 2026-11-03
**Preregistration:** `PRISM_D2_MULTIPLIER_MASS_PREREGISTRATION_V1.md`, frozen before any arm ran.
**Verdict:** **K4 fires and its recommendation holds.** **M2 (delete the target multipliers)
passes every gate that can be passed** and beats the preregistered primary arm on the population
the defect actually touches. **K5 fires on M3/M4.** Promotion requires prospective confirmation —
see §6.
**Artifacts:** `data/research/d2_multiplier_mass_v1.csv`,
`scripts/research/evaluate_d2_multiplier_mass_v1.py`.

---

## 1. Mechanical gates

| check | requirement | observed |
|---|---|---|
| G1 — M1 team sum == pre-multiplier sum | ≤1e-12 | **2.2e-16** |
| G1 — M3 team sum == 1.0 | ≤1e-12 | **1.1e-16** |
| G1 — M4 team sum == 1.0 | ≤1e-12 | **1.1e-16** |
| G2 — identity | analytic | M1 = C0·(sb/sc), and sb ≡ sc when every multiplier is 1 |

**Team sums:** C0 **1.5110** → M1/M2 **1.3772** → M3/M4 **1.0000**, against a **realized 0.9536.**
Even forcing the sum to 1.0 overshoots the truth, because the universe contains players who draw
no targets at all — the composition fault from `PRISM_MC_OPPORTUNITY_LEAK_V1`.

## 2. G3 — every arm beats production

| arm | mean \|bias\| | gain vs C0 | 95% CI | G3 |
|---|---|---|---|---|
| C0 | 0.05539 | — | — | — |
| **M1** | 0.05135 | +0.00405 | [+0.00310,+0.00499] | PASS |
| **M2** | 0.05098 | +0.00441 | [+0.00308,+0.00574] | PASS |
| M3 | 0.04623 | +0.00916 | [+0.00572,+0.01266] | PASS |
| M4 | 0.04600 | +0.00940 | [+0.00609,+0.01279] | PASS |

**K4 fires:** M1 − M2 = **−0.00037 [−0.00146,+0.00070]**, CI spanning zero. **The multipliers
carry no information on the mean**, exactly as predicted — they are threshold step functions on
*team-level* coverage rates, applied identically to every TE on a team regardless of which TE
he is.

## 3. A correction to my own output, which reverses the arm ranking

The run printed M2's breadth as **33.3%** against M1's 80.6%. **That comparison is invalid.**
M2 leaves every WR and QB **bit-identical** to C0 — 438 of 769 rows, 57% — and an unchanged row
can never count as "improved." Breadth was penalising M2 for correctly leaving alone what it
should not touch.

Restricted to rows each arm actually changes:

| arm | n changed | improved | mean gain on changed rows |
|---|---|---|---|
| M1 | 459 | 80.6% | +0.00405 |
| **M2** | **204** | **75.0%** | **+0.00993** |
| M3 | 459 | 73.6% | +0.00916 |
| M4 | 459 | 73.0% | +0.00940 |

And on **TE/RB/FB — the only positions the multipliers inflate:**

| arm | improved | mean gain |
|---|---|---|
| M1 | 79.0% | +0.00543 |
| **M2** | 74.6% | **+0.00988** |

**M2 gains nearly double M1 on the affected population.** M1's pooled parity with M2 comes from
somewhere else entirely — **WR**, who have no multiplier at all:

| arm | WR improved | WR mean gain |
|---|---|---|
| M1 | 73.1% | +0.00432 |
| **M2** | **0.0%** | **+0.00000** (unchanged by construction) |

**M1 helps WRs only because WRs are independently over-projected**, and mass-preserving
renormalisation scales them down along with everyone else. **So M1 conflates the D2 fix with an
across-the-board level correction, charging WRs for inflation they did not cause.** M2 is the
clean, targeted fix.

**My preregistered primary arm was M1, and the corrected analysis beats it with M2.** Recorded as
such: the conservatism argument for M1 does not survive contact with the per-position breakdown
that the preregistration itself required be reported.

## 4. G4 — no tier harmed by M1 or M2; K5 fires on M3/M4

Tiers from the C0 estimate (pregame), **top tier first**:

| tier | n | actual | C0 | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|---|---|
| **Q4 TOP** | 192 | 0.1595 | +0.0407 | +0.0229 | +0.0257 | **−0.0274** | **−0.0254** |
| Q3 | 192 | 0.0656 | +0.0605 | +0.0493 | +0.0497 | +0.0180 | +0.0183 |
| Q2 | 192 | 0.0386 | +0.0475 | +0.0400 | +0.0376 | +0.0185 | +0.0168 |
| Q1 low | 193 | 0.0045 | +0.0080 | +0.0069 | +0.0061 | +0.0039 | +0.0033 |

Per-tier **|bias|** change vs C0 — Q4: M1 **+0.0030**, M2 **+0.0036**, M3 **−0.0020** (fails the
−0.002 bar at the boundary), M4 −0.0012 (inside tolerance).

**K5 fires.** M3/M4 win on aggregate but **invert the top tier's sign, +0.0407 → −0.0254** —
over-projection becomes under-projection. The |bias| gate barely registers it because the
magnitude is similar, but the direction flips, and the top tier is the one unsolved population in
this program. **No feasibility-enforcing arm may be promoted on this study**, exactly as §5 of
the preregistration required.

**Conditioning note, to prevent a false contradiction:** Q4 here is the top tier *by prediction*
and is **over**-projected (+0.0407). `IRREDUCIBLE_FLOOR_AND_ALPHA_HAIRCUT_V1` §5 found the top
tier *by true rate* **under**-projected (−0.0392). Both are correct — tiering on prediction
selects players the model likes, tiering on truth selects players who produced. These tables do
not conflict.

## 5. G5 — descriptive only, and it cannot decide anything

| arm | side accuracy | Δ vs C0 | 95% CI |
|---|---|---|---|
| C0 | 55.74% | — | — |
| M1 | 56.60% | +0.85pp | [−0.80,+2.55] |
| **M2** | 54.89% | **−0.86pp** | [−2.61,+0.80] |
| M3 | 54.47% | −1.30pp | [−4.04,+1.33] |
| M4 | 53.19% | −2.58pp | [−5.32,+0.16] |

**Minimum detectable effect 1.70pp.** Every interval spans zero and the whole M1–M2 spread
(1.71pp) is exactly one MDE. **This gate was preregistered as unpassable and it is: nothing here
licenses or blocks any arm.** It is reported because the preregistration required it, and because
the *direction* — M2 and M4 negative — is a standing reminder that the P-versus-S tension found
twice before has not gone away.

## 6. What this supports, and the limitation that caps it

**Recommendation: delete the matchup target multipliers (M2).** It passes G3 and G4, gains
**+0.00988 on the positions it touches**, leaves every other row **bit-identical** so it carries
no collateral, and K4 says the multipliers contain no information to lose.

**The limitation, stated plainly: there is no holdout.** The defect was found on W7+W8 and tested
on W7+W8 — 769 rows, 54 team-weeks. The arms contain **no fitted parameters** (they are
deterministic transforms of values already in the committed universes), so this is far less
dangerous than an in-sample fit of a tuned quantity, and G1/G2 are exact rather than estimated.
**But it is the same two weeks, and that is not out-of-sample.** W3–W5 2026 are permanently lost
(AGENTS.md §5a), so the only honest routes are a prospective week or reconstruction from prior
seasons. **Promotion requires one of those first.**

## 7. Prediction scored

§6 predicted: **M1 beats C0 — RIGHT.** **M1 ≈ M2 on the mean — RIGHT** (K4, CI spanning zero).
**M3/M4 improve aggregate bias and harm the top tier — RIGHT** (K5).

**Named risk — WRONG, and informatively so.** I predicted M1 would *raise* WR shares and might
worsen them. It *lowered* them (+0.0503 → +0.0372), because the multipliers inflate the sum
*above* the Bayesian sum, so returning to that sum scales everyone **down**. I had the sign
backwards on my own arm's construction. The risk's underlying point — that the per-position
breakdown must be reported rather than pooled — **was correct and is what produced §3.**

**Three of four predictions correct.** First study in six where the predictions carried the
analysis rather than the named risk rescuing it.

## 8. Next

Per the user's sequencing: **D1** (an OUT player keeping 50% of his opportunity) then **D3**
(phantom yards at zero opportunity). Both also *remove* mass, so the audit's interaction warning
stands: **the measured excess is finite (+0.5110 per team here) and D1, D2, D3 and the
`ROLE_SHRINKAGE` arms are competing to remove the same mass.** D1 and D3 must each be measured
with D2 **off**, and a joint study must follow before any combination ships.

No sportsbook variable entered any arm; market lines graded only. No production code modified.
