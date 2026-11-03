# PRISM — ALPHA-TIER USAGE SHARE, RESULTS V1 — **B1 PASSES ALL GATES**

**Date:** 2026-11-03
**Preregistration:** `PRISM_ALPHA_TIER_PREREGISTRATION_V1.md`, frozen in commit `ab4836fb`
**before** any arm was scored.
**Status:** POSITIVE RESULT, first of this lane. **Nothing promoted.** Per preregistration §9 a
pass licenses exactly one follow-up: a flagged implementation plus prospective grading.

---

## 1. Sample

Pooled walk-forward, REG only: 2024 W2–18 (prior 2023), 2025 W2–18 (prior 2024), 2026 W2–8
(prior 2025). 7,253 player-weeks, 552 players.

**ALPHA (C1 posterior ≥ 0.20): 1,078 player-weeks** against a preregistered minimum of 400 —
460 / 500 / 118 by season. Powered, unlike the Week 8 test.

---

## 2. Phase 1 — the diagnostic confirmed the prediction

Signed error = prediction − actual. **Negative = production under-projects.**

| population | n | mean signed error | 95% CI |
|---|---|---|---|
| **ALPHA** | 1,078 | **−0.01435** | [−0.02048, −0.00776] |
| NON-ALPHA | 6,175 | −0.00618 | [−0.00827, −0.00405] |

**K0 did not fire.** The alpha gap is **systematic downward bias**, not dispersion, and the bias
is ~2.3× larger for alphas than for everyone else. This is compression toward the position
prior, exactly as hypothesised.

By season: 2024 −0.02188, 2025 −0.00647 (CI includes 0), 2026 −0.01833. The effect is weakest
in 2025.

---

## 3. Phase 2 — gates

| arm | G1 beats C1 | G2 bias→0 | G3 no harm | G4 beats C0 | G5 all seasons | verdict |
|---|---|---|---|---|---|---|
| **B1** tier-aware group weight | **+0.00058** [+0.00016,+0.00102] | **+0.00651** [+0.00550,+0.00751] | +0.00001 | +0.00134 | [+,+,+] | **PASS ALL** |
| B2 logit blend | −0.00053 | −0.00350 | pass | −0.00022 | [−,−,−] | FAIL |
| B3 Beta-Binomial | +0.00049 [−0.00014,+0.00113] | +0.00654 | pass | +0.00124 | [+,+,−] | FAIL |

Mean error on ALPHA: C0 0.07408 / **C1 0.07333** / **B1 0.07274** / B2 0.07386 / B3 0.07284.
Signed: C1 **−0.01435** → B1 **−0.00784**.

**B1 roughly halves the alpha under-projection bias.** The absolute-error gain is small
(+0.00058) because dispersion dominates at this tier — which is precisely why G2 exists, and
why a study judged on MAE alone would have missed this.

### The two failures are informative

- **B2 (logit) made bias worse** (−0.01785 vs −0.01435) and failed in all three seasons.
  Logit blending compresses the top *more*, not less. The scale is not the problem.
- **B3 (Beta-Binomial) cut bias as much as B1** (+0.00654) but failed G1 (CI straddles 0) and
  flipped sign in 2026. **This was the arm the author predicted would win**, on the argument
  that a proportion from counts wants a conjugate update. The conjugate form fixes the bias but
  does not reliably improve error, and is not season-stable. Prediction recorded in the
  preregistration; prediction wrong.

---

## 4. The mechanism check

B1 is `G_eff = G * min(1, g / C)` — it reduces the pull toward the position default in
proportion to how far above that default the player already sits. If compression is the cause,
the correction should **grow with predicted share**. It does:

| C1 predicted share | n | C1 bias | B1 bias | bias cut | abs gain |
|---|---|---|---|---|---|
| .20–.23 | 555 | −0.01022 | −0.00607 | +0.00415 | +0.00042 |
| .23–.26 | 316 | −0.01832 | −0.01118 | +0.00714 | +0.00076 |
| .26–.30 | 189 | −0.01958 | −0.00840 | **+0.01118** | +0.00125 |
| .30+ | 18 | −0.01668 | **+0.00252** | +0.01416 | **−0.00438** |

Monotone through the first three bins. **At .30+ the correction overshoots** — bias flips
positive and absolute error gets worse. n=18, so this is a caution rather than a finding, but
it is the obvious place for the follow-up to cap the correction.

### 2026 spot-check on the players V21 flagged

| player | n | actual | C1 (bias) | B1 (bias) |
|---|---|---|---|---|
| Justin Jefferson | 3 | 0.2848 | 0.2542 (−0.0306) | 0.2630 (−0.0218) |
| Puka Nacua | 3 | 0.3479 | 0.2772 (−0.0706) | 0.2931 (−0.0548) |
| Amon-Ra St. Brown | 4 | 0.3175 | 0.2764 (−0.0411) | 0.2903 (−0.0271) |
| **Ja'Marr Chase** | 4 | 0.2424 | 0.2485 (**+0.0061**) | 0.2536 (+0.0112) |

B1 helps three of four. **Chase is the exception and it is diagnostic:** his *target share* is
already well projected (0.2485 vs 0.2424 actual). So the V21 finding that production projects
Chase at 4.49 receptions against a market implying 6+ **is not a target-share problem for him** —
it must come from team target volume or the target→reception conversion. The elite-receiver
under-projection is therefore **at least two distinct defects**, and only one of them is this one.

---

## 5. Practical size

Teams average ~35 targets per game, so a 0.0065 bias reduction is **~0.23 targets per game**
for an alpha receiver — roughly 3% of an 8-target workload. Real but modest. It will not by
itself close a 4.49-vs-6.0 reception gap.

---

## 6. What this licenses

Per preregistration §9, exactly one follow-up:

1. implement B1 in `bayesian_v2._posterior_mean` **behind a version flag**, default off;
2. re-run the full production validation suite;
3. **grade it prospectively** before any live use;
4. cap the correction above ~0.30 predicted share, where §4 shows it overshoots.

It does **not** license changing live pricing, and it does **not** establish that downstream
yardage or reception projections improve — a separate question with a separate test. `rush_share`
was not evaluated here and the result must not be assumed to transfer.
