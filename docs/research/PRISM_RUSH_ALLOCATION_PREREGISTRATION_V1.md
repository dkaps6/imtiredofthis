# PRISM — RUSH CARRY ALLOCATION, PREREGISTRATION V1

**Frozen:** 2026-11-04, **before any arm is computed.**
**Status:** PREREGISTRATION. No result here. Nothing promoted.
**Source:** `PRISM_JOINT_D1_D2_D3_RESULTS_V1.md` §5 — D3 is correct but exposes a 14% rushing
under-projection that blocks it from shipping.

---

## 0. The gap, decomposed

`rush_yards = carries × ypc`. With D3 on:

| quantity | model | realized | ratio |
|---|---|---|---|
| total carries | 1333.8 | 1465.0 | **0.910** |
| implied ypc | 4.173 | 4.369 | **0.955** |
| **rush yards** | **5565** | **6400** | **0.870** |

**Roughly two-thirds of the gap is carries, one-third efficiency.** This study addresses the
carries. **`ypc` is explicitly out of scope** and gets its own preregistration.

### The carry defect is a misallocation, not a shortfall

Allocated rush share by position, against realized carry share (W7+W8):

| position | model | realized | error | mechanism |
|---|---|---|---|---|
| RB | 76.9% | **85.3%** | **−8.4pp** | crowded out by the three below |
| QB | 17.1% | 11.8% | **+5.3pp** | `DEFAULTS["rush_share"]["QB"] = 0.14` |
| **FB** | **6.0%** | **0.1%** | **+5.9pp** | **`_position_group` maps FB → RB**, so FBs inherit the **0.32** RB prior; mean allocated FB share is **0.2255** |
| WR | **0.0%** | 2.2% | −2.2pp | **`_top_n_shares(·, 5)`** zeroes everyone outside the team's top five |
| TE | **0.0%** | 0.5% | −0.5pp | same |

**WR and TE survive the top-5 filter in 0.0% of rows** (0 of 301 and 0 of 165). RB survives 99.4%,
QB 67.6%, FB 82.6%.

## 1. Research question

**Does correcting the positional rush-share allocation close the carry component of the gap — and
how much of the 14% can carries alone recover?**

## 2. Arms

- **C0** production.
- **A1** **FB split out of the RB group**, prior calibrated on W1–W6.
- **A2** **top-N widened 5 → 8**, so WR and TE can receive carries.
- **A3** **positional rush-share recalibration**: per-position multiplicative factors =
  (realized positional carry share) / (model positional carry share), **fitted on W1–W6 only**.
- **A4** **A2 + A3** — widening is a prerequisite for WR/TE to receive anything, so the
  recalibration cannot reach them without it.

**All arms are scored with D3 ON**, because that is the combination that would ship and the
defect only matters there. D3-OFF is reported as a secondary.

## 3. Mechanical gates — arms named

- **M1 — all arms.** Team total rush *attempts* must be unchanged from C0 (the 0.95 cap pins it).
  This verifies the harness, not the arm. **Failure voids the run.**
- **M2 — A1 only.** Only FB rows and their teammates' renormalised shares may change; no
  non-FB team without an FB may differ from C0 by more than 1e-12. **Failure voids A1.**
- **M3 — A2 only.** On teams with ≤5 rush-share-positive players, A2 must be bit-identical to C0
  to 1e-12. **Failure voids A2.**
- **M4 — A3, A4.** The calibration must read **W1–W6 only**. Verified by asserting the fit
  excludes weeks 7 and 8. **Failure voids the run** — this is the one place lookahead could enter.

## 4. Statistical gates — arms named

- **S1 — all arms.** Per-player bias on `rush_yards` must improve versus C0, 95% CI excluding 0,
  bootstrapped **over players**.
- **S2 — all arms.** Positional carry shares must move **toward** realized for RB, QB and FB.
- **S3 — all arms.** `rush_yards` projected/realized must move **toward 1.000 from 0.870** and
  **must not exceed 1.05.**
- **S4 — all arms.** No tier degrades by more than **2% relative**, **both ends reported.**
- **S5 — all arms, DESCRIPTIVE.** Effect on `rec_yards`, to confirm this does not disturb
  receiving. Not a gate, but a regression check: receiving should be untouched except through the
  shared `plays` budget.

## 5. The structural ceiling — stated before running

**This study cannot close the gap, and I am recording the arithmetic now rather than presenting a
shortfall as a surprise later.**

The 0.95 cap costs **5%** of carries unconditionally. Even a perfect positional fix therefore
caps the carries ratio near **0.95**, and with `ypc` at **0.955** the best achievable yards ratio
is about **0.95 × 0.955 ≈ 0.91.**

**So the expected best case is ≈0.91, not 1.00.** Closing the remainder requires (a) the separate
`ypc` study and (b) a decision about the 0.95 cap itself, which is a deliberate 5% haircut on
every market and has never been justified in any document I have read. **Neither is in scope
here.**

## 6. Prediction

**A4 is the best arm**, taking the carries ratio from 0.910 toward ≈0.95 and the yards ratio from
0.870 to **≈0.90–0.92** — short of 1.0, per §5. **A1 (FB) and A3 (recalibration) carry most of
it**, since FB's +5.9pp and QB's +5.3pp are the two largest positional errors and both move
carries to RBs, who have a higher realized `ypc` (4.44) than QBs (3.55) — so the yards gain should
**exceed** the carries gain. **A2 (widening) is worth little alone** — 2.7pp of share — but is
required for A3 to reach WR/TE.

**Named risk:** moving carries toward RBs raises their counts, and `_sharpen_rb_rush_shares`
concentrates the top five further. **The lead back could overshoot** — S4's top tier is the place
to watch, and if it breaches, the admissible fix is the FB/QB correction without the
recalibration.

A second risk: the FB prior is being fitted from a realized share of **0.1%**, which on W1–W6 is a
handful of carries. **That cell is thin and the fitted factor may be unstable**; if so, excluding
FBs from the rush pool entirely is the more robust choice than calibrating them.

My record this session: **eight hits, fifteen misses**, with the named risk landing in nearly
every study. **This prediction carries no evidentiary weight; the risks above are the part worth
reading.**

## 7. Scope

Carry allocation only. **`ypc` out of scope. The 0.95 cap out of scope. D1 and D2 held at their
joint-study settings (D1 on, D2 off).** No sportsbook variable enters any arm. The RB P3 Week-1
`WEEK1_STACK_OVERRIDE` route stays frozen and must be bit-identical after any arm. No production
code is modified by this study.

**No holdout for the yards scoring** — W7+W8 only, as with every study in this sequence; the
*calibration* has a genuine W1–W6 holdout, the scoring does not. Promotion requires a prospective
week.
