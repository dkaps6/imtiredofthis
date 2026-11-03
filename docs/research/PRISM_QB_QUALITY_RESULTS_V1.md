# PRISM — QB QUALITY → RECEIVER PRODUCTION, RESULTS V1 — **NULL**

**Date:** 2026-11-03
**Preregistration:** `PRISM_QB_QUALITY_PREREGISTRATION_V1.md`, frozen before any arm was
computed or any outcome inspected.
**Status:** NEGATIVE. **K1 fires.** Nothing promoted. **Banked so this is not re-run.**

---

## 1. Verdict

Walk-forward within season, 2023/2024/2025, **6,457 receiver-weeks, 490 players, 822
player-seasons with ≥3 evaluated weeks**, 237 QB-changed rows.

| | mean per-player |err| |
|---|---|
| **B** receiver's own prior baseline | **19.3481** |
| **Q** baseline + global β·QB CPOE | 19.3516 |
| T baseline + global β·team points | 19.3592 |
| Q_epa (secondary) | 19.3608 |

| gate | result | |
|---|---|---|
| **G1** Q beats B | **−0.0035** CI [−0.0092, +0.0019] | **FAIL** |
| **G2** breadth ≥60% | 50% [46%] | **FAIL** |
| G3 no tier harmed | all tiers within −0.5 yd | PASS *(trivially — nothing moved)* |
| **G4** same sign all seasons | **[−1, +1, −1]** | **FAIL** |
| **G5** QB-changed subgroup | +0.0026, CI lower −0.0276 | **FAIL** |
| **G6** beats team-points control | +0.0076, CI lower −0.0096 | **FAIL** |

**Every gate that could fail, failed.** The effect is **0.0035 yards** against a baseline error
of 19.35 — indistinguishable from zero.

K5 (endogeneity check) does not rescue it: the EPA specification is **worse** (−0.0127) than
CPOE, so there is no "it works on EPA" story either. Both are null.

---

## 2. Why — and this is the useful part

**The receiver's own baseline already contains his quarterback.**

A receiver's prior-weeks yards-per-game were *produced with that quarterback throwing to him*.
QB quality is therefore not new information — it is already encoded in the baseline the arm is
trying to improve. Adding it is redundant by construction.

That reading predicts the effect should appear **only where the QB changes**, breaking the
encoding. **G5 tested exactly that** on 191 player-seasons and found **+0.0026** with a CI lower
bound of −0.0276.

**Honest limitation:** G5 is the one gate where a real effect could hide. n=191 with a CI
spanning −0.028 to roughly +0.033 cannot rule out an effect smaller than ~0.03 yards — which is
itself negligible — but it is **underpowered rather than a strong null**. A dedicated QB-change
study with more seasons could revisit it. The full-sample result is a strong null; the
QB-change subgroup is a weak one.

---

## 3. What this does and does not say

**Does say:** prior-CPOE QB quality adds nothing to receiving-yards prediction at the
player level, once the receiver's own recent baseline is in hand, across three seasons.

**Does NOT say:**
- that QB quality is irrelevant to football. It says it is **redundant with the baseline**;
- anything about receiver **opportunity** (target share) — only receiving yards, per §7 of
  the preregistration;
- that a different QB metric would also fail. CPOE and EPA/attempt were tested; aDOT,
  pressure-adjusted metrics and time-to-throw were not;
- anything about a QB *downgrade* specifically (starter → backup), which was not isolated as
  its own subgroup.

---

## 4. Ledger entry

**Added to the banked list in `PRISM_AVAILABILITY_SIGNAL_PREREGISTRATION_V1.md` §0:**

> *"QB quality conditions receiver production"* — **NULL.** Prior CPOE, one global coefficient,
> walk-forward 2023–2025, 822 player-seasons. Gain −0.0035 yd, CI [−0.0092, +0.0019]; breadth
> 50%; season signs [−, +, −]; fails the team-points control and the QB-change identification
> gate. Mechanism: the receiver's own baseline already encodes his quarterback. The QB-change
> subgroup (n=191) is underpowered and is the only place a small effect could still hide.

This closes the first of the two leads logged as genuinely open on 2026-11-03. The second —
**NGS receiving advanced stats** (`avg_separation`, `avg_cushion`,
`percent_share_of_intended_air_yards`, `avg_yac_above_expectation`, available for 2026 and used
nowhere, while `route_rate` and `yprr` are NaN for every player) — remains open and untested.
