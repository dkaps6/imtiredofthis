# PRISM — NGS AS AN OPPORTUNITY SIGNAL, RESULTS V1

**Date:** 2026-11-03
**Preregistration:** `PRISM_NGS_OPPORTUNITY_PREREGISTRATION_V1.md`, frozen before any arm.
**Status:** **Part A NULL. Part B UNDERPOWERED — K1 and K5 fire, but the mechanism gate passes.**
Nothing promoted.

---

## PART A — NGS air-yards share → target share: **NULL**

1,880 evaluated weeks, **246 player-seasons with ≥3 weeks** (minimum 100 — met).

| arm | M1 mean per-player \|err\| |
|---|---|
| **B** prior mean target share *(incumbent)* | **0.05452** |
| C naive: prior targets ÷ 35 | 0.05713 |
| A1 air-yards share replacing the baseline | 0.05854 |
| A2 disagreement term | 0.05448 |

| arm | G1 | G2 | G3 | G5 | G6 |
|---|---|---|---|---|---|
| C | FAIL −0.00261 | FAIL 42% | PASS | *(control)* | FAIL |
| A1 | **FAIL** −0.00402 | FAIL 48% | FAIL | FAIL | FAIL |
| A2 | FAIL +0.00004 | FAIL 54% | PASS | PASS | FAIL |

**K1 fires.** A2 is a wash (+0.00004, CI [−0.00014, +0.00023]); A1 is materially **worse** than
simply using the player's own past target share.

**The opportunity hypothesis fails on its own terms.** The reasoning was that NGS might beat the
baseline at predicting *opportunity* even though it failed at *production*. It does not. **The
receiver's own prior target share is the best available predictor of his next target share**, and
a depth-weighted role metric does not improve on it.

**Coverage limitation, material:** the NGS panel covers ~60 qualifying receivers per week, and
the tier table is degenerate as a result — **165 of 246 player-seasons sit in the 22%+ tier and
only one below 15%.** This is a study of high-volume receivers only. It says nothing about the
mid- and low-share players where `PRISM_MC_LEAK_FIX_RESULTS_V1` §7 found E1 doing harm.

---

## PART B — rushing NGS (RYOE) → rush yards: **UNDERPOWERED**

906 evaluated weeks, **110 player-seasons with ≥3 weeks** (minimum 80 — met, but thin).

| arm | M1 |
|---|---|
| B prior mean rush yards *(incumbent)* | 26.46434 |
| C naive: prior carries × league YPC | 25.94476 |
| **R1** B + β·RYOE/att | **25.91307** |
| R2 decomposed carries × (YPC + β·RYOE) | 26.82495 |

| gate | R1 | |
|---|---|---|
| **G1** beats B | **+0.55127, CI [−0.01681, +1.12202]** | **FAIL — by 0.017** |
| **G2** breadth | **66% [57%]** | **PASS** |
| **G3** no tier harmed | | **PASS** |
| **G4** same sign all seasons | [−1, +1, +1] | **FAIL** |
| **G5** beats naive control C | +0.03169, CI [−0.52834] | **FAIL** |
| **G6** mechanism concentration | **+1.7736 vs +0.55 full sample** | **PASS** |

**K1 and K5 fire. Nothing is promoted.**

### But this is not the same as Part A, and the distinction matters

R1 **passes the mechanism gate decisively**: the gain is **+1.77 yards** in the top tercile of
|RYOE| against **+0.55** overall. That is exactly the signature G6 was written to detect —
the effect concentrating where the signal is strongest, rather than a flat rescaling.

It also passes breadth (66%) and harms no tier. **G1 misses by 0.017 yards** on a CI, with
**110 player-seasons** — the rushing NGS panel is only ~85 players per season and cannot be
widened from available data.

**This is underpowered, not null.** Reported as a failure because the preregistration says so,
and the preregistration is not renegotiated after seeing the number. But a future study with
more seasons (2020–2022 rushing NGS, untested here) is the one genuinely live NGS lead left.

**G5 remains the harder problem even so:** R1 does not beat *carries × league YPC*. Whatever
RYOE contributes may be contributed just as well by counting carries — the same result Part A
and the receiving-yards study both produced.

`efficiency` was carried into no arm, as preregistered. Descriptively its correlation with
next-week rush yards is **−0.073**; excluding it was correct.

---

## Ledger entries

> *"NGS air-yards share predicts target share better than past target share"* — **NULL.**
> 246 player-seasons, walk-forward 2023–2025. Baseline best; the air-share arm materially worse.
> Covers high-volume receivers only (165/246 above 22% share).

> *"Rushing NGS RYOE improves rush-yards projection"* — **UNDERPOWERED, K1/K5.** 110
> player-seasons. R1 gain +0.551, CI [−0.017, +1.122] — misses G1 by 0.017. Passes breadth (66%),
> tiers, and the mechanism gate (+1.77 in the top |RYOE| tercile). Does not beat carries × YPC.
> **The one live NGS lead; needs 2020–2022 to resolve.**

---

## What all four NGS/QB studies together now say

Receiving yards (null), target share (null), QB quality (null), rush yards (underpowered).
In every case **the player's own prior baseline was the hardest thing to beat**, and in three of
four a naive opportunity count matched or beat every sophisticated arm.

That is consistent with the day's structural finding and sharpens it: **the model's recoverable
error is not in choosing better per-player predictors — it is in the allocator that decides who
gets opportunity at all** (`PRISM_MC_OPPORTUNITY_LEAK_V1`: ~30% of target mass to players who
record none, replicated across two weeks).

**Recommendation: stop adding predictors. The remaining headroom measured today is in
allocation, not in signal.**
