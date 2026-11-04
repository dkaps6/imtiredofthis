# PRISM — RECEIVING MASS BIAS, RESULTS V2 — **NO ARM PROMOTED; MECHANISM LOCALISED**

**Date:** 2026-11-03
**Preregistration:** `PRISM_RECEIVING_MASS_PREREGISTRATION_V2.md`, frozen in commit `07589c58`
**before** any arm was computed.
**Status:** NEGATIVE on all arms. **K1 and K4 both fire.** Nothing promoted.
But the defect is now located.

---

## 1. Verdict

Leave-one-week-out cross-fitting, 2026 weeks 1/2/6/7/8. Receptions: 636 rows, **124 players
with ≥3 evaluated weeks** (minimum 100 — met).

| arm | G1 bias | G2 breadth | G3 MAE guard | G4 vs D0 | G5 harm | |
|---|---|---|---|---|---|---|
| **D0** add a constant | PASS +0.0987 | PASS 60% [52%] | PASS | *(control)* | **FAIL 38%** | ✗ |
| **D1** pure ML | PASS +0.1301 | **FAIL** 57% [48%] | **FAIL** +0.097 | **FAIL** | PASS 21% | ✗ |
| **D2** bias-targeted blend | identical to D1 | FAIL | FAIL | FAIL | PASS | ✗ |

**K1 fires** — no arm passes all gates.
**K4 fires** — D1/D2 do not beat D0 (G4 CI [−0.0307] includes 0). Per the preregistration:
*the mechanism is not ensemble mis-weighting, and no reweighting may be promoted.*

`rec_yards` (127 players) is worse: D0 fails G1 outright; D1/D2 pass only G1 and fail
everything else. **K5 does not fire** — both markets agree in direction.

---

## 2. The result the player-first design was built to catch

**D0 — adding a single constant — passes G1, G2 and G3.** On aggregate bias, on breadth, and on
the MAE guardrail, it looks like a fix.

**It fails G5: 38% of individual players are made worse by more than 0.25 receptions.**

That is the entire argument for the player-first principle, demonstrated. A correction that
improves the pooled number *and* a bare majority of players still damages more than a third of
named projections. Had this study been judged on pooled bias and MAE — the metrics the existing
ensemble was fit to — **D0 would have been promoted and would have degraded 38% of players.**

AGENTS.md: *"Do not erase meaningful player differences to make a position-wide number look
cleaner."* G5 is that rule made enforceable, and it is the gate that stopped this.

---

## 3. D2 collapsed into D1 — and that is the key structural finding

The bias-targeted grid selected **w_ml = 1.00 in every fold**. There is no blend of
`mc_proj` and `ml_proj` that removes the bias, because **every component is biased**.

Team-level reception mass, same player set both sides, 80 team-weeks:

| component | team sum | actual | bias |
|---|---|---|---|
| `mc_proj` | 14.41 | 18.06 | **−3.65 (−20.2%)** |
| `state_proj` | 16.03 | 18.06 | −2.03 (−11.2%) |
| `ensemble` | 15.59 | 18.06 | −2.47 (−13.7%) |
| **`ml_proj`** | **17.73** | 18.06 | **−0.33 (−1.8%)** |

**`mc_proj` under-allocates team receptions by 20%.** The ensemble inherits it and cannot
reweight its way out, because the only near-unbiased component (`ml_proj`) fails the
player-level gates on its own — its breadth is 57% with a CI lower bound of 48%, and it breaches
the MAE guardrail.

So the defect is **upstream of the ensemble, in the Monte Carlo allocation step.**

This is the same shape as the RB lane: `PRISM_RB_RUSH_SHARE_MASS_BUDGET_V1.md` found the same
simulator over-allocating rush mass (`rules_rush_share` summing to 1.2729 against a 0.95 cap,
haircut 0.7574, RB carries 19.77 vs 23.15 actual). **Rushing over-allocates then gets
haircut; receiving under-allocates.** Both are the finite-opportunity allocator, in opposite
directions.

---

## 4. What is established

1. The receiving mass bias is **real and large** — −0.63 receptions per player, −13.7% of team
   reception mass.
2. It is **not an ensemble weighting problem** (K4). Reweighting cannot beat adding a constant.
3. It is **not fixable by a constant either** — that harms 38% of individual players (G5).
4. It originates in **`mc_proj`, which under-allocates team receptions by 20.2%**, while
   `ml_proj` is nearly correct at team level (−1.8%).

## 5. What is NOT established

- That fixing the MC allocation would fix the player-level distribution. `ml_proj` is nearly
  unbiased *in aggregate* yet still fails breadth — aggregate correctness does not imply
  player-level correctness, which is this study's own central lesson.
- Anything about `rush_rec_yards` (out of scope, uncalibrated, −17.49 bias).
- That the 2026 five-week sample generalises. Weeks 3–5 are permanently lost (AGENTS.md §5a).

## 6. Next

The target is now specific: **why does the MC allocator under-produce receiving opportunity by
~20% at team level, when the same allocator over-produces rushing opportunity?**

That is a question about `scripts/simulation_v2.py`'s opportunity allocation, not about
weights, priors or post-hoc corrections — all three of which have now been tested and rejected
in this lane. It should be investigated as a **code-path and conservation question first**
(does receiving opportunity get normalised, capped, or truncated the way `_allocate_counts`
handles rushing?), with measurement before any modelling.

Nothing in this study touched `data/model_ensemble_weights.csv` or any production path.
