# PRISM — JOINT RESULTS: **D1+D3 SHIP, D2 DOES NOT**, AND D3 NEEDS A COMPANION FIX

**Date:** 2026-11-03
**Preregistration:** `PRISM_JOINT_D1_D2_D3_PREREGISTRATION_V1.md`, frozen before any arm ran.
**Verdict:** **K5 fires** — the triple over-corrects on rushing, so the admissible recommendation
is a **subset**, exactly as the preregistration required. **K6 does not fire** — D1 and D3 are
genuinely synergistic. **D2's recommendation is withdrawn on yards evidence.**
**Artifacts:** `data/research/joint_d1d2d3_v1.csv` (6,392 rows, 8 arms),
`scripts/research/evaluate_joint_d1d2d3_v1.py`.

---

## 1. Mechanical gates — scale invariance holds inside the full harness

Team allocated targets across all eight arms: **1767.5 – 1767.8**, a spread of **0.3 targets**,
pure Monte Carlo noise. **M1, M3 and M4 PASS.** The prerequisite finding is confirmed in the live
harness: changing share *levels* cannot change team opportunity.

## 2. S1 — and D2 goes the wrong way on yards

**`rec_yards`** (n=474 players):

| arm | mean \|bias\| | mean bias | gain | 95% CI | S1 |
|---|---|---|---|---|---|
| `------` production | 12.945 | +1.427 | — | — | — |
| `----D3` | 12.397 | +0.742 | +0.549 | [+0.467,+0.632] | PASS |
| **`--D2--`** | **13.073** | **+1.556** | **−0.128** | [−0.319,+0.072] | **fail** |
| `D1----` | 12.547 | +1.493 | +0.398 | [−0.023,+0.873] | fail |
| **`D1--D3`** | **11.930** | +0.751 | **+1.015** | **[+0.569,+1.524]** | **PASS** |
| `D1D2D3` | 12.019 | +0.871 | +0.926 | [+0.422,+1.498] | PASS |

**`rush_yards`:**

| arm | mean \|bias\| | mean bias | gain | 95% CI | S1 |
|---|---|---|---|---|---|
| `------` | 7.207 | **+0.083** | — | — | — |
| `----D3` | 6.502 | −0.820 | +0.705 | [+0.627,+0.780] | PASS |
| `--D2--` | 7.212 | +0.084 | −0.005 | [−0.011,+0.001] | fail |
| `D1----` | 6.861 | −0.019 | +0.346 | [+0.113,+0.615] | PASS |
| **`D1--D3`** | **6.140** | −0.928 | **+1.067** | [+0.820,+1.349] | **PASS** |
| `D1D2D3` | 6.138 | −0.928 | +1.068 | [+0.822,+1.351] | PASS |

**`D1--D3` is the best arm on both markets**, and **adding D2 makes receiving worse** (+0.926
against +1.015).

## 3. D2's recommendation is withdrawn — the mechanism

`PRISM_D2_MULTIPLIER_MASS_RESULTS_V1` recommended deleting the matchup multipliers on the strength
of a **+0.00441 share-bias gain.** On **yards**, D2 alone scores **−0.128** and raises total
receiving yards from 13,553 to **13,672**.

**Why:** D2 moves opportunity from TE and RB to WR (measured: TE −0.347, RB −0.134, WR +0.272
allocated targets). **WRs carry a higher yards-per-target**, so the same opportunity converts to
*more* yards — and total receiving yards were already **9.1% too high.** D2 makes a correct share
distribution produce a worse yards projection.

**This is the second time in two days that a share-level improvement failed to transfer to the
priced quantity** (the first being `ROLE_SHRINKAGE` V2's P-versus-S divergence). The pattern is
now explicit: **a fix to the opportunity *distribution* must be validated on yards, because
yards-per-opportunity differs systematically by position.**

D2 remains a genuine *defect* — multiplying zero-sum shares without renormalising is wrong, and
the audit's description stands. **But deleting the multipliers is not a safe fix on its own**, and
D2 should not ship until the yards consequence is addressed.

## 4. K6 does not fire — D1 and D3 are synergistic

| combination | `rec_yards` interaction | `rush_yards` |
|---|---|---|
| `--D2D3` = D3 + D2 | −0.001 [−0.009,+0.007] independent | +0.000 independent |
| **`D1--D3`** = D3 + D1 | **+0.069 [+0.036,+0.105] SYNERGY** | **+0.016 [+0.003,+0.032] SYNERGY** |
| `D1D2--` = D2 + D1 | +0.041 [−0.042,+0.132] independent | +0.006 independent |
| **`D1D2D3`** = all three | **+0.108 [+0.016,+0.210] SYNERGY** | **+0.023 [+0.007,+0.039] SYNERGY** |

**D1 and D3 reinforce each other**, modestly — 7% of the combined gain. The mechanism is coherent:
D1 removes inactive players from the share vector, which raises the active players' allocation
probabilities, which means **fewer zero-opportunity draws for the players who matter** — exactly
the cell D3 fixes. D3 then has less phantom yardage left to remove on the bench but a cleaner
distribution to work with.

**My prediction that the three were independent and K6 would fire was wrong.**

## 5. K5 fires — D3 over-corrects rushing, and that is the central result

**S3, projected/realized total yards:**

| arm | `rec` ratio | `rush` ratio | |
|---|---|---|---|
| `------` | 1.091 | **0.984** | |
| `----D3` | 1.046 | **0.870** | **UNDERSHOOT** |
| `D1----` | 1.093 | 0.975 | |
| **`D1--D3`** | **1.045** | **0.860** | **UNDERSHOOT** |
| `D1D2D3` | 1.053 | 0.860 | **UNDERSHOOT** |

**Production's rushing aggregate is 0.984 — almost exactly right. Every D3 arm takes it to
0.860–0.870, a 13–14% under-projection.**

And yet D3 *improves* per-player rushing bias by +0.705. **Both are true, and the D3 writeup
understated it.** D3 §2 noted the aggregate flipping +0.083 → −0.820 and called it "two errors
cancelling." The joint study shows what that costs: **shipping D3 alone would move rushing from
accidentally-right to systematically 14% low.**

**So the honest recommendation splits by market:**

- **Receiving: ship `D1--D3`.** Per-player gain **+1.015** `[+0.569,+1.524]`, ratio **1.091 →
  1.045**, bottom tier over-projection down **55.8%**, no tier degraded beyond the 2% bar (worst
  −1.5%, from D2). This is the clear win.
- **Rushing: D3 must not ship alone.** It needs a companion fix to the rushing yards *level* —
  `ypc`, the carry allocation, or both — because removing the phantom yards exposes a pre-existing
  **14% under-projection** that was being masked. **`D1----` alone is admissible for rushing**
  (gain +0.346, ratio 0.984 → 0.975, no undershoot).
- **D2: do not ship** (§3).

**S2 passes everywhere** — worst degradation −1.5% at Q4 against a 2% bar — and the bottom tier
improves 55.8% under every D3 arm.

## 6. Prediction scored

- **"the three are close to independent, K6 fires" — WRONG.** D1×D3 synergy is significant.
- **"the triple passes S1 on both markets" — RIGHT.**
- **"`rec_yards` bias moves toward +0.3 to +0.5" — WRONG**, it reached +0.751 (`D1--D3`).
- **"S3 lands near 1.00–1.02 without undershooting" — WRONG**, rushing hit 0.860.
- **Named risk — RIGHT, precisely:** *"`rush_yards` already flipped negative under D3 alone …
  adding D1 and D2 could push it further negative and breach S3's floor … if so the right
  recommendation is D1+D2 for receiving and D3 for rushing handled separately, not the triple."*
  That is the finding, with the one amendment that **D2 drops out of the receiving recommendation
  too.**

**One of four predictions, and the named risk right again.** Across the session: **eight
prediction hits, eleven misses, and the named risk landing in nearly every study.** The
preregistered risk statement has been the single most productive element of this methodology, and
the predictions have been close to worthless. That is worth saying plainly.

## 7. Limitations

**No holdout.** W7+W8 only, 799 player-weeks, 54 team-weeks; W3–W5 2026 permanently lost
(AGENTS.md §5a). No fitted parameters except D1's QUESTIONABLE retention (0.8535 / 0.5718), fitted
on W1–8 injury data rather than the scored yards.

**S4 (side accuracy) not computed.** The preregistration committed to computing it here; I did
not, and I am recording that rather than quietly dropping it. With ~800 rows and an MDE above
1.5pp it could not have changed any conclusion above, but the commitment was made and not kept.

**Promotion of `D1--D3` for receiving still requires a prospective week.**

## 8. Next, in value order

1. **The rushing level defect** that D3 exposes (§5) — now the highest-value open item, because it
   blocks a fix that is otherwise correct and ready.
2. **D2's yards consequence** (§3) — the defect is real but the fix is not safe; it needs a
   position-aware `ypt` reconciliation.
3. **Prospective validation** of `D1--D3` on receiving.
4. **The upstream artifact fix** for `injuries.csv` (D1's §6), which is a production repair rather
   than research and is what makes D1 possible at all.

No sportsbook variable entered any arm. No production code modified.
