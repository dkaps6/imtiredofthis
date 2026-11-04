# PRISM — MC OPPORTUNITY LEAK FIX, RESULTS V1 — **E1 PASSES ALL GATES**

**Date:** 2026-11-03
**Preregistration:** `PRISM_MC_LEAK_FIX_PREREGISTRATION_V1.md`, frozen in commit `73b86b7a`
**before** any arm was computed.
**Status:** POSITIVE. **Nothing promoted.** Per §7 a pass licenses only a flagged
implementation plus prospective grading.

---

## 1. Verdict

769 player-weeks, W7 + W8 production universes, **310 players with ≥2 evaluated weeks**
(minimum 100 — met).

| arm | G1 bias | G2 breadth | G3 starved | G4 vs F0 | G5 both weeks | |
|---|---|---|---|---|---|---|
| **F0** drop never-targeted | PASS +0.00799 | PASS 61% [56%] | PASS 4.2% | *(control)* | PASS | ✓ |
| **E1 soft availability weighting** | **PASS +0.01591** | **PASS 75% [71%]** | **PASS 4.2%** | **PASS +0.00792** | **PASS** | **✅ ALL** |
| E2 hard filter A≥0.5 | PASS +0.01397 | PASS 70% | **FAIL 19.8%** | PASS | PASS | ✗ |
| E3 top-7 (rushing analogue) | PASS +0.00391 | PASS 62% | **FAIL 27.9%** | **FAIL −0.00408** | PASS | ✗ |

**E1 — multiply each share by the fraction of prior weeks in which the player drew a target,
then normalise — passes every gate.** **K4 does not fire:** E1 genuinely beats the naive
control (+0.00792, CI lower bound +0.00520), so the gain is not merely "the universe is too
large."

M1 (mean per-player |bias|): C1 **0.04622** → F0 0.03823 → **E1 0.03031** → E2 0.03226 →
E3 0.04231.

---

## 2. The leak is roughly halved

Share of team targets by destination:

| arm | to players who drew ≥1 target | **leaked to zero-target players** |
|---|---|---|
| **C1 (production)** | 0.6544 | **0.2956** |
| F0 | 0.7295 | 0.2205 |
| **E1** | **0.7583** | **0.1494** |
| E2 | 0.7849 | 0.1366 |
| E3 | 0.7269 | 0.2231 |

**E1 cuts leaked mass from 0.2956 to 0.1494** and lifts mass reaching real receivers from
0.6544 to 0.7583.

**It does not close the leak.** Targeted players still receive 0.7583 against an actual share
near 0.95–0.98. E1 is a substantial improvement, not a solution, and should not be described as
fixing the defect.

---

## 3. G3 did exactly the job it was written for

G3 — *at most 5% of players who actually drew a target may receive zero allocation* — was added
because hard filters can buy pooled mass by starving real receivers. It caught both of them:

- **E2 starves 19.8%** of genuinely targeted players.
- **E3 starves 27.9%.**

Both pass G1 and G2. On bias and breadth alone, **both would have been promoted.** Nearly a
third of real receivers would have been projected at zero opportunity.

This is the second time in this session a player-level gate has stopped an arm that looked good
on aggregate — the first was D0 in the receiving-mass study (38% of players harmed).

---

## 4. The rushing analogue is the worst arm, and that matters

**E3 — the direct analogue of `_top_n_shares(..., 5)` that rushing already applies — fails G4:
it loses to simply dropping never-targeted players** (−0.00408, CI upper bound −0.00828).

The preregistration warned that truncating a tail is not modelling it. That is now measured:
a top-N cut is **worse than the naive control** and starves 27.9% of real receivers.

**Implication for the rushing lane, which is out of scope here and untested:** the existing
`_top_n_shares(..., 5)` on rushing may itself be a poor mitigation rather than a feature worth
copying. The RB lane's smaller leak (−14.6% vs receiving's −29.3%) may be bought at a cost
nobody has measured. **That is a hypothesis, not a finding — rushing was not evaluated.**

---

## 5. What is established, and what is not

**Established:**
1. E1 passes all five preregistered gates on two independent weeks.
2. It roughly halves leaked opportunity mass and improves 75% of named players.
3. It beats the naive control, so availability weighting adds value beyond trimming the universe.
4. Hard filters (E2, E3) are unsafe at the player level regardless of their aggregate numbers.

**NOT established:**
- That downstream receptions, yards, or pricing improve. **The allocation rule was scored in
  expectation only**; no Monte Carlo was run and no production code was modified. Each
  downstream market needs its own test.
- That E1 generalises beyond two weeks. W7 and W8 are the only committed production universes.
- That A_j — fraction of prior weeks with ≥1 target — is the best availability signal. It is
  simply a leakage-safe one that works. Injury designations, snap counts and inactives lists
  were not tested.
- Anything about `rush_shares` (§4 is a hypothesis).

---

## 6. Next

Per §7, exactly one follow-up is licensed:

1. implement E1 in `scripts/simulation_v2.py` **behind a version flag, default off**;
2. re-run the full production validation suite
   (`compileall`, `audit_repo.py --strict`, `pytest -q`, `audit_2026_production_readiness.py`);
3. **grade it prospectively** before any live use;
4. measure the downstream effect on receptions and `rec_yards` separately — §5 is explicit that
   this study does not establish it.

Residual question worth its own study: E1 leaves 0.1494 of mass still leaking. Whether a better
availability signal (snap counts — available weekly, confirmed in
`PRISM_SNAP_SIGNAL_AND_NOISE_FLOOR_V1.md` §1 — or official inactives) closes the remainder is
open, and connects directly to V20/V21 **AUDIT TARGET 6**.

---

## 7. QUALIFICATION (added 2026-11-03, same day) — the headline number oversells itself

Prompted by the question *"are individual player projections actually getting better, or are we
still aggregating?"* — a named-player read of the E1 result materially qualifies §1 and §2.
**This section is part of the result, not a footnote.**

### 7a. M1 is diluted by players who never play

**124 of 308 evaluated players (40%) were never targeted in either week.** M1 counts each of
them equally with Jaxon Smith-Njigba. E1's easiest win — correctly zeroing players who were
never going to catch anything — therefore carries a large part of the headline.

| population | n | C1 | E1 | gain | breadth |
|---|---|---|---|---|---|
| all players (**as published in §1**) | 308 | 0.04603 | 0.03051 | **+0.01552** | **75%** |
| **excluding never-targeted** | 184 | 0.05436 | 0.04203 | **+0.01233** | **62%** |
| targeted, ≥5% share | 124 | 0.06308 | 0.04751 | — | — |

**The gain survives** (CI [+0.00751, +0.01713] still clears 0) **but breadth falls from 75% to
62%** — barely above the G2 threshold of 60%. E1 passes honestly; it passes far more narrowly
than §1 reads.

### 7b. The effect is tier-dependent, and one real tier gets WORSE

| actual-share tier | n | C1 bias | E1 bias | mean gain | % improved |
|---|---|---|---|---|---|
| 0 (never targeted) | 124 | +0.0337 | +0.0134 | +0.0203 | **94%** |
| <5% | 60 | +0.0353 | +0.0219 | +0.0056 | 60% |
| **5–12%** | **49** | **−0.0052** | **+0.0018** | **−0.0080** | **31%** |
| 12–20% | 46 | −0.0599 | −0.0298 | +0.0208 | 80% |
| 20%+ | 29 | −0.1357 | −0.0821 | **+0.0472** | **93%** |

**C1 was already nearly unbiased for the 5–12% tier (−0.0052) and E1 over-corrects it to
+0.0018.** Only 31% of that tier improves. Named regressions are exactly that population —
**Cade Otton, Mark Andrews, Sam LaPorta, Cooper Kupp, Josh Jacobs, Zach Charbonnet**: tight ends
and pass-catching backs with moderate, variable roles.

The mechanism is the renormalisation, not the availability signal. A full-time tight end who
draws three targets a game has `A_j = 1.0`, identical to an alpha receiver, so E1 does not
distinguish them — it simply redistributes the freed mass proportional to share, which hands
the mid-tier more than it should get.

### 7c. What this changes

- **§1 and §2 stand as computed** — the gates were preregistered and E1 cleared them.
- **The magnitude in §2 must be cited from the 184-player figure (+0.01233, 62% breadth), not
  the 308-player one.**
- **The flagged implementation in §6 must not proceed without addressing the 5–12% tier.**
  Shipping E1 as-is would knowingly degrade a identifiable group of real contributors — which is
  the failure G3 and G5 were written to prevent, arriving in a form neither gate tested for.

### 7d. The standing methodological lesson

A metric can be *player-level in construction* — each player counted once, bootstrapped by
player — and still be an **aggregate that conceals which players move and in which direction.**
Both the alpha-tier study (B1 won by conditioning on level) and this one (E1 fails for one tier)
point the same way: **corrections conditioned on player role or tier beat universal ones**, and
a single number over players cannot reveal that. Future studies in this lane should report the
tier table alongside M1 by default.

