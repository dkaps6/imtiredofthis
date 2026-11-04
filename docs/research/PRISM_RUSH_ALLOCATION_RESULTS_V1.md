# PRISM — RUSH ALLOCATION RESULTS: THE ALLOCATION IS FIXABLE AND **IT BARELY MATTERS**

**Date:** 2026-11-04
**Preregistration:** `PRISM_RUSH_ALLOCATION_PREREGISTRATION_V1.md`, frozen before any arm ran.
**Verdict: NOTHING PROMOTED**, for two reasons recorded in §6 — the preregistration **omitted its
kill conditions**, and S4's bar is mis-scaled. **The substantive finding stands: positional
reallocation recovers +0.9% of a 14% gap and this lane is close to a dead end.**
**Artifacts:** `data/research/rush_allocation_v1.csv`,
`scripts/research/evaluate_rush_allocation_v1.py`.

---

## 1. The real decomposition — my §0 was incomplete

§0 split the gap into carries (0.910) and `ypc` (0.955). **The carries term itself splits in
two**, and the full chain is:

| step | value | ratio |
|---|---|---|
| team rush **attempts** the model generates (`plays − pass_att`) | **1404.5** | — |
| carries actually **allocated** to players | **1322.1** | **0.941** of attempts |
| carries **realized** | **1465.0** | — |

| factor | ratio | per team-week |
|---|---|---|
| **attempts vs realized carries** | **0.959** | model 26.0, realized 27.1 |
| **the 0.95 cap + residual bucket** | **0.941** | 26.0 → 24.5 |
| **`ypc`** | **0.953** | 4.164 vs 4.369 |
| **product** | **0.860** | ✓ matches observed |

**The 14% rushing gap is three roughly equal structural factors — too few rush plays, the cap,
and efficiency.** Positional misallocation, which is this study's entire subject, is **not one of
them.**

## 2. Arms — A3 works and it does not matter

| arm | carries | rush yards | yards ratio | per-player gain | 95% CI | S1 |
|---|---|---|---|---|---|---|
| C0 | 1322.1 | 5505 | 0.860 | — | — | — |
| A1 FB split out | 1318.5 | 5488 | **0.857** | +0.1913 | [−0.0434,+0.4471] | fail |
| **A2 top-N 5→8** | 1325.2 | 5508 | 0.861 | **−0.1893** | **[−0.3489,−0.0351]** | **fail, worse** |
| **A3 recalibration** | 1323.5 | 5537 | **0.865** | **+0.4599** | [+0.1579,+0.7672] | **PASS** |
| A4 = A2+A3 | 1325.4 | 5553 | 0.868 | +0.3635 | [+0.0679,+0.6657] | PASS |

**A3 fixes the allocation almost exactly** (S2, positional carry share):

| group | realized | C0 | **A3** |
|---|---|---|---|
| RB | 0.8410 | 0.7526 | **0.8560** |
| QB | 0.1270 | 0.1895 | **0.1433** |
| **FB** | 0.0020 | **0.0571** | **0.0000** |
| WR | 0.0246 | 0.0008 | 0.0007 |
| TE | 0.0055 | 0.0000 | 0.0000 |

**And the yards ratio moves 0.860 → 0.865.** The allocation defect was real, it is now corrected,
and it is worth **half a percent of a fourteen-percent gap.**

### Why fixing *who* carries does almost nothing

**Because the model's `ypc` barely varies by position.** It assigns ~4.2 to everyone, while
realized `ypc` differs sharply — **RB 4.44 against QB 3.55.** Moving a carry from a quarterback to
a running back changes the model's yards by about **0.15**, when reality says **0.89**.

So the model **does** know who should carry (after A3) and **does not** know that it matters. This
is the position-over-player pattern again, in a new place: a per-player `ypc` that does not
distinguish a quarterback scramble from a running back's handoff.

## 3. A2 is actively harmful

Widening the top-N filter from 5 to 8 **loses** 0.1893 with a CI excluding zero, and A4 is worse
than A3. Widening lets WR and TE in (WR share 0.0008 → 0.0154) but **dilutes a fixed carry pool
across more players**, so the backs who actually carry get fewer. **The top-5 filter is crude but
it is not the problem**, and loosening it costs more than the 2.7pp of WR/TE share it recovers.

## 4. S4 fails for every arm — and the bar is mis-scaled

Bottom-tier (Q1) relative degradation: **A1 −17.2%, A2 −45.8%, A3 −11.2%, A4 −33.6%**, against a
2% bar. Top tier *improves* for every arm (A3 +5.1%).

**Q1's actual mean is 0.86 yards.** A 2% relative bar on a 0.86-yard base is **0.017 yards** —
absurd precision, and any change at all breaches it. **The gate is mis-scaled for a near-zero
denominator**, the same error class as the +250% QB `rush_yards` figure in the coverage audit.

I am recording it as mis-specified rather than rescoring it on an absolute bar. **It blocks
promotion**, which is the correct outcome here anyway.

## 5. Prediction scored

- **"A4 best, yards ratio 0.90–0.92" — WRONG.** A3 was best, at 0.865.
- **"A1 and A3 carry most of it" — HALF.** A3 yes; A1 failed and was slightly negative.
- **"A2 worth little alone" — RIGHT**, and understated: it is harmful.
- **"the yards gain should exceed the carries gain" — RIGHT** (+0.6% yards on +0.25% carries).
- **Named risk, lead-back overshoot at the top tier — WRONG direction.** The top tier improved;
  the **bottom** tier broke.
- **Named risk, "the FB cell is thin, excluding fullbacks may be more robust than calibrating
  them" — RIGHT.** A3's factor of 0.020 effectively zeroes FB share, and that is what worked.

## 6. Why nothing is promoted — including a new kind of process failure

**First: this preregistration has no kill conditions.** Every other preregistration in this
sequence defined what a gate failure meant; **I omitted the section entirely.** So there is no
preregistered rule for what passing S1 would license, and **inventing one after seeing the
numbers is exactly what the discipline exists to prevent.** Nothing can be promoted from a study
that never said what promotion would require.

**Second: S4 is mis-scaled** (§4).

**That is the eighth process error of this sequence, and the first by omission rather than
mis-specification.** The arm-scoping fix I adopted for D3 worked — every gate here was correctly
scoped — but I fixed the gates and dropped the kill conditions. **The check that would have caught
it is mechanical: a preregistration must contain a research question, arms, gates, kill
conditions, a prediction with named risks, and scope.** I am recording that list here so it can be
applied as a checklist rather than remembered.

## 7. Where the rushing gap actually lives

In descending value:

1. **`ypc` is position-blind** (§2). The model's ~4.2 for everyone against a realized RB 4.44 /
   QB 3.55 spread. This is both a third of the gap *and* the reason A3's correct allocation does
   nothing. **Highest value, and it is a player-identity problem, not a calibration one.**
2. **The model generates 4% too few rush plays** — 26.0 attempts per team-week against 27.1
   realized carries. Lives in `_team_inputs`: `plays` clipped to [50,80], `pass_rate` to
   [0.35,0.75], with a 64-play fallback.
3. **The 0.95 cap costs 5.9%** on every market, not just rushing. **No document in this repository
   justifies it.** It is the largest single unexplained constant in the simulation and deserves its
   own study — including whether the residual bucket should exist at all.
4. **A3's recalibration** is correct and cheap, and should ride along with whichever of the above
   ships — but on its own it buys half a percent and should not be sold as a rushing fix.

`rec_yards` was untouched across all arms (12,977–12,982), so S5's regression check passes.

No sportsbook variable entered any arm. No production code modified.
