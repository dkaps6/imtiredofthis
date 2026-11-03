# PRISM — NGS RECEIVING SIGNAL, RESULTS V1 — **NULL**

**Date:** 2026-11-03
**Preregistration:** `PRISM_NGS_RECEIVING_PREREGISTRATION_V1.md`, frozen before any arm.
**Status:** NEGATIVE. **K1 fires.** Nothing promoted. **Banked so this is not re-run.**

---

## 1. Verdict

Walk-forward within season, 2023–2025, 2,031 evaluated player-weeks, **258 player-seasons with
≥3 weeks** (minimum 100 — met).

| arm | M1 mean per-player \|err\| |
|---|---|
| **B** prior-weeks mean receiving yards *(incumbent)* | **27.5686** |
| **C** naive control: prior targets × league yards-per-target | **27.0685** |
| N1 air-yards baseline | 28.2747 |
| N2 B + separation | 27.6061 |
| N3 B + air-yards share | 27.6705 |

| arm | G1 vs B | G2 | G3 | G5 vs C | G4 |
|---|---|---|---|---|---|
| C | −0.1347 lower bound → **FAIL** | PASS 61% | FAIL | *(control)* | PASS |
| N1 | **FAIL** | FAIL | FAIL | FAIL | FAIL |
| N2 | **FAIL** | FAIL | PASS | FAIL | FAIL |
| N3 | **FAIL** | FAIL | PASS | FAIL | FAIL |

**All three NGS arms are worse than simply using the receiver's past yards.** No arm passes G1.

---

## 2. The design rationale was wrong, and that is the finding worth keeping

The study was motivated by a measured fact: three NGS metrics are **more stable** than receiving
yards (`avg_intended_air_yards` r=+0.754, `percent_share_of_intended_air_yards` +0.730,
`avg_separation` +0.545, against `yards` +0.457). The reasoning was *"a more stable quantity is
a better thing to build a baseline on."*

**That reasoning is false, and this is the cleanest possible demonstration.** The single most
stable metric — intended air yards, r=+0.754, two-thirds more stable than the baseline — produced
the **worst** arm (N1, 28.2747 vs 27.5686).

**Stability is not relevance.** `avg_intended_air_yards` measures *where* a receiver is targeted
— a genuine, persistent stylistic trait. Receiving yards depend on *how often* he is targeted and
what happens after. A deep threat and a volume slot receiver can have identical, highly stable
air-yards profiles and completely different yardage. The metric is stable because it is a style
fingerprint, and a style fingerprint does not forecast volume.

**The correct design criterion was correlation with the future outcome, not stability of the
predictor.** Stability is necessary but nowhere near sufficient. Recorded because the stability
table is seductive and a future session will be tempted by it again.

---

## 3. The control beat everything

**C — prior mean targets × league yards-per-target — is the best arm** (27.0685), ahead of both
the yards baseline and every NGS specification. Its own G1 CI straddles zero
([−0.1347, +1.1135]) so it is **not** a validated improvement, and it fails G3, so nothing is
promoted from it either.

But the direction is consistent with the whole day: **opportunity volume is the lever, and
efficiency-style metrics are not.** The same conclusion arrived from four unrelated directions:

- the MC opportunity leak — 30% of target mass allocated to players who record none;
- the alpha-tier bias — an *allocation* compression, not an efficiency error;
- the prior-anchoring and ensemble studies — estimator corrections all failed;
- this study — stable efficiency/style metrics fail; counting targets does best.

**The model's defect is who gets the opportunity, not what they do with it.**

---

## 4. What this does and does not say

**Does say:** NGS receiving — intended air yards, air-yards share, separation — adds nothing to
receiving-yards prediction at the player level once the receiver's own recent baseline is in
hand, across three seasons, 258 player-seasons.

**Does NOT say:**
- that NGS is useless for other markets. **Rushing NGS was not tested** (`expected_rush_yards`,
  RYOE, `percent_attempts_gte_eight_defenders`) and the RB lane has a different, documented
  defect;
- that these metrics are useless for *opportunity* prediction — only receiving **yards** was the
  outcome. `percent_share_of_intended_air_yards` is a depth-weighted role measure and its value
  as a **target-share** predictor is untested, which is a materially different question and one
  the MC leak work makes relevant;
- that `avg_yac_above_expectation` fails — it was excluded a priori as least stable (+0.295) and
  **was never scored**.

---

## 5. Ledger entry

> *"NGS receiving metrics improve receiving-yards projection"* — **NULL.** Air yards, air-yards
> share and separation, one global coefficient each, walk-forward 2023–2025, 258 player-seasons.
> All three arms **worse** than the plain yards baseline; the naive targets × league-YPT control
> beat every one of them. Mechanism: stability ≠ relevance — the most stable metric gave the
> worst arm. **Untested and still open: rushing NGS, and NGS as a predictor of target share
> rather than yards.**
