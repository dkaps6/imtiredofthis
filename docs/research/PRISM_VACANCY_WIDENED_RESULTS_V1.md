# PRISM — VACANCY: **NO EDGE, BUT A MEASURED, FIXABLE DEFECT THAT COSTS MONEY**

**Date:** 2026-11-04
**Preregistration:** `PRISM_VACANCY_WIDENED_PREREGISTRATION_V1.md`, frozen first.
**Verdict: K1 fires — all six cells fail. H1 is now REFUTED, not merely untestable.**
**But the pattern inverts the hypothesis, and that inversion is the valuable result.**
**Artifacts:** `data/research/vacancy_widened_v1.csv`,
`scripts/research/evaluate_vacancy_widened_v1.py`.

---

## 1. All six cells fail, and two had adequate size

| cell | n | players | win% | EV/$1 | 95% CI (clustered) | p |
|---|---|---|---|---|---|---|
| V-STRICT × B-SAME | 88 | 14 | 52.27% | −0.0337 | [−0.4792, +0.4402] | 1.000 |
| V-STRICT × B-SKILL | 194 | 28 | 54.64% | +0.0201 | [−0.2270, +0.2688] | 0.883 |
| V-MID × B-SAME | 136 | 17 | 45.59% | −0.1532 | [−0.4620, +0.1626] | 1.000 |
| **V-MID × B-SKILL** | **310** | **39** | 45.81% | **−0.1408** | [−0.3206, +0.0379] | 1.000 |
| V-LOOSE × B-SAME | 196 | 24 | 46.94% | −0.1322 | [−0.3778, +0.1212] | 1.000 |
| **V-LOOSE × B-SKILL** | **504** | **55** | 48.81% | **−0.0946** | [−0.2365, +0.0551] | 1.000 |

**Best p-value is 0.883 against a Holm threshold of 0.0083.** Nothing is close. **Two cells pass
G4 on size (310/39 and 504/55 players), so this is a refutation rather than a power failure** —
which is the difference from the narrow test.

**The controls settle it:**

| | EV/$1 | 95% CI |
|---|---|---|
| model side, vacancy | **−0.0946** | [−0.2365, +0.0551] |
| **C1 always-OVER, vacancy** | **−0.0082** | [−0.2165, +0.1914] |
| C2 non-vacancy baseline | −0.0248 | [−0.0778, +0.0279] |
| **C3 PLACEBO** (QUESTIONABLE who played) | **+0.0297** | [−0.1358, +0.1987] |

**The placebo is the best cell of all.** If vacancies created an edge, the treatment would beat
the placebo. It loses to it by **12 points**. There is no vacancy edge to harvest.

## 2. The inversion — the model is *actively worse* in vacancy spots

The hypothesis was that the model would do **better** in vacancy spots. It does **worse** than its
own baseline (−0.0946 against −0.0248). And the mechanism is unambiguous:

| population | n | model says UNDER | actual went UNDER | lean |
|---|---|---|---|---|
| **vacancy beneficiaries** | 504 | **68.3%** | **47.2%** | **+21.1pp** |
| non-vacancy baseline | 6348 | 60.9% | 51.9% | +9.0pp |

**The model's under-lean more than doubles when a starter is out.** And the side preference
reverses completely:

| population | model side | always-OVER | gap |
|---|---|---|---|
| **vacancy** | −0.0946 | **−0.0082** | **+0.0865** (always-over wins) |
| non-vacancy | −0.0248 | −0.0862 | **−0.0614** (model wins) |

**A 15-point swing.** In normal spots the model beats a blind over by 6 points. In vacancy spots a
blind over beats the model by 9.

**This is D1 in the betting record.** The model cannot see who is inactive — an OUT player retains
his full target share — so the freed opportunity never reaches his teammates, the model
under-projects them, and it takes the wrong side at twice its normal rate.

## 3. What this is worth, stated with its uncertainty

**The precise measurement is the side lean: 68.3% unders against 47.2% of outcomes, on 504 rows.**
That is not noise, and it is a direct consequence of a defect we have already located and fixed.

**The EV figure is much softer.** The +0.0865 gap is a point estimate whose component intervals
are ~0.3 wide. **"D1 is worth 8.6 points of EV in vacancy spots" is the right order of magnitude
and the wrong level of confidence.** What is solid: the direction, the mechanism, and the fact
that it is a *defect*, not a market property.

**Vacancy rows are 504 of 7786 — about 6.5% of the board.** Even a complete fix there moves the
whole board by roughly 0.5 points. **It is not the profitability gap. It is a real leak with a
known cause and a validated fix.**

## 4. Why this matters for the roadmap

`D1+D3` was justified on accuracy grounds (+1.015 yards per player). **It now has a betting
justification as well, and a falsifiable one:**

> **Fixing D1 should close the +0.0865 model-versus-always-over gap in vacancy spots, and should
> pull the 68.3% under-lean down toward the 60.9% baseline. If it does not, D1's value is
> accuracy-only.**

**That is prospectively testable on W9 and beyond at no cost**, and it is a far better test of the
fix than anything in-sample.

## 5. Prediction scored

- **"only V-LOOSE × B-SKILL reaches G4; expect G1 to fail after Holm" — RIGHT**, and two cells
  reached size rather than one.
- **"carry vacancies show a larger point estimate than target vacancies"** — not separable in the
  output as run; the cells mix both. **Recorded as not evaluated**, a design gap.
- **"C1 ≈ the model-side cell; if anything is there it is the market, not the model" — WRONG in
  direction and more informative for it.** C1 **beat** the model side by 8.6 points. The model is
  not merely failing to see an edge; it is actively taking the wrong side.
- **Named risk, "the placebo is the real test"** — **exactly right.** The placebo came in best of
  all four populations, which is what closed the hypothesis.

## 6. Disposition

**H1 is closed.** Vacancy underreaction is not a betting edge: all six cells fail, two at adequate
power, and the placebo outperforms the treatment.

**Still untested and explicitly open:** alt lines (the only out-of-sample-validated positive in
this programme, pending an authorization decision on alt prices), correlated parlays, live.

No odds fetched. **No bet authorized.** No production code modified.
