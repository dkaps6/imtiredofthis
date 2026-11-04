# PRISM — VOLUME-CONDITIONAL RECALIBRATION: NOTHING PROMOTED, BUT THE SPREAD IS CORRECTABLE

**Date:** 2026-11-04
**Preregistration:** `PRISM_VOLUME_CONDITIONAL_PREREGISTRATION_V1.md`, frozen before any arm ran.
**Verdict:** **no market passes all four gates. NOTHING PROMOTED. No production change.**
**But the headline is not the refusal** — the compression spread was cut by up to **83%**
with per-player MAE and both quartile ends intact. The blocker is a **level overshoot**, and
`receptions` misses by about **1 percentage point**.
**Artifacts:** `data/research/volume_conditional_v1_results.json`,
`scripts/research/evaluate_volume_conditional_v1.py`

---

## 1. A measurement bug I caught in my own harness

The first run assigned volume strata using each arm's **corrected** projection against the
**baseline** cut points, so rows moved between bins and every arm was scored on a different
population. It also produced `NaN` spreads for two arms, which is what exposed it.

Fixed: the stratum is now fixed by the **baseline** projection, and the corrected value is only
the numerator. **Every number below is post-fix.** The pre-fix run showed S1 failing everywhere;
that reading was an artifact and is withdrawn.

## 2. Results, blind 2025

`spread` = (high-quartile proj/actual) − (low-quartile proj/actual), median-of-ratios, which
named risk 1 designated as governing. Fit 2024, strata from the fit sample.

| market | arm | ppMAE | level | spread V0 → arm | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|---|---|---|
| **`receptions`** | V3 isotonic | 1.4304 → 1.4732 | 0.709 → **1.062** | **0.3247 → 0.0558** | **P** | **P** | **P** (+1.2%) | **f** (1pp over) |
| `receptions` | V1 linear | → 1.4906 | → 1.059 | → −0.0454 | P | P | P | f |
| `rec_yards` | V3 isotonic | — | ✓ | ✓ | P | **f** | P | P |
| `rush_yards` | V2 tiered | 18.24 → 18.77 | 0.750 → **1.034** | 0.1935 → −0.0313 | P | P | **f** (+9.9% Q0) | P |
| `rush_rec_yards` | V2 tiered | 28.40 → 27.97 | 0.621 → 0.889 | 0.0228 → −0.2947 | **f** | P | P | f |

**The spread is correctable.** S1 passed for every arm in `receptions` and `rec_yards` and for
V2 in `rush_yards`. `receptions` fell from 0.3247 to 0.0558 — an **83% reduction** — while S2
and S3 both held.

**What kills it is over-correction.** Every arm overshoots the level: `receptions` 0.709 → 1.06,
`rush_yards` 0.750 → 1.03, `rush_rec_yards` 0.621 → 0.89, and the spread flips from positive to
negative. A correction fitted on 2024 lands past the target on 2025 — the same season drift the
alt-line work measured independently (`rush_rec_yards` clearance moves 0.0632 between those two
seasons).

**M4 could not be verified.** `pass_yards` has **35 blind players** against the 40-player bar, so
K5 fires and the positive control is **unevaluable, not undamaged**. Nothing could promote on that
basis alone.

## 3. What this refutes, including my own prediction

**Named risk 2 did NOT land, and it was my main structural worry.** I wrote that conditioning on
the model's own output could only relabel a feature-level error. It did not: recalibrating
`ensemble_proj` against itself genuinely reduced the measured spread. **K3's premise — "the spread
is not correctable by recalibration on the model's own output" — is false**, and that is the most
useful thing this study produced.

**Named risk 1 landed exactly.** At baseline for `rush_rec_yards` the three spread readings
disagreed *in sign*: mean-of-ratios −0.0835, median +0.0228, mean/mean +0.1072. The preregistration
anticipated this and designated the robust reading as governing, which is the only reason the
result is interpretable.

**Named risk 3 landed for `rec_yards`.** De-compressing adds variance, and `rec_yards` is the one
market where per-player MAE got significantly worse (S2 fail).

**Prediction scored:** "V1 and V3 pass S1; V2 overfits" — **wrong on V2**, which was the *best* S1
arm for `rush_yards` and matched the others elsewhere. "S3 is where this is decided and I am
genuinely unsure" — **wrong**; S3 passed everywhere except `rush_yards`, and the decider was S4.

## 4. The thing I am deliberately not doing

**`receptions` misses promotion by ~1pp of level.** Shrinking the correction by a few percent
would very likely clear S4 and promote it.

**I am not going to do that.** Choosing a shrinkage coefficient after seeing which gate missed and
by how much is precisely what preregistration exists to prevent, and this programme has a measured
record of such adjustments not surviving contact with a holdout. If it is worth doing it is worth
doing properly: a new preregistration with the shrinkage rule stated in advance, validated on a
genuinely independent sample.

**That sample now exists.** 2026 weeks 1/2/6/7/8 board lines joined to nflverse actuals give a
third season that was not used here — the same route that converted the `receptions` line cap from
post-hoc to evidence-backed. A shrunk correction fitted on 2024+2025 and tested on 2026 would be a
clean test. **That is the next study.**

## 5. Scope

No production code or artifact changed. No sportsbook variable entered any arm — the conditioner
is `ensemble_proj` and the target is `actual`, asserted by name against a 19-item forbidden list.
