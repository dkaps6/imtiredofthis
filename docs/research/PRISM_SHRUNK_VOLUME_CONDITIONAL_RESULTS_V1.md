# PRISM — SHRUNK VOLUME-CONDITIONAL: NOTHING PROMOTED, AND THE LEVEL GAP IS A TRADE-OFF, NOT A BUG

**Date:** 2026-11-04
**Preregistration:** `PRISM_SHRUNK_VOLUME_CONDITIONAL_PREREGISTRATION_V1.md`, frozen before any fit.
**Verdict:** **no market passes all four gates on 2026. NOTHING PROMOTED. No production change.**
**One real positive:** `rush_rec_yards` per-player MAE **27.802 → 27.016, CI [−1.463, −0.114]** —
a significant accuracy gain on a fully independent season, the first in this programme.
**Artifacts:** `data/research/shrunk_volume_conditional_v1_results.json`,
`scripts/research/evaluate_shrunk_volume_conditional_v1.py`

---

## 1. M5 saved the study, and named risk 1 deserved its weighting

The preregistration rated the baseline reconstruction as "the single most likely way this study
produces a confidently wrong answer rather than a null." Measured:

| market | recorded `ensemble_proj` level | reconstructed calibrated level | delta |
|---|---|---|---|
| `rec_yards` | 0.7067 | 0.8020 | **+0.0953** |
| `receptions` | 0.6888 | 0.8274 | **+0.1386** |
| `rush_rec_yards` | 0.6570 | 0.8035 | **+0.1465** |
| `rush_yards` | 0.7380 | 0.7380 | +0.0000 |

**Fitting on the recorded baseline would have trained a correction for a 0.66–0.71 starting point
and applied it to a 0.77–0.84 production one.** That is a 10–15pp phantom gap, and every arm would
have over-corrected 2026 by roughly that much. I would have read it as the method failing.

`rush_yards` is unchanged because it was **already calibrated** in the 2024/25 benchmark (frozen
STACK1 weights), which is exactly the consistency check that makes the other three credible.

## 2. LOSO selected genuine shrinkage — and said "don't bother" for the biggest market

| market | arm | **λ** | LOSO ppMAE λ=0 → λ\* |
|---|---|---|---|
| `rec_yards` | V1 linear | **0.0** | 18.6277 → 18.6277 |
| `receptions` | V3 isotonic | 0.2 | 1.4144 → 1.4129 |
| `rush_yards` | V3 isotonic | 0.4 | 18.3986 → **18.1302** |
| `rush_rec_yards` | V1 linear | 0.3 | 25.3352 → 25.2181 |

**`rec_yards` selected λ = 0.0 — no correction is best. K6 fires for the largest market in scope.**

## 3. 2026, touched exactly once

| market | n / players | ppMAE | level | spread | worst Q | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|---|---|---|---|
| `rec_yards` | 638/221 | 20.674 → 20.674 | 0.813 → 0.813 | +0.223 → +0.223 | +0.0% | f | P | P | **f** |
| `receptions` | 633/219 | 1.554 → 1.547 | 0.840 → 0.874 | +0.198 → +0.173 | +0.3% | **f** | P | P | **f** |
| `rush_yards` | 343/119 | 17.711 → 17.995 | 0.811 → 0.927 | +0.120 → **+0.019** | +5.1% | **P** | P | **f** | **f** |
| **`rush_rec_yards`** | 148/58 | **27.802 → 27.016** | 0.765 → 0.823 | −0.035 → −0.020 | −2.2% | **P** | **P** | **P** | **f** |

**`rush_rec_yards` passes S1, S2 and S3 and improves per-player MAE significantly** — CI
[−1.463, −0.114], excluding zero — on a season the correction never saw. **It fails only S4.**

## 4. The actual finding: you cannot have both

**S4 fails everywhere, and in the opposite direction from the parent.** The parent over-shot to
1.06; here every arm **under-shoots**: 0.813, 0.874, 0.927, 0.823 against a ±5pp band around 1.0.

That is not a tuning miss. It is the two objectives pulling apart:

- **MAE-optimal λ is 0.0–0.4.** Selected by leave-one-season-out, on accuracy alone.
- **Level-optimal λ would be ≈1.0 or higher**, since 2026 baselines sit at 0.765–0.840.
- **The parent already measured what λ=1.0 does**: it overshoots the level *and* damages the low
  quartile by +8.7% to +12.9%.

**So the compression is substantially optimal shrinkage, not pure error.** A projection minimising
squared error *should* sit below the realized mean when the outcome is noisy and right-skewed.
Removing that bias costs accuracy, and the amount it costs is now measured rather than assumed.

**Named risk 4 called this exactly:** *"Selecting on MAE may systematically under-shrink for the
level gate… if they disagree, S4 fails and that is the honest answer rather than a reason to change
the criterion."* It disagreed. I am not changing the criterion.

**Named risk 1 was right to be weighted first** (§1). **My prediction was wrong**: I said λ would
land 0.3–0.7 and `receptions` would pass all four; `receptions` failed S1 *and* S4, and `rec_yards`
selected zero. The one thing I got right was `rush_yards` failing S3 again (+5.1% at Q0).

## 5. What this closes and what it leaves

**Closed.** The level gap in non-QB markets is **not** a calibration defect that a monotone
recalibration can remove for free. Three studies have now tried — unnormalised weights, an
intercept, and shrunk volume-conditional maps — and all three hit the same wall from different
directions. **Stop trying to fix the level with a transform of the model's own output.**

**Open, and now better evidenced.** `rush_rec_yards` is the one market where a correction produced
a significant, independent accuracy gain. It is also the market with no calibrated ensemble until
today, the worst projection cell in the programme, and the smallest 2026 sample here (148 rows / 58
players). **That combination is worth one targeted study, not another sweep.**

**For betting specifically, this changes nothing and should not.** The alt-line decision layer takes
its probabilities from realized outcomes, not from the projection, precisely because the projection
carries this bias. That design choice is now independently vindicated.

## 6. Scope

No production code or artifact changed. No sportsbook variable entered any arm. `pass_yards` was
out of scope by design and emitted no coefficient (M4), which is the fix for the parent's
unverifiable control.
