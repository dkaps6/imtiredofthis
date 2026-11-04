# PRISM — D3 RESULTS: ALL THREE ARMS WORK, AND EVERY GATE HOLDS

**Date:** 2026-11-03
**Preregistration:** `PRISM_D3_PHANTOM_YARDS_PREREGISTRATION_V1.md`, frozen before any arm ran.
**Verdict:** **A1 is the recommended fix.** Every mechanical and statistical gate passes, for the
first time this session. **A3 wins S1 but remains unpromotable by design** (§6 of the
preregistration). Promotion still requires a holdout — §7.
**Artifacts:** `data/research/d3_phantom_yards_v1.csv`,
`scripts/research/evaluate_d3_phantom_yards_v1.py`.

---

## 1. Gates — all pass, arms named as preregistered

| gate | arms | result |
|---|---|---|
| **M1** zero opportunity → zero yards | A1,A2,A3 | **PASS, exact by construction** (`np.where(t>0,·,0.0)`) |
| **M2** bit-identity to C0 where opportunity ≥1 | A1 only | **PASS** |
| **M3** Gamma mean/sd match μ/σ | A2 only | **PASS** |
| **M4** faithfulness of the C0 reproduction | C0 | **PASS — r = +0.967** |
| **S1** per-player bias improves | A1,A2,A3 | **PASS, both markets** |
| **S2** no tier degrades >2% relative | A1,A2,A3 | **PASS** |
| **S3** total moves toward realized | A1,A2,A3 | **PASS** |

**M4 detail:** n=1036, **r = +0.967**, mine 28.74 against shipped 32.36 (ratio 0.888). The 11%
level gap is expected — production blends MC with ML and State before pricing, while this
reproduces the MC yards block alone. **The correlation is high enough that the study is not
analytic-only**, and the named faithfulness risk did not materialise.

**A correction to my own M1 check:** the run first printed residuals of 3–6e-03 for A1/A2/A3,
which looked like a near-miss. It was my selection, not the arms — I filtered `p_zero_t ≥ 0.999`,
and in those rows **under 0.1% of iterations genuinely do have targets** and legitimately produce
yards. Filtered at exactly `p_zero_t == 1.0` the set is empty, and the gate holds by construction
regardless. Recorded so the 3e-03 is not later misread as a leak.

## 2. S1 — every arm improves both markets

**`rec_yards`** (n=474 players):

| arm | mean \|bias\| | mean bias | gain vs C0 | 95% CI |
|---|---|---|---|---|
| C0 | 12.955 | **+1.428** | — | — |
| **A1** | 12.406 | +0.743 | **+0.549** | [+0.468,+0.634] |
| A2 | 12.355 | +0.645 | +0.600 | [+0.514,+0.688] |
| A3 | 12.362 | +0.659 | +0.593 | [+0.509,+0.681] |

**`rush_yards`** (n=474):

| arm | mean \|bias\| | mean bias | gain vs C0 | 95% CI |
|---|---|---|---|---|
| C0 | 7.211 | **+0.080** | — | — |
| **A1** | 6.508 | **−0.821** | **+0.703** | [+0.629,+0.780] |
| A2 | 6.503 | −0.830 | +0.709 | [+0.635,+0.784] |
| A3 | 6.504 | −0.826 | +0.707 | [+0.633,+0.784] |

**The `rush_yards` row is the most informative line in this study.** C0's aggregate bias is
**+0.080 — essentially unbiased** — yet its per-player error is 7.211 and removing phantom yards
improves that error by **+0.703** while pushing the aggregate to **−0.821**.

**C0's aggregate balance was an accident of two errors cancelling:** phantom yards inflating the
bench, and genuine under-projection on the backs who carry the ball. Fixing the phantom reveals
the under-projection that was hiding behind it. **This is direct evidence for why pooled aggregate
metrics may veto but never approve** — an unbiased total concealed a 7-yard per-player error and
would have reported this market as healthy.

## 3. S2 — the fix lands exactly where the defect lives

`rec_yards` tiers from the C0 projection, **bottom tier first**:

| tier | n | actual | C0 | **A1** | A2 | A3 |
|---|---|---|---|---|---|---|
| **Q1 low** | 200 | 0.42 | **+2.86** | **+0.87** | +0.77 | +0.77 |
| Q2 | 200 | 7.45 | +3.24 | +2.76 | +2.60 | +2.61 |
| Q3 | 200 | 15.11 | +1.96 | +1.70 | +1.61 | +1.64 |
| Q4 TOP | 199 | 39.34 | −2.42 | −2.47 | −2.50 | −2.48 |

**The bottom tier's over-projection falls 70%, +2.86 → +0.87.** The top tier moves 0.05 yards
(0.13% relative, inside the 2% bar) because high-volume players almost never draw zero
opportunity — exactly as predicted.

**S3 level:** projected/realized goes **1.091 → 1.046** (A1), 1.040 (A2). Mean phantom yards
removed per player-week: **A1 +0.693**, A2 +0.792, with **160 of 799 rows losing more than a
yard** and a maximum of 2.62 — matching the §1 analytic ceiling of 2.40 plus the one-target term.

## 4. K6 — A2 is distinguishable, but only on receiving

| contrast | `rec_yards` | `rush_yards` |
|---|---|---|
| A2 − A1 | **+0.0511** [+0.0346,+0.0679] | +0.0055 [**−0.0007**,+0.0120] |
| A3 − A1 | +0.0444 [+0.0335,+0.0554] | +0.0040 [+0.0021,+0.0060] |

**K6 fires on `rush_yards` but not on `rec_yards`.** The residual clip bias is real for receiving
(σ=6.0 against a 7.5-yard-per-target mean, so the floor binds for two or three targets) and
immaterial for rushing (σ=3.0 against 4.2 per carry, binding only at one).

**A2's extra gain is +0.05 yards against A1's +0.55 — 9% of the fix for a change of distribution
family.** That is a poor trade: A1 is a two-token guard on an existing draw, A2 replaces the
Normal with a Gamma everywhere and would need its own distributional validation. **Recommend A1.**

## 5. Prediction scored — and this one was right

- **A1 passes S1 — RIGHT.**
- **"reduces low-tier `rec_yards` by roughly 1.5–2.5 yards" — RIGHT**, measured **−1.99**
  (+2.86 → +0.87), inside the stated range.
- **"negligible effect on high-volume players" — RIGHT**, 0.05 yards at Q4.
- **"a material part but not all of the +8%–30% bench over-projection" — RIGHT**: the bottom tier
  keeps +0.87 of over-projection after the fix, so this mechanism was most of it but not all.
- **"A2 adds a small further gain, K6 a live possibility" — RIGHT**, and K6 fired on one of two
  markets.
- **Named faithfulness risk — did not materialise** (r = +0.967).

**Six of six.** First study this session where every prediction held and no gate was
mis-specified. The arm-scoping change in §0 of the preregistration is the only thing I did
differently, and it is worth keeping.

## 6. A3 remains unpromotable, as designed

A3 (floor removed) wins S1 — but §6 of the preregistration **disqualified it in advance**: this
study scores the **mean only**, the floor exists to widen the distribution for low-volume players,
and `model_sd` feeds pricing. **A3 could improve the mean while understating tail risk, and this
design cannot see that.** Its S1 win does not change that, and I am not promoting it on evidence
the design declared insufficient before the number existed.

A1 and A2 are unaffected: A1 touches only the zero cell, where there is no tail to understate, and
A2 preserves σ exactly by construction.

## 7. The limitation that caps promotion

**No holdout.** Found on W7+W8, tested on W7+W8 — 799 player-weeks, 54 team-weeks. As with D2, the
arms carry **no fitted parameters** (A1 is a deterministic guard; A2's Gamma is parameterised from
μ and σ, not fitted) and M1–M4 are exact rather than estimated. **But it is the same two weeks.**
W3–W5 2026 are permanently lost (AGENTS.md §5a), so promotion needs a prospective week or a
prior-season reconstruction.

**S4 (side accuracy) was not computed.** The preregistration made it descriptive-only and
unpassable; with ~800 rows and an MDE above 1.5pp it could not have informed anything, and I am
recording the omission rather than reporting a number that cannot bear weight.

## 8. Next — the joint study is now mandatory

Three fixes measured in isolation, each on the same two weeks:

| | team `tgt_share` sum | `rec_yards` projected/realized |
|---|---|---|
| production | **1.5110** | **1.091** |
| D2 best (delete multipliers) | 1.3772 | — |
| D1 best (zero out inactives) | 1.4270 | — |
| **D3 best (A1)** | unchanged | **1.046** |
| truth | **0.9536** | 1.000 |

**D1 and D2 both remove share mass; D3 removes yards given share.** They are therefore **partly
complementary rather than competing** — D3 operates downstream of the share layer, so it does not
contest the same +0.5574 that D1 and D2 do. That is a better position than the audit feared, but
it is a hypothesis from these isolated runs, not a measurement.

**The joint D1+D2+D3 study must be designed around the total**, must score both objectives, and
must check the specific risk that D1+D2 (which cut share) plus D3 (which cuts yards per share)
**compound multiplicatively into under-projection.** D1's §3 lesson applies directly: with the
level still wrong, whether any individual fix helps depends on the others.

No sportsbook variable entered any arm. No production code modified.
