# PRISM — ALT-LINE ANCHOR: THE COMMITTED TABLE WAS ANCHORED ON THE WRONG NUMBER

**Date:** 2026-11-04
**Preregistration:** `PRISM_ALT_LINE_ANCHOR_PREREGISTRATION_V1.md`, frozen before any arm was scored.
**Label:** **MARKET-ASSISTED.** No football projection modified. No production code changed.
**Verdict:** **S1 passes decisively and reproduces prospectively. The incumbent table
`alt_line_clearance_curve_2024_2025_v1.csv` is DEPRECATED.** It is replaced by
`data/research/alt_line_market_anchored_table_v1.csv`.
**And one prediction of mine was wrong in the model's favour** — see §4.
**Artifacts:** `data/research/alt_line_market_anchored_table_v1.csv`,
`data/research/alt_line_anchor_v1_results.json`,
`scripts/research/evaluate_alt_line_anchor_v1.py`.

---

## 1. Mechanical gates

| gate | result |
|---|---|
| **M1** no sportsbook odds/EV variable as an input | **PASS** — 27 forbidden columns checked by name; feature set is `proj_use, line_use, actual_use, player, market, season, week, era` |
| **M2** no 2025/2026 row in the 2024 fit | **PASS** — fit `[2024]`, validate `[2025]`, prospective `[2026]` |
| **M3** B0 independent of every model column | **PASS** — recomputed with all model columns dropped, bit-identical |
| **M4** unverified outcomes VOID, not losses | applied — **463 of 17,715** historical rows and **106 of 2,007** 2026 rows voided (89 unmatched, 17 zero-volume) |
| **K7** support floor | applied as a filter — 150 of 200 candidate table cells survive ≥100 rows **and** ≥40 players |

Samples: fit **9,401**, validate **7,851** (365 players), prospective **1,480** (269 players).
Line-quartile boundaries were estimated on the 2024 fit sample alone and applied unchanged
downstream. 2026 `z` was standardised from prior weeks only, so **W1 is excluded** from every
prospective contrast.

## 2. S1 — the anchor must be the sportsbook line, not our projection

Held-out 2025, Brier, bootstrapped over **players**. Negative = the market anchor is better.

| market | Δ Brier (B0−B1) | 95% CI | players | |
|---|---|---|---|---|
| `receptions` | **−0.01932** | [−0.02236, −0.01661] | 302 | **PASS** |
| `rush_rec_yards` | **−0.01619** | [−0.02154, −0.01085] | 114 | PASS · K5 |
| `rec_yards` | **−0.01381** | [−0.01607, −0.01148] | 299 | **PASS** |
| `rush_yards` | **−0.01321** | [−0.01701, −0.00981] | 137 | PASS · K5 |
| `pass_yards` | −0.00567 | [−0.00748, −0.00389] | 35 | PASS · K5 |

**Five of five, and it reproduces on 2026** (`rec_yards` −0.01344, `receptions` −0.01993,
`rush_yards` −0.02354, `rush_rec_yards` −0.01410, all CIs excluding 0; `pass_yards` −0.00170 ns,
same sign). **S5 holds and K6 does not fire.**

**This was nearly guaranteed and the preregistration said so.** §0 had already measured 32–54pp of
volume compression in the projection and a near-unbiased line, so replacing the anchor removes a
large signed error. **It is a measurement of a known defect, not a discovery.** Note that
`pass_yards` — the one market with a promoted synthesis and only 14.4pp of compression — shows the
smallest gain and does not reach significance prospectively. The effect tracks compression, which
is the mechanism it should track.

## 3. What the incumbent table actually cost — in price terms

This is the decision-relevant number. The incumbent expresses the threshold as a fraction of the
**projection** and pools over volume. Measured on held-out 2025:

**`rush_yards`, `frac_of_proj = 0.6`. Table promises 70.4% → "beat −238".**

| volume | line ~ | n | realized | gap | true break-even | |
|---|---|---|---|---|---|---|
| Q0 | 8.5 | 274 | **60.9%** | −9.5pp | **−156** | **overpaying at −238** |
| Q1 | 22.5 | 366 | 71.3% | +0.9pp | −249 | ok |
| Q2 | 43.5 | 338 | 79.6% | +9.1pp | −390 | value left |
| Q3 | 69.0 | 292 | **90.4%** | **+20.0pp** | **−943** | **value left** |

**`rush_rec_yards`, `frac_of_proj = 0.6`. Table promises 82.5% → "beat −473".**

| volume | line ~ | n | realized | gap | true break-even | |
|---|---|---|---|---|---|---|
| Q0 | 26.5 | 54 | 77.8% | −4.8pp | −350 | overpaying |
| Q1 | 43.5 | 93 | 76.3% | −6.2pp | −323 | overpaying |
| Q2 | 62.5 | 160 | 89.4% | +6.8pp | −841 | value left |
| Q3 | 90.5 | 347 | **96.0%** | **+13.4pp** | **−2379** | **value left** |

`rec_yards` at 0.6 runs −9.2pp (Q0) to +9.5pp (Q3) against a promised 72.6%.

**The error has two signs and both cost money.** On low-volume players the table told us to accept
prices 5–10pp worse than the proposition is worth — those bets were losers by construction. On
high-volume players it told us to refuse bets worth up to **−943** and **−2379** because it quoted
−238 and −473. **The safe tier the model kept failing to find was being filtered out by the
table's own anchor.**

## 4. S3 — does the model add anything? Partly, and I predicted it would not

Δ Brier (B3 − B2), 2025, over players. B2 is already volume-stratified and model-free.

| market | Δ | 95% CI | players | 2025 | 2026 |
|---|---|---|---|---|---|
| `receptions` | **−0.00418** | [−0.00545, −0.00305] | 302 | **PASS** | −0.00214 [−0.00453,+0.00007] ns |
| `rec_yards` | **−0.00102** | [−0.00134, −0.00070] | 299 | **PASS** | **−0.00048 [−0.00095,−0.00006] holds** |
| `rush_rec_yards` | +0.00008 | [−0.00147, +0.00137] | 114 | ns · K5 | ns |
| `rush_yards` | +0.00134 | [−0.00001, +0.00264] | 137 | ns · K5 | ns |
| `pass_yards` | +0.00068 | [+0.00005, +0.00143] | 35 | **worse** · K5 | ns |

**I predicted S3 would fail. It passes in both markets that have the power to detect it** —
`rec_yards` and `receptions` are the only two clearing K5's 150-player bar, and both pass, with
`rec_yards` reproducing on 2026. The three markets where S3 fails are the three that are
**K5-underpowered**, so their failure is weakly evidenced, not established.

**But the effect is small and I will not inflate it.** 0.001–0.004 Brier on a base of ~0.21 is
**0.5–2% relative**, and `receptions` loses significance prospectively. **A 1% Brier improvement is
not demonstrated to convert into profit** — that requires alt-line prices, which §6 says we do not
have. The defensible statement is: *the projection carries a little information about clearance
beyond volume, in receiving markets, and it is not enough to build a tier on.*

**S2** (volume stratification over the pooled market anchor) passes for `rec_yards` (−0.00178) and
`receptions` (−0.00385) and is ns for the other three.

## 5. S4 — the arm being shipped missed its own calibration gate

Max |predicted − realized| across deciles of predicted probability, 2025:

| arm | max gap | |
|---|---|---|
| B3 (market + volume + model) | 0.0295 | PASS |
| B0 (market anchor, pooled) | 0.0298 | PASS |
| **B2 (market + volume) — shipped** | **0.0338** | **FAIL vs the ±3pp bar** |
| **B1 (projection anchor) — incumbent** | **0.0892** | **FAIL badly** |

**K4 does not fire** — not every arm fails — but B2 is the arm the table ships and it missed by
0.38pp. **Disclosed, not worked around.** The mitigation is to treat
`breakeven_american_conservative` (the lower 95% bound, clustered over players) as the operative
number rather than the point estimate; it absorbs more than 0.38pp. B1's **8.92pp** miss is a
second, independent reason the incumbent is withdrawn.

## 6. The deliverable, and the half of the calculation we do not have

`data/research/alt_line_market_anchored_table_v1.csv` — 150 cells, threshold as a fraction of the
**sportsbook line**, stratified by line quartile, with player-clustered CIs and break-even prices.
Internal check: **at `frac = 1.00` clearance is 0.497–0.561 in every quartile of every market**,
which is the line behaving as a median exactly as §0(a) said.

The short-price end, 2024 fit, ≥75% clearance:

| market | vol | line ~ | × line | clears | beat | conservative |
|---|---|---|---|---|---|---|
| `rush_rec_yards` | Q3 | 84.5 | 0.40 | **94.6%** | −1748 | **−840** |
| `rush_rec_yards` | Q3 | 84.5 | 0.50 | 90.2% | −918 | −499 |
| `receptions` | Q3 | 4.5 | 0.40 | 91.9% | −1141 | −634 |
| `rush_yards` | Q3 | 66.5 | 0.40 | 89.5% | −853 | −419 |
| `receptions` | Q0 | 1.5 | 0.60 | 88.7% | −787 | −527 |
| `rec_yards` | Q3 | 57.5 | 0.40 | 86.0% | −616 | −373 |
| `rush_yards` | Q3 | 66.5 | 0.50 | 85.4% | −586 | −305 |
| `rush_rec_yards` | Q2 | 60.5 | 0.40 | 84.7% | −553 | −350 |

**What this table cannot tell you.** It gives the price you **need**. It does not give the price the
book **offers**, because the boards carry main-line odds only — no alt-line prices exist in this
repository. **Converting this into realized profit requires reading the offered alt price and
comparing.** That is the single remaining blocker and it is a data-acquisition question, not a
modelling one.

Two further limits, stated rather than buried:

- **`receptions` is parameterised wrongly.** It is an integer market, so `0.40 × 1.5` and
  `0.60 × 1.5` collapse to the same threshold; Q0 reads 88.7% at three different fractions and
  quartile 1 does not exist. **`receptions` should be rebuilt on absolute integer thresholds**
  before anything is bet off its rows.
- **These are per-leg, and legs correlate.** Five 85% legs are 0.85⁵ = **44%** if independent, and
  two legs in the same game are positively correlated, so a same-game slip is worse than that.

## 7. Predictions scored

- **"S1 passes clearly"** — **RIGHT**, 5/5 with prospective reproduction. I also said it would be
  weak evidence of skill, which stands.
- **"S2 passes modestly, most where compression is steepest, least for `pass_yards`"** —
  **RIGHT**; `rec_yards`/`receptions` pass, `pass_yards` is the one point-negative arm.
- **"S3 fails"** — **WRONG** in the two markets with the power to test it. Both pass, one
  reproduces prospectively. This is the first result in this programme where the model beat my
  expectation of it.
- **Named risk 1, "the era shift threatens B3 most and is the likeliest reason nothing ships"** —
  **WRONG**; `rec_yards` survived the era change. The risk was real and did not land.
- **Named risk 2, "S1 will look like a win and is not the content"** — **RIGHT**, and §2 says so.
- **Named risk 3, "B2's gain over B0 is partly the market's information, not ours"** — stands;
  it is why §4 reports only B3 vs B2 as PRISM's contribution.
- **Named risk 4, "`rush_rec_yards` may trip K5"** — **RIGHT**, 114 players.

## 8. What changes and what does not

**Changes:** the incumbent clearance table is deprecated in favour of the market-anchored,
volume-stratified one. That is the extent of what the preregistration licensed.

**Does not change:** no production code, no pricing path, no football projection, no ensemble
weight, no odds fetched. The QB synthesis, the RB P3 Week-1 route and the 0.95 cap are untouched.

**Open, and now sharper:** §0(d) recorded that `rush_rec_yards` has **no calibrated ensemble** —
`data/model_ensemble_weights.csv` has no row for it — and that composing it from the two calibrated
components recovers only 2.3 of its 14.4-yard shortfall. That is a production defect with a
measured cost, it is independent of everything above, and it is the next thing worth fixing.
