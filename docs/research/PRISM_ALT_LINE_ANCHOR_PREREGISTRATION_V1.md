# PRISM — ALT-LINE ANCHOR CHOICE, PREREGISTRATION V1

**Frozen:** 2026-11-04, **before any arm is scored.**
**Status:** PREREGISTRATION. No result here. Nothing promoted.
**Label:** **MARKET-ASSISTED.** This is a downstream bet-selection layer. It modifies **no**
football projection and supplies **no** feature to any upstream model. AGENTS.md §3 permits
market-assisted outputs provided they are separately labeled and do not silently replace the
football model. Both conditions are met: nothing in production changes, and the output is a
decision table, not a projection.

---

## 0. What prompted this, stated as exploratory

Four measurements were taken **before** this preregistration and are therefore **descriptive, not
tested**. They are stated here so the motivation is auditable and so no later document can present
them as confirmatory.

**(a) The sportsbook main line is a near-perfect median.** Realized `P(actual > line)` by line
quartile, 2024+2025:

| market | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| `rec_yards` | 0.5084 | 0.5036 | 0.4875 | 0.4760 |
| `rush_rec_yards` | 0.5015 | 0.4904 | 0.5031 | 0.4869 |

**(b) The model's projection is compressed in volume, in every market.** Production 2026 boards,
median `model_proj / vegas_line` by line quartile:

| market | Q1 low | Q2 | Q3 | Q4 high | spread |
|---|---|---|---|---|---|
| `rec_yards` | 1.366 | 1.035 | 0.961 | 0.827 | **53.9pp** |
| `rush_yards` | 1.221 | 1.070 | 0.817 | 0.790 | 43.1pp |
| `rush_rec_yards` | 1.078 | 0.821 | 0.807 | 0.730 | 34.8pp |
| `receptions` | 1.116 | 0.922 | 0.854 | 0.791 | 32.5pp |
| **`pass_yards`** | 1.077 | 1.026 | 1.012 | 0.933 | **14.4pp** |

**(c) The model's probabilities are badly miscalibrated on volume**, 2024+2025 — model `p_over`
against realized over-rate:

| market | Q1 gap | Q2 gap | Q3 gap | Q4 gap |
|---|---|---|---|---|
| `rec_yards` | −0.090 | +0.177 | +0.235 | **+0.260** |
| `rush_rec_yards` | +0.047 | +0.214 | +0.373 | **+0.448** |

On Q4 `rush_rec_yards` the model asserts a **3.9%** chance of the over and reality is **48.7%**.

**(d) `rush_rec_yards` has no calibrated ensemble.** `data/model_ensemble_weights.csv` carries
`pass_yards`, `rush_att`, `rush_yards`, `rec_yards`, `receptions` — **not** `rush_rec_yards`,
which therefore ships `mc_fallback_no_oos_weights`. Deriving it from the two calibrated components
recovers only **2.3 of its 14.4-yard** shortfall, so missing calibration is not the whole cause.

**The consequence for the committed deliverable.**
`data/research/alt_line_clearance_curve_2024_2025_v1.csv` expresses the threshold as a **fraction
of the model projection** and pools across volume. Given (b), `0.60 × proj` is **0.48× the line**
for a Q4 `rec_yards` player and **0.82× the line** for a Q1 player. The pooled rate is therefore
wrong per player in a **signed, predictable** direction — too pessimistic on high volume, too
optimistic on low volume. This is consistent with the graded ledger's SAFE tier falling from
**93.3% (W7) to 25.0% (W8)**, though that is two weeks and is not evidence of the mechanism.

## 1. Research question

**Two questions, in order of decision value:**

**Q1 — anchor.** Should an alt-line threshold be expressed as a fraction of the **sportsbook line**
rather than a fraction of the **model projection**?

**Q2 — does the model add anything?** After the line is used as the anchor, does the model's
projection carry **any** incremental information about clearance? **If it does not, PRISM has no
role in bet selection, and that must be published rather than worked around.**

## 2. Arms — nested, with a model-free naive control

Target: the clearance indicator `1[actual > f × anchor]` over a frozen fraction grid
`f ∈ {0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5}`.

- **B0 — naive control, no model at all.** Anchor = line. Pooled empirical clearance per
  `(market, f)`. **Uses zero model inputs.** This is the arm to beat; if nothing beats it, the
  model is not needed for bet selection.
- **B1 — the incumbent.** Anchor = model projection, pooled per `(market, f)`. This is the
  committed table, unchanged.
- **B2 — stratified market anchor.** Anchor = line, estimated per
  `(market, f, line-quartile)`. Volume-aware, still model-free in the anchor.
- **B3 — B2 plus the model.** B2's cell rate adjusted by a single logistic coefficient on
  `z = standardized(proj / line)`, standardized **within `(market, era, line-quartile)`** so that
  the 2024/25 `mc_only` era and the 2026 `calibrated` era are placed on a common scale. One
  coefficient per `(market, f)`; no interactions. **This is the Q2 test.**

## 3. Sample and split

| | |
|---|---|
| fit | **2024** (`clean_v1_full_stack_vegas_benchmark_detail.csv`) |
| validate | **2025**, untouched by the fit |
| prospective | **2026 W1, W2, W6, W7, W8** committed boards, joined to nflverse weekly actuals |

Markets: `rec_yards`, `receptions`, `rush_yards`, `rush_rec_yards`, `pass_yards`.
**`anytime_td` is excluded by standing instruction** and is absent from the historical set.
All bootstrapping is **clustered over players**, never over rows.

## 4. Gates — each scoped to the arms it governs

- **M1 — all arms.** No sportsbook **odds** variable enters any arm. The **line** is used
  deliberately and is the subject of the study; `over_odds`, `under_odds`, `over_novig`,
  `under_novig`, `ev_over`, `ev_under`, `edge_pct` are **forbidden as inputs** and may be used only
  to report break-even prices. Asserted by name; **failure voids the run.**
- **M2 — all arms.** No 2025 or 2026 row may enter the 2024 fit. Asserted on the fitting frame's
  season values; **failure voids the run.**
- **M3 — B0 only.** B0's prediction must be independent of every model column. Verified by
  recomputing B0 from a frame with all model columns dropped and requiring bit-identity.
- **M4 — all arms.** No row may be scored whose `actual` is unverified. A player absent from the
  weekly table, or present with `attempts + targets + carries == 0`, is **VOID**, not a loss — the
  grading defect already corrected once in this programme.
- **S1 — B0 vs B1 (Q1).** B0 must beat B1 on held-out 2025 **Brier score**, 95% CI excluding 0,
  clustered over players.
- **S2 — B2 vs B0.** Volume stratification must beat the pooled market anchor on 2025 Brier,
  CI excluding 0.
- **S3 — B3 vs B2 (Q2).** The model term must beat B2 on 2025 Brier, CI excluding 0.
- **S4 — all arms.** **Calibration:** predicted clearance within **±3pp** of realized in every
  decile of predicted probability on 2025.
- **S5 — all arms.** The 2025 arm ranking must **reproduce** on the 2026 prospective sample.

## 5. Kill conditions

**K1** M1, M2, M3 or M4 breached → **run void.**
**K2** S1 fails → the projection anchor is not worse than the market anchor; the incumbent table
stands and §0(b) does not transmit to decisions.
**K3** **S3 fails → the model adds nothing to alt-line selection.** Publish it. The betting layer
is then market-anchored and empirical, and PRISM's role is projection research only, not bet
selection. **This is the outcome I must not flinch from.**
**K4** S4 fails for every arm → clearance is not forecastable at this granularity; no table ships
at any fraction.
**K5** fewer than **150 players** in the 2025 validation for a market → that market is
underpowered and is reported as such, not promoted.
**K6** S5 fails — the 2026 ranking reverses 2025 → regime instability between the `mc_only` and
`calibrated` eras dominates, and **nothing ships** regardless of the 2025 result.
**K7** any cell promoted on fewer than **100 rows** or **40 players** → insufficient support.

## 6. Prediction

**S1 passes clearly.** §0(a) and §0(b) together almost force it: the line is an unbiased median and
the projection is compressed by 32–54pp, so replacing the anchor removes a large signed error. I
would be surprised by anything else, which is precisely why S1 is weak evidence of skill.

**S2 passes modestly.** Stratification should help most where compression is steepest
(`rec_yards`, `rush_yards`) and least for `pass_yards`.

**S3 — I expect it to FAIL.** This is the prediction that matters and the one I least want to make.
The model's `p_over` is off by up to 44.8pp in exactly the cells where the money is, and the
irreducible-floor study placed 93% of receivers at the information limit. A single standardized
coefficient on `proj/line` is unlikely to survive a player-clustered CI once volume is already
controlled. **If S3 fails, the honest conclusion is that the alt-line table should be rebuilt
market-anchored and the model should be kept out of bet selection entirely.**

**Named risks, weighted:**

1. **The era shift is the biggest threat, and it threatens B3 most.** 2024/25 receiving is
   `mc_only`; 2026 is `calibrated`. Within-era standardization is a mitigation, not a fix — if the
   *shape* of the proj/line relationship changed, B3's 2025 coefficient is meaningless for 2026 and
   **K6 fires.** I rate this the single most likely reason this study ships nothing.
2. **S1 is nearly guaranteed and will look like a win.** It is not: it is a measurement of a defect
   I already described in §0. The real content is S3. **I must not present a passing S1 as the
   study's result.**
3. **Line-quartile strata are defined by the market**, so B2 and B3 condition on a market variable.
   That is legitimate for a market-assisted layer but means **B2's gain over B0 is partly the
   market's information, not ours.** B3 vs B2 is the only contrast that isolates PRISM.
4. **`rush_rec_yards` has only 654 rows in 2025** against 1,937 in 2024. It is the market with the
   largest apparent defect and the weakest validation support; K5 may well fire on it.

My record in this programme: **~10 prediction hits, ~21 misses**, with the **named risk landing in
nearly every study.** Weight risk 1 heavily.

## 7. Scope

No production code modified. No odds fetched — prices come from the committed boards and from
nflverse actuals, and are used only to report break-even prices, never as inputs. The QB synthesis,
the RB P3 Week-1 route, the frozen ensemble weights and the 0.95 cap are all untouched.

A pass licenses **one** follow-up: replacing
`data/research/alt_line_clearance_curve_2024_2025_v1.csv` with a market-anchored, volume-stratified
table, clearly labeled market-assisted. **It does not license a pricing-path change, a production
model change, or any ticket construction beyond what the table itself states.**
