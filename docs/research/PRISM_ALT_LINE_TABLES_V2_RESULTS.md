# PRISM — ALT-LINE TABLES V2: ALL FIVE MARKETS, AND A CORRECTION TO V1

**Date:** 2026-11-04
**Parent:** `PRISM_ALT_LINE_ANCHOR_RESULTS_V1.md`
**Label:** MARKET-ASSISTED. No football projection touched.
**Artifacts:** `data/research/alt_line_clearance_v2.csv`,
`data/research/alt_line_receptions_absolute_v2.csv`,
`scripts/research/build_alt_tables_v2.py`

---

## 1. The correction — v1's conservative column was too generous

**v1's `breakeven_american_conservative` was a 1.96σ sampling bound and nothing
else.** It assumed the only uncertainty was how many rows we had. That is wrong:
realized clearance also moves between seasons.

Measured at the same fraction of line, 2024 against 2025:

| market | max \|drift\| | |
|---|---|---|
| **`rush_rec_yards`** | **0.0632** | **exceeds the 3pp calibration bar** |
| `rec_yards` | 0.0286 | within |
| `receptions` | 0.0284 | within |
| `pass_yards` | 0.0255 | within |
| `rush_yards` | 0.0188 | within |

**`rush_rec_yards` is the market carrying v1's headline "94.6% clears" row.**
v2's conservative figure subtracts the sampling bound **and** the measured
drift. What that costs:

| row | cleared | drift | v2 conservative | break-even v1 → v2 |
|---|---|---|---|---|
| `rush_rec_yards` Q0 ×0.4 | 0.7752 | **+0.1270** | 0.5827 | −345 → **−140** |
| `rush_rec_yards` Q2 ×0.4 | 0.8619 | +0.0594 | 0.7425 | −624 → **−288** |
| `rush_rec_yards` Q3 ×0.4 | 0.9409 | −0.0122 | 0.8792 | −1592 → **−728** |

**A v1 user would have accepted −345 on a proposition whose honest floor is
−140.** That is the correction and it matters more than the two markets added
below.

## 2. `pass_yards` — pooled, now priceable

It failed v1's support floor **only because of stratification**: 36 players in
the 2024 fit is ~9 per quartile against a 40-player bar.

The parent study had already measured that stratification **does not help this
market** — S2 was **+0.00049 [−0.00028, +0.00131]**, non-significant and
point-*negative* — and that the pooled arm passed decile calibration at 2.98pp
where the stratified arm **failed** at 3.38pp. **Pooled is the better model
here, not a fallback.** Both seasons give **812 rows / 43 players**, clearing the
bar.

| × line | cleared | drift | conservative | need | safe |
|---|---|---|---|---|---|
| 0.40 | 0.9704 | +0.007 | 0.9133 | −3283 | −1053 |
| 0.50 | 0.9495 | +0.012 | 0.8725 | −1880 | −684 |
| 0.60 | 0.9015 | +0.019 | 0.7939 | −915 | −385 |
| 1.00 | 0.4852 | +0.026 | — | +106 | — |

At 1.00× it clears **48.5%** — the main line is a coin flip here too.

## 3. `receptions` — absolute thresholds, and a disclosed post-hoc restriction

A fraction of an integer line is the wrong parameterisation: `0.4×`, `0.5×`,
`0.6×` and `0.7×` of a 2.5 line **all mean "≥2"**. Books quote "3+ receptions",
so the table now does too.

**The preregistered bar failed and I am not hiding it.** Fit 2024 → predict
2025, decile calibration came back **0.0525** against a stated **0.03** bar.

The failure is concentrated: main line **5.5** (55 validation players) misses by
−0.20, −0.13, −0.12. Every other line sits within ~6pp and mostly 2–3pp, and
the market **pooled** drifts only 0.0284 — so the *market* is stable and the
**2-D line×k cell table is simply finer than two seasons can support at the high
lines.**

**Emitted cells are therefore restricted to main lines ≤ 4.5. That restriction
was chosen after seeing which cell failed — it is post-hoc**, it is flagged in
the artifact as `post_hoc_line_restriction`, per-cell drift is published, and it
should be re-validated on a clean split once a third season exists.

P(receptions ≥ k):

| ≥k | line 1.5 | 2.5 | 3.5 | 4.5 |
|---|---|---|---|---|
| 2 | 0.532 | 0.731 | 0.858 | 0.915 |
| 3 | 0.287 | 0.493 | 0.678 | 0.770 |
| 4 | 0.120 | 0.283 | 0.473 | 0.609 |
| 5 | 0.051 | 0.148 | 0.299 | 0.436 |

## 4. What is now priceable

**All five markets.** `scripts/price_alt_line_offers_v1.py` routes
`pass_yards` to the pooled row, `receptions` to the absolute table
(k = ⌊T⌋+1, so "over 3.5" is "≥4"), and the other three to the stratified
fraction table.

It still **refuses rather than guesses**: a fraction outside support, a
`receptions` main line above the 4.5 cap, and a player with no main-line anchor
all return a named reason and no number.

## 5. Honest limits

- **The drift adjustment is backward-looking.** It subtracts the drift *already
  observed* between 2024 and 2025. A 2026 regime shift larger than that is not
  covered, and two seasons is a thin basis for estimating regime risk at all.
- **The `receptions` line cap is post-hoc** (§3).
- **None of this has met live alt-line data.** The capture path has never
  executed; column shapes are read from the provider's documented schema, not
  observed.
- **The model's own distribution is still not used anywhere here**, and should
  not be: `_prob_over` is a Normal approximation whose `p_over` is wrong by up
  to +0.448, worst in the tails where alt lines live.
