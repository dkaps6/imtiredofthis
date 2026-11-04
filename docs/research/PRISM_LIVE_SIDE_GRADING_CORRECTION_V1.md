# PRISM — LIVE SIDE-GRADING CORRECTION V1

**Date:** 2026-11-02
**Status:** CORRECTION + MEASUREMENT. Nothing promoted. No production behaviour changed.
**Data:** `data/research/live_2026/backtest_2026_live_calibration_v1_rows.csv` — 1,571 graded
live rows, 2026 weeks 1, 2, 6, 7.

---

## 1. The defect — `side` is NOT the model's pick

The `side` column in the graded live calibration file is a **board listing orientation**, not
the model's directional view. Measured against `sign(model_proj - vegas_line)`:

| | agrees |
|---|---|
| `side` vs model's actual direction | **622 / 1571 = 39.6%** |

Cross-tab:

| `side` column | model implies | n |
|---|---|---|
| over | over | 579 |
| over | **under** | **930** |
| under | over | 19 |
| under | under | 43 |

In 930 of 1,571 rows the column says `over` while `model_proj` sits **below** the line.

**Any study that grades this file by the `side` column measures the wrong direction.** One was
run on 2026-11-01 and produced a pooled hit rate of 48.31% with a CI excluding breakeven,
which was briefly treated as evidence that the model had no side-picking edge. That reading is
**withdrawn**. It was an artifact of the grading bug.

**Rule going forward:** derive the pick as

```python
pick = 'over' if model_proj > vegas_line else 'under'
```

Never grade this file (or any board-derived file) by its `side` column without first
verifying the column against the projection.

---

## 2. Corrected result — the model's directional pick

| scope | record | hit | 95% CI (Wilson) |
|---|---|---|---|
| **all rows** | 832 / 1571 | **52.96%** | [50.5%, 55.4%] |

Breakeven at standard −110 vig is 52.4%. The point estimate clears it; the interval straddles
it narrowly.

### By week — the trend is real

| week | record | hit | 95% CI |
|---|---|---|---|
| 1 | 222/432 | 51.4% | [46.7%, 56.1%] |
| 2 | 210/414 | 50.7% | [45.9%, 55.5%] |
| 6 | 181/336 | 53.9% | [48.5%, 59.1%] |
| 7 | 219/389 | **56.3%** | [51.3%, 61.1%] |

Monotone across the two observed blocks. This is **four weeks and is not yet significant** —
no individual week's CI excludes breakeven, and weeks 3–5 are permanently missing (see
AGENTS.md §5a; those boards expired at `retention-days: 7`). Treat the uptrend as a live
hypothesis with a prospective test available, not as an established result.

### By market cell — this is where the structure is

| market | pick | record | hit | 95% CI |
|---|---|---|---|---|
| pass_yards | **over** | 38/62 | **61.3%** | [48.8%, 72.4%] |
| pass_yards | under | 27/49 | 55.1% | [41.3%, 68.1%] |
| rush_yards | **under** | 101/171 | **59.1%** | [51.6%, 66.2%] |
| rush_yards | over | 55/113 | 48.7% | [39.7%, 57.8%] |
| receptions | over | 94/170 | 55.3% | [47.8%, 62.6%] |
| receptions | under | 192/356 | 53.9% | [48.7%, 59.0%] |
| rec_yards | over | 117/234 | 50.0% | [43.6%, 56.4%] |
| rec_yards | under | 153/296 | 51.7% | [46.0%, 57.3%] |

`rush_yards OVER` is the one cell materially **below** breakeven, and it is the mirror of the
RB rushing mass-deflation defect already documented in
`PRISM_RB_RUSH_SHARE_MASS_BUDGET_V1.md`: the model under-projects RB rushing volume, so its
rush OVERs are manufactured by a known defect while its rush UNDERs ride it.

---

## 3. Model **confidence** carries no information

Computing the model's own probability for its chosen side from `N(model_proj, model_sd)`:

| claimed p | n | mean claimed | realized | miscalibration |
|---|---|---|---|---|
| 0.50–0.55 | 337 | 52.4% | 52.8% | +0.4% |
| 0.55–0.60 | 299 | 57.6% | 48.8% | −8.7% |
| 0.60–0.65 | 226 | 62.3% | 56.6% | −5.7% |
| 0.65–0.70 | 203 | 67.3% | 55.7% | −11.6% |
| 0.70–0.80 | 275 | 74.2% | 52.0% | −22.2% |
| 0.80–1.01 | 231 | 88.3% | 53.7% | −34.6% |

**Realized accuracy is flat at ~49–57% across every confidence band.** A row the model calls
88% lands at 53.7% — the same as a row it calls 52%. The distribution's *location* carries
signal; its *width* does not.

`edge_pct` is no better. Deciles, lowest to highest edge:
51.6, 51.6, 51.6, 52.2, 58.6, 54.1, 58.0, 43.9, 52.9, **55.1%**. Top decile CI
[47.3%, 62.6%]. No monotone relationship.

**Consequence:** do **not** use `model_proj`/`model_sd` normal probability, or `edge_pct`, as a
betting probability or as a ranking key. Select on **market cell**, which does discriminate.

---

## 4. Empirical alt-line curve — validated out-of-sample

For thresholds expressed as a fraction of the model projection, trained on weeks 1/2/6 and
tested on week 7. This measurement never touches `side`, so it is unaffected by §1.

| market | threshold | train | test (W7) | error |
|---|---|---|---|---|
| rush_yards | 50% of proj | 74.5% | 69.1% | −5.4 |
| rush_yards | 65% | 69.9% | 57.4% | −12.6 |
| rush_yards | 80% | 61.6% | 51.5% | −10.1 |
| receptions | 50% | 84.2% | 79.1% | −5.1 |
| receptions | 65% | 77.6% | 73.1% | −4.4 |
| receptions | 80% | 69.1% | 60.4% | −8.7 |
| rec_yards | 50% | 76.0% | 73.9% | −2.1 |
| rec_yards | 65% | 69.4% | 66.4% | −3.0 |
| rec_yards | 80% | 61.9% | 56.0% | −5.9 |

**All nine cells came in below their training estimate; mean absolute error 6.4pp.** The
method transfers directionally but is systematically optimistic and requires a haircut of
roughly 6–10pp, worst on `rush_yards`.

Practical reading: a threshold at **~50% of the model projection** clears ~69–79% of the time.
Thresholds at **75–80% of projection** — which is where sportsbook "popular" alt tiers sit —
clear only ~51–60%.

---

## 5. Per-player side records — NOT established

Using the corrected grading, 88 players have n≥8 graded props.

- median player hit rate **52.8%** (it was 50.0% under the broken grading)
- **4 of 88** clear a 95% Wilson lower bound above 52.4%

At a 95% bound, ~2 of 88 clear **by chance alone**. Four is barely above that. Named extremes
(Marvin Harrison Jr. 8/8; Courtland Sutton, Tyler Shough, Rome Odunze each 7/8) are **not
established skill** — they are what 88 near-coin-flippers produce.

The negative tail is more usable, because it corroborates a known defect rather than standing
alone: **Kenneth Walker III 4/16 (25%)** and **Sam Darnold 0/4**.

**Verdict:** per-player selection is a **tiebreaker at most**. Four weeks cannot identify which
players the model is reliably right about. Do not build a per-player prior on this.

---

## 6. New player-level lead — elite receivers are under-projected

Week 8 board (`data/market_track_record/boards/2026_wk08.csv`) against sportsbook alt ladders:

| player | model `receptions` proj | market alt | implied |
|---|---|---|---|
| Ja'Marr Chase | **4.49** | 4+ @ −1800 | 94.7% |
| Justin Jefferson | **4.12** | 4+ @ −800 | 88.9% |

A market pricing 4+ receptions at 94.7% implies a mean well above 6. A ~2-reception gap on two
alpha receivers is **model error, not edge**, and it is the receiving-side analogue of the
existence-ceiling / WR-hierarchy compression already open in the RB lane. Any alt-line method
keyed on `model_proj` is unusable for these players until this is understood.

**This is the most valuable finding in this document for the research lane**, because it is
player-level, falsifiable, and points at a structural compression rather than a tuning
parameter.

---

## 7. What is promoted

**Nothing.** This document corrects a measurement error and records what the corrected
measurement shows. No coefficient, weight, feature, or pricing path changed.

## 8. Next action — a real prospective test exists

The 2026 Week 8 board is committed (`data/market_track_record/boards/2026_wk08.csv`, 2,820
rows) and its games are now complete. Grading it with the **corrected** direction:

1. extends the series to five weeks and gives the W1→W7 uptrend its first out-of-sample test;
2. tests `pass_yards OVER` 61.3% and `rush_yards UNDER` 59.1% prospectively on rows chosen
   before kickoff, which is the strongest form of evidence this lane can produce;
3. costs nothing — the board is already frozen in git, so this is a clean preregistered read.

Freeze the hypothesis before scoring: **pass OVER > 52.4%** and **rush UNDER > 52.4%** on W8,
with the naive control being "all model picks pooled".
