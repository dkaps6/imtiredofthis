# PRISM — THE `pass_yards` EDGE IS **NOT REAL**. THERE IS NO CURRENTLY EXPLOITABLE EDGE.

**Date:** 2026-11-04
**Preregistration:** `PRISM_PASS_YARDS_EDGE_DURABILITY_PREREGISTRATION_V1.md`, frozen before any
statistic was computed.
**Verdict: K1, K2, K3 and K4 all fire. All four gates fail.**
**This retracts the headline of `PRISM_EDGE_DECOMPOSITION_EV_V1` §2.**
**Artifacts:** `data/research/pass_yards_edge_durability_v1.csv`,
`scripts/research/evaluate_pass_yards_edge_durability_v1.py`.

---

## 1. G1 — the edge dies under correct bootstrapping

554 rows, **37 quarterbacks**, median 16 rows per QB.

| bootstrap unit | EV/$1 | 95% CI | |
|---|---|---|---|
| **rows** (what I used earlier) | +0.0817 | **[+0.0038, +0.1606]** | appears profitable |
| **quarterbacks** (correct) | +0.0817 | **[−0.0707, +0.2251]** | **spans zero** |

**The cluster interval is 1.89× wider and contains zero.** Rows for one quarterback are not
independent observations — same player, same offence, correlated weekly form. **My earlier
"profitable, CI excludes zero" claim was an artifact of treating 554 correlated rows as 554
independent ones.**

## 2. G2 — it is a two-week result

| wk | n | win% | EV/$1 |
|---|---|---|---|
| **1** | 108 | **72.22%** | **+0.3617** |
| **2** | 124 | 59.68% | +0.1252 |
| 6 | 102 | 52.94% | −0.0006 |
| 7 | 108 | 48.15% | **−0.0924** |
| 8 | 112 | 53.57% | +0.0100 |

**3 of 5 weeks positive, and the entire edge is W1 and W2.** From W6 onward it is flat to
negative. A 72% win rate in W1 is not a repeatable property of a model.

## 3. G3 — 77% of the profit comes from three quarterbacks

| | n | EV/$1 | total |
|---|---|---|---|
| Malik Willis | 16 | **+0.8851** | +14.16 |
| Brock Purdy | 12 | +0.8864 | +10.64 |
| Dak Prescott | 20 | +0.5104 | +10.21 |
| … | | | |
| Sam Darnold | 12 | **−1.0000** | −12.00 |
| Michael Penix Jr | 12 | **−1.0000** | −12.00 |
| Kyler Murray | 12 | **−1.0000** | −12.00 |

**Top 3 quarterbacks contribute +35.01 of +45.67 total profit — 77%.** Dropping only the single
best leaves EV +0.0584, CI **[−0.0907, +0.1977]**.

**Three quarterbacks have an EV of exactly −1.0000: they lost every single bet, 12 for 12.**
Another two won nearly every bet. **That is not a market edge — it is a handful of players the
model reads consistently right or consistently wrong.** With 12–20 bets on the same quarterback
across five weeks, each quarterback is close to **one** independent observation, not sixteen.

## 4. G4 — the model has directional skill, and it is not enough

| | EV/$1 | 95% CI (clustered) |
|---|---|---|
| model's side | +0.0824 | [−0.0707, +0.2251] |
| **inverse side** | **−0.1959** | **[−0.3417, −0.0437]** |

The inverse **loses significantly**; the model's side does not. So **the model genuinely picks the
better side** — which is what `PRISM_LIVE_SIDE_GRADING_CORRECTION_V1` already established. But the
two sides sum to roughly **−0.11**, which is just the vig. **Directional skill is real. It is not
large enough to clear the price.** G4's gate required non-overlapping intervals and that fails;
the substantive reading is the sentence above, not that the model is bad.

**T5 note:** every `pass_yards` row priced in a single band (−120 to −110), so price bands are
uninformative here. No rule could have been derived even if K5 permitted it.

## 5. The conclusion the preregistration required

§1 stated: *"If not, the honest answer is that the model has no currently exploitable edge
anywhere, and that must be said plainly."*

**There is no currently exploitable edge.**

- the full board is **−0.0211/$1** with a row-level CI excluding zero — and clustering would only
  widen that too;
- `pass_yards`, the one apparent exception, **fails all four durability gates**;
- every other market is flat or negative, and `rush_rec_yards` loses **−0.1665**;
- model confidence and `edge_pct` were already closed as selection keys (V21 §3).

**What survives is narrower and still worth something:** the model has **measurable directional
skill** — it beats its own inverse, and it cuts the house edge from about −4.5% to −2%. **It is
roughly half way to profitability and the remaining half is not currently bridged by anything we
have.**

## 6. What this changes about the roadmap

**The projection fixes remain worth shipping — but for accuracy, not for money.** `D1+D3` makes
receiving projections measurably better (+1.015 yards per player, bottom tier −55.8%). **Nothing
in this study suggests that converts to profit**, and the rec_yards synthesis result says the
noise ceiling there blocks it.

**Bet selection cannot be solved right now, and that is not a selection problem.** You cannot
select a profitable subset out of a population that has no profitable subset we can identify
pregame. Every pregame key tested — confidence, `edge_pct`, disagreement size, market cell, price
— either fails or fails to survive clustering.

**`rush_rec_yards` remains the most interesting target** for the same reason as before: low noise
(dispersion 0.548) and a bad model (−0.1665, worst projection cell). **But its −0.1665 was also
row-bootstrapped**, and on this evidence the honest expectation is that a cluster bootstrap widens
it too. **Fixing the projection there is justified by the projection error, not by a promised EV
gain.**

## 7. Prediction scored

- **"G1 fails or passes marginally; I expect the CI to span zero" — RIGHT.**
- **"G3 is the informative failure — a handful of quarterbacks carry it" — RIGHT**, 77% from three.
- **Named risk — "I may want this to be true … if G1 passes and G2/G3 fail, the answer is NO"** —
  handled as preregistered. The clustering correction was written into the design *before* seeing
  the number, which is the only reason this was caught at all.

**2 for 2 on this study.** The useful move was preregistering the methodological fix rather than
the hoped-for outcome.

## 8. A standing correction for this repository

**Every EV and win-rate figure in this programme computed by bootstrapping over rows is
overstated in precision**, wherever rows cluster on a player. That includes
`PRISM_EDGE_DECOMPOSITION_EV_V1` throughout and the per-market table in particular. **Future
betting-side measurement must cluster on the natural unit — player for props, team-week for team
markets — and any figure that cannot survive clustering must not be reported as an edge.**

No odds were fetched. No bet is authorized by this study. No production code modified.
