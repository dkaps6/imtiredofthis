# PRISM — CORRECTION: production does NOT price from a constant σ or a normal

> **This retracts a central claim repeated across several documents, commits, handoff
> checkpoints V14/V15, and three GitHub #535 comments.** Read this before citing any of
> them.

## What I claimed

That `scripts/pricing_v2.py:244` — `sigma = SIGMA[market]`, `rec_yards: 26.0` — meant
**every receiver in the league is priced with the same distribution width**, through a
normal CDF that forces each player's floor and ceiling equidistant from his mean. I
illustrated it with McLaurin and Wilson being "priced identically" despite a 2.7×
difference in realised dispersion.

## What is actually true

The production path is:

```
.github/workflows/full-slate.yml:573
  -> run_pricing_with_full_roster_universe_v3.py
     -> v4_production / v5_production / v6_production
        -> run_pricing_with_full_roster_universe_v1.py
           -> import scripts.run_pricing_v2 as pricing
```

and `scripts/run_pricing_v2.py:400,410`:

```python
p_over   = float(np.mean(adjusted_outcomes > line))
model_sd = float(np.std(adjusted_outcomes, ddof=1)) if len(adjusted_outcomes) > 1 else 0.0
```

Production prices from the **empirical Monte Carlo outcome distribution**. `model_sd` is
a **per-player reported diagnostic**, not an input.

`run_pricing_v2` imports only `_fair_market_prob` and `_fair_odds` from `pricing_v2` and
**never references `SIGMA`**. The only importers of `scripts.pricing_v2` are
`run_pricing_v2` itself and my own research script. **The constant never reaches
production.**

The live Week 7 board confirms it — `model_sd` is not constant in any market:

| market | distinct `model_sd` values | observed range | constant I claimed |
|---|---|---|---|
| `rec_yards` | **139** | 5.64 – 33.79 | 26.0 |
| `receptions` | 139 | 0.88 – 2.80 | 1.8 |
| `rush_yards` | 71 | 3.23 – 29.76 | 24.0 |
| `pass_yards` | 27 | 50.72 – 68.30 | 48.0 |

So I was wrong on **both** counts: dispersion **is** player-specific, and the
distribution is **not** forced symmetric — the simulated distribution can take any
shape.

## This is the same error as the rate-scale guard, repeated

In `PRISM_RATE_SCALE_PATH_ATTRIBUTION_V1` I found a defect in `team_context_v3.py` that
turned out not to be on the path that produces player target share. I documented the
lesson and then **made the identical mistake one layer over**, locating a constant in
`pricing_v2.py` without first confirming the workflow calls it. The check that would
have caught it in one step — grep the workflow for the entry point it actually runs —
is the same check that resolved the first case.

## What survives, and what does not

**Survives — the data findings.** These were measured from outcomes and say nothing
about production code:

- a player's floor rises **1.7–2.4× more slowly** than his ceiling, in all three markets
  (walk-forward, 2016–2025);
- role centrality predicts floor-as-a-share-of-mean at `r = +0.620`, and loads more on
  the ceiling than the floor;
- per-player volatility is weakly persistent (`r ≈ 0.25`, positive in 5/5 tests);
- the three opponent-side nulls (`PRISM_OPPONENT_SIGNAL_NULLS_V1`) are untouched.

**Does NOT survive — every comparison against "production".** All four quantile studies
used `Q0` = constant σ + normal as the production baseline. **That is a textbook
baseline, not ours.** So:

- "well-calibrated players 42.2% → 69.9%" is an improvement over a **straw man**, not
  over production;
- the McLaurin/Wilson "priced identically" claim is **false** for live boards;
- `PRISM_PLAYER_DISPERSION_IS_A_CONSTANT_V1` is wrong in its title and premise;
- the walk-forward "H1 SUPPORTED in all three markets" does **not** establish any
  improvement over the deployed system.

**The direction of the error, however, was right** — and the live backtest now
establishes it properly, against actual outcomes rather than against an assumed form.

## What the live data says instead

`scripts/research/backtest_2026_live_calibration_v1.py`, 2026 weeks 1, 2, 6, 7 — 1,571
deduped player-market-weeks.

### Production's probabilities are severely overconfident

Calibration of `fair_prob` (its real MC-derived output) against realised outcomes:

| predicted bucket | n | mean predicted | actual over-rate | gap |
|---|---|---|---|---|
| 0.00–0.10 | 76 | 0.061 | **0.447** | **−0.387** |
| 0.10–0.20 | 140 | 0.156 | 0.486 | −0.330 |
| 0.20–0.30 | 247 | 0.253 | 0.486 | −0.232 |
| 0.30–0.40 | 308 | 0.353 | 0.455 | −0.102 |
| 0.40–0.50 | 336 | 0.447 | 0.473 | −0.026 |
| 0.50–0.60 | 235 | 0.549 | 0.481 | +0.068 |
| 0.60–0.70 | 136 | 0.642 | 0.574 | +0.069 |
| 0.70–0.80 | 60 | 0.742 | 0.600 | +0.142 |
| 0.80–0.90 | 26 | 0.844 | **0.538** | **+0.305** |

When production says **6%**, the outcome goes over **45%** of the time. When it says
**84%**, it goes over **54%**. The realised rate is nearly flat (0.45–0.60) across the
entire predicted range, so the probabilities are compressed-signal and far too extreme —
**the simulated distributions are much too narrow.**

There *is* some ordering (0.447 at the bottom vs 0.600 near the top), so the model is not
noise — but the usable spread is ~0.15 wide while it reports ~0.78.

Brier: **model 0.2800** vs **market 0.2491** (+0.0309 worse). Overall predicted over-rate
0.406 against an actual 0.488, so it is also **biased toward unders**.

**This vindicates the direction and refutes the mechanism.** Distributions too narrow —
yes. Because of a constant in a dict — no. Because the Monte Carlo is too tight.

### Mean accuracy versus the market

| market | n | model MAE | vegas MAE | diff | model closer |
|---|---|---|---|---|---|
| `rec_yards` | 530 | 21.77 | 21.13 | +0.64 | 45.7% |
| `receptions` | 526 | 1.64 | 1.58 | +0.06 | 49.6% |
| `rush_yards` | 284 | 19.03 | 17.87 | +1.16 | 47.5% |
| `rush_rec_yards` | 120 | 31.86 | 26.99 | **+4.87** | 40.8% |
| **`pass_yards`** | 111 | **53.28** | 55.18 | **−1.91** | **54.1%** |
| overall | 1571 | 17.53 | 16.85 | +0.68 | 47.5% |

Production is modestly behind the market on means overall, **beats it on pass yards** —
the one market with a promoted synthesis (M89/M90) — and is worst on `rush_rec_yards`.

**And it is improving week over week:** MAE gap +1.24 → +0.66 → +0.57 → **+0.19**, with
model-closer rising 43.8% → 46.9% → 49.4% → **50.9%**.

## Operational gap found while doing this

Bet-level grading is **blocked for weeks 6 and 7**.
`data/market_track_record/PRODUCTION_DECISION_GATES_V1.json` preserves decision-gate
evidence for **weeks 1–2 only**, so the canonical grader fails closed on later weeks —
correctly, since grading a whole board without knowing which rows were blocked would
overstate the decision set. Weeks 3–5 are already permanently lost (AGENTS.md §5a).

**Every live run should preserve gate evidence at the same time it archives the board.**
Otherwise the paid boards survive but remain ungradable as a betting record.

## Corrected next step

The dispersion lane's target changes. It is no longer "replace a constant" — it is
**widen and calibrate the Monte Carlo distribution**, measured by `fair_prob` calibration
and Brier against the market, on live data. The per-tail quantile work remains useful as
a *description* of how floors and ceilings actually scale, but it is not a drop-in
replacement for anything, and its reported gains against `Q0` must not be quoted as
gains against production.

## Scope

No production code changed. No sportsbook variable entered any projection; lines appear
only as a comparison benchmark and as the graded boundary.
