# PRISM — Every player's floor and ceiling comes from one constant (V1)

> ⚠️ **CORRECTED — read `PRISM_CORRECTION_PRODUCTION_IS_NOT_CONSTANT_SIGMA_V1.md` first.**
> This document's `Q0` baseline assumes production prices from a constant σ through a
> normal CDF. **It does not.** The live path runs `run_pricing_v2.py`, which prices from
> the empirical Monte Carlo distribution and reports a per-player `model_sd`;
> `pricing_v2.SIGMA` is never referenced on the production path. Every comparison below
> is therefore against a textbook baseline, **not** against the deployed system, and its
> gains must not be quoted as gains over production. The measurements of how floors and
> ceilings scale in the data remain valid.

> **STATUS: diagnosis is code-verified and trait measurements are descriptive.**
> No mechanism is promoted. A sigma change requires its own preregistered validation.

## 1. The defect, in code

`scripts/pricing_v2.py:244`

```python
sigma = SIGMA[market]
p_over = _prob_over(mu, sigma, line)     # normal CDF
```

with `SIGMA` a per-market dict at `pricing_v2.py:36-43`: `rec_yards` **26.0**,
`rush_yards` 24.0, `pass_yards` 48.0, `receptions` 1.8, `rush_att` 3.2.

So **every receiver in the league is priced with the same distribution width**, and the
distribution is symmetric, so each player's floor and ceiling are forced to be
equidistant from his mean. Dispersion is keyed on *market*, not on player, role,
position, or opponent. Nothing player-specific enters it.

This is the same defect class as the positional gates, one layer further down: a
constant where a player property belongs.

## 2. What it costs — same mean, different player

121 receivers, 2025, ≥10 games and ≥40 targets. Realized per-player SD spans
**10.0 to 53.8** (median 27.1). **29 players are more than 30% under-dispersed** by
production; 8 are more than 30% over-dispersed.

Within a single narrow mean band (45–60 yards, 32 players), realized SD spans
**19.0 to 51.5** and production assigns 26.0 to every one:

| player | mean | realized SD | production SD | actual floor (p20) | actual ceiling (p80) |
|---|---|---|---|---|---|
| Terry McLaurin | 58.2 | **19.0** | 26.0 | 46.6 | 70.0 |
| Jakobi Meyers | 52.2 | 21.3 | 26.0 | 39.0 | 68.0 |
| Stefon Diggs | 59.6 | 42.7 | 26.0 | 26.0 | 101.0 |
| Michael Wilson | 59.2 | **51.5** | 26.0 | 15.2 | 97.0 |

McLaurin and Wilson differ by **1.0 yard of mean** and **2.7× in realized SD**.
Production prices them identically, and assigns both a floor of ~37. McLaurin's actual
floor is 46.6; Wilson's is 15.2 — a **31-yard** difference production cannot express.

## 3. The bias is systematic, not random

| mean tercile | realized SD | production SD | floor bias | ceiling bias |
|---|---|---|---|---|
| low mean | 21.7 | 26.0 | **−4.3** | **+7.8** |
| mid | 28.3 | 26.0 | +0.6 | +2.4 |
| high mean | 35.9 | 26.0 | **+6.1** | **−4.7** |

Production **over-disperses role players** and **compresses the stars**: it claims a
high-mean receiver has a higher floor (+6.1) and a lower ceiling (−4.7) than he has.
Mean absolute per-player error is 5.7 yards on the floor and 7.7 on the ceiling.

For over/under pricing this is the most costly possible place to be wrong, because the
floor and ceiling *are* the bet.

## 4. Centrality buys a floor — the player property that drives shape

| centrality tercile | target share | mean | floor as × of mean | ceiling as × of mean | rel. volatility |
|---|---|---|---|---|---|
| low | 0.145 | 27.6 | **0.295×** | 1.564× | 0.882 |
| mid | 0.208 | 40.6 | 0.432× | 1.505× | 0.697 |
| high | 0.302 | 61.5 | **0.554×** | 1.411× | 0.569 |

`corr(centrality, floor_ratio) = +0.620`;
`corr(centrality, relative volatility) = −0.656`.

The two archetypes the user named, measured:

| | share | mean | floor | ceiling |
|---|---|---|---|---|
| Drake London | 0.392 | 76.6 | 43.4 (**0.57×**) | 116.4 (1.52×) |
| Josh Downs | 0.210 | 35.4 | 13.0 (**0.37×**) | 54.0 (1.53×) |

Nearly identical ceiling *ratios* (1.52 vs 1.53), radically different floors. A player
central to his offense has protected downside; a complementary player does not. That is
distribution **shape** differing by role, which no choice of a single constant can
represent.

## 5. How much of this is reliably estimable — the honest limit

Disjoint-half (weeks 1–9 vs 10–18) and cross-season persistence:

| test | r(SD) | r(CV) | r(MEAN) for scale |
|---|---|---|---|
| 2023 halves | +0.566 | +0.314 | +0.736 |
| 2024 halves | +0.152 | +0.276 | +0.592 |
| 2025 halves | +0.238 | +0.028 | +0.629 |
| 2023 → 2024 | +0.213 | +0.188 | +0.625 |
| 2024 → 2025 | +0.310 | +0.428 | +0.692 |

Volatility is a **real but weak** trait: positive in 5 of 5 tests, so not noise, but far
weaker than the mean (~0.25 vs ~0.65) and unstable between years (0.566 in 2023,
0.152 in 2024).

**Therefore a player's own measured SD must not be used raw.** At r ≈ 0.25 only about a
quarter of the observed spread is signal, so Michael Wilson's 51.5 is mostly
estimation noise on 15 games.

**What is reliable:** SD rises monotonically with the mean (21.7 / 28.3 / 35.9), and the
mean is strongly persistent (r ≈ 0.6–0.74). So the dependable player-specific dispersion
is a function of **projected mean and role centrality**, not of the player's own noisy
realized volatility. That is still fully player-specific — London and Downs get
different distributions — but it is estimated from the stable part of their identity.

Note the relationship is **sublinear**, not proportional: CV falls as the mean rises
(0.79 / 0.70 / 0.58 by tercile), so `sigma = k × mu` is wrong; the form needs an
intercept.

## 6. What this does and does not establish

- **Established, code-verified:** dispersion is a per-market constant; nothing
  player-specific enters it; the resulting floor/ceiling bias is systematic by mean
  tercile.
- **Established, descriptive:** centrality predicts floor ratio (+0.620) and relative
  volatility (−0.656); per-player volatility is weakly persistent across 5 of 5 tests.
- **NOT established:** any specific replacement form. 2023–2025 have now been inspected
  for these trait measurements, so a proposed `sigma(mu, centrality)` must be
  preregistered and validated with its own held-back data, player-level scoring, and
  distributional calibration (coverage of the 20th/80th percentile per player), not
  averaged MAE.
- **NOT established:** that environment sensitivity varies predictably by player. The
  per-player slopes spread widely (−42.9 to +34.5) but are fit on 10–17 games each, and
  the trait correlations are weak (+0.231 centrality, −0.094 volatility).

## 7. Why averaged MAE could never have found this

Mean absolute error on the point projection is insensitive to dispersion: a model can
have every player's mean right and every player's floor and ceiling wrong, and score
identically to one that gets both right. Every metric used earlier in this line of work
averaged across players, which is why a constant sigma survived unexamined. The correct
evaluation for a player-level distribution is per-player quantile calibration, reported
as a distribution across players rather than as a mean.

## 8. Scope

No production code changed. No promoted model, weight, or pricing path modified. No
sportsbook variable read. Betting remains CLOSED. The `PERCENTY_COLS` scale bug and
SLOT unreachability remain separate open items.
