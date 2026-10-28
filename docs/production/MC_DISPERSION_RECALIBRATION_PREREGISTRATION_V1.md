# MC dispersion recalibration — PREREGISTRATION (V1)

**Frozen before any scoring.** Production change proposal; no code merged until the
decision rule below passes and the user authorises.

## 0. The measured defect

`run_pricing_v2.py` prices from the empirical Monte Carlo distribution
(`p_over = np.mean(adjusted_outcomes > line)`). On live 2026 boards those distributions
are **too narrow**, which makes the probabilities severely overconfident: `fair_prob`
0.061 → actual 0.447, and 0.844 → actual 0.538, with Brier **0.2800 vs the market's
0.2491**.

The deficit is multiplicative and market-specific. `sd(resid / model_sd)` should be
**1.0** for a calibrated predictive distribution; measured on 2026 weeks 1/2/6/7:

| market | n | sim sd | realized resid sd | ratio |
|---|---|---|---|---|
| `receptions` | 526 | 1.72 | 2.08 | 1.20 |
| `pass_yards` | 111 | 55.52 | 69.71 | 1.26 |
| `rec_yards` | 530 | 17.65 | 29.56 | 1.68 |
| `rush_rec_yards` | 120 | 21.10 | 37.69 | 1.79 |
| `rush_yards` | 284 | 14.42 | 26.56 | 1.84 |

**Structural cause, stated but NOT claimed as proven:** `simulation_v2.simulate()` draws
outcome noise conditional on **point-estimate** parameters — `ypt`, `ypc`, `catch_rate`
and target share are fixed per player, with no sampling of their own estimation
uncertainty. That omission is a known source of under-dispersion. The evidence here
establishes the *magnitude*, not the mechanism: the one test attempted used projection
terciles as an evidence proxy, which does not separate the cause.

## 1. The change

A **mean-preserving** dispersion scale applied where `adjusted_outcomes` is finalised in
`run_pricing_v2.py` — i.e. **after** every promoted synthesis has set the mean, so the
mean is untouched:

```
out = mu + k_market * (out - mu)        # mu = out.mean() before scaling
out = clip(out, 0, None)                # markets are non-negative
```

Receptions continue through the existing `discrete_count_alignment_v1` so integer
alignment is preserved.

Chosen over restructuring the simulation because the measured deficit is a clean
multiplicative factor, roughly flat across projection terciles, and because touching the
draw mechanics risks the promoted QB/RB mean invariants. A structural
parameter-uncertainty fix remains the better long-term answer and is **not** foreclosed.

## 2. Where `k` comes from — derived, not tuned

`k_market = sd(resid / model_sd)` computed on **2026 weeks 1, 2 and 6 ONLY**.
**Week 7 is never used to derive `k`.** One scalar per market, no per-player or
per-position variation, no search.

## 3. Validation — on real draws, on data `k` never saw

Week 7 is re-simulated from the archived fixture
(`week7_trace__data__model_rule_simulation_inputs.csv`, 1,447 rows), which
`simulation_v2.simulate()` runs on, producing genuine draws for all six markets. Then:

1. recompute `p_over` from baseline draws against the archived Week 7 vegas lines;
2. recompute `p_over` from **widened** draws, same lines, same seed;
3. grade both against actual Week 7 outcomes via the canonical GSIS identity resolver.

## 4. Decision rule — frozen, ALL must hold on Week 7

1. **Brier improves** versus baseline;
2. **calibration gaps shrink** — the mean absolute gap between predicted bucket and
   realised over-rate decreases;
3. **`sd(z)` moves toward 1.0** and does not overshoot past 1.0 in the other direction;
4. **the mean is preserved** — per-row `|mean_after − mean_before| / max(mean_before, 1)`
   ≤ **0.02**, and for `pass_yards` and Week-1 `rush_yards` the promoted-synthesis mean
   invariant (AGENTS.md §7) must hold within the same tolerance. Clipping at zero can
   shift a low-mean player's mean upward, so this is measured, not assumed.

If any fails, the change is **not** supported and is not merged. No per-market tuning of
`k` after seeing Week 7, no market dropped, no tolerance relaxed.

## 5. Scope

Research validation only until it passes and is authorised. No sportsbook variable
enters any projection; lines are the graded boundary and the comparison benchmark.
Betting remains CLOSED.
