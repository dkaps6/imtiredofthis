# PRISM — Player-specific dispersion: RESULTS (V1)

> ⚠️ **CORRECTED — read `PRISM_CORRECTION_PRODUCTION_IS_NOT_CONSTANT_SIGMA_V1.md` first.**
> This document's `Q0` baseline assumes production prices from a constant σ through a
> normal CDF. **It does not.** The live path runs `run_pricing_v2.py`, which prices from
> the empirical Monte Carlo distribution and reports a per-player `model_sd`;
> `pricing_v2.SIGMA` is never referenced on the production path. Every comparison below
> is therefore against a textbook baseline, **not** against the deployed system, and its
> gains must not be quoted as gains over production. The measurements of how floors and
> ceilings scale in the data remain valid.

**Preregistration:** `PRISM_PLAYER_DISPERSION_PREREGISTRATION_V1.md`, frozen at
`cd8117d0` **before** scoring.
**Script:** `scripts/research/measure_player_dispersion_v1.py`
**Record:** `data/research/player_dispersion/player_dispersion_v1.json`
**Per-player rows:** `data/research/player_dispersion/player_dispersion_v1_players.csv`
**Design:** fit 2020–2021 (2,880 rows), score **2022** — forward in time and never
inspected. 1,420 holdout rows, 96 players with ≥8 games.

## Verdict: H1 NOT SUPPORTED — but the constant is now *proven* wrong

The frozen four-part rule fails on rule 2 for both arms. Reported as the result; no
arm re-specified, no subgroup searched, no player carved out.

**What the study did establish, and this is the part that matters:**

> Production's intervals contain **50.0%** of outcomes when they are supposed to
> contain **60%**. They are systematically too narrow, worst for the best players.

That is no longer an inference from realized SDs — it is measured per player on naive
holdout data.

## The numbers

| arm | well-calibrated players | coverage median | sigma median | sigma range |
|---|---|---|---|---|
| **S0** (production, σ=26.0) | **41.7%** | **0.500** | 26.0 | 26.0–26.0 |
| **S1** (σ = a + b·μ) | **52.1%** | **0.608** | 32.5 | 20.1–56.4 |
| S2 (σ = a + b·μ + c·centrality) | 51.0% | 0.620 | 33.4 | 20.3–55.7 |

Fitted forms (×√(π/2) from the mean-absolute-deviation estimator):
`S1: σ = 11.488 + 0.2977·μ`;
`S2: σ = 10.596 + 0.2530·μ + 14.168·centrality`.

The intercept matters, as preregistered: `b ≈ 0.30` with an intercept near 11 is
sublinear, confirming `σ = k·μ` was correctly falsified in advance.

**Floor/ceiling bias by mean tercile** (want flat, especially high-mean floor):

| arm | low | mid | **high** |
|---|---|---|---|
| S0 | fb −6.3 / cb +5.7 | fb +5.2 / cb +5.9 | **fb +9.4** / cb −2.7 |
| S1 | fb −5.9 / cb +5.3 | fb +0.1 / cb +11.0 | **fb −3.9** / cb +10.5 |

Star compression is substantially repaired: the high-mean floor bias moves from
**+9.4 → −3.9**. Production really does claim its best players have a higher floor
than they do, and mean-scaling fixes most of that.

## Why it fails, precisely

| arm | players improved at p20 (floor) | players improved at p80 (ceiling) |
|---|---|---|
| S1 | **77.1%** | **31.2%** |
| S2 | 75.0% | 34.4% |

Rule 2 required a majority at **both** tails. Widening a **symmetric** normal fixes the
floor for three quarters of players and breaks the ceiling for two thirds. The
high-mean ceiling bias goes +(−2.7) → **+10.5**: it overshoots.

The empirical distribution puts the floor **further** from the mean than the ceiling
(for high-mean players, roughly 31 down vs 25 up). A normal centred on the mean forces
those distances equal, so no single σ can satisfy both tails at once. **The blocker is
the distribution family, not the σ function.**

## The preregistered secondary rules out the obvious fix

A lognormal matched to the same mean and σ (reported only, cannot change the verdict):

| arm | well-calibrated | coverage median | asymmetry (down vs up) |
|---|---|---|---|
| S0 | 33.3% | 0.417 | 19.0 vs 14.7 |
| S1 | 49.0% | 0.500 | 23.3 vs 17.6 |
| S2 | 51.0% | 0.500 | 23.4 vs 17.7 |

It produces asymmetry in the **right direction** but **narrows the central interval**,
dropping coverage back to 0.500 and scoring below the normal S1 (52.1%). Matching a
lognormal on (mean, σ) pushes variance into the long right tail and leaves the middle
too tight. So the natural skewed family is **not** the answer either.

## Where this leaves the lane

Established, on naive holdout data:

1. the per-market constant is **wrong and measurably so** — 50% coverage against a 60%
   target;
2. scaling σ by the player's own projected mean fixes most of it — well-calibrated
   players 41.7% → **52.1%**, coverage median 0.500 → **0.608**, star compression
   +9.4 → **−3.9**;
3. **role centrality adds nothing beyond the mean** (S2 ≈ S1; 51.0% vs 52.1%). The
   centrality→floor correlation of +0.620 is apparently carried by the mean, which
   centrality predicts. The simpler arm wins, as the preregistration specified;
4. a symmetric family **cannot** calibrate both tails, and a mean/σ-matched lognormal
   does not fix it.

Not established: any replacement that passes a two-tailed per-player calibration test.
**No production change is justified by this study.**

## Next step, requiring its own preregistration

Stop routing through a σ and a distributional family. **Fit the two quantiles
directly** — separate quantile (pinball) regressions for p20 and p80 as functions of
the player's projected mean, with the per-tail asymmetry free rather than implied by a
family. That addresses the exact failure above instead of working around it, and it
makes the floor and the ceiling independently answerable, which is what an over/under
actually needs.

That is a new frozen study, not an edit to this one.

## Caveats

- WR/TE receiving yards only; rushing and receptions untested.
- 96 players clear ≥8 holdout games. Sensitivity at ≥6 and ≥10 is recorded in the JSON.
- `MIN_HOLDOUT_GAMES = 8` was **not** fixed by the preregistration and is disclosed as
  an implementation choice, with the sensitivity reported rather than hidden.
- μ is untouched in every arm, so nothing here speaks to point-projection accuracy.

## Scope

No production code changed. No promoted model, weight or pricing path touched. No
sportsbook variable read. Betting remains CLOSED.
