# PRISM — Direct per-tail quantile fit: RESULTS (V1)

> ⚠️ **CORRECTED — read `PRISM_CORRECTION_PRODUCTION_IS_NOT_CONSTANT_SIGMA_V1.md` first.**
> This document's `Q0` baseline assumes production prices from a constant σ through a
> normal CDF. **It does not.** The live path runs `run_pricing_v2.py`, which prices from
> the empirical Monte Carlo distribution and reports a per-player `model_sd`;
> `pricing_v2.SIGMA` is never referenced on the production path. Every comparison below
> is therefore against a textbook baseline, **not** against the deployed system, and its
> gains must not be quoted as gains over production. The measurements of how floors and
> ceilings scale in the data remain valid.

**Preregistration:** `PRISM_DIRECT_QUANTILE_PREREGISTRATION_V1.md`, frozen at
`abe11db5` **before** scoring.
**Script:** `scripts/research/measure_direct_quantile_fit_v1.py`
**Record:** `data/research/direct_quantile/direct_quantile_fit_v1.json`
**Per-player rows:** `data/research/direct_quantile/direct_quantile_fit_v1_players.csv`
**Design:** fit 2020–2022 (4,245 rows), score **2017–2019** — never inspected —
3,992 rows, **184 players** with ≥8 games.

## Verdict: H1 SUPPORTED — arm Q2

All four frozen rules pass on naive data. This is the first validated improvement in
this line of work.

| rule | Q1 (mean only) | **Q2 (mean + role)** |
|---|---|---|
| 1. more well-calibrated players | PASS | **PASS** |
| 2. pinball majority at **both** tails | **FAIL** | **PASS** |
| 3. high-mean floor better, ceiling not worse | PASS | **PASS** |
| 4. no crossed quantiles | PASS | **PASS** |
| | not supported | **SUPPORTED** |

## The numbers

| arm | well-calibrated players | coverage median | median interval width |
|---|---|---|---|
| **Q0** (production, σ=26 normal) | **38.6%** | **0.520** | 43.8 |
| Q1 (per-tail, mean only) | 63.6% | 0.603 | 49.4 |
| **Q2** (per-tail, mean + centrality) | **59.8%** | **0.597** | 49.6 |

Players improved versus production:

| arm | p20 (floor) | p80 (ceiling) |
|---|---|---|
| Q1 | 84.8% | **45.7%** ← fails the majority |
| **Q2** | **83.7%** | **59.8%** |

**Floor/ceiling bias by mean tercile:**

| arm | low | mid | **high** |
|---|---|---|---|
| Q0 | fb −5.7 / cb +3.4 | fb +3.1 / cb +1.2 | **fb +11.7** / cb −3.1 |
| **Q2** | fb −1.2 / cb +4.0 | fb −0.7 / cb +3.3 | **fb −4.7** / cb **+0.2** |

Star compression is largely repaired — high-mean floor bias **+11.7 → −4.7** — and
unlike the symmetric arm it does **not** pay for that at the ceiling (**−3.1 → +0.2**).

## Why it works — the thing production gets structurally wrong

Production's `sigma = 26.0` with a normal gives `p20 = mu − 21.88` and
`p80 = mu + 21.88`, so **both tails move 1:1 with the mean** and role is irrelevant.
The fitted tails disagree on both counts:

| | floor slope on μ | ceiling slope on μ | centrality on floor | centrality on ceiling |
|---|---|---|---|---|
| production (implied) | **1.000** | **1.000** | 0 | 0 |
| Q1 fitted | 0.472 | 1.039 | — | — |
| **Q2 fitted** | **0.335** | **0.698** | **+43.3** | **+107.3** |

Two findings, both invisible to a single σ:

1. **A receiver's floor rises far more slowly than his average.** Being better raises
   your ceiling roughly twice as fast as your floor. Production assumes they move
   together, which is exactly why it overstates the floor of its best players by
   ~12 yards.
2. **Role centrality drives the ceiling about 2.5× more than the floor** (+107.3 vs
   +43.3). Being central to an offence is mostly an *upside* property.

That second point also explains a result in the prior study. There, centrality added
nothing (S2 ≈ S1) — because it entered through a **single σ**, which cannot lift one
tail more than the other. Once the tails are fitted independently, centrality's effect
is revealed to be tail-specific. The earlier null was an artefact of the functional
form, not evidence against role.

## Illustration (not evidence — two named players)

| | mean | centrality | production floor | **Q2 floor** | actual floor |
|---|---|---|---|---|---|
| Drake London | 76.6 | 0.392 | 54.7 | **36.3** | 43.4 |
| Josh Downs | 35.4 | 0.210 | 13.5 | **14.6** | 13.0 |

Production is 11 yards too high on London's floor and roughly right on Downs's. Q2 is
7 low on London and 1.6 high on Downs. The constant happens to work for mid/low-volume
players and fails on the stars, which is precisely what the bias table says.

## Robustness

**Sensitivity to the one choice left unfrozen last time** (minimum holdout games — frozen
at 8 this study, reported at 6 and 10):

| min games | Q0 | Q1 | Q2 |
|---|---|---|---|
| ≥6 | 37.3% | 61.3% | 57.6% |
| ≥8 | 38.6% | 63.6% | 59.8% |
| ≥10 | 39.3% | 64.4% | 61.3% |

**Era stability** (2023–2025, contaminated, REPORTED ONLY — cannot change the verdict):

| season | well-calibrated Q0 → Q2 | coverage median Q0 → Q2 | p20 improved | p80 improved |
|---|---|---|---|---|
| 2023 | 40.0% → **61.0%** | 0.500 → 0.608 | 74.0% | **65.0%** |
| 2024 | 50.5% → **53.4%** | 0.500 → 0.615 | 72.8% | **61.2%** |
| 2025 | 44.2% → **53.7%** | 0.545 → 0.600 | 75.8% | **54.7%** |

Q2 beats production in all three, and — the property that matters — the **p80 majority
holds in every one**. Zero crossed quantiles anywhere.

## Transparency: the headline metric alone would have picked the wrong arm

Q1 scores **higher** than Q2 on the headline (63.6% vs 59.8%) at every threshold. It is
nevertheless **not supported**, because it improves the ceiling for only 45.7% of
players — it buys floor accuracy at the ceiling's expense, which is the exact failure
mode of the previous study. Rule 2 was frozen in advance to prevent that, and it did.

Anyone reading only the coverage column would choose Q1 and repeat the earlier mistake.

## What this does NOT establish

- **WR/TE receiving yards only.** Rushing, receptions and pass yards are untested, and
  each has its own constant in `SIGMA`.
- **Calibration, not accuracy.** `mu` is untouched in every arm; nothing here improves
  a point projection.
- The holdout is **earlier** than the fit, disclosed and justified in the
  preregistration: these were the only naive seasons left, and every arm predicts from
  a player's own within-season prior mean, so the target is a structural relationship
  rather than an era level. The 2023–2025 reported results corroborate era stability.
- **No production change is authorised by this study.** Integration into
  `pricing_v2.py` requires its own change, its own tests, and explicit user
  authorisation.

## Recommended next step

Propose replacing `_prob_over`'s constant-σ normal for `rec_yards` with the Q2 per-tail
quantiles, as a separate reviewed change: a per-tail interval needs a probability
model for arbitrary lines, not just p20/p80, so the integration question is how to turn
two fitted quantiles into a full distribution without reintroducing a symmetric family.
That is a design question, not a research one, and it should be posed before any code
moves.

Then repeat this study for `rush_yards` and `receptions`, which have the same defect
and have not been tested.

## Scope

No production code changed. No promoted model, weight or pricing path touched. No
sportsbook variable read. Betting remains CLOSED.
