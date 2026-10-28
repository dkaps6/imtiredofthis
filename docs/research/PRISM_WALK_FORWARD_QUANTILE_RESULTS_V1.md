# PRISM — Walk-forward per-tail quantiles, all three markets: RESULTS (V1)

> ⚠️ **CORRECTED — read `PRISM_CORRECTION_PRODUCTION_IS_NOT_CONSTANT_SIGMA_V1.md` first.**
> This document's `Q0` baseline assumes production prices from a constant σ through a
> normal CDF. **It does not.** The live path runs `run_pricing_v2.py`, which prices from
> the empirical Monte Carlo distribution and reports a per-player `model_sd`;
> `pricing_v2.SIGMA` is never referenced on the production path. Every comparison below
> is therefore against a textbook baseline, **not** against the deployed system, and its
> gains must not be quoted as gains over production. The measurements of how floors and
> ceilings scale in the data remain valid.

**Preregistration:** `PRISM_WALK_FORWARD_QUANTILE_PREREGISTRATION_V1.md`, frozen
`42097573` **before** scoring.
**Script:** `scripts/research/measure_walk_forward_quantiles_v1.py`
**Record:** `data/research/walk_forward/walk_forward_quantiles_v1.json`
**Predictions:** `data/research/walk_forward/walk_forward_predictions_{market}.csv`

Expanding-window walk-forward: coefficients for target season *S* fitted on seasons
`[2013, S−1]` only; within a season `mu_hat` still uses weeks 1…W−1. Every prediction is
out-of-sample in **both** dimensions. Target seasons **2016–2025**. 2026 excluded.

## Verdict: H1 SUPPORTED in ALL THREE markets, under BOTH rule-3 forms, era-stable

| market | Q0 well-calibrated | preferred arm | arm well-calibrated | p20 improved | p80 improved |
|---|---|---|---|---|---|
| `rec_yards` | **42.2%** | **Q2** | **69.9%** | 88.2% | 56.6% |
| `rush_yards` | **38.3%** | **Q1** | **64.8%** | 85.9% | 64.8% |
| `receptions` | **51.6%** | **Q1** | **69.2%** | 82.2% | 53.5% |

Coverage median moves to the 0.60 target in every market (0.556 → 0.604, 0.468 → 0.607,
0.568 → 0.600). Zero crossed quantiles anywhere. `STRICT` and `JOINT` rule-3 forms agree
in every market — the dual reporting was not needed to reach the verdict, which is the
cleanest possible outcome for it.

**Preferred arm by the frozen tiebreak** (simpler wins if it suffices, and must be
era-stable):

| market | pooled Q1 / Q2 | era-stable Q1 / Q2 | **chosen** |
|---|---|---|---|
| `rec_yards` | ✗ / ✓ | ✗ / ✓ | **Q2** |
| `rush_yards` | ✓ / ✓ | ✓ / ✓ | **Q1** |
| `receptions` | ✓ / ✓ | ✓ / **✗** | **Q1** |

Role centrality earns its place **only** for receiving yards. For receptions it passes
pooled but **reverses in 2023–2025**, so the preregistration's era clause selects Q1 —
exactly the case that clause was written for.

## The mechanism, now measured at scale in every market

Production's constant σ with a normal implies **both tails move 1:1 with the mean** and
role is irrelevant. Fitted tails (2025 target season):

| market | arm | floor slope | ceiling slope | **ratio** | centrality → floor | → ceiling |
|---|---|---|---|---|---|---|
| **production** | Q0 | **1.000** | **1.000** | **1.00** | 0 | 0 |
| `rec_yards` | Q2 | 0.294 | 0.696 | **2.37** | +55.50 | +98.87 |
| `rush_yards` | Q1 | 0.419 | 0.821 | **1.96** | — | — |
| `rush_yards` | Q2 | 0.336 | 0.701 | **2.08** | +13.18 | +19.07 |
| `receptions` | Q1 | 0.550 | 0.954 | **1.73** | — | — |
| `receptions` | Q2 | 0.412 | 0.764 | **1.85** | +3.00 | +4.28 |

**A player's floor rises 1.7–2.4× more slowly than his ceiling, in every market and every
arm.** Production assumes 1.00. And wherever centrality enters, it loads more heavily on
the ceiling than the floor. The structural claim is now established on 14,024 receiving
rows and 3,888 rushing rows, out-of-sample by construction, across ten seasons.

## Era stability — measured, not assumed

| market | 2016–2019 | 2020–2022 | 2023–2025 |
|---|---|---|---|
| `rec_yards` Q2 | 66% (✓) | 66% (✓) | 66% (✓) |
| `rush_yards` Q1 | 65% (✓) | 57% (✓) | 56% (✓) |
| `receptions` Q1 | 70% (✓) | 63% (✓) | 66% (✓) |

Q0 in the same windows never exceeds 56%. No market's chosen arm reverses in the most
recent era.

## The rushing reversal — the earlier failure was underpowered, not real

`PRISM_DIRECT_QUANTILE_RUSH_REC_RESULTS_V1` reported rushing as **not supported**, on
**51** players from **2014–2016**, failing rule 3 with the fitted floor landing 15 yards
below the empirical floor for high-volume backs.

Under walk-forward with **128** players scored across 3,888 rows, rushing passes **all
four rules under both rule-3 forms and in all three eras**, with the high-tercile floor
bias moving `+10.66 → −2.48` and the ceiling `−10.03 → −2.01`. Both tails improve.

That earlier result was an artefact of a sample that was simultaneously the smallest and
the least relevant available — decade-old football, before the shift to RB committees.
The diagnosis offered at the time ("inconclusive and underpowered, which is not the same
as evidence the constant is adequate") is confirmed.

**This is the direct consequence of abandoning the season-quarantine design.** Samples
grew 2.2–2.5× (415 / 415 / 128 players scored versus 184 / 191 / 51), and the receiving
result also *improved* on the better design — 59.8% under the three-season holdout
versus **69.9%** here.

## Honest weak points

- **p80 is the weakest link everywhere.** The ceiling majority is 56.6% / 64.8% / 53.5% —
  a majority, as required, but a thin one for receiving yards and receptions. The floor is
  repaired far more convincingly (82–88%).
- **Rushing remains the thinnest market** at 128 players scored; 2.5× the old sample and
  era-consistent, but still a third of the receiving pool.
- **`receptions` Q2 reverses in the current era**, which is why Q1 is chosen. The reversal
  itself is unexplained.
- `pass_yards` and `rush_rec_yards` remain untested and carry their own constants.
- The anomaly gate still requires snap counts, so 2013 is the earliest fit season.
  `load_rosters_weekly` reaches 2005 and would add ~8 seasons; deliberately not bundled.

## Integration remains blocked, and this result does not unblock it

A per-tail fit yields p20 and p80. `pricing_v2._prob_over` needs a probability for an
**arbitrary** sportsbook line. Turning two fitted quantiles into a full distribution
**without reintroducing the symmetric family that failed** is unsolved, and no amount of
further market replication addresses it. That is the next question, and it is a design
question rather than a research one.

## Scope

No production code changed. No promoted model, weight or pricing path touched. No
sportsbook variable read. Betting remains CLOSED.
