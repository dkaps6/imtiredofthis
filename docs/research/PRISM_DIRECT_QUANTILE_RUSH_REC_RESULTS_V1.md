# PRISM — Per-tail quantiles for rushing yards and receptions: RESULTS (V1)

> ⚠️ **CORRECTED — read `PRISM_CORRECTION_PRODUCTION_IS_NOT_CONSTANT_SIGMA_V1.md` first.**
> This document's `Q0` baseline assumes production prices from a constant σ through a
> normal CDF. **It does not.** The live path runs `run_pricing_v2.py`, which prices from
> the empirical Monte Carlo distribution and reports a per-player `model_sd`;
> `pricing_v2.SIGMA` is never referenced on the production path. Every comparison below
> is therefore against a textbook baseline, **not** against the deployed system, and its
> gains must not be quoted as gains over production. The measurements of how floors and
> ceilings scale in the data remain valid.

**Preregistration:** `PRISM_DIRECT_QUANTILE_RUSH_REC_PREREGISTRATION_V1.md`, frozen
`50b4c4f0` **before** scoring. Form, estimator, evaluation and the four rules were
imported from the validated receiving-yards study rather than reimplemented, so they
provably could not drift per market.
**Script:** `scripts/research/measure_direct_quantile_rush_rec_v1.py`
**Record:** `data/research/direct_quantile/direct_quantile_rush_rec_v1.json`
**Design:** fit 2020–2022, score **2014–2016 — never inspected for any market.**

## Verdict: H1 NOT SUPPORTED for either market

| market | rule 1 | rule 2 | **rule 3** | rule 4 | verdict |
|---|---|---|---|---|---|
| `rush_yards` | PASS | PASS | **FAIL** | PASS | **not supported** |
| `receptions` | PASS | PASS | **FAIL** | PASS | **not supported** |

Both fail on rule 3 alone. Reported as the result; no arm re-specified, no rule moved,
no market dropped after seeing its outcome.

## What DID replicate — the mechanism, 3 for 3

This is the durable finding. Production's constant σ with a normal implies **both tails
move 1:1 with the mean** and role is irrelevant. Every market disagrees, in the same
direction and similar proportion:

| market | floor slope on μ | ceiling slope on μ | centrality → floor | centrality → ceiling |
|---|---|---|---|---|
| **production (implied)** | **1.000** | **1.000** | **0** | **0** |
| `rec_yards` *(validated)* | 0.335 | 0.698 | +43.3 | +107.3 |
| `rush_yards` | 0.361 | 0.633 | +10.76 | +26.49 |
| `receptions` | 0.450 | 0.816 | +2.17 | +3.69 |

In all three markets the **floor rises roughly half as fast as the ceiling**, and
**centrality loads 1.7–2.5× more on the ceiling than the floor**. The structural claim —
a player's downside is not a mirror of his upside, and role is an upside property —
holds for yards, carries-derived yards, and catch counts alike.

## Rules 1 and 2 also pass in both markets

| market | well-calibrated players | coverage median (target 0.60) | p20 improved | p80 improved |
|---|---|---|---|---|
| `rush_yards` Q0 → Q1 | **29.4% → 60.8%** | **0.467 → 0.636** | 78.4% | **68.6%** |
| `receptions` Q0 → Q2 | **50.8% → 64.9%** | **0.529 → 0.606** | 80.6% | **57.6%** |

So the core defect — intervals too narrow, in both directions of the mean — replicates,
and per-tail fitting repairs coverage in both markets with a majority improving at
**both** tails. Zero crossed quantiles anywhere.

## Why rule 3 failed, per market — and what that says about the rule

### `receptions`: the ceiling criterion is near-unsatisfiable by construction

| | |floor bias| high tercile | |ceiling bias| high tercile |
|---|---|---|
| Q0 → Q2 on holdout | 0.59 → **0.32** (better) | 0.06 → **0.15** (worse) |

The floor improves materially. The ceiling "worsens" by **0.09 receptions**, because
production's high-mean ceiling bias was already **0.06** — essentially zero. Rule 3 as
frozen requires `|ceiling_bias|` not to increase, so when the baseline is already ~0 the
rule can only be held or failed, never won.

Across all seven windows the floor improves in **6 of 7** (0.59→0.32, 0.62→0.33,
0.85→0.54, 0.73→0.55, 0.76→0.52, 0.65→0.42) while the ceiling baseline sits between
0.00 and 0.30. Rule 3 passes in exactly one window (2023, where the ceiling bias
happened to be 0.09 → 0.09).

**This is a flaw in my rule design, not evidence for the constant.** It is named here as
a limitation to fix in a *future* preregistration — a tolerance band, or a joint
criterion — and explicitly **not** used to overturn this verdict. Re-scoring with a
relaxed rule after seeing the result is the post-hoc rescue this whole programme exists
to avoid.

### `rush_yards`: a genuine overshoot, on a badly underpowered sample

| | |floor bias| high tercile | |ceiling bias| high tercile |
|---|---|---|
| Q0 → Q1 on holdout | 4.03 → **15.03** (worse) | 6.90 → **1.84** (better) |

Here the failure is substantive, not a technicality: the fitted floor lands **15 yards
below** the empirical floor for high-volume backs, where production was only 4 yards
above it. The ceiling improves markedly, but the floor overshoots.

It is also **unstable across windows**, passing rule 3 in 3 of 6 reported seasons
(2017 PASS, 2018 fail, 2019 PASS, 2023 fail, 2024 fail, 2025 PASS) with high-tercile
`|fb|` moving 14.75→1.77, 8.16→10.30, 16.55→7.32, 4.87→9.44, 11.72→11.80, 13.44→6.63.
That is what an underpowered estimate looks like.

**Power is the real problem.** Only **51 players** clear the gate on the 2014–2016
holdout, and 21–34 per reported season, versus 184 for receiving yards. A per-tercile
bias estimate on ~17 players per tercile is not a sound basis for a verdict either way.
Rushing should be treated as **inconclusive and underpowered**, not as evidence that the
constant is adequate.

## Disposition

| market | status |
|---|---|
| `rec_yards` | **VALIDATED** (`PRISM_DIRECT_QUANTILE_RESULTS_V1`, arm Q2) |
| `rush_yards` | **inconclusive — underpowered** (n=51); rules 1, 2, 4 pass, rule 3 unstable |
| `receptions` | **not supported**; floor improves in 6/7 windows, ceiling criterion near-unsatisfiable |
| `pass_yards`, `rush_rec_yards` | untested |

**No production change is authorised.** Integration remains blocked on the open design
question regardless of these results: a per-tail fit yields p20/p80, but
`pricing_v2._prob_over` needs a probability for an **arbitrary** line, and turning two
quantiles into a full distribution without reintroducing a symmetric family is unsolved.

## What I am deliberately NOT doing

- **not** re-specifying rule 3 and re-scoring. If the rule needs a tolerance band, that
  is a new frozen study with its own justification;
- **not** claiming rushing supports the constant — it is underpowered, which is a
  different statement;
- **not** pooling markets to manufacture power, since the preregistration judges them
  separately;
- **not** treating the 3-for-3 slope replication as a pass. It is strong evidence about
  the *mechanism* and no evidence at all about the *decision rule*.

## Data runway, stated plainly

Holdout seasons are nearly exhausted: 2014–2016 (this study), 2017–2019 (receiving
yards), 2020–2022 (fit), 2023–2025 (contaminated). Snap counts — required by the anomaly
gate — begin in 2012, leaving only **2012–2013** genuinely naive. Any further
preregistration in this lane should either use that remaining window deliberately or
adopt a cross-validated design, and should say which in advance.

## Scope

No production code changed. No promoted model, weight or pricing path touched. No
sportsbook variable read. Betting remains CLOSED.
