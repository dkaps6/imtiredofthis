# PRISM — Walk-forward per-tail quantiles, all three markets: PREREGISTRATION (V1)

> ⚠️ **CORRECTED — read `PRISM_CORRECTION_PRODUCTION_IS_NOT_CONSTANT_SIGMA_V1.md` first.**
> This document's `Q0` baseline assumes production prices from a constant σ through a
> normal CDF. **It does not.** The live path runs `run_pricing_v2.py`, which prices from
> the empirical Monte Carlo distribution and reports a per-player `model_sd`;
> `pricing_v2.SIGMA` is never referenced on the production path. Every comparison below
> is therefore against a textbook baseline, **not** against the deployed system, and its
> gains must not be quoted as gains over production. The measurements of how floors and
> ceilings scale in the data remain valid.

**Frozen before any scoring.**

## 0. Why this supersedes the three-season holdout design

The earlier studies quarantined whole seasons as "naive holdouts" and spent three per
study, then declared the runway exhausted. That was wrong on three counts:

1. **It cost ~13× the available power.** Validated on **51** RB players and **184**
   receivers, when 2013–2025 holds **652** RB and **2,325** WR/TE player-seasons
   (8,299 and 31,288 player-weeks).
2. **It conflated two different risks.** *Data leakage* (the model seeing its own
   future) and *researcher degrees of freedom* (the analyst tuning a form to results
   already seen) need different remedies. Preregistration fixes the second — and was
   already in place. Season quarantine is the remedy for the first, and it is
   unnecessary once evaluation is **out-of-sample by construction**. The quarantine was
   paying a large power cost for protection already held by other means.
3. **It drove the design toward older, less relevant football.** The rushing study's
   holdout (2014–2016) was simultaneously the smallest **and** the least relevant sample
   available — a decade removed, before the shift to RB committees and pass-catching
   backs. Its failure may be an era artefact and at n=51 that is indistinguishable from
   noise.

This study replaces the quarantine with rolling-origin walk-forward, which makes every
season usable, repeatedly, with no leakage.

## 1. Hypothesis

**H1 (per market, judged separately).** Fitting p20 and p80 directly, each as its own
function of the player's projected mean and role share, calibrates individual floors and
ceilings better than production's constant-σ normal — at **both** tails, under
walk-forward evaluation.

**H0.** The constant is adequate for that market.

## 2. Walk-forward structure, frozen

**Expanding window by season.** For each target season *S*, the per-tail coefficients are
fitted on **all seasons in [2013, S−1]** and used to predict **all of season *S***. Then
*S* advances. Coefficients never see season *S* or anything later.

Within a season, `mu_hat` and `centrality` continue to use **only weeks 1…W−1**, exactly
as before. So every prediction is out-of-sample in **both** dimensions — across seasons
and within them.

- **Earliest fit season: 2013** (the first year snap counts exist, which the anomaly gate
  requires).
- **Minimum prior seasons: 3** → **first target season 2016**.
- **Target seasons: 2016–2025** — ten seasons of walk-forward evaluation per market.
- **2026 is EXCLUDED.** It is partial and is the live production season; it gets its own
  separate backtest.
- Season length: week 17 before 2021, week 18 from 2021.
- There is **no holdout to spend** and no contaminated-season bookkeeping. Nothing is
  quarantined because nothing needs to be.

## 3. Markets, arms, eligibility — carried over unchanged

All three markets, each judged separately:

| market | production σ | positions | volume | min prior volume | min game volume |
|---|---|---|---|---|---|
| `rec_yards` | 26.0 | WR, TE | targets | 20 | 1 |
| `rush_yards` | 24.0 | RB | carries | 60 | 5 |
| `receptions` | 1.8 | WR, TE | targets | 20 | 1 |

Arms, estimator and `mu_hat` construction are **imported from the validated study**, not
reimplemented, so the form provably cannot drift: **Q0** production constant-σ normal,
**Q1** per-tail on `mu_hat`, **Q2** per-tail on `mu_hat` + `centrality`. Week ≥ 4,
`MIN_HOLDOUT_GAMES = 8` for per-player coverage.

Anomaly gate unchanged: `DNP_INACTIVE` excluded from prior and outcome, `PARTIAL_GAME`
excluded from outcomes, `UNRESOLVED_CONTEXT` fails closed.

## 4. Rule 3 is reported BOTH ways — stated in advance, nothing hidden

The previous study's rule 3 ("high-tercile `|floor_bias|` decreases AND `|ceiling_bias|`
does not increase") has a **design flaw**: when a baseline bias is already ≈0 it can only
be held or failed, never won. Receptions' high-tercile ceiling bias was **0.06**, so the
rule was effectively unsatisfiable there for reasons unrelated to model quality.

Both forms are therefore computed and **both reported**:

- **rule 3-strict** — the original, verbatim.
- **rule 3-joint** — high-tercile `|floor_bias| + |ceiling_bias|` must **decrease**, AND
  neither tail's `|bias|` may increase by more than **0.25 × that market's production σ**
  (6.5 yards for `rec_yards`, 6.0 for `rush_yards`, 0.45 receptions).

Why the joint form is principled rather than a rescue: the **sum must fall**, so it
cannot be satisfied by trading one tail for the other — the exact failure mode rule 3
existed to catch. The tolerance is expressed in units of each market's own σ so it means
the same thing across markets. And it does **not** excuse the substantive rushing
failure: `|floor_bias|` there went 4.03 → 15.03, an increase of **11 yards** against a
6-yard tolerance, so rushing still fails under the joint rule on that sample.

Two verdicts are published per market, `SUPPORTED_STRICT` and `SUPPORTED_JOINT`. Neither
is suppressed and neither is presented as the only answer.

## 5. Evaluation, frozen — per-player, never averaged

Averaged MAE remains **disallowed** as a primary metric. Headline is the fraction of
players whose predicted p20–p80 contains within **±0.10** of **0.60** of their games, with
the full distribution of per-player coverage, per-player pinball at p20 and p80, and
floor/ceiling bias by mean tercile.

**Additionally, and required:** all metrics are reported **by era** — 2016–2019,
2020–2022, 2023–2025 — so era stability is *measured* rather than assumed or caveated.
This is the question the old design could not answer, and it is why the rushing result
was uninterpretable.

## 6. Decision rule, frozen

**H1 is supported for a market only if, on the pooled 2016–2025 walk-forward predictions:**

1. the fraction of well-calibrated players **increases** versus Q0;
2. per-player pinball improves for a **majority** at **p20 AND p80**;
3. rule 3 passes — reported under **both** strict and joint forms;
4. `p20_hat < p80_hat` on **every** row.

If any fails, H1 is not supported for that market. No arm re-specified, no threshold
moved, no market dropped, no era excluded after seeing its result. Q1 vs Q2 by the same
rules; if Q1 suffices the simpler form wins.

A market that passes pooled but **reverses in the most recent era (2023–2025)** must be
reported as era-unstable and is **not** recommended for production regardless of the
pooled verdict.

## 7. What this does NOT do

- does not change production; integration stays blocked on the open design question
  (two fitted quantiles → a probability for an arbitrary line, without reintroducing a
  symmetric family);
- does not alter `mu` in any arm;
- does not extend the anomaly gate to pre-2013 via weekly rosters. That is available
  (`load_rosters_weekly` reaches 2005) and would add ~8 further seasons, but swapping the
  gate mechanism is a separate change and is deliberately **not** bundled here;
- does not touch 2026, which gets its own backtest.
