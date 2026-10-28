# PRISM — RB rushing yards: the gap is Monte Carlo LEVEL, and ML already has it right

**Live 2026 data, weeks 1/2/6/7, n=173 RB `rush_yards` rows graded against actuals.**
All component columns come from the archived boards; no fixture, no replay needed.

## The finding

| component | mean projection | bias | MAE |
|---|---|---|---|
| `mc_proj` | **32.75** | **−16.15** | 25.38 |
| `ml_proj` | **49.57** | **+0.79** | 26.40 |
| `state_proj` | 45.43 | −3.68 | 26.66 |
| `ensemble_proj` | 39.84 | −9.06 | 24.47 |
| `model_proj` (final) | 40.44 | −8.46 | 24.39 |
| vegas line | 46.47 | −2.43 | **22.78** |
| **actual** | **48.90** | | |

**The Monte Carlo under-projects RB rushing yards by 16 yards. The ML component is
essentially unbiased (+0.79).** The ensemble lands at −9.06 because it is anchored to a
badly biased MC, and the final inherits −8.46.

Our excess bias versus the market on identical rows is **−6.03 [−8.03, −4.03]**,
n=173 — significant, and selection-immune because the line faces the same rows.

**The blend is not the problem.** Final MAE 24.39 beats the best single component
(`mc_proj`, 25.38) by 0.99, so the ensemble is doing its job. MC also has the *best* MAE
of the three despite the *worst* bias — good discrimination, wrong level. That is the
signature of a level defect, not a weighting defect.

## Why MC is low: the prop universe is selected toward featured backs

Raw-simulation diagnostics on the W7 fixture all came back clean:

- **team RB carry volume is right**: 22.59 simulated vs 23.11 actual RB carries per team
  (bias −0.52);
- **YPC is right**: model 4.093 vs actual 4.252;
- **the model picks the right back**: within-team Spearman ρ = **+0.498**
  [+0.273, +0.723], and its top projected back is the actual lead back **20/26 = 76.9%**
  of the time against a ~38% chance baseline.

Team total right + correct back identified + per-featured-back low leaves one
explanation: **the carry distribution is too flat.** The model's mean over its whole RB
universe is fine, but sportsbook props exist mainly for featured backs, and those are
exactly the players a flat distribution shortchanges.

Measured directly, the lead back's share of team carries is model **0.558** vs actual
**0.606** — a −0.048 gap whose 95% CI is [−0.104, +0.009], so **suggestive but not
significant at n=28 teams**. It is stated at that strength deliberately: it is the only
mechanism consistent with every other diagnostic, not an independently established one.

## Corrections made along the way, recorded so they are not repeated

Three intermediate conclusions in this diagnosis were **wrong** and were killed by
cleaner measurement. Each was a selection artifact of filtering on the outcome:

1. "team carry volume is 7.3 low" — an artifact of filtering teams to `actual >= 15`
   carries. Clean value: **−0.52**.
2. "lead-back concentration is 7.7pp low" — same filter. Clean value: **−4.8pp, not
   significant**.
3. "projected carries are 4.20 low per back" — an artifact of requiring ≥5 *actual*
   carries, which excludes backs who were projected carries and got none.

A fourth error was caught before it reached a conclusion: a position filter silently
matched nothing (the backtest rows carry no `position` column), so numbers labelled "TE"
were actually all rows. It was noticed only because RB and TE returned identical values.

**The one number immune to all of this is the vegas-benchmarked excess bias**, because
the line is quoted on exactly the same selected rows.

## Two candidate fixes

**A. Sharpen RB carry concentration in the simulation** — addresses the root cause, and
has direct precedent: WR hierarchy target-share sharpening is already a promoted concept
doing the analogous job for receivers. There is a promoted top-five rushing opportunity
pool but no concentration sharpening for backs.

**B. Reweight the ensemble toward ML for `rush_yards`** — faster, and `ml_proj` is already
unbiased on this market. But `data/model_ensemble_weights.csv` is promoted/frozen
calibration (AGENTS.md §5), so this needs its own preregistered validation and must not
be hand-tuned.

**A is the real fix; B is the fast one.** Neither is implemented. Both need a frozen
design and an out-of-sample check before touching live projections, and the measured
prize is bounded: closing the full −6.03 excess would bring RB `rush_yards` MAE from
24.39 toward roughly the market's 22.78.

## What this does NOT claim

Nothing here is a production change. The −16.15 MC bias is measured on 173 rows across
four weeks of one season; `rush_rec_yards` inherits it (its −18.33 bias decomposes
exactly as rush −9.17 + rec −6.32 + a −2.84 conservation haircut), so a rushing-level fix
should improve both markets, but that is an expectation and not yet a result.

## Scope

No production code changed. No sportsbook variable entered any projection; lines are the
comparison benchmark only. Betting remains CLOSED.
