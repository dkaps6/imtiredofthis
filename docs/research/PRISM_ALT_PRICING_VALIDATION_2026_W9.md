# PRISM alt-line pricing: what is actually validated, and what was not

**Date:** 2026-11-08 · **Rows:** `data/research/prism_probability_calibration_2026_v1.csv`,
`data/research/alt_parlay_tickets_2026_wk09_v2.csv`
**Status:** research finding. Three production defects identified. No production
behaviour changed by this document.

This supersedes the selection method in
`PRISM_ALT_PARLAY_TICKETS_2026_WK09.md`. That document's tickets were built from
the clearance table plus a raw own-record count. The analysis below shows that
method could not establish a priced edge, and 18 of its 24 legs are rejected by
the football model.

## Finding 1 — the alt clearance backtest never used a price

`scripts/research/backtest_alt_clearance_table_v1.py` contains zero references
to `offered_american`. It generated synthetic rungs as fractions of the main
line and measured P(clear) against outcomes. It established that the table's
**probabilities are calibrated** (13,990 cells, max band deviation 3.9pp) and
nothing whatsoever about whether those probabilities beat an offered price.

A calibrated probability with no price comparison carries no profitability
claim. Week 9 is the first week real alt prices were archived
(`data/market_track_record/alt_props/`), so **no out-of-sample test of alt
profitability exists for any method in this repository.** Every +EV number
quoted for an alt ladder, including in the superseded ticket document, rests on
that untested premise.

## Finding 2 — `model_sd` is far too narrow (production defect)

`run_pricing_v2.py` sets `p_over = mean(adjusted_outcomes > line)` from the
simulation draws. A gamma matched to (`model_proj`, `model_sd`) reproduces the
resulting `fair_prob` to MAE 0.009–0.048 across line/mean ratios from 0.03x to
22x, so the published summary faithfully describes the simulated distribution.

That distribution is badly overconfident. Graded against actuals for 2026 weeks
2/6/7/8, with the played-denominator taken from `load_snap_counts`
(`offense_snaps > 0`) so players who played and recorded nothing count as real
zeros (n=1,447, `anytime_td` excluded — it has no actual in this join):

| model says | n | actual | gap |
|---|---|---|---|
| 0-10% | 38 | 39.5% | **+33.5** |
| 10-20% | 109 | 50.5% | **+35.1** |
| 20-30% | 251 | 44.2% | +18.7 |
| 30-50% | 650 | 45.1% | +5.3 |
| 50-70% | 326 | 49.7% | -7.8 |
| 70-100% | 73 | 49.3% | **-30.3** |

Everything collapses toward ~46%. That is the signature of a mean carrying
modest information inside a distribution that is far too tight. The measured
spread of `actual/model_proj` confirms it — for `rec_yards` the ratio runs 0.47
at p25 to 1.67 at p75, and the player records a literal zero **11.4%** of the
time (20% in the low-projection tercile). The simulated sd never approaches
that width, and the bust rate is not represented at all.

Consequence: any probability `run_pricing_v2.py` reports outside roughly
30-50% is overstated in the direction of its own mean. `edge_pct` on the
priced board inherits this.

## Finding 3 — PRISM beats a naive baseline but not the market line

Predicting actuals, same sample:

| market | n | MAE model | MAE line | MAE prior-mean | r model | r line | r prior |
|---|---|---|---|---|---|---|---|
| pass_yards | 108 | **56.81** | 58.02 | 70.57 | **0.308** | 0.239 | 0.086 |
| rec_yards | 478 | 21.89 | **21.24** | 24.90 | 0.532 | **0.573** | 0.441 |
| receptions | 476 | 1.66 | **1.61** | 1.81 | 0.529 | **0.571** | 0.481 |
| rush_rec_yards | 110 | 30.27 | **26.61** | 31.29 | 0.518 | **0.609** | 0.541 |
| rush_yards | 262 | 18.37 | **17.98** | 21.64 | 0.678 | **0.701** | 0.606 |

PRISM beats a strictly-prior trailing mean on every market, so the model does
real work. It beats the **market line** only on `pass_yards`. Taking the
model's side wherever it disagrees with the line wins **52.1%** (n=1,447);
breakeven at -110 is 52.4%. Stratifying by disagreement size does not rescue
it (52.8 / 52.3 / 50.4 / 53.6% across <5% / 5-15% / 15-30% / >30%).

**There is no demonstrated main-line edge.** This independently reproduces the
anchor result in `PRISM_ALT_LINE_ANCHOR_RESULTS_V1.md`.

An empirical predictive distribution — the measured `actual/model_proj`
distribution conditioned on market and projection tercile, fit on weeks 2/6/7
and tested on week 8 — scores Brier **0.2622** at the market line, worse than
constant 0.5 (0.2500). Correcting the spread fixes the calibration but does not
create discrimination at the line.

## What the week 9 v2 tickets therefore rest on

Not a validated edge. A **robustness intersection**: a leg is included only
when three estimates computed by different routes all exceed the offered
breakeven, and the ticket is priced off the *lowest* of the three.

1. `p_prism` — PRISM's `model_proj` under the measured empirical spread
   (football only; no sportsbook input, AGENTS.md §3).
2. `p_side` — the calibrated clearance table.
3. `p_own` — the player's own 2026 rate, Beta-shrunk toward the table with
   m=5 pseudo-games, denominator from snap counts.

Plus: ≥6 games of own evidence, one leg per game, and an injury gate. 178 of
5,275 priced alt OVERs survive; 45 players across 12 games.

The injury gate removed **MarShawn Lloyd** (Questionable, knee, limited in
practice) — a 7/8 leg in two tiers of the superseded document, never checked
against the week 9 report. That is the AGENTS.md player-state gate catching a
real miss.

`pass_yards` is the one market where PRISM has measured skill over the line,
and there it rejects every long QB rung in the superseded lottery tier
(Stroud >349.5 at 0.4% against an 11.3% table claim). That rejection has the
strongest evidential support of anything computed here.

## Honest residual risks

- **The price-beating premise is untested**, per Finding 1. The intersection is
  a conservatism device, not evidence.
- **Independence is assumed across legs.** One-per-game removes same-game
  correlation but not the league-wide passing environment, and the HIGH and
  LOTTERY tiers are almost entirely receptions, so they are one correlated bet
  on pass volume. Their EV figures compound that assumption 5-6 times and
  should not be read as reliable.
- **Own-record samples are 6-8 games**; shrinkage mitigates but does not cure.
- `p_min` is deliberately pessimistic, so stated win probabilities bracket
  rather than estimate.

## Recommended production follow-ups

1. Widen the simulation's dispersion to the measured residual spread, per
   market and projection tercile, and model the bust rate explicitly. Re-audit
   every `edge_pct` consumer afterwards.
2. Grade week 9 alt rungs against the archived prices — the first real test of
   alt profitability this repository will have.
3. Extend `backtest_alt_clearance_table_v1.py` to require a price, so a
   calibration claim can never again be reported as an edge claim.
