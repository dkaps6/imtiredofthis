STATUS: RESEARCH ONLY — NOT PROMOTED — AWAITING USER APPROVAL. UNVERIFIED — flagged for GPT-5.6 falsification, not treated as real yet.

# Full-Market Holdout Scan V1 — Analysis

Companion to `FULL_MARKET_HOLDOUT_SCAN_V1_RESULT.md` (raw output). This extends
the receptions-only situational search to all four non-QB markets, with the
holdout discipline built in from the start (fit quantile thresholds on one
season, test blind on the other, both directions, require both positive)
instead of the pooled-then-checked approach that had to be corrected for the
receptions candidate.

## What came back

**Categorical dimensions** (side/home-away/week-bucket, no leakage risk): only
2 weak candidates, both receptions, both marginal (+1.5-1.7% pooled). Nothing
new or alarming here.

**Quantile-holdout dimensions** (prob_edge, component_sd — fit on one season,
tested blind on the other): 7 candidates, and a pattern that needs scrutiny
before anyone gets excited about it:

| market | dimension | slice | fit2024→test2025 n / ROI | fit2025→test2024 n / ROI |
|---|---|---|---|---|
| rec_yards | prob_edge | top quartile, OVER | 257 / +3.26% | 277 / +3.90% |
| receptions | prob_edge | top quartile, UNDER | 509 / +1.09% | 506 / +2.02% |
| receptions | prob_edge | top quartile, OVER | 193 / +3.23% | 95 / +4.58% |
| rush_rec_yards | component_sd | top quartile | 258 / +5.84% | 292 / +2.62% |
| rush_rec_yards | component_sd | top quartile, UNDER | 218 / +7.97% | 261 / +1.17% |
| rush_yards | component_sd | top quartile, UNDER | 197 / +6.41% | 225 / +3.15% |

## Why I am not treating this as good news yet

Three of these six (rush_yards, rush_rec_yards x2) are driven by
**`component_sd`** — the spread across `mc_proj`/`ml_proj`/`state_proj`, i.e.
how much the model's own internal components disagree. That variable feeds
**directly** into the exact fidelity gap already disclosed in every doc in this
thread: fair probability is computed as `Normal(mean=proj, sd=component_sd)`,
not production's actual simulated Monte Carlo distribution. A higher
`component_sd` mechanically widens that Normal approximation and changes the
computed `p_over`/`p_under` — which is also what feeds `prob_edge` and the
`STRONG_EDGE` gate itself. **It is entirely possible that "top quartile of
component_sd shows an edge" is not a football signal at all — it may be an
artifact of how this specific historical benchmark approximates probability**,
one that wouldn't exist against production's real simulated distribution.

This is exactly the kind of thing I cannot self-falsify well, because I built
the approximation. It's precisely GPT-5.6's assigned lane in #535
("fair-probability calibration... historical MC distribution reconstruction").
I'm posting this to the issue asking GPT-5.6 to specifically check: does this
pattern survive if you replace the Normal(mean, component_sd) approximation
with something closer to the real simulated distribution, or is the "edge"
mechanically created by the approximation itself?

The `prob_edge`-driven candidates (rec_yards/receptions OVER, receptions
UNDER) are one step more removed from the raw `component_sd` mechanism but
still ultimately derive from the same Normal-approximation probability
calculation, so the same caution applies to all six, just more directly to
the three `component_sd` ones.

## Status: `UNVERIFIED`, flagged as possibly artifactual, not `QUALIFIED_CANDIDATE`

Unlike the receptions-UNDER `prob_edge` candidate (which survived a holdout
test and is merely statistically thin), these six have a specific, named,
plausible alternative explanation — a measurement artifact — that has not
been ruled out. Do not treat any of these as edges. Posted to Issue #535 for
independent audit before any further characterization work.
