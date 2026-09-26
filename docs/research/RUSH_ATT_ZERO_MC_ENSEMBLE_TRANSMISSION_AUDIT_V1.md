# Rush-Attempt Zero-MC Ensemble Transmission Audit V1

Date: 2026-09-26

Status: **DIAGNOSTIC CONFIRMED — CAUSE/REPAIR NOT YET FROZEN**

Authority:
- historical fair-probability reconstruction run `36275245917`
- compact artifact `10917407106`
- digest `sha256:35ad18b1a3ea96439558045a12801ea5c30a06247fce94f1ff83f6e3082c5d04`
- seasons 2024-2025
- no Week-3 outcomes
- no sportsbook information upstream

## Structural finding

Current pricing computes a calibrated ensemble mean but can only transmit that mean into the Monte Carlo distribution when:

`mc_proj > 0`.

When MC mean is zero, `run_pricing_v2.py` leaves the raw MC distribution unchanged. Therefore a nonzero ML/State contribution can appear in `ensemble_proj` while the actual priced distribution and final `model_proj` remain exactly zero.

This is a real mean-transmission contradiction.

## Historical count

Count-market rows:
- receptions: 9,253 total; zero-MC / nonzero-ensemble = **0**
- rush_att: 10,592 total; zero-MC / nonzero-ensemble = **7,137**

The issue is therefore concentrated in rushing attempts.

## Position-level evidence on the 7,137 blocked rush-att rows

| position | rows | actual > 0 | forced-zero MAE | intended ensemble MAE | ensemble closer |
|---|---:|---:|---:|---:|---:|
| QB | 520 | 420 | 2.2481 | 1.5108 | 77.88% |
| RB | 161 | 93 | 1.4472 | 1.2684 | 52.17% |
| WR | 4,247 | 611 | 0.1884 | 0.2441 | 14.34% |
| TE | 2,209 | 89 | 0.0634 | 0.1208 | 4.03% |

QB result independently persists:
- 2024: 2.2472 -> 1.5519 MAE; ensemble closer 77.12%
- 2025: 2.2490 -> 1.4661 MAE; ensemble closer 78.71%

RB result:
- pooled 1.4472 -> 1.2684 MAE;
- direction favorable in both seasons, though materially smaller than QB.

## What this does NOT authorize

A blanket "inject the ensemble whenever MC is zero" repair is **not authorized** because that would materially worsen WR and TE rush-attempt predictions.

Likewise, this does not authorize:
- a QB carveout chosen after seeing these results;
- an RB carveout;
- new pool thresholds;
- alternate top-N values;
- position-specific rescue coefficients;
- production changes.

The position asymmetry is diagnostic evidence about the architecture, not a fitted routing rule.

## Deeper unresolved inconsistency

The zero-MC state is not explained simply by low `rules_rush_share`.

Among blocked rows:
- QB: 95.38% have their reported positive rush share ranked within the team's top five;
- RB: 96.89% rank within the top five.

Example:
- 2024 Week 1 ATL Kirk Cousins:
  - `rules_rush_share = 0.101538`
  - third-highest reported ATL rush share
  - historical MC `rush_att = 0.0`
  - calibrated ensemble `rush_att = 1.6759`
  - actual = 1 carry

Across all historical positive-share rush-att rows, the probability of receiving positive MC carry mass declines sharply by reported share rank:
- rank 1: 100%
- rank 2: 98.7%
- rank 3: 75.9%
- rank 4: 36.2%
- rank 5: 4.0%
- rank 6+: 0%

That pattern does not resemble the documented literal top-five selector and requires an exact allocation-lineage audit before any repair is designed.

## Current Week-3 relevance known so far

In the already-frozen Week-3 RB/FB cross-market diagnostic:
- zero-MC / nonzero-ensemble RB/FB rows = 3
- all three are FB:
  - Alec Ingold
  - Andrew Beck
  - Kyle Juszczyk

No broad Week-3 tailback conclusion follows from this.

The Week-3 QB side has not yet been authoritatively materialized under this audit and must not be inferred from the historical result.

## Next legitimate action

Diagnostic only:
1. reconstruct the exact player-selection/allocation trace that produced historical zero-MC QB/RB rows;
2. compare selected `rules_rush_share`, top-five membership, final multinomial probability, and realized MC mean;
3. locate whether the contradiction is in selector inputs, deduplication/order, identity, allocation, or post-allocation lookup;
4. only then freeze a repair hypothesis.

No production/model/weight/threshold change.
