# QB M89/M90 Opportunity / Efficiency Decomposition V1 — Result

**STATUS: COMPLETE DIAGNOSTIC. RESEARCH ONLY. NO PRODUCTION CHANGE. NO CANDIDATE AUTHORIZED.**

Authoritative workflow:
- run: `36330107392`
- conclusion: **SUCCESS**
- head: `1d3b872fd57b5e18053d11e58a25fdc16f2b4851`
- artifact: `10935138252`
- artifact digest: `sha256:8699230a732a754c3487af650d793503c3a6683eb070b730e604c2cbb32251c1`

Frozen authority:
`docs/research/QB_M89_M90_OPPORTUNITY_EFFICIENCY_DECOMPOSITION_V1_PLAN.md`

## Integrity

All frozen integrity checks passed.

- evaluation seasons: exactly 2024 + 2025
- rows: **894**
- players: **61**
- sportsbook inputs used: **false**
- candidate models fit: **0**
- production changed: **false**
- max actual-yards trace/log identity gap: **0.0**
- max exact Shapley decomposition identity gap: **5.68e-14**
- final disposition: `QB_M89_OPP_EFF_DECOMPOSITION_COMPLETE`

## Pooled 2024-2025 result

Promoted M89/M90 football-only mean:
- MAE: **60.4901 yd**
- RMSE: **77.1715 yd**
- bias: **+4.3436 yd**

Absolute error-mass attribution:
- passing opportunity / attempts: **42.37%**
- passing efficiency / YPA: **34.01%**
- non-factor M89-vs-(attempts×YPA) residual: **23.62%**

Dominant row counts:
- opportunity: **426 / 894**
- efficiency: **332 / 894**
- non-factor residual: **136 / 894**

Oracle diagnostics, retaining the non-factor residual:
- actual attempts + predicted YPA: MAE **49.5476**, recovery **10.9425 yd**
- predicted attempts + actual YPA: MAE **49.4583**, recovery **11.0318 yd**
- actual attempts + actual YPA: MAE **29.7664**, recovery **30.7237 yd**

The two primitives have major interaction/cancellation; the pooled cancellation rate is **86.02%**. Therefore contribution shares are diagnostic and must not be converted into fitted weights.

## Catastrophic misses

For absolute M89 error >=100 yd:
- rows: **160**
- opportunity-dominant: **98 (61.25%)**
- efficiency-dominant: **56 (35.00%)**
- non-factor residual-dominant: **6 (3.75%)**
- actual-attempt oracle MAE: **60.4959** vs baseline **139.5008**
- actual-YPA oracle MAE: **93.1481**
- both-primitives oracle MAE: **29.7099**

For absolute M89 error >=75 yd:
- rows: **286**
- opportunity-dominant: **156**
- efficiency-dominant: **112**
- non-factor residual-dominant: **18**

This is the strongest routing result from the audit: the largest M89/M90 misses are disproportionately driven by passing-opportunity error, while efficiency remains a substantial secondary component.

## Season stability

2024:
- M89 MAE **60.6478**
- opportunity share **43.52%**
- efficiency share **33.49%**
- residual share **22.99%**
- opportunity-oracle recovery **12.3873 yd**
- efficiency-oracle recovery **10.6972 yd**

2025:
- M89 MAE **60.3339**
- opportunity share **41.19%**
- efficiency share **34.54%**
- residual share **24.27%**
- opportunity-oracle recovery **9.5106 yd**
- efficiency-oracle recovery **11.3634 yd**

The broad ordering is stable across both evaluation seasons; neither season may be cherry-picked into a candidate.

## Interpretation under the M82 anti-reinvention ledger

This result does **not** authorize:
- generic pass-rate/game-script retuning;
- another attempt model using the same historical tendency families;
- another generic efficiency-volatility model;
- a catastrophic-miss router;
- M89/M90 coefficient/cap changes;
- C2 changes;
- market-assisted correction;
- same-data model-zoo rescue.

Those families were already tested/closed.

What this result does establish is that the next QB mean-information search, if any, should prioritize **genuinely new pregame week-specific passing-opportunity / gameplan intent information**, because opportunity owns the largest pooled error mass and the majority of catastrophic misses.

The existing public-intent source lane remains conceptually aligned with this diagnosis, but its prior V1B work is parked as a retrieval/transport failure rather than a scientific rejection. No new public-intent candidate may be fitted until source/schema reliability is established and a separate plan is frozen.

Efficiency remains material (~34% of pooled error mass), so it stays a secondary frontier, but prior generic efficiency/volatility families remain closed. Any future efficiency candidate also requires materially new information.

## Final disposition

`QB_M89_OPP_EFF_DECOMPOSITION_COMPLETE`

`next_candidate_authorized = false`

Next legitimate step:
1. preserve this as diagnostic authority;
2. source/schema-audit materially new pregame opportunity/intent information;
3. only if a trustworthy new source exists, freeze a separate predictive candidate before observing candidate results.

No production mutation follows from this run.
