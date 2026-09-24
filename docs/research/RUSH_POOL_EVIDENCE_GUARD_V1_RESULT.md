# Rush Pool Evidence Guard V1 — Frozen Result

Disposition: **RUSH_POOL_EVIDENCE_GUARD_V1_QUALIFIED**

Research qualification only. **Production changed: false.**

## Frozen authority

- parent production main at freeze: `e5f630d7ece7762a71ae238c210408fb9d4fb3cb`
- research branch: `research-rush-pool-evidence-guard-v1`
- frozen plan commit: `156fb0a2d98f8822be95f011d384795701990073`
- frozen Issue #535 checkpoint: `5820317798`
- first/only frozen scoring run: `36045359696`
- workflow head: `24236c40919920a3a12d2045aa1668c3ebbb0fa1`
- job: `107787459855`
- result artifact: `10828600510`
- artifact digest: `sha256:d338f9ecca49dddd02e3e501f70b7b4b2efc25d084064299aea4b5e4b3f46e3b`
- parameters fit: `0`
- candidate variants scored: `1`
- sportsbook inputs used: `0`
- Week-1 no-op verified: `true`

The workflow completed mechanically with conclusion **SUCCESS**.

## Frozen candidate

From Week 2 onward:

- preserve every existing raw `rules_rush_share`;
- preserve top-five pool size;
- preserve team rush volume;
- preserve 95% player-mass cap and residual semantics;
- select positive-share players with player-specific evidence before players whose Bayesian state is only `position_prior_only`;
- use `position_prior_only` players only to fill unused slots;
- preserve all existing efficiencies.

No depth weighting, position weighting, share threshold, pool-size search, outcome fit, sportsbook input, or 2026 result tuning was used.

## Independent 2024 result

Scope: 2024 Weeks 2-18, with 2023 as prior season.

- team-games: `512`
- selector-changed team-games: `177`
- baseline omitted evidenced positive players: `5,048`
- candidate omitted evidenced positive players: `4,795`
- mean position-prior-only carry mass: `0.082784 -> 0.000000`

### ALL positions

- n: `3,304`
- MAE: `3.396212 -> 3.117177`
- improvement: `0.279035 attempts`
- p90 absolute error: `6.763334 -> 6.526283`
- changed-row candidate closer rate: `55.5062%`

### RB / FB / HB

- n: `1,907`
- MAE: `4.450521 -> 4.016440`
- improvement: `0.434081 attempts`
- p90 absolute error: `8.990499 -> 8.730309`
- changed-row candidate closer rate: `58.2090%`

### QB

- n: `1,019`
- MAE: `2.241966 -> 2.142317`
- improvement: `0.099649 attempts`
- p90 absolute error: `4.239973 -> 4.104421`

### Other positions

- n: `378`
- MAE: `1.188829 -> 1.208417`
- regression: `0.019588 attempts`
- p90 absolute error: `2.000000 -> 2.000000`

The small OTHER-position regression is preserved explicitly. It was not a preregistered primary gate, but it must be guarded in the separate production-integration test.

## Independent 2025 result

Scope: 2025 Weeks 2-18, with 2024 as prior season.

- team-games: `512`
- selector-changed team-games: `226`
- baseline omitted evidenced positive players: `5,174`
- candidate omitted evidenced positive players: `4,893`
- mean position-prior-only carry mass: `0.095854 -> 0.000000`

### ALL positions

- n: `3,300`
- MAE: `3.317911 -> 3.014478`
- improvement: `0.303433 attempts`
- p90 absolute error: `6.611318 -> 6.327268`
- changed-row candidate closer rate: `55.0962%`

### RB / FB / HB

- n: `1,946`
- MAE: `4.382224 -> 3.927140`
- improvement: `0.455084 attempts`
- p90 absolute error: `8.957913 -> 8.430163`
- changed-row candidate closer rate: `56.5169%`

### QB

- n: `984`
- MAE: `2.019613 -> 1.896226`
- improvement: `0.123388 attempts`
- p90 absolute error: `4.000000 -> 3.717959`

### Other positions

- n: `370`
- MAE: `1.172973 -> 1.188319`
- regression: `0.015346 attempts`
- p90 absolute error: `2.000000 -> 2.000000`

Again, the small OTHER-position regression is explicit and must be checked downstream.

## Pooled primary result

Across `6,604` scored player-games:

- ALL-position rush-attempt MAE:
  `3.357085 -> 3.065859`
- improvement:
  **`0.291226 attempts`**

## Frozen gates

All preregistered gates passed:

- ALL rush-attempt MAE improves in 2024: **PASS**
- ALL rush-attempt MAE improves in 2025: **PASS**
- RB/FB/HB MAE improves in 2024: **PASS**
- RB/FB/HB MAE improves in 2025: **PASS**
- pooled ALL improvement >= 0.02: **PASS**
- ALL p90 non-worse in both years: **PASS**
- RB/FB/HB p90 non-worse in both years: **PASS**
- QB MAE non-worse in both years: **PASS**
- changed-row candidate closer > 50% in both years: **PASS**
- omitted evidenced positive players strictly decrease in both years: **PASS**
- integrity gates: **PASS**

Therefore the exact frozen candidate qualifies for a **separate full-stack production-integration test**.

## What qualification does NOT mean

This does **not** authorize production.

Before any promotion, freeze a separate integration contract and prove:

- exact current simulator wiring;
- Week 1 no-op;
- team rush volume invariant;
- rush attempts improve under canonical MC;
- rush yards do not regress;
- RB rush+receiving does not regress;
- QB rushing does not regress;
- OTHER-position rushing is explicitly guarded because the historical allocator-only test showed a small MAE regression there;
- receiving/pass markets are unchanged;
- RB Rush+Receiving Conservation V2 identity remains intact;
- Full Slate lineage/fail-closed audit behavior is correct.

## Stopping rule remains frozen

Do not search:

- another evidence definition;
- another pool size;
- minimum-share thresholds;
- position priorities;
- depth-chart gates;
- rookie/QB exceptions;
- injury-conditioned exceptions;
- Bayesian prior strengths;
- 2026 outcome-based rescue/tuning.
