# Rush Pool Evidence Guard V1 — Production Integration Result

Disposition: **RUSH_POOL_EVIDENCE_GUARD_V1_PRODUCTION_INTEGRATION_FAILED_CLOSED**

Production changed: **false**

The upstream allocator-only candidate remains a valid research finding, but its exact production integration did not satisfy the preregistered current-stack football gates. The candidate is closed with no rescue.

## Authority

- production main entering integration: `b5b1816e7d11cadca412faf841fbfe5336798c2e`
- research qualification: `RUSH_POOL_EVIDENCE_GUARD_V1_QUALIFIED`
- original frozen science run: `36045359696`
- integration branch: `integrate-rush-pool-evidence-guard-v1`
- integration head: `808072634603c03e6c86fd3c45ea311a39b13fa6`
- full-stack run: `36049144898`
- full-stack job: `107800031207`
- full-stack artifact: `10830277740`
- digest: `sha256:b6a0715d38984fa91b8815561b4e72679684affcc142ee773a8b7428342412a5`
- Repo CI on integration head: `36049328629` = **SUCCESS**
- preserved paid-artifact Full Slate replay: `36049328727` = **SUCCESS**
- production PR #633: **CLOSED / NOT MERGED**
- no new paid odds acquisition

## What passed

The integration architecture behaved exactly as intended.

Across both seasons:

- Week-1 baseline and candidate arrays were bit-identical;
- all non-rushing arrays were bit-identical;
- max non-rushing element gap = `0.0`;
- target allocations were unchanged;
- team rush-attempt arrays were unchanged;
- raw `rules_rush_share` values were unchanged;
- sportsbook inputs used = `0`;
- RB Rush+Receiving Conservation V2 pathwise identity gap = `0.0`;
- WR historical future-participation violations = `0`.

So this is **not** a plumbing, leakage, RNG, identity, or integration-integrity failure.

## 2024 full-stack result

### Rush attempts

- ALL MAE: `1.304463 -> 1.296773` — improved
- ALL p90: `3.939111 -> 4.002644` — **worsened**
- RB/FB/HB MAE: `3.623225 -> 3.596394` — improved
- RB/FB/HB p90: `8.228238 -> 8.175675` — improved
- QB MAE: `1.836032 -> 1.828781` — improved
- QB p90: `3.888789 -> 3.838940` — improved
- OTHER MAE: `0.187041 -> 0.187577` — worse
- OTHER p90: `0.589443 -> 0.595308` — **worsened**
- 10+ attempt misses: `75 -> 71` overall/RB

### Rush yards

- ALL MAE: `8.208893 -> 8.156087` — improved
- ALL p90: `23.525439 -> 23.948763` — worse
- RB/FB/HB MAE: `21.378350 -> 21.150249` — improved
- RB/FB/HB p90: `48.951195 -> 46.982269` — improved
- RB/FB/HB 30+ misses: `299 -> 292`
- QB MAE: `12.001147 -> 12.040055` — **worsened**
- QB p90: `30.008560 -> 29.116218` — improved
- OTHER MAE: `1.702668 -> 1.707076` — worse
- OTHER p90: `2.460827 -> 2.531166` — **worsened**

### RB rush+receiving yards

- MAE: `25.859070 -> 25.468813` — improved
- p90: `56.244047 -> 54.915846` — improved
- 30+ misses: `375 -> 376` — one worse

## 2025 full-stack result

### Rush attempts

- ALL MAE: `1.249158 -> 1.253482` — **worsened**
- ALL p90: `3.757901 -> 3.728187` — improved
- RB/FB/HB MAE: `3.442835 -> 3.456101` — **worsened**
- RB/FB/HB p90: `7.874987 -> 7.768205` — improved
- RB 10+ misses: `62 -> 63` — one worse
- QB MAE: `1.726906 -> 1.733759` — **worsened**
- QB p90: `3.794853 -> 3.750734` — improved
- OTHER MAE: `0.212070 -> 0.212043` — effectively flat/slightly better
- OTHER p90: unchanged `0.645542`

### Rush yards

- ALL MAE: `7.845055 -> 7.855806` — **worsened**
- ALL p90: `21.324949 -> 21.238577` — improved
- RB/FB/HB MAE: `20.671533 -> 20.596408` — improved
- RB/FB/HB p90: `47.014453 -> 46.432120` — improved
- RB/FB/HB 30+ misses: `257 -> 256`
- QB MAE: `10.782066 -> 11.001698` — **worsened**
- QB p90: `24.349466 -> 23.132160` — improved
- OTHER MAE: `1.751110 -> 1.754818` — worse
- OTHER p90: `2.343064 -> 2.385445` — **worsened**

### RB rush+receiving yards

- MAE: `25.133456 -> 24.930727` — improved
- p90: `54.343826 -> 54.118428` — improved
- 30+ misses: `333 -> 325`

## Frozen failed gates

The exact candidate failed multiple preregistered gates, including:

- 2024 ALL rush-att p90 non-worse;
- 2024 OTHER rush-att p90 non-worse;
- 2024 QB rush-yard MAE non-worse;
- 2024 OTHER rush-yard p90 non-worse;
- 2025 ALL rush-att MAE improvement;
- 2025 RB rush-att MAE improvement;
- 2025 QB rush-att MAE non-worse;
- 2025 ALL rush-yard MAE non-worse;
- 2025 QB rush-yard MAE non-worse;
- 2025 OTHER rush-yard p90 non-worse.

Therefore the exact V1 production integration is scientifically disqualified.

## Interpretation

The allocator-only research signal was real, but it was not sufficiently stable after the current production stack's stochastic simulation, ensemble weighting, positional interactions, and downstream yardage construction.

Notably:

- RB rush-yard MAE improved in both seasons;
- RB rush+receiving-yard MAE and p90 improved in both seasons;
- but rushing opportunity accuracy did not replicate in 2025;
- QB rushing-yard mean accuracy worsened in both seasons;
- some OTHER-position tails worsened.

That mixed pattern is useful architecture evidence, but it does not authorize a carveout or rescue.

## Permanent stopping rule

Do **not** rescue this exact family with:

- RB-only routing;
- QB exclusions;
- OTHER-position exclusions;
- another evidence-state definition;
- another top-N size;
- minimum rush-share thresholds;
- depth-chart priority;
- injury-conditioned exceptions;
- rookie exceptions;
- retuned Bayesian strengths;
- 2026 result fitting.

Any future rushing work must be a genuinely different information/mechanism hypothesis.

## Production state

Main remains:
`b5b1816e7d11cadca412faf841fbfe5336798c2e`

The V1 integration code was never merged. Existing production science is unchanged.
