# Rush Post-Ensemble Reconciliation V1 — Frozen Result

Date: 2026-09-26

Disposition: **RUSH_POST_ENSEMBLE_RECONCILIATION_V1_FAILED_CLOSED**

Production changed: **false**

## Authority

- branch: `research-rush-post-ensemble-reconciliation-v1`
- frozen plan commit: `8b794b17206399714bce65219a33b779404a8908`
- implementation commit: `bae947b8024ae39ef711c7068bea1e5b6a02b043`
- workflow head: `776a01605d81d3e6d91648622191a4c8fd6753c5`
- run: `36253116006`
- job: `108434747167`
- artifact: `10910042122`
- artifact digest: `sha256:00a361e78db9383864b39a3a8cd9396670bf2a68dbc73f8490136f57c2aea4cb`

Historical baseline authority:

- run `36049144898`
- job `107800031207`
- artifact `10830277740`
- digest `sha256:b6a0715d38984fa91b8815561b4e72679684affcc142ee773a8b7428342412a5`
- source main `b5b1816e7d11cadca412faf841fbfe5336798c2e`

Only the preserved baseline columns were consumed. The old Rush Pool candidate columns were explicitly discarded.

## Candidate

For every 2024/2025 Week-2-18 team-game across all positions:

- `M = sum(baseline_mc_rush_att)`
- `E = sum(baseline_final_rush_att)`
- `F = M/E`
- candidate rush attempts = baseline final rush attempts × `F`
- candidate rush yards = baseline final rush yards × `F`

The same factor was applied to carries and yards, preserving every player's existing final implied YPC exactly.

RB rush+receiving kept its baseline receiving component unchanged.

No position carveout, threshold, cap, fit, sportsbook input or Week-1 change existed.

## Integrity

All mechanical invariants passed:

- max team player-carry mass gap: **3.55e-15**
- max player implied-YPC change: **7.11e-15**
- max preserved receiving-component gap: **1.42e-14**
- sportsbook inputs: **0**
- parameters fit: **0**
- candidate variants scored: **1**
- Week-1 rows: **0**
- production changed: **false**

So the failure is scientific, not mechanical.

## 2024

### ALL positions

Rush attempts:
- MAE `1.304463 -> 1.489554` — worse
- p90 `3.939111 -> 4.661606` — worse

Rush yards:
- MAE `8.208893 -> 8.638554` — worse
- p90 `23.525439 -> 25.550229` — worse

### RB family

Rush attempts:
- MAE `3.623225 -> 4.258956`
- p90 `8.228238 -> 10.744321`

Rush yards:
- MAE `21.378350 -> 23.236811`
- p90 `48.951195 -> 58.931787`

Rush+receiving yards:
- MAE `25.859070 -> 27.827834`
- p90 `56.244047 -> 65.348272`

### QB

- rush-att MAE `1.836032 -> 2.024198`
- rush-yard MAE `12.001147 -> 12.617420`

### OTHER

The only broad positive slice:
- rush-att MAE `0.187041 -> 0.175683`
- rush-yard MAE `1.702668 -> 1.472911`

This does not rescue the failed all-position / RB / QB gates.

## 2025

### ALL positions

Rush attempts:
- MAE `1.249158 -> 1.457922`
- p90 `3.757901 -> 4.652613`

Rush yards:
- MAE `7.845055 -> 8.251436`
- p90 `21.324949 -> 24.675290`

### RB family

Rush attempts:
- MAE `3.442835 -> 4.211076`
- p90 `7.874987 -> 10.344172`

Rush yards:
- MAE `20.671533 -> 22.786732`
- p90 `47.014453 -> 55.954382`

Rush+receiving yards:
- MAE `25.133456 -> 27.130781`
- p90 `54.343826 -> 63.688313`

### QB

- rush-att MAE `1.726906 -> 1.890003`
- rush-yard MAE `10.782066 -> 10.933493`

### OTHER

Again improved, but cannot rescue:
- rush-att MAE `0.212070 -> 0.191401`
- rush-yard MAE `1.751110 -> 1.480905`

## Frozen gate disposition

- passed: **9 / 33**
- failed: **24 / 33**

The candidate failed the central all-position, RB-family, QB, p90 and RB rush+receiving gates in both seasons.

Therefore:

**RUSH_POST_ENSEMBLE_RECONCILIATION_V1_FAILED_CLOSED**

No rescue.

## Failure diagnosis

The post-result team-volume diagnostic explains why the mathematically coherent reconciliation is a bad predictive repair.

Across 512 team-games per season, summing the preserved historical player rush-attempt rows:

### 2024

Actual player carry mass:
- mean **26.8125**

Joint-MC player carry mass:
- mean **17.4561**
- team-level MAE vs actual **10.0826**
- bias **-9.3564**

Final independently ensembled player carry mass:
- mean **23.8111**
- team-level MAE vs actual **6.7316**
- bias **-3.0014**
- closer than MC on **75.0%** of team-games

### 2025

Actual:
- mean **26.6504**

Joint-MC:
- mean **17.4596**
- MAE **10.0517**
- bias **-9.1908**

Final ensemble:
- mean **23.8631**
- MAE **6.4806**
- bias **-2.7873**
- closer than MC on **75.98%** of team-games

So the final independent player ensembles are not merely violating a good finite MC total; they are partially correcting an upstream MC rushing-volume state that is far too low in these historical folds.

That is why forcing final means back to the MC finite mass degrades RB/QB accuracy badly.

## Permanent stopping rule

Do not rescue this exact family with:

- partial reconciliation factors;
- factor caps/floors;
- RB-only routing;
- QB/OTHER exclusions;
- high-volume routing;
- alternate residual percentages;
- different carry/yard factors;
- 2026 outcome tuning.

The existing STACK6 team-rush-context slicing stop rule also remains binding.

Future work must use genuinely different information/mechanism or preserve the superior marginal means and address only joint/distributional consistency.
