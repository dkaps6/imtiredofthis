# QB Pass-Opportunity Rate Designed-Run D1 — Result

## Disposition

`QB_DESIGNED_RUN_PASS_RATE_D1_FAIL_NO_CONFIRMATION`

The designed-run linkage produced a real but insufficient 2024 development improvement under the frozen gates. It does not advance to 2025 confirmation. Production remains unchanged.

## Canonical lineage

- Branch: `research-qb-pass-rate-designed-run-d1`
- Frozen development plan: `d2b512951c45bc8d617ebb5e0fcab451717a17f8`
- Run-1 evaluator: `7908aed52da9a25070f2d0a012caee8f712d1430`
- Run-1 tested head: `99c2e8abf359f3fc52f60fd7d0ad205cd2411236`
- Run 1: `34538083332`
- Job 1: `103074236226`
- Run-1 evidence artifact: `10176183646`
- Run-1 artifact digest: `sha256:07a8a86bdd12be8632938127499b34bb1d5e05dbafb3656ca068ea2aeb7d9bcb`
- Run-1 disposition: `MECHANICAL_SCHEMA_CASE_MISMATCH_NO_SCIENCE`
- Frozen Run-2 schema repair wrapper: `e945bc66d2be72850f3f28dadc0b73b7ee86c922`
- Canonical Run-2/tested head: `54d43ab161076d18735942c8a520ef23bbb995d9`
- Canonical Run: `34538238737`
- Canonical Job: `103074716072`
- Canonical Artifact: `10176238015` (`qb-pass-rate-designed-run-d1-r2`)
- Canonical Artifact digest: `sha256:e33a1a9bdc1c04eeff2283a36d1827f85dc6a648340aea9e4646bea3c5567bac`

## Integrity

All frozen scientific integrity gates passed on Run 2:

- exact 2024 rows: `444`
- 2025 scored/summarized: `false`
- sportsbook inputs: `0`
- model fitting: `0`
- production changes: `0`
- candidate coefficient: exactly `1.0`
- baseline pass-opportunity rate: exactly `0.57`
- M89/M90 QB passing-yard mean identity max gap: `0.0`
- baseline team-D identity max gap: `7.105427357601002e-15`
- baseline QB-attempt identity max gap: `1.7763568394002505e-14`
- source strict-prior contract reused: PASS

Run 1 is preserved separately as a mechanical/no-science failure; no candidate metric was calculated before its schema-case error.

## Frozen candidate

For source-eligible QBs, the candidate used:

`qb_prior_designed_runs_per_game = recent8_designed_runs / prior_games`

and centered that value on the same target week's strict-prior QB mean. One excess designed QB run displaced exactly one pass opportunity at fixed projected team plays:

`candidate_pass_rate = 0.57 - (QB designed runs/game - weekly reference) / pred_plays`

with coefficient exactly `1.0`, fixed sanity clip `[0.35, 0.75]`, and no adjustment for unresolved/no-history rows.

Eligible rows: `441/444` (`99.3243%`).

No clip was hit.

## 2024 development result

### Pass-opportunity rate

- baseline MAE: `0.0882520861`
- candidate MAE: `0.0857579808`
- MAE gain: `0.0024941052`
- frozen required gain: `>=0.0030` — **FAIL**
- baseline RMSE: `0.1090171629`
- candidate RMSE: `0.1055843562`
- baseline p90 abs error: `0.1830256410`
- candidate p90 abs error: `0.1751682056` — PASS/non-worse
- baseline correlation: `0.0227823707`
- candidate correlation: `0.2531514805`

The candidate adjustment had Spearman `0.1551541374` against the realized pass-rate residual, clearing the frozen `>=0.10` directional-information gate.

Bootstrap over 5,000 paired resamples:

- mean pass-rate MAE gain: `0.0025334462`
- P(gain > 0): `0.9754`

So the signal is real directionally, but its effect size misses the preregistered minimum.

### Team pass opportunity

- baseline MAE: `7.4034391293`
- candidate MAE: `7.2943953600`
- MAE gain: `0.1090437694`
- frozen required gain: `>=0.15` — **FAIL**
- baseline RMSE: `9.4023404989`
- candidate RMSE: `9.2571463424`
- baseline p90 abs error: `14.9447697812`
- candidate p90 abs error: `14.7722442736` — PASS/non-worse
- baseline correlation: `0.0062344615`
- candidate correlation: `0.1390520998`

Bootstrap:

- mean D-MAE gain: `0.1104320883`
- P(gain > 0): `0.9364`

### QB official attempts

- baseline MAE: `7.3396804700`
- candidate MAE: `7.2238586286`
- MAE gain: `0.1158218414`
- frozen required gain: `>=0.10` — **PASS**
- baseline RMSE: `9.5350515989`
- candidate RMSE: `9.3915915819`
- baseline p90 abs error: `15.0547287719`
- candidate p90 abs error: `15.0175147135` — PASS/non-worse
- 10+ attempt miss rate: `26.8018% -> 25.9009%` — PASS/non-worse
- 8+ attempt miss rate: `39.1892% -> 38.7387%`

Bootstrap:

- mean QB-attempt MAE gain: `0.1167086426`
- P(gain > 0): `0.9670`

### Shared receiver diagnostic

Spearman between candidate team-pass-opportunity adjustment and 2024 WR reception-mass residual:

- `0.0857084541`
- frozen requirement: `>=0.10` — **FAIL**

This is the third failed advance gate.

## Frozen gate outcome

Passed:

- all integrity gates;
- pass-rate p90 non-worse;
- team-D p90 non-worse;
- QB-attempt MAE gain >=0.10;
- QB-attempt p90 non-worse;
- 10+ attempt miss rate non-worse;
- pass-rate correction directional Spearman >=0.10;
- all three bootstrap probability gates.

Failed:

1. pass-rate MAE gain `0.002494 < 0.0030`;
2. team-D MAE gain `0.1090 < 0.15`;
3. WR reception-mass coupling `0.0857 < 0.10`.

Therefore D1 does not advance and 2025 remains unopened for this candidate.

## Scientific meaning

Strict-prior QB designed-run burden contains genuine pass-intent information: it materially increases pass-rate correlation, improves all three physical opportunity metrics, and clears strong paired-bootstrap directional support.

However, the signal is too small under the exact one-for-one conservation mapping to clear the preregistered population-level effect-size gates, and it does not carry enough of the independent receiver-opportunity miss. The family therefore cannot be promoted, confirmed on 2025, or retuned.

Do not rescue this result by fitting a coefficient, changing the recent window, defining a mobile-QB subgroup, changing the weekly center, or combining it with previously failed schedule/rest or personnel candidates.

## Next research direction

The fixed 57% pass-opportunity-rate bottleneck remains unresolved.

The next authorized diagnostic should decompose the realized pass-opportunity rate into **football state occupancy that is distinct from the already-closed score-state M64/M65 family**. A promising next layer is down-and-distance / drive-state composition: how often an offense reaches obvious-pass states versus neutral choice states, and whether that composition is the missing shared QB/receiver opportunity mechanism.

Before any predictive experiment, audit M40-M42, M64-M65, M73 and later QB work specifically for down/distance state occupancy so this is not a renamed retest.
