# RB R26I — Week-1 Selective Restoration V1 Result

Status: **WEEK1_SELECTIVE_RESTORATION_MIXED_NO_SHADOW**
Date: 2026-09-09
Production authority protected: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Valid execution head: `5d28fa3b3dbe19a5314524e15c4ae85999c02535`

## Authoritative lineage

Frozen plan commit: `7c8d07e472cc7225a2e84ffc53cf8b648bab84a5`
Initial workflow launch head: `c737f54045ec90c9ce6b69733b8186a3d453cc43`

### Mechanical first attempt
- run `34369824459`
- failed before scientific scoring because boolean masks were created before a pandas merge and no longer aligned with the post-merge index;
- frozen plan, protected-production boundary, and immutable-parent verification had already passed;
- no R26I metrics/gates were graded.

Mechanical repair only:
- commit `5d28fa3b3dbe19a5314524e15c4ae85999c02535`
- change: derive identical frozen boolean room masks after the R26H room-state merge;
- no child logic, thresholds, cohorts, metrics, or gates changed.

### Valid run
- workflow: `RB R26I Week 1 Selective Restoration V1`
- run `34370018586`
- job `102528540110`
- artifact `10111502087`
- artifact digest `sha256:5389c77d747cf1862e4423ea5d65519b7a1393897a89cce21b5615b38ab0b4b6`
- workflow conclusion: mechanically `success`
- scientific disposition: **`WEEK1_SELECTIVE_RESTORATION_MIXED_NO_SHADOW`**
- frozen gates passed: **22 / 27**

## Frozen child logic

R26I introduced no new fitted value. Every Week-1 RB room selected one complete immutable endpoint.

1. non-vacancy -> production baseline exact;
2. unbalanced vacancy -> original frozen R26 exact;
3. balanced turnover:
   - meaningful exit + veteran entrant -> original R26 exact;
   - meaningful exit + no-prior-NFL entrant -> original R26 exact;
   - unsupported balanced states -> production baseline exact.

The R26D meaningful-exit threshold was unchanged.

## Structural integrity

All structural/inheritance protections passed:
- R26 immutable parent structural integrity: pass;
- R26H authorized states verified: pass;
- no prediction regeneration: true;
- no R9 refit: true;
- sportsbook inputs added: 0;
- production parameters changed: false;
- receiving-yard means changed: false;
- R22 changed: false;
- non-vacancy child vs baseline max delta: 0;
- unbalanced vacancy child vs R26 max delta: 0;
- authorized balanced child vs R26 max delta: 0;
- unsupported balanced child vs baseline max delta: 0;
- max child RB-room target-mass gap: `5.329070518200751e-15`.

Room selections:
- authorized balanced restoration rooms: 61;
- balanced fallback rooms: 11;
- unbalanced vacancy rooms: 98.

## Pooled Week-1 result

Vacancy-active same-team incumbents, `n=246`.

### Receptions
Baseline:
- MAE `1.4244850719`
- RMSE `1.8993682539`
- bias `-0.8528102135`
- p90 abs error `3.3473247651`

Original R26:
- MAE `1.3122994738`
- RMSE `1.7411176321`
- bias `-0.4455093698`
- p90 `2.9753086015`

R26I:
- MAE `1.3283563090`
- RMSE `1.7748590281`
- bias `-0.4754570178`
- p90 `2.9753086015`

R26I therefore remained materially better than production baseline, but weaker than original R26 pooled.

### Targets
Baseline MAE `1.6878420656`
Original R26 MAE `1.5887600865`
R26I MAE `1.6156497984`

Again, R26I improved baseline but surrendered part of the original R26 gain.

## Role safety

RB1 (`n=95`):
- baseline reception MAE `1.8034912996`
- R26I `1.6371603796`

RB2+ (`n=151`):
- baseline `1.1860374452`
- R26I `1.1340756024`

Both role groups improved baseline.

## Global Week-1 safety

All RB rows `n=509`:
- baseline reception MAE `1.3126034423`
- original R26 `1.2352843870`
- R26I `1.2421773247`

R26I remained a strong global Week-1 improvement versus baseline but did not preserve original R26 within the frozen 0.5% preservation tolerance.

## Season results

2020 (`n=46`):
- baseline `1.3036397526`
- R26 `1.4205343030` (+8.97% harmful)
- R26I `1.4347169964` (**+10.05% harmful**)

2021 (`n=42`):
- baseline `1.8005896572`
- R26I `1.6314732382` (**9.39% better**), exact R26.

2022 (`n=41`):
- baseline `1.6702176650`
- R26 `1.6060504998`
- R26I `1.6144601219` (**3.34% better than baseline**).

2023 (`n=41`):
- baseline `1.1989315529`
- R26 `1.0959788089`
- R26I `1.1026020309` (**8.03% better than baseline**).

2024 (`n=45`):
- baseline `1.4988190216`
- R26 `1.2342229063`
- R26I `1.2938058166` (**13.68% better than baseline**, but materially weaker than R26).

2025 (`n=31`):
- baseline `0.9596503083`
- R26I `0.7301930376` (**23.91% better**), exact R26.

Five of six seasons remained improved. The unresolved regime is still 2020.

## Failed gates

Five frozen gates failed:

1. `16_no_w1_season_worsens_more_than_2pct`
   - 2020 worsened 10.05%.
2. `24_preserve_r26_pooled_inc_rec_mae_within_0p5pct`
3. `25_preserve_r26_pooled_inc_target_mae_within_0p5pct`
4. `26_preserve_r26_2021_2025_inc_rec_mae_within_0p5pct`
5. `27_preserve_r26_global_w1_rec_mae_within_0p5pct`

The remaining 22 gates passed.

## Scientific interpretation under component preservation

Do **not** discard original R26 Week-1 logic.

The evidence remains strong that R26 improves Week-1 RB receiving allocation broadly:
- pooled targets/receptions improve materially;
- RB1 and RB2+ both improve;
- all-RB Week-1 MAE improves;
- 5 of 6 seasons improve.

R26I shows that the R26H pooled role-state distinctions are not sufficient to repair the anomalous 2020 regime. In fact, restoring R26 in the R26H-supported balanced states increased 2020 harm while conservative fallback cost too much genuine signal in later seasons.

Therefore the next question is not another threshold/router tweak. It is whether **2020 Week 1 is structurally different in pregame football state or source semantics** from the other Week-1 seasons.

This must be investigated before any attempt to exempt 2020 or authorize a 2026 shadow.

## Authorized next step

R26I authorizes only a diagnostic/source audit, tentatively R26J:
- no prediction changes;
- no R9 refit;
- no new router;
- no shadow/production;
- compare 2020 Week-1 roster/room-transition structure against 2021-2025 using strict-prior/pregame-only state;
- inspect room continuity, entrant composition, number and summed significance of exits, established receiving identities, and any source/timing irregularity specific to the 2020 COVID offseason;
- freeze all dimensions before looking at R26 error slices.

Do not exclude 2020 from future qualification unless a separate predeclared comparability/governance study supports that conclusion.

No production files were changed by R26I.
