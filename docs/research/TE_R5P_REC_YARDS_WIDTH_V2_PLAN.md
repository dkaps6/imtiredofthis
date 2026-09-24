# TE-R5P RECEIVING-YARDS WIDTH V2 — FROZEN PLAN

**Status:** FROZEN BEFORE V2 EXECUTION  
**Branch:** `research-te-live-entitlement-efficiency-v1`  
**Parent production main:** `69e9c6211710c60de9aaf07d06aca6cafeff5ccf`  
**Production mutation from this run:** NONE

## Why this is the next question

TE Live Entitlement vs Efficiency V1 established on the exact paid Week-2
production trace that:

- strict-prior Week-1 2026 snaps improve Week-2 all-TE target-share MAE by
  **2.32%** and TE-room-share MAE;
- the canonical selected Week-2 TE receiving-yard cohort improves target-share
  MAE by **2.84%** under that current-snap counterfactual;
- nevertheless receiving-yard point MAE does not improve when only entitlement
  changes;
- perfect target entitlement recovers **4.11 yd** of selected-cohort absolute
  error on average, while perfect realized efficiency recovers **5.68 yd** on
  nonzero-target rows;
- raw MC was slightly closer than the final blend (20.63 yd vs 20.87 yd).

Historical TE-R1 independently assigned ~45% of TE error mass to targets and
~55% to catch-rate + YPR efficiency.

A separate, already-completed production-order historical replay (PR #549,
run `34722725629`, compact artifact `10307242156`) shows that TE-R5P
improves TE receiving means but the specialist receiving distributions remain
too narrow.

On the exact TE × receiving-yard rows from that artifact:

| season | arm | n | MAE | mean model SD | realized residual SD | implied k |
|---|---|---:|---:|---:|---:|---:|
| 2024 | TE-R5P specialist | 682 | 19.819 | 14.781 | 25.545 | **1.728** |
| 2025 | TE-R5P specialist | 671 | 19.219 | 14.957 | 25.250 | **1.688** |

The under-width is both material and stable across seasons. This V2 tests that
specific mechanism without touching the mean.

## Scientific question

> If we estimate one TE receiving-yard width multiplier from one historical
> season using football outcomes only, does applying it unchanged to the other
> season improve the fold-safe TE-R5P specialist distribution out of sample?

Both directions are required:

- fit 2024 -> blind test 2025;
- fit 2025 -> blind test 2024.

No factor search is allowed.

## Exact authority

Reconstruct the already-certified PR #549 production-order replay:

- historical replay run: `34722725629`;
- compact artifact: `10307242156`;
- digest:
  `sha256:d5a991bd76df5b053e6411e9873b12bdaada1458592c2586e3c1416c6fe37044`;
- replay code/lineage remains unchanged;
- TE-R5P fold-safe OOS authority:
  run `34152797603`, artifact `10029942404`;
- TE treatment is authorized in both 2024 and 2025;
- WR-R15 behavior remains whatever the frozen replay requires, but V2 grades
  **TE + rec_yards only**.

The expired raw distribution artifact is not substituted with an approximation.
The exact replay is rerun deterministically to regenerate its 2,000-draw
specialist arrays under the frozen PR #549 contract.

## Candidate construction

For fit-season TE `rec_yards` rows only:

1. mean-align each exact specialist MC array to the frozen final football mean
   using production's existing multiplicative alignment semantics;
2. compute each row's sample SD;
3. compute fit-season:
   `k = SD(actual - final_projection) / mean(row_MC_SD)`;
4. freeze that one scalar k.

For the other season:

`widened = final_mean + k * (aligned_draw - final_mean)`

Then re-anchor to the exact final mean at floating precision.

No lower/upper clipping is introduced beyond whatever the already-replayed
football distribution itself contains; the scientific variable is width only.

## Primary football-only evaluation

On every exact TE receiving-yard row in the blind test season:

- empirical CRPS;
- 80% interval coverage and absolute coverage gap from 80%;
- 90% interval coverage and absolute coverage gap from 90%;
- average interval width;
- point MAE (must be identical);
- maximum absolute mean shift.

CRPS is computed directly from the empirical 2,000 draws.

## Secondary downstream calibration evaluation

Only after k is frozen from football outcomes in the fit season:

- join the already-existing historical market archive;
- compute empirical `p_over` from base vs widened arrays;
- Brier score and log loss on line outcome;
- STRONG coverage / ROI are descriptive only.

Sportsbook fields never enter k, the projection mean, or the draw transform.

## Frozen qualification gates

The exact V2 candidate qualifies as a **future-only distribution-calibration
candidate**, not an automatic production promotion, only if:

1. point MAE is unchanged to <=1e-10 in both blind directions;
2. max absolute row mean shift <=1e-8 in both directions;
3. mean CRPS strictly improves in **both** blind directions;
4. 80% absolute coverage gap improves in both directions;
5. 90% absolute coverage gap improves in both directions;
6. pooled blind Brier is non-worse;
7. pooled blind log loss is non-worse;
8. no sportsbook field is used in fitting k;
9. all PR #549 replay/conservation/fold-authority gates pass.

If any primary football-only gate fails, this exact width candidate closes.
There is no k search or rescue.

## Future Week-3 factor if V2 qualifies

Qualification does not itself edit production.

If both blind directions qualify, the predeclared future-only factor to evaluate
for Week 3+ integration is the same formula fit once on the pooled 2024+2025
fold-safe TE-R5P specialist rows:

`k_future = pooled residual SD / pooled mean row MC SD`.

That value may be carried into a separate production-integration validation,
but it is never fit from 2026 Week-1/Week-2 outcomes.

## Explicit exclusions

- no TE entitlement coefficient refit;
- no current-2026 efficiency fitting;
- no sportsbook line in football construction;
- no global receiving-yard factor across WR/RB/TE;
- no reuse of PR #548's base-stack `rec_yards` k;
- no threshold/edge-gate tuning;
- no QB change;
- no RB M96 retrospective reopening.
