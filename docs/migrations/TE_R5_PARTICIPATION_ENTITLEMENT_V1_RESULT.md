# TE-R5 Participation Entitlement V1 — Result

## Status

**Disposition:** `TE_PARTICIPATION_ENTITLEMENT_V1_PASS`

This is a scientific pass of the frozen TE-R5 research candidate. It does **not** directly change production. It authorizes a separate frozen full-stack / production-integration confirmation for the TE entitlement mechanism.

## Exact lineage

- Branch: `research-te-r5-participation-entitlement-v1`
- Launch SHA: `999c29d543e6854a903c5a0a4ee6fecbe69dce61`
- Run: `34132127351`
- Job: `101774469114`
- Artifact: `10022512461`
- Artifact digest: `sha256:4f6d649492d2a08c4deeccd7944a4731e3d463e3f8d1bd40dcf8b9b82797af3d`
- OOS player-games: `3214`
- Test seasons: 2023, 2024, 2025
- Sportsbook inputs used: `0`
- Same/future participation observations used: `0`
- Join coverage: `1.0`
- Max team target mass gap: `1.7763568394002505e-15`

## Frozen candidate

TE-R5 modeled **relative individual TE entitlement** inside the finite TE target pool rather than applying a universal TE target correction. It used the exact frozen TE-R3/TE-R4 source lineage and a single inspectable `StandardScaler + Ridge(alpha=20.0)` candidate.

Feature family:

- current B0 TE room share;
- finite TE pool level / ratio / room size;
- strict-prior same-team and any-team offensive participation percentage;
- strict-prior offensive snap counts;
- prior history counts / availability;
- strict-prior room-relative snap-share features.

No post-result coefficient, blend, alpha, threshold or window search was performed.

## Pooled result

### Targets

- B0 MAE: `1.6825301068`
- TE-R5 MAE: `1.5412697533`
- Improvement: `0.1412603536`
- B0 RMSE: `2.4532070178`
- TE-R5 RMSE: `2.0942900516`
- B0 bias: `-0.9447466381`
- TE-R5 bias: `-0.1042076048`
- Correlation: `0.6137904077 -> 0.6483802567`
- p90 absolute target error: `4.0366412750 -> 3.3262610097`

### Receiving yards

- B0 MAE: `16.2679709448`
- TE-R5 MAE: `15.8580416890`
- Improvement: `0.4099292558`
- B0 RMSE: `23.4986170248`
- TE-R5 RMSE: `21.6200422044`
- B0 bias: `-6.6108343024`
- TE-R5 bias: `-0.3500159515`
- Correlation: `0.5117488366 -> 0.5365538499`
- p90 absolute receiving-yard error: `37.2272331958 -> 33.2847375938`
- 30+ yard miss rate: `13.8768% -> 12.4144%`
- 40+ yard miss rate: `8.7430% -> 6.5028%`

The candidate therefore improved both the center and the upper error tail while nearly eliminating the large pooled underprojection bias.

## Season replication

Receiving-yard MAE improved in **all 3/3 OOS seasons**:

- 2023: `16.4609205225 -> 15.6371514380` (`-0.8237690846`)
- 2024: `16.6027106593 -> 16.2840633437` (`-0.3186473155`)
- 2025: `15.7659008721 -> 15.6461205168` (`-0.1197803553`)

The maximum season regression was negative (`-0.1197803553`), meaning no OOS season regressed.

## Latest-era 2024-2025

- Receiving-yard MAE: `16.1783966305 -> 15.9605870948`
- RMSE: `23.4968812246 -> 21.7656277993`
- bias: `-6.6738892484 -> -0.2798557165`
- correlation: `0.5164713138 -> 0.5289639809`
- p90 absolute error: `37.4643846469 -> 33.6912313397`
- 30+ miss rate: `13.9408% -> 12.4829%`
- 40+ miss rate: `8.7472% -> 6.8337%`

## High-volume TE tier

On the high projected-volume Q4 subgroup (`n=804`):

- receiving-yard MAE: `24.0730761539 -> 22.2557042135`
- RMSE: `32.7318792521 -> 29.1478453403`
- bias: `-14.9653220392 -> -2.9763158155`
- p90 absolute error: `55.9225066458 -> 44.9475761717`
- 40+ miss rate: `18.1592% -> 14.9254%`

This is especially important because the largest TE projections were one of the areas where generic underprojection had been most damaging.

## Gate disposition

All frozen integrity gates passed:

- all three OOS seasons present;
- B0 values unchanged;
- duplicate rate zero;
- join coverage >= 0.90;
- OOS rows >= 3000;
- sportsbook inputs zero;
- finite team target mass preserved;
- zero same/future participation leakage.

All frozen scientific gates passed:

- target MAE improvement >= 0.08;
- target p90 guard;
- receiving-yard MAE improvement >= 0.25;
- receiving-yard p90 guard;
- 30+ and 40+ miss guards;
- receiving-yard wins >= 2/3 seasons (actual 3/3);
- latest 2024-2025 receiving-yard improvement;
- high-volume Q4 receiving-yard improvement;
- no single-season regression > 1.0;
- bias-magnitude guard.

## What this means

TE-R5 is the first dedicated TE candidate in the current lane that demonstrates the architecture we are trying to build:

`finite TE pool -> player-specific participation / role entitlement -> receiving outcome`

Unlike TE-R3, it is not simply a universal upward correction. The player-allocation layer is materially more differentiated, and the improvement survives through actual individual receiving-yard outcomes and error tails.

## Authorized next step

Freeze a separate **TE full-stack integration / confirmation** migration that:

1. embeds the TE-R5 entitlement mechanism into the canonical joint opportunity / receiving architecture;
2. preserves all non-TE production anchors unless explicitly being tested;
3. checks 2026 Full Slate feature availability and fail-closed behavior;
4. re-scores targets, receptions and receiving yards at the individual TE level;
5. checks interaction with shared QB passing opportunity and C2 conservation;
6. requires a fresh frozen production-promotion gate before any production change.

Do not directly promote TE-R5 from this result file.