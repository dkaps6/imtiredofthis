# RB R26B — Immutable Artifact Precheck Amendment V1

Status: **MECHANICAL SOURCE AMENDMENT BEFORE NEW OUTCOME SLICING**
Date: 2026-09-09
Parent artifact: `10106271075`
Parent digest: `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`

## Precheck finding

The frozen R26B forensic plan requested `prior_rb_room_share` as one diagnostic dimension. The authoritative parent R26 V1 prediction artifact does not persist that feature. It persists the fitted `r8_raw_residual`, R9 reliability/calibrated residual, baseline and candidate room shares, vacancy/continuity state, predictions, and outcomes, but not the underlying `prior_rb_room_share` feature value.

The safe-transition artifact likewise does not contain prior receiving-room target share; it contains roster/depth transition state only.

This was discovered during schema preflight before any new R26B outcome slice was successfully produced.

## Fail-closed decision

R26B will **not**:

- reconstruct `prior_rb_room_share` from target-game outcomes;
- re-download/rebuild historical receiving logs from a new external source;
- infer the feature from candidate results;
- substitute a different post-hoc variable;
- alter any R26 V1 model or prediction.

Frozen diagnostic dimension C.8 (`prior_rb_room_share` bands) is therefore marked **NOT_OBSERVABLE_FROM_IMMUTABLE_PARENT_ARTIFACT** and omitted from slice scoring.

Every other predeclared diagnostic dimension remains unchanged and must be reported.

## Scientific consequence

This amendment does not make the forensic test easier and does not alter any success criterion. It only documents that one internal model feature was not serialized by the parent run.

If the remaining immutable-artifact dimensions identify a coherent replicated role/allocation mechanism, R26B may still receive `FORENSIC_MECHANISM_IDENTIFIED` under the original forensic disposition language. Any future candidate that explicitly needs prior receiving-room share must first create a separately governed, timing-safe serialization/source step; it cannot borrow an unrecorded value from this atlas.

The original frozen plan remains preserved in commit history.