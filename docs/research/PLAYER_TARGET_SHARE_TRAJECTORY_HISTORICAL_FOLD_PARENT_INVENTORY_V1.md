# PLAYER TARGET SHARE TRAJECTORY — HISTORICAL FOLD PARENT INVENTORY V1

Date: 2026-10-08. Research-only; source/provenance inventory, **no historical target scoring, no new model fit, no production promotion**.

## Exact reconciliation to Gate 0

Gate-0 run 37780057649 correctly found that **current deployed FINAL FIT** TE-R5P and WR-R15 production model JSON assets each trained on 2022–2025. An exact 2023–2025 historical application of *those final fitted coefficients* is not OOS.

However, the repository also contains an already-frozen **SEPARATE fold-safe historical execution**:
- `docs/research/WR_TE_PRODUCTION_ORDER_HISTORICAL_REPLAY_V1_PLAN.md`
- `scripts/research/persist_wr_te_production_order_historical_v1.py`
- `.github/workflows/research-wr-te-production-order-historical-replay-v1.yml`
- `tests/test_wr_te_production_order_historical_v1.py`

This existing code uses **earlier model fold coefficients**, rather than the final deployed coefficients. It is **not disallowed by Gate 0**, because it is not the same parent model/version under examination. Preserve this distinction: no rebranding of final-fit models as historical OOS and no dismissal of the separate properly fold-trained sources.

## Exact fold scope

| Test season | TE-R5P parent | WR-R15 parent | What this means |
|---|---|---|---|
| 2023 | Original OOS fold was listed in historical authority artifacts, but no production-order full-stack replay contract here | 2022 -> 2023 fold in original WR authority | **No verified exact combined production-order historical baseline** on this branch. Do not invent one. |
| 2024 | TE-R5P production-contract **test-2024 fold** | WR-R15 **train-2023 -> test-2024 fold** | Supported model-parameter lineage for a bounded WR+TE historical production-order test, **subject to recovering exact TE coefficients and validating pregame/source parity and prior trajectory state**. |
| 2025 | TE-R5P production-contract **test-2025 fold** | **Not authorized**; original WR-R15 validation contract forbids 2025 | TE-only historical integration at most; cannot promote combined WR+TE test under this fold as a complete 2025 candidate. |

The historic code requires exact pool conservation, M38 WR1 anchoring, strict-prior participation, authorized source, baseline array parity. Reuse rather than rewriting it.

## Primary fold authorities, current GitHub retention evidence

1. WR-R15 original OOS authority: run `34238301577`, artifact `10061328722`, name `wr-r15-wr1-anchor-participation-v1`, original ZIP digest `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`. On 2026-10-08 it was still downloadable, with expiration `2026-10-08T14:38:25Z`.
2. The original artifact was **successfully recovered and extracted unchanged**. Historical fold coefficients (test seasons 2023/2024) archived in this branch at `docs/research/authority_snapshots/wr_r15_fold_coefficients_original_10061328722.csv`. Original file length **3600 bytes**, SHA256 `e0c95ed6a302495e4b2b06eecc4d9e2e5a66882d49393d853cfa70b337d4dab6`; dedicated test `tests/test_wr_r15_fold_archive_provenance_v1.py` asserts exact bytes and train/test pairs.
3. TE-R5P production-contract fold authority: run `34152797603`, artifact `10029942404`, original archive digest `sha256:f9951441b748ef72514dbc81adf6fbe9cd023c9bf64ecb52a016840989ab4cdb`. **Artifact list on October 8 is empty**. Do not claim exact TE fold coefficients are currently recovered or substitute 2022–2025 final-fit JSON.
4. Previously successful combined production-order replay: run `34722725629`, compact artifact `10307242156` (`sha256:d5a991bd76df5b053e6411e9873b12bdaada1458592c2586e3c1416c6fe37044`), raw-draw artifact `10306649017` previously expired. **No artifacts currently listed for run `34722725629`**. The frozen code/plan persists in GitHub; historical target-grade rows and replay output artifacts are not currently verifiable from those expired artifacts.
5. Existing WR-R15 model JSON may say `SUPPORTED_OOS_AUTHORIZED_FOR_PRODUCTION_INTEGRATION`, but the *final-fit coefficient vector* contains 2023–25 training data and must not be confused with the OOS fold coeff vectors.

## Exact next action and hard gates

- No 2026 W1–W4 trajectory eligibility relaxation: four prior same-season team games.
- Locate TE-R5P exact fold coeffs from another provably identical artifact or immutable commit. If not found, record **SOURCE_ARTIFACT_UNAVAILABLE**, not a retrospective V1 performance failure.
- Ensure historical source roster ACT-only parity; the 2026 W1–W4 corrected replay established a historical `ACT + INA` problem and must not be silently reused.
- Only with original or source-parity-verifiable folds, reconstruct the parent **as-of** baseline *before* outcome scoring, then apply the exact Week-5 frozen target-share trajectory formula inside TE room and WR2+ while leaving WR1, all room pools, team target mass and efficiency untouched.
- 2024 historical comparison is at most a **separate, source-verified, OOS-fold parent** test, not an exact 2026 final-fit model replay. Its results do not displace the immutable four-week/400-player-game/120-identity/80-room **prospective** trajectory acceptance.
- If TE folds cannot be recovered, prioritize the sealed 2026 prospective trajectory route; do not refit specialists, silently resurrect expired combined metrics, or run a generic same-data share-rescue search.

## Related operational issue (independent)

Week-5 Full Slate automatic run `37778827288` failed because QB C2 state-context builder insists on 32 active teams while authoritative bye schedule has 30. Fix needs versioned C2 source-integrity/availability-seam reconciliation, not naive numeric replacement. Tracked separately as [P0 #673](https://github.com/dkaps6/imtiredofthis/issues/673); no production edit in this research branch.

Disposition: `OOS_FOLD_PARENT_DISCOVERED__WR_FOLD_ORIGINAL_RECOVERED__TE_FOLD_ARCHIVE_MISSING__COMBINED_REPLAY_ARTIFACT_EXPIRED__ASOF_RECONSTRUCTION_GATED`.
