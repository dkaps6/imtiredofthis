# WR/TE Production-Order Historical Replay V1 — Frozen Plan

Status: **RESEARCH ONLY — FROZEN BEFORE RESULT INSPECTION**

Canonical freeze: Issue #535, GPT-5.6 checkpoint 18 (`5649047009`).
Base `main` SHA: `9daebe1b075a3d8c71380316f13e263338e55ad5`.

## Question

Does replaying the already-promoted WR-R15 / TE-R5P entitlement logic **upstream of the historical joint Monte Carlo**, in the same order used by production, improve the identity-clean 2024-2025 historical benchmark relative to the certified base empirical-MC reconstruction?

This is not a new football model. It is a fidelity reconstruction of already-authorized entitlement mechanisms under fold-safe historical parameters.

## Production order

The replay preserves:

`historical pregame state -> Bayesian/rules -> M38 -> explicit finite target entitlement -> TE-R5P -> WR-R15 (authorized season only) -> joint MC -> frozen ensemble mean -> empirical fair probability -> sportsbook comparison`

Sportsbook lines/odds are forbidden before the final comparison.

## Fold-safe model authority

### TE-R5P

Use the production-contract OOS artifact, not the final all-history 2026 coefficients:

- run `34152797603`
- artifact `10029942404`
- name `te-r5p-production-contract-refit-v1`
- digest `sha256:f9951441b748ef72514dbc81adf6fbe9cd023c9bf64ecb52a016840989ab4cdb`

The artifact is the `pool_ratio=1.0` / conserved-existing-TE-room version that matches production semantics. Use the exact fold scaler means/scales/Ridge coefficients/intercepts for test 2024 and test 2025.

### WR-R15

Use:

- run `34238301577`
- artifact `10061328722`
- name `wr-r15-wr1-anchor-participation-v1`
- digest `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`

Use the exact `train_season=2023 -> test_season=2024` fold parameters.

**Do not apply WR-R15 to 2025.** Its frozen science explicitly forbids 2025 confirmation. 2025 WR remains the M38 explicit baseline and is labeled untreated.

## Historical football reconstruction

For each week:

1. Build the exact historical context used by the canonical walk-forward.
2. Rebuild the canonical market-expanded MC metrics and base simulation under `iterations=2000`, `seed=42+week`.
3. Reduce to the exact one-player/game/team state consumed internally by `simulation_v2` (`sort + drop_duplicates(keep='last')`).
4. Materialize `TEAM_TARGET_ENTITLEMENT_V1_PROJECTION_NEUTRAL`, which lifts the existing M38 allocation outside the simulator.
5. Require exact array parity (key universe, shape, mean, element values <=1e-12) between legacy simulation and explicit-entitlement baseline before applying specialists.
6. Apply TE-R5P with the historical test-season fold parameters and strictly-prior participation features.
7. Apply WR-R15 after TE-R5P for 2024 only, with M38 WR1 anchor frozen and WR2+ pool conserved.
8. Run `simulation_explicit_entitlement_v1` with the same iterations/seed.

## Entitlement integrity gates

All must pass:

- TE room target mass preserved <=1e-10;
- TE non-TE entitlement preserved <=1e-10;
- WR1 anchor preserved <=1e-10;
- WR2+ pool preserved <=1e-10;
- total WR room mass preserved <=1e-10;
- WR non-WR entitlement preserved <=1e-10;
- modeled-player target mass / residual bucket preserved <=1e-10;
- same/future participation usage = 0;
- 2025 WR specialist applications = 0.

## Hybrid distribution artifact

To isolate the authorized specialist effect without contaminating unchanged players with irrelevant Monte Carlo draw-order differences:

- use specialist-simulation arrays for authorized receiving rows only;
- use canonical legacy arrays for every untreated row;
- save one complete distribution lineage covering the same rows as the clean empirical benchmark.

Authorized rows:

- markets: `rec_yards`, `receptions`, `rush_rec_yards`;
- 2024: WR + TE;
- 2025: TE only.

This hybridization is valid for single-player prop grading; it is **not** an assertion of a coherent cross-player portfolio covariance matrix.

## Mean / probability semantics

- Replace only `mc_proj` on authorized rows with the specialist-array mean.
- Keep ML/state byte-identical.
- Keep `data/model_ensemble_weights.csv` unchanged.
- Rebuild final football mean through existing `apply_ensemble()` semantics.
- Mean-align each raw MC array to the final football mean using production multiplicative scaling.
- Fair probability is empirical `P(outcome > line)` from the aligned array.
- No PR #545 research weights.
- No PR #548 widening factor is transported. Re-measure specialist distribution dispersion from scratch.

## Baseline reproduction gates

Before specialist science is interpreted:

- untouched baseline component / projection identities must match the canonical fair-probability reconstruction;
- legacy -> explicit-entitlement simulation must be exactly projection neutral;
- untreated rows in the final specialist grade must reproduce baseline football mean and empirical probability exactly (<=1e-12 where floating operations are identical, otherwise hard fail above 1e-10);
- all 17,715 historical graded row identities must remain unchanged.

## Evaluation

On exact same rows report:

- mean MAE and signed bias;
- empirical Brier / log loss;
- mean model SD, realized residual SD, and ratio;
- STRONG coverage;
- win rate / flat-unit ROI with unchanged gates;
- season / market / position / side diagnostics;
- whole-cohort descriptive summary;
- primary specialist-authorized cohort summary.

Primary inference is on the authorized cohort, not on post-hoc slices.

## Stopping / interpretation

- Negative results are preserved.
- No threshold, parameter, cohort, fold, weight, or widening-factor search after result inspection.
- V1 cannot be called a full current-2026-stack historical replay because 2025 WR-R15 is intentionally absent and QB C2/M89 plus 2026-only RB layers are outside this test.
- Any mechanical/source failure may be repaired without changing the frozen scientific contract.
