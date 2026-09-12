# Fair Probability × Heldout Weight Interaction V1 — Frozen Plan

Status: **FROZEN BEFORE B1 RESULT**

Branch: `research-fair-prob-weight-interaction-v1`, cut from `main` at
`b7bb2eb5d81691a43fbdb3d711735fe21fe5ab95` after PR #546 merged.

This is a pure 2×2 interaction of two independently completed research
ablations. Nothing is refit, resimulated, threshold-tuned, or promoted.

## Cells

- A0: current/fallback ensemble weights × legacy `component_sd` Normal translator.
- A1: 2023-only heldout weights for `rec_yards`, `receptions`, `rush_rec_yards`
  × legacy translator.
- B0: current/fallback weights × empirical historical MC translator.
- B1: same frozen heldout weights for the same three markets × empirical translator.

## Frozen artifact provenance

- Historical identity-clean projection/market/distribution artifact: workflow run
  `34712931786`, artifacts `historical-fair-probability-reconstruction-v1` and
  `historical-simulated-outcome-shards-v1`.
- 2023-only heldout weights: workflow run `34715724032`, artifact
  `ensemble-weight-2023-fit-v1`.
- A1 canonical summary: `docs/research/overnight/ensemble_weight_2023_holdout_summary.csv`.

## Fail-closed pre-result gates

1. Exact ALL_NO_FILTER row identity across A0/A1/B0/B1.
2. A0 reproduces the prior legacy summary.
3. B0 reproduces the prior empirical summary.
4. A1 reproduces the prior heldout-weight legacy summary.
5. Heldout weights alter only the three previously-unweighted markets.
6. Empirical distribution lineage remains exact and every row uses 2,000 draws.

## Frozen readout

By market and aggregate report MAE, Vegas MAE, Brier, log loss, STRONG coverage,
win rate, units, ROI, side counts, and interaction deltas.

Primary question: whether the `rush_rec_yards` ROI regression under A1 survives
or improves/reverses in B1 once the known-bad probability translator is removed.

No production/model/weight/threshold change is authorized by this experiment.
