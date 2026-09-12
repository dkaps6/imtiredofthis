# Fair Probability × Heldout Weight Interaction V1 — Result

Status: **RESEARCH ONLY — NO PRODUCTION/MODEL/WEIGHT/THRESHOLD CHANGE**

Canonical run: `34717930355`

Canonical head: `40a96a01a997ddf5ca7286a34fec1bfb651d0dca`

Artifact: `fair-prob-weight-interaction-v1`, ID `10305426871`, digest
`sha256:dab24ede566449e9241c1bdfd2d029aa3498691b4acdb7b04152b3e0073891b6`.

## Frozen cells

- A0 = current/fallback means × legacy `component_sd` Normal translator.
- A1 = PR #545 published heldout means × legacy translator.
- B0 = current/fallback means × empirical historical MC translator.
- B1 = PR #545 published heldout means × empirical historical MC translator.

All four cells use the same identity-clean 2024-25 rows, the same historical
sportsbook archive, the same decision gates, and the same 2,000-draw historical
MC arrays in the empirical arms. A0/B0/A1 reproduction gates passed before the
B1 result was accepted.

### A1 provenance note discovered by the reproduction gate

PR #545's published A1 summary used the committed six-decimal rounded heldout
weights and a fixed weighted sum where a missing component contributes zero
without renormalizing the remaining weights. This differs from production
`apply_ensemble()` missing-component semantics. The interaction preserves the
published A1 object exactly rather than silently redefining it after the fact.
The limitation is emitted in `published_a1_application_semantics_audit.csv`.

## STRONG-tier result

| Market | A0 legacy/current | A1 legacy/heldout | B0 empirical/current | B1 empirical/heldout |
|---|---:|---:|---:|---:|
| rec_yards ROI | -3.364% | -3.128% | **-2.433%** | -3.465% |
| rec_yards MAE | 21.603 | **20.810** | 21.856 | 21.348 |
| receptions ROI | -0.868% | **-0.795%** | -1.052% | -2.167% |
| receptions MAE | 1.707 | **1.597** | 1.725 | 1.606 |
| rush_rec_yards ROI | -4.424% | -6.096% | **-4.328%** | -4.971% |
| rush_rec_yards MAE | 33.040 | **30.456** | 33.485 | 30.769 |
| ALL_MARKETS ROI | -2.766% | -2.920% | **-2.311%** | -3.112% |
| ALL_MARKETS MAE | 18.482 | **17.941** | 18.525 | 18.394 |

STRONG row counts also move materially because probability translation and mean
changes alter the frozen gate membership: A0 16,322; A1 15,601; B0 15,324; B1
13,844.

## Primary frozen question: rush_rec_yards

The translator repair **does recover part of the A1 ROI damage**:

- A1: -6.0963%
- B1: -4.9715%
- B1 minus A1: **+1.1248 percentage points**

But it does **not** rescue the heldout-weight arm:

- B0: -4.3278%
- B1: -4.9715%
- B1 minus B0: **-0.6437 percentage points**

So the known-bad legacy translator amplified some of the apparent
`rush_rec_yards` damage, but it was not the whole explanation. The ROI remains
negative and does not flip positive; heldout weights are still worse than the
current/fallback mean arm under the better empirical translator.

## Other affected markets

The interaction is unfavorable for ROI despite better point accuracy:

- `rec_yards`: B1 vs A1 ROI **-0.337pp**; B1 vs B0 **-1.032pp**.
- `receptions`: B1 vs A1 ROI **-1.371pp**; B1 vs B0 **-1.114pp**.
- `ALL_MARKETS`: B1 vs A1 ROI **-0.192pp**; B1 vs B0 **-0.801pp**.

This is not because the heldout means lose their point-accuracy signal. B1 still
improves MAE versus B0 in each affected market:

- rec_yards: 21.856 -> 21.348
- receptions: 1.725 -> 1.606
- rush_rec_yards: 33.485 -> 30.769

## Probability quality

B1 has the best aggregate Brier/log-loss of the four cells:

- A0: Brier 0.3549 / log loss 1.4409
- A1: 0.3324 / 1.2686
- B0: 0.3066 / 0.9005
- B1: **0.2815 / 0.7898**

B1 likewise improves Brier/log loss versus B0 in all three heldout-weight
markets. Better point accuracy and better probability calibration therefore do
**not** automatically create profitable selection under the current historical
base stack and frozen betting gates.

## Disposition

1. **Translator finding remains REPRODUCED:** `component_sd` was the wrong outcome
   variance object and materially degraded calibration/selection.
2. **Heldout-weight point-accuracy signal remains real:** 2023-only weights improve
   raw OOS means in the three previously-unweighted markets.
3. **Full ROI-rescue hypothesis is FALSIFIED:** combining the two fixes does not
   produce a positive or even uniformly improved betting result. The
   `rush_rec_yards` regression is partially reduced but survives; aggregate B1
   is worse than B0.
4. This remains a **base-stack historical interaction**, not a verdict on the
   current full 2026 production architecture. Promoted WR-R15/TE-R5P in actual
   production order, historically legitimate M89/M90, and reproducible
   C2/distribution routing remain the next fidelity layers.

No production authority changes from this experiment.
