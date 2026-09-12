# Historical Fair-Probability Reconstruction V1 — Frozen Plan

Status: **FROZEN BEFORE RESULT**

Purpose: isolate the historical betting-translation layer without changing
football means, ensemble weights, sportsbook thresholds, or model science.

## Contract

1. Football means remain upstream and frozen. Sportsbook lines/prices enter only after a football distribution exists.
2. Target-game outcomes are grading-only and never construct the target-row simulation.
3. Fair probabilities come from empirical simulated outcome arrays where the historical simulation is reproducible.
4. The simulated distribution is rescaled to the frozen football mean using the same non-negative multiplicative mean-alignment semantics as production pricing.
5. C2 is not called production-equivalent unless its historical routing can be reconstructed honestly; this V1 is the base historical MC distribution reconstruction.
6. The experiment is a same-row A/B: legacy `component_sd` Normal translator versus empirical MC translator on identical identity-clean rows.
7. Existing PLAY/LEAN/STRONG gates remain unchanged. No threshold tuning is allowed in this pass.
8. Missing or inconsistent distribution lineage fails closed; there is no invented fallback sigma.
9. WR-R15 / TE-R5P remain outside the final probability claim until replayed upstream of historical simulation in production order.
10. RB P3/R26/R22 are not backported to 2024-25 under this study.

## Measurement

Report, by market and aggregate where meaningful:

- matched rows and exact same-row parity;
- probability calibration bins;
- Brier score and log loss for OVER probability on non-push outcomes;
- STRONG coverage;
- win rate, units, and ROI under frozen gates;
- empirical simulated SD versus realized residual dispersion;
- season / market / side diagnostics from the emitted detail.

The reconstructed raw arrays are emitted as a temporary workflow artifact so
Claude/GPT can independently audit the exact probability calculations.

No production/model/weight/threshold change is authorized by this study.
