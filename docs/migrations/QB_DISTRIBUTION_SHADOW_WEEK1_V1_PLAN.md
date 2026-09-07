# QB Distribution Shadow Week-1 V1 — Frozen Deployment Plan

## Purpose

Validate the deployable Phase-J QB distribution selector inside the actual 2026 Week-1 Full Slate data path without changing production pricing.

This is a deployment/parity test, not a new scientific feature search and not a mean-model migration.

## Frozen authorities

- QB point mean: `QB_PASS_SYNTHESIS_V1` / M89-M90 authority.
- QB distribution-state selector: `QB_DISTRIBUTION_STATE_SELECTOR_V1` from Phase J.
- Candidate QB distribution: frozen C2 QB conservation distribution only.
- WR: M38 authority unchanged.
- TE: TE-R5 authority unchanged.
- RB rushing: RB-P3 Week-1 authority unchanged.
- Sportsbook information is downstream audit only and may not enter the selector or football distribution construction.

## Exact deployment test

1. Launch the existing Full Slate workflow for 2026 Week 1 with live odds enabled only so the current production pricing artifact is materialized.
2. Do not modify `scripts/run_pricing_v2.py` or `scripts/simulation_v2.py`.
3. Download the exact Full Slate run artifact.
4. Rebuild strict-prior Phase-J team environment context from completed historical PBP and the authoritative 2026 team-week map.
5. Reconstruct the canonical QB Monte Carlo from the exact `data/model_rule_simulation_inputs.csv` produced by Full Slate using `simulate_with_states`, which must retain byte parity with `simulation_v2` when C2 is off.
6. Convert the reconstructed QB pass-opportunity distribution using the exact `qb_attempt_conversion` and `qb_pass_att_share` written by production pricing.
7. Verify the reconstructed pre-synthesis MC mean equals the production `mc_proj`.
8. Anchor both the canonical shadow distribution and any selected C2 distribution to the exact production `model_proj` / `qb_synthesis_proj` mean.
9. Apply `QB_DISTRIBUTION_STATE_SELECTOR_V1` using only:
   - pass opportunity spot
   - pass efficiency spot
   - rush opportunity spot
   - rush efficiency spot
   - promoted predicted QB attempts
   - week
10. Expose the C2 QB array in the shadow sidecar only when the selector predicts a positive pass-attempt delta. Otherwise retain the canonical shadow distribution.
11. Never write the selected shadow distribution back into `outputs/props_priced_clean.csv` in this migration.

## Frozen gates

All must pass for `QB_DISTRIBUTION_SHADOW_WEEK1_PASS`:

1. Real-slate reconstructed `mc_proj` parity: maximum absolute gap <= `1e-8`.
2. Canonical shadow mean anchor: maximum absolute gap to promoted M89/M90 mean <= `1e-8`.
3. Selected shadow mean anchor: maximum absolute gap to promoted M89/M90 mean <= `1e-8`.
4. Selector activates for at least one priced QB.
5. Every priced QB distribution is covered by the shadow audit.
6. Selector sportsbook inputs are exactly zero.
7. Production pricing is not modified by the shadow evaluator.

No gate may be relaxed after results.

## Outputs

- `outputs/qb_distribution_shadow_v1.csv`
- `outputs/qb_distribution_shadow_v1_result.json`

The sidecar records player/team/opponent, Phase-J state inputs, predicted attempt delta, selector decision, canonical and selected distribution moments/quantiles, exact mean/parity gaps, and downstream-only probability deltas versus the posted line when available.

## Interpretation

- PASS authorizes a subsequent explicit production-integration migration for the QB distribution selector. It does not itself modify production.
- FAIL stays failed. Repair only mechanical data/path/parity defects; do not tune selector coefficients, C2 distribution parameters, thresholds, or gates based on this Week-1 result.
