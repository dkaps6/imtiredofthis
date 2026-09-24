# One-Pass-State Integration V1 — Historical Selector Amendment

Date: 2026-09-24

Status: **FROZEN BEFORE OUTCOME SCORING**

This is a methodological clarification to the already-frozen
`ONE_PASS_STATE_INTEGRATION_V1_PLAN.md`. It changes no candidate football
mechanism and is being recorded before any candidate outcome scoring.

## Why the amendment is required

The deployed `QB_DISTRIBUTION_STATE_SELECTOR_V1` JSON is the final production
fit trained on eligible historical data through 2025.

Applying that final-fit JSON retrospectively to 2024-2025 would allow target
games to influence their own selector coefficients through training and would
therefore violate the frozen no-target-game-outcome rule.

That is not permitted.

## Historical selector authority

Historical A/B scoring must use the original frozen Phase-J walk-forward
selector construction:

- source Phase-C all-row authority: run `34147777341`, artifact
  `10028332887`;
- source Phase-J authority: run `34151640191`, artifact `10029560958`;
- exact six features:
  1. `pass_opportunity_spot`
  2. `pass_efficiency_spot`
  3. `rush_opportunity_spot`
  4. `rush_efficiency_spot`
  5. `pred_qb_attempts`
  6. `week`
- exact `StandardScaler`;
- exact `Ridge(alpha=20.0)`;
- exact target:
  `actual_qb_attempts - pred_qb_attempts`;
- for each scored week, train only on chronologically earlier eligible QB
  team-games;
- require at least 128 training rows, otherwise emit selector delta = 0;
- exact route:
  `C2_SELECTED` iff predicted delta > 0.

## Reproduction gate

The reconstructed 2025 walk-forward selector must reproduce the preserved
Phase-J 2025 selector casebook before candidate scoring is accepted.

Required:
- identical 2025 team-game universe;
- identical selected/unselected decisions;
- max absolute predicted-delta difference <= `1e-9`.

If this reproduction gate fails, the experiment stops as a mechanical/provenance
failure.

## 2024 extension

The same already-frozen Phase-J algorithm is applied to 2024 in strict
walk-forward form using only chronologically earlier Phase-C QB rows.

Because Phase-C QB rows begin in 2024, early 2024 weeks that do not meet the
frozen minimum 128-row training requirement receive delta = 0 and therefore
remain on the canonical receiver state.

No threshold, minimum-row rule, feature, alpha, or selection rule is changed.

## Prospective production distinction

This historical fold-safe selector is used only to create leakage-safe
historical evidence.

For 2026 prospective/shadow operation, the actual deployed final-fit
`QB_DISTRIBUTION_STATE_SELECTOR_V1` remains the authority.

The candidate mechanism itself is unchanged:
when C2 is selected, receiver `receptions` and `rec_yards` use that exact
same C2 completed-pass realization; when C2 is not selected, canonical receiver
arrays remain untouched.
