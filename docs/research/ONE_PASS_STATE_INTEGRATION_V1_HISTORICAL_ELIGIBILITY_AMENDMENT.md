# One-Pass-State Integration V1 — Historical Eligibility Coverage Amendment

Date: 2026-09-24

Status: **FROZEN BEFORE OUTCOME SCORING**

This is a provenance/mechanical clarification recorded after the first historical
run stopped before scoring at 2024 Week 1. It changes no football parameter,
scientific gate, selector threshold, or candidate mechanism.

## Mechanical stop that exposed the issue

Run `36075299047` stopped before any candidate outcome scoring with:

`RuntimeError: 2024 W01 selector team count mismatch selector=31 football=32`

All upstream historical-input and authority-download steps had passed.

The cause is not a missing current football team. The preserved Phase-C / Phase-J
research authority did not contain an eligible QB selector row for every
historical team-game.

Examples:
- 2024 Week 1: 31 eligible QB rows; NYJ absent.
- preserved 2025 Phase-J state casebook: 440 eligible QB team-games total, not
  every regular-season team-game.

This is part of the original selector research eligibility universe and must not
be filled with fabricated decisions.

## Frozen historical eligibility rule

For historical ONE_PASS_STATE_INTEGRATION_V1 evaluation:

1. Reconstruct the exact original Phase-J walk-forward selector only on
   team-games present in the preserved Phase-C QB source.
2. The reconstructed 2025 eligible rows must still reproduce the preserved
   Phase-J state casebook exactly:
   - same eligible team-game universe;
   - max absolute delta difference <= `1e-9`;
   - zero selected/unselected decision mismatches.
3. Any football team-game with **no eligible Phase-C QB selector row** is
   fail-closed to:
   - `selector_c2_selected = false`;
   - canonical QB distribution;
   - canonical receiver arrays;
   - no candidate receiver replacement.
4. Missing selector eligibility is reported by season/week and pooled.
5. No missing row may be imputed from outcomes, nearby weeks, depth charts,
   sportsbook information, or the final-fit 2026 production selector.

## Why this is the correct conservative treatment

The candidate only acts when the frozen historical selector has legitimate
pregame authority to choose C2.

Inventing a historical selector decision for an ineligible row would expand the
experiment beyond the preserved Phase-J research universe and could introduce
outcome-informed routing.

Leaving those rows canonical does the opposite: it minimizes candidate scope and
preserves exact baseline behavior where historical selector provenance does not
exist.

## Scientific gates unchanged

Every scientific gate in
`docs/research/ONE_PASS_STATE_INTEGRATION_V1_PLAN.md` remains unchanged.

No outcome from ONE_PASS_STATE_INTEGRATION_V1 was inspected before this
clarification. The failed run produced no scorecard/result artifact.
