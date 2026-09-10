# Current Player Availability Timing V1 — Run1 Mechanical Failure

Status: `PRESERVED_MECHANICAL_FIXTURE_EXPECTATION_FAILURE / NO SEMANTIC CHANGE`

- run: `34437394807`
- job: `102745226871`
- head: `3219f584092946d9be81f8b30c2abd2b5f4b4019`
- protected-production boundary: PASS
- timing fixture step: FAIL

## Cause

The cross-window fixture used as-of `19:00Z` for a later game kicking at `20:25Z`. The frozen T-90 threshold for that game is `18:55Z`, so at `19:00Z` the later team is already inside the official-inactive-required window. The test incorrectly expected `NOT_YET_AVAILABLE`; the frozen design requires `REQUIRED_MISSING_FAIL_CLOSED` when no complete section exists.

The implementation behavior is consistent with the frozen T-90 rule. This is a fixture expectation defect, not a semantic/design failure.

## Minimum value-neutral repair

Change only the later-window expected state in `test_early_window_cannot_certify_late_window` from `NOT_YET_AVAILABLE` to `REQUIRED_MISSING_FAIL_CLOSED`. Preserve the assertion that an early-window team's complete section cannot certify another team/window.

No production code, availability precedence, T-90 threshold, R26, R22, or predictive model changes are authorized.