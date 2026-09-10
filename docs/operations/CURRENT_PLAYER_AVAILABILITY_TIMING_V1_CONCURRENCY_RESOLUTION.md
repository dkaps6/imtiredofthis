# Current Player Availability Timing V1 — Concurrent Freeze Resolution

Status: `CONFLICT RESOLVED BEFORE FIRST TIMING TEST / T-75 PLAN CANONICAL`

Two automation workers wrote conflicting timing contracts to the same implementation branch within seconds. Neither timing implementation had been tested when the conflict was discovered.

## Chronology

Canonical first freeze:
- file: `docs/operations/CURRENT_PLAYER_AVAILABILITY_TIMING_V1_FROZEN_PLAN.md`
- commit: `905f1bbe55d51676587d941295d357a1d2c31e9b`
- commit time: `2026-09-10T04:28:18Z`
- threshold: T-75 minutes
- rationale: NFL inactive list is delivered at the T-90 meeting; fixed 15-minute publication/ingestion allowance before fail-closed certification begins.

Conflicting later concurrent freeze:
- file: `docs/operations/CURRENT_PLAYER_AVAILABILITY_TIMING_CERTIFICATION_V1_LOCK.md`
- creation commit: `106113bf61a189f91546628d79afa37d30daec29`
- commit time: `2026-09-10T04:28:39Z`
- threshold: T-90 minutes
- later implementation commit: `78436b80e4e1e6432fd0b435b78a898ea271b1ee`
- later fixture commit: `cdbbc27f88e7ed72801925db860c47e855787b8d`
- later pin commit: `3219f584092946d9be81f8b30c2abd2b5f4b4019`

The T-90 freeze did not reference, formally supersede, or justify changing the already-frozen T-75 contract. It was concurrent conflicting work, not an authorized amendment.

## Resolution rule

Under the repository's anti-retuning / freeze-before-results discipline, the first complete frozen contract controls unless a separately documented pre-result amendment explicitly identifies and supersedes it for a non-result-driven reason.

Therefore:
- T-75 is canonical for V1;
- the T-90 lock, certifier and fixtures are preserved in Git history as superseded concurrent drafts;
- no result from T-90 code may be treated as V1 evidence;
- executable validator/tests/workflow must be aligned exactly to the T-75 frozen plan before the first timing test;
- no threshold change is permitted after timing test results are observed.

No production code, model, R26, R22, Full Slate or sportsbook path changes in this resolution.
