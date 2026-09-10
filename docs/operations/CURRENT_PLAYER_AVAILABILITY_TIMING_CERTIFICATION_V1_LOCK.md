# Current Player Availability — Timing Certification V1 Lock

Status: `FROZEN BEFORE TIMING-CERTIFIER TEST EXECUTION`

Parent operational plan: `docs/operations/CURRENT_ROSTER_LATE_WEEK_ROLE_FIX_V1_FROZEN_PLAN.md`.
Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`.

Pinned implementation for first fixture execution:
- certifier: `scripts/build/certify_current_player_availability_timing_v1.py`
- certifier commit: `78436b80e4e1e6432fd0b435b78a898ea271b1ee`
- frozen fixtures: `tests/test_current_player_availability_timing_v1.py`
- fixtures commit: `cdbbc27f88e7ed72801925db860c47e855787b8d`
- workflow creation commit: `63388b51dfffbe5c729109ab969a8449203299a5`

## Purpose

Make the already-frozen game-window semantics executable without changing availability precedence or any predictive model.

## Frozen publication threshold

The official NFL Game Day Administration Report, which includes each club's inactive list, is delivered at the league's 90-minute pre-kickoff meeting. Therefore the V1 production-certification threshold is exactly **90 minutes before scheduled kickoff**.

For each scheduled team/game at an as-of timestamp:

1. `asof < kickoff - 90 minutes` => `NOT_YET_AVAILABLE`; official inactive coverage is not required and missing/invalid payload is not a failure.
2. `kickoff - 90 minutes <= asof < kickoff`:
   - complete validated official team section with snapshot timestamp <= kickoff => `CERTIFIED_OFFICIAL_SECTION`;
   - otherwise => `REQUIRED_MISSING_FAIL_CLOSED`.
3. `asof >= kickoff` => `POST_KICKOFF_NOT_PRICEABLE`; pregame pricing certification is closed.

A team section certifies only that team's exact scheduled game window. Endpoint reachability, another team's section, or an earlier/later game window cannot certify it.

A complete team section may contain zero listed inactive players and is still valid absence evidence. An incomplete section is never absence evidence.

## Frozen outputs

Team-game ledger fields:
- season, week, game_id when available
- team, opponent
- kickoff_utc
- official_required_from_utc
- asof_utc
- official_section_complete
- official_snapshot_utc
- certification_state
- priceable_now
- certification_reason

Summary sidecar:
- scheduled teams
- not-yet-required teams
- certified teams
- required-missing/fail-closed teams
- post-kickoff teams
- source endpoint reachability/payload validity retained as metadata only
- sportsbook inputs used = 0

## Frozen fixture cases

1. T-91 minutes + no section => NOT_YET_AVAILABLE / priceable under non-official lower authorities.
2. T-90 minutes + no section => REQUIRED_MISSING_FAIL_CLOSED.
3. T-30 minutes + complete section => CERTIFIED_OFFICIAL_SECTION.
4. T-30 minutes + incomplete section => REQUIRED_MISSING_FAIL_CLOSED.
5. Complete section for early-window Team A cannot certify later-window Team B before B's section exists.
6. As-of after kickoff => POST_KICKOFF_NOT_PRICEABLE.
7. Complete section snapshot after kickoff cannot provide pregame certification.
8. Sportsbook inputs = 0.

This lock changes no R26/R22/QB/WR/TE/RB predictive mechanics and does not wire production Full Slate.