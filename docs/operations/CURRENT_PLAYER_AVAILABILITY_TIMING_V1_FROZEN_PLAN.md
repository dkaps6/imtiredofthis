# Current Player Availability Timing Certification V1 — Frozen Plan

Status: `FROZEN BEFORE TIMING VALIDATOR TESTS / PRODUCTION UNWIRED`

## Parent authority

- Frozen roster/late-week fix architecture: `docs/operations/CURRENT_ROSTER_LATE_WEEK_ROLE_FIX_V1_FROZEN_PLAN.md`
- Fix-plan commit: `b2206e7ad693148623447bcf9a3ad6b594033500`
- Active implementation branch: `ops-current-player-availability-v1`
- Clean semantic fixture run: `34436970099`
- Live source smoke: `34437032282`, artifact `10136545256`, digest `sha256:decb703afe4769befba790d8b1adceb0accb5a49eec4f5fd8f5d2adc6c7eb75a`
- Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`

## External timing authority used before choosing threshold

NFL Football Operations states that at the 90-minute pre-kickoff officiating meeting, club PR representatives deliver the Game Day Administration Report, which includes the inactive list. NFL support also describes inactive designation as occurring about 90 minutes before scheduled kickoff.

The production certification threshold is frozen at **75 minutes before scheduled kickoff**. This intentionally allows a fixed 15-minute publication/ingestion buffer after the league's 90-minute submission point. This threshold is frozen before testing the validator against any current game-window payload and may not be tuned based on pass/fail results.

## Per-game timing states

For each scheduled game at evaluation time `asof_utc`:

- `NOT_YET_REQUIRED`: more than 75 minutes remain before kickoff. Missing official inactive sections do not block that game.
- `REQUIRED_AND_CERTIFIED`: 75 minutes or less remain before kickoff, kickoff is still in the future, and complete official inactive sections exist for both scheduled teams from a snapshot timestamp strictly before kickoff.
- `REQUIRED_MISSING_FAIL_CLOSED`: 75 minutes or less remain before kickoff and one/both scheduled-team sections are missing/incomplete, or the official snapshot timestamp is absent/invalid/not strictly before kickoff.
- `KICKED_OFF_LOCKED`: kickoff is at or before `asof_utc`. This V1 current pregame pipeline must not newly price/re-certify that game from a post-kickoff availability snapshot.

## Frozen game-level rule

A game's current-player availability certification is production-eligible iff:
1. state is `NOT_YET_REQUIRED`, or
2. state is `REQUIRED_AND_CERTIFIED`.

`REQUIRED_MISSING_FAIL_CLOSED` and `KICKED_OFF_LOCKED` are not eligible for new production pricing.

Failure is scoped to the affected game/teams, not the entire slate.

## Official section rules

- Endpoint HTTP reachability is never enough.
- Each scheduled team must have a complete parser-validated official section inside the required window.
- Absence of a player from a complete team section can be used as active evidence.
- Absence from an incomplete/missing section cannot be used as active evidence.
- Official source timestamp must be parseable UTC and strictly before the game's kickoff.
- A source snapshot may certify multiple same-window teams only where each team's section is independently complete.

## Required validator output

Create a game-level certification artifact containing at least:
- season
- week
- game_id when available
- away_team
- home_team
- kickoff_utc
- asof_utc
- minutes_to_kickoff
- official_required
- away_official_section_complete
- home_official_section_complete
- official_snapshot_asof_utc
- certification_state
- production_eligible
- failure_reason

And a JSON summary with counts of each state and withheld teams/games.

## Frozen fixtures

1. 120 minutes before kickoff + no official sections => `NOT_YET_REQUIRED`, eligible.
2. 76 minutes before kickoff + no sections => `NOT_YET_REQUIRED`, eligible.
3. Exactly 75 minutes before kickoff + no sections => `REQUIRED_MISSING_FAIL_CLOSED`, ineligible.
4. 60 minutes before kickoff + both complete sections + pre-kickoff snapshot => `REQUIRED_AND_CERTIFIED`, eligible.
5. 60 minutes before kickoff + only one complete section => fail closed for that game.
6. 60 minutes before kickoff + both sections complete but snapshot timestamp >= kickoff => fail closed.
7. as-of >= kickoff => `KICKED_OFF_LOCKED`, ineligible for new pricing.
8. Multiple games with different kickoff windows => only games inside the required window can be withheld; later games remain `NOT_YET_REQUIRED`.

## Boundary

No Full Slate production workflow, PlayerForm, P3, R26, R22, QB/WR/TE production model, simulation rule, sportsbook path, or historical backtest changes during timing validation.
