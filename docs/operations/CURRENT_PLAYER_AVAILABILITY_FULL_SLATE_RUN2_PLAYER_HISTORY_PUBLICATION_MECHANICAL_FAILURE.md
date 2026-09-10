# Current Player Availability Full Slate — Run 2 Mechanical Failure

Status: `MECHANICAL_FAILURE_AFTER_AVAILABILITY_BEFORE_35_GATE_RESULT`

## Exact lineage

- branch: `ops-current-player-availability-full-slate-v1`
- head: `ec01f7f8087679d6650cae98bf765258b073a261`
- workflow run: `34443710690`
- job: `102763847787`
- artifact: `10138897760`
- artifact digest: `sha256:165c0e431e4771eba05472b17d6680457fedac37d5769bfc5c0b478aad830b1f`

## What succeeded before failure

- locked candidate boundary: PASS
- raw Ourlads + authoritative schedule: PASS
- repaired Week-1 TeamForm + promoted QB context: PASS
- weather/injuries: PASS with declared warnings only
- availability resolved before opportunity: PASS
- T-75 timing: 15 eligible games, 1 already-kicked-off game withheld (`NE`/`SEA`)
- reconciled production-eligible roles: 437 rows across 30 eligible teams
- PlayerForm unavailable-player exclusion assertion: PASS

## Exact failure

The first command in `Build canonical model stack`, `python scripts/run_model_context_bridge.py`, failed inside `validate_2026_provider_artifacts.validate_player_history` with:

`RuntimeError: player_game_logs contains same/future-week rows`

The sample rows were 2026 Week 1 players from the already-completed `NE`/`SEA` game.

## Root cause

`player_form_v2.build()` correctly constructs model inputs from strict-prior active-season history only (`week < target_week`). Therefore `player_form.csv` itself did not use the NE/SEA Week-1 outcomes. However, the function returned and published the broader `all_logs` and totals built from `all_logs`, which included current-season Week-1 rows. The published `player_game_logs.csv` / `player_season_totals.csv` therefore violated the existing provider validator contract even though the actual PlayerForm blend was strict-prior.

## Disposition

This is not an availability-science failure and not a 35-gate integration result. No integration disposition is authorized.

A permitted repair must be value-neutral: publish only the same strict-prior player-history rows already used by the model form, recompute the published season totals from that strict-prior set, and leave model formulas, role semantics, T-75 semantics, R22/R26, sportsbook boundary, and all frozen 35 gates unchanged.
