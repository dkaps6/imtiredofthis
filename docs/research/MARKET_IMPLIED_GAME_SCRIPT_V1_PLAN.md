# Market-Implied Game Script V1 — Frozen Plan

**STATUS: FROZEN BEFORE ANY CANDIDATE OUTPUT. RESEARCH ONLY. NO PRODUCTION CHANGE.**

## Motivation

`scripts/modeling/rules_v2.py::project_game_script()` is the single shared
function every position's volume projection routes through
(`scripts/modeling/simulation_rules.py` calls it once per player-game and
uses the result for QB pass rate/plays, and — via the same `script` object —
for the play-count ceiling that RB/WR/TE opportunity shares are computed
against). It is built entirely from historical team-strength inputs:

- `success_diff()` = `offense.success_rate_off - defense.success_rate_def`
  (season-level efficiency, not this week's game);
- `estimate_plays()` = a blend of season `plays_est` and recent pace, again
  with no live-game input;
- `pass_share` is a hardcoded constant (0.57) applied to every team in every
  game, not modulated by expected game competitiveness at all.

It never reads Vegas's spread or total for the specific upcoming game. A
prior narrow QB-only diagnostic
(`scripts/backtest/audit_qb_gamescript_market_signals.py`, "Migration 50")
already confirmed `spread_line`/`total_line`/moneylines are available,
leakage-safe (known from the nflverse schedule before kickoff), and computed
implied team totals from them — but that work was diagnostic-only and was
never integrated into `project_game_script()` itself, and was never tested
for RB/WR/TE at all.

**User-requested scope**: extend market-implied game-script signal to
RB/WR/TE. Given the shared-function finding above, the correct target is
`project_game_script()` itself, which benefits all four positions at once
rather than a position-specific bolt-on.

## Design

**Step 1 (this PR): diagnostic only, no feature injection yet.** Before
touching `rules_v2.py`, establish whether the market-implied signal actually
carries incremental information beyond what `success_diff`/`estimate_plays`
already capture, on real historical outcomes:

- Join `spread_line`/`total_line` (nflverse schedule, leakage-safe) onto the
  historical cohort already used throughout tonight (2024-2025, extendable
  to 2020-2023 if useful).
- Compute `market_team_implied = (total_line - team_spread) / 2` and
  `market_abs_spread = abs(team_spread)` per team-game, exactly as Migration
  50 already did.
- Regress/correlate these against ACTUAL team plays, actual pass rate, and
  actual RB carry share / WR+TE target share for that game, controlling for
  (i.e., alongside, not replacing) the existing `success_diff`/`estimate_plays`
  predictions — the question is incremental value, not a replacement.
- Genuine train/test holdout: fit any coefficient on one season, freeze,
  test blind on another, matching every other experiment tonight.

**Step 2 (separate, later PR, only if Step 1 shows real incremental value)**:
design the actual injection into `project_game_script()` — e.g., blending
`market_team_implied`/`market_abs_spread` into the play-count and pass-share
estimate, or replacing the historical-only `diff` with a blend of
`success_diff` and market spread. Not attempted in this PR.

## Integrity

- Market data used only as of what nflverse's schedule table carries
  pregame (spread_line/total_line/moneyline) — never target-game PBP.
- No sportsbook prop odds involved (this is game-level spread/total from
  schedule data, a different, more liquid market than player props, and
  legitimate as a feature — not circular with the props being priced).
- No change to `rules_v2.py`, `simulation_rules.py`, or any live pricing path
  in this step. Diagnostic script + doc only.

## Collaboration

Per the user's explicit request, proposing this as a joint lane with
GPT-5.6 before building further — see Issue #535 checkpoint for the
proposed split.
