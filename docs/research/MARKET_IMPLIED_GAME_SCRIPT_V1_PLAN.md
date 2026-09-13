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

**Step 1 (this PR): diagnostic only, no feature injection yet, narrower than
originally scoped.** Before touching `rules_v2.py`, establish whether the
market-implied signal actually carries incremental information beyond a
historical-tendency baseline, on real historical outcomes:

- Join `spread_line`/`total_line` (nflverse schedule, leakage-safe) onto
  2024-2025 team-week history.
- Compute `market_team_implied` and `market_abs_spread` per team-game.
  **Sign convention, verified empirically against real 2023 results** (e.g.
  DAL home `spread_line=+17.5`, won 49-17): nflverse's `spread_line` is
  POSITIVE when the home team is favored. `team_spread` is therefore
  "points this team is favored by" (positive=favored) per side, and the
  favored team's implied total is the larger half:
  `market_team_implied = (total_line + team_spread) / 2`, not minus — an
  earlier draft of this plan (and the first committed script) had the sign
  backwards, caught in review before any result was reported, and fixed.
- Compare against ACTUAL team plays and actual team pass rate only in this
  pass — **not** RB carry share / WR+TE target share as originally scoped
  here. Measuring those requires aggregating player-level actual carries/
  targets to team level, additional plumbing not yet built; deferred to an
  explicit Step 1b rather than silently treated as covered by this PR. Team
  plays/pass-rate are the two quantities `project_game_script()` itself
  estimates, so this pass still directly tests the shared engine's inputs.
- Genuine single-direction holdout: fit on 2024 only, freeze, evaluate blind
  on 2025. **Not two-directional.** A clean reverse direction (fit 2025,
  test 2024) would require the training fold's own rolling-history feature
  to never draw on 2024's outcomes, which it cannot avoid with only two
  seasons of history available (2025's early weeks' prior-8 window reaches
  directly into 2024, the season being held out in that direction) — caught
  in review, and dropped rather than reported as if it were independent.
  Extending to a true two-directional test would require additional
  backstop seasons (2022/2023) purely for history, not attempted here.
- Three-arm comparison, not two: (1) the raw historical rolling baseline
  unfit, (2) that same baseline re-fit through a plain linear regression
  (isolates how much of any improvement is just correcting the baseline's
  own scale/bias), (3) baseline + market signal fit jointly. Incremental
  market value is (2) vs (3), not (1) vs (3) — an unfit baseline would make
  the market arm look better than it is purely by getting to fit an
  intercept/slope the baseline arm never had a chance to.

**Step 2 (separate, later PR, only if Step 1 shows real incremental value)**:
design the actual injection into `project_game_script()` — e.g., blending
`market_team_implied`/`market_abs_spread` into the play-count and pass-share
estimate, or replacing the historical-only `diff` with a blend of
`success_diff` and market spread. Not attempted in this PR.

## Amendment log

**Amendment 1 (before any result was reported)**: Codex review (via GitHub,
PR #558) found four real issues, all fixed before interpreting any output:
1. the implied-total formula had the wrong sign (subtracting instead of
   adding `team_spread`), verified and corrected against real 2023 data;
2. the two-directional holdout's reverse direction leaked 2024 (test-season)
   outcomes into 2025 (train-season) rows' own history feature — dropped,
   single-direction only;
3. the original two-arm comparison let the market arm win credit for merely
   fitting the baseline's own scale/bias — added a fitted-baseline-only arm
   as the correct comparator;
4. this plan's Step 1 promised RB carry-share/WR+TE target-share evaluation
   the script never implemented — scope corrected above; deferred to Step 1b.

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
