# Game-Script-Confirmed Player Usage V1 — Frozen Plan

**STATUS: FROZEN BEFORE ANY CANDIDATE OUTPUT. RESEARCH ONLY. NO PRODUCTION CHANGE.**

## Motivation

PR #559 (`VEGAS_LINE_GAMESCRIPT_CALIBRATION_V1`) measured whether Vegas's
closing spread/total, on its own terms, describes what actually happens in a
game. Result: real but noisy signal — moderate correlation (0.3-0.5), MAE
~10 points on both margin and total, monotonic bucket ordering (bigger
spread -> more lopsided actual result and higher favorite win rate; bigger
total -> higher actual combined score), and no further exploitable linear
bias after recalibration.

User's direct follow-up: in the games where Vegas's implied script actually
happened (the market was right, not just directionally but close in
magnitude), how did specific *players* perform — and is that a usable
predictive thesis (which kind of game -> which kind of player benefits)?

This is a genuinely different question from #558 (team plays/pass-rate vs a
historical baseline) and from #559 (is the market's number itself accurate).
It asks: conditional on script correctness, does the well-known football
theory (leading teams run more, trailing teams pass more, high-total games
mean more overall passing volume) show up cleanly at the player-position
level, and does knowing the market got it right sharpen that pattern versus
using the market's raw pregame label blind?

## The honest framing, stated before any output

"Script-confirmed" is defined using the *actual* game outcome. This analysis
therefore measures a **ceiling**: does game script, once realized, actually
drive player-position volume in a clean, exploitable way? It does **not**,
by itself, tell us how to identify *before kickoff* which games will land in
the confirmed bucket — that remains unsolved and is explicitly out of scope
here. The value of this diagnostic is deciding whether that harder follow-on
problem is worth pursuing at all: if even the confirmed-only subset shows a
weak/noisy player-usage relationship, chasing pregame confirmation
prediction is not worth it; if it shows a strong, clean relationship, that
motivates future work on predicting confirmation likelihood itself (e.g.
via line consistency across books, key-number distance) — not attempted
here.

## Design

**Data**: reuses `load_game_outcomes()` from
`scripts/research/diagnose_vegas_line_gamescript_calibration_v1.py`
(2023-2025 nflverse schedule, `predicted_margin_home`/`predicted_total`/
`actual_margin_home`/`actual_total`, leakage-safe pregame market data) —
no new schedule fetch. Player-game actuals come from
`scripts/backtest/historical_player_logs.py::build_historical_player_logs`,
the same canonical historical player-log builder already used elsewhere in
this repo's backtests (position, rushes/rush_yards, targets/rec_yards, and
team-level rushes/targets/dropbacks per team-game).

**Team-game expansion**: each game becomes two team-perspective rows (home,
away), with `predicted_team_margin`/`actual_team_margin` signed from that
team's own perspective (positive = that team favored/leading). This mirrors
PR #558's `load_market_schedule` team-expansion pattern.

**Confirmation definition, pre-registered with a sensitivity check (not
tuned after seeing results)**: a team-game's margin is "confirmed" if
`abs(actual_team_margin - predicted_team_margin) <= T`; total is
"confirmed" if `abs(actual_total - predicted_total) <= T`, evaluated at
**two frozen thresholds, T=3 and T=7** (3 = one field goal, the single most
common NFL scoring increment and the standard backdoor-cover buffer; 7 = one
touchdown, a looser tolerance) — both reported, neither cherry-picked.

**Position-level metrics** (aggregated to team-game level from player logs):
- `rb_rush_att`, `rb_rush_yards` — sum over players tagged position `RB`.
- `wrte_targets`, `wrte_rec_yards` — sum over players tagged position `WR`
  or `TE`.

**Two hypotheses, each tested across three arms**:

1. **Margin -> RB volume**: favored/leading teams should see more rush
   volume than trailing teams.
   - Arm A (ground truth): split by sign of `actual_team_margin`.
   - Arm B (Vegas alone, unconditional): split by sign of
     `predicted_team_margin`, no filtering — what we could act on pregame,
     blind.
   - Arm C (Vegas, confirmed-only): same split as Arm B, restricted to rows
     where the margin is confirmed at threshold T.

2. **Total -> WR/TE volume**: higher-scoring games should show more overall
   passing volume for both teams.
   - Split point frozen once: `TOTAL_SPLIT_CUTOFF` = median of
     `predicted_total` over the full pooled 2023-2025 sample (computed once,
     used identically for both the predicted-based and actual-based splits
     so "high" means the same thing in every arm).
   - Arm A (ground truth): split by `actual_total >= TOTAL_SPLIT_CUTOFF`.
   - Arm B (Vegas alone, unconditional): split by
     `predicted_total >= TOTAL_SPLIT_CUTOFF`, no filtering.
   - Arm C (Vegas, confirmed-only): same split as Arm B, restricted to rows
     where the total is confirmed at threshold T.

**Reporting**: for every (hypothesis, arm, threshold, metric), report
`n_high`/`n_low`, `mean_high`/`mean_low`, `diff`, and Cohen's d (pooled-std
effect size) — pooled across all three seasons AND per-season, so a result
that only shows up in one season is visible as such rather than hidden in
the pooled number. `INSUFFICIENT_ROWS` (< 30 rows in either bucket) is
reported explicitly, never silently dropped.

**What determines the read**: Arm B substantially weaker than Arm A is
consistent with #559's ~10-point MAE (Vegas's raw pregame label is a noisy
proxy for actual script). Arm C closing most of the gap to Arm A means
market-confirmation genuinely recovers the ground-truth effect; Arm C
staying close to Arm B means confirmation doesn't add much beyond the raw
label, at least at the tested thresholds.

## Amendment log

**Amendment 1 (before any result was reported)**: Codex review (via GitHub,
PR #560) found that the "confirmed" definition used error tolerance alone
(`abs(actual - predicted) <= threshold`), which lets a sign reversal or
cutoff crossing through: e.g. predicted margin +3, actual margin -3 has
abs error 6, passing the T=7 threshold even though the predicted favorite
actually lost; analogously for total, predicted=43/actual=45 with a
cutoff of 44 has abs error 2 but the two land on opposite sides of the
split. Left unfixed, this would have silently included directionally-wrong
games in Arm C's "confirmed" cohort, diluting the very effect the arm is
meant to isolate and potentially changing the reported conclusion. Fixed
by requiring, in addition to the error tolerance, that predicted and
actual land on the same side (matching sign for margin, same side of the
frozen cutoff for total) — for both hypotheses, since the total hypothesis
had the identical class of bug even though Codex's comment anchored on the
margin hypothesis's line. Covered by two new tests constructing an explicit
reversal/crossing row and asserting it is excluded from the confirmed
cohort's row count.

## What this is not

- Not a claim that we can identify confirmed games before kickoff — see
  "honest framing" above. That is a distinct, harder, unsolved problem.
- Not a production change — diagnostic script, tests, this doc, and a CI
  workflow only. No touch to `rules_v2.py`, `simulation_rules.py`, pricing,
  or any live path.
- Not a repeat of #558 or #559 — independent question, independent code
  path (though it reuses #559's already-reviewed `load_game_outcomes`
  function rather than re-deriving the market sign convention).

## Collaboration

Per standing practice, posting this plan to Issue #535 alongside #559 and
inviting GPT-5.6 to review before/alongside the result.
