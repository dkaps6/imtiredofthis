# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## ACTIVE PRODUCTION CHECKPOINT — 2026-09-11

Before doing anything else, read:

1. `docs/handoffs/NFL_HANDOFF_2026-09-11_FULL_SLATE_LIVE_REPAIR_MERGED_CURRENT.md`
2. `NFL_MASTER_CONTINUITY_RECORD.md`

GitHub is canonical; chat memory is secondary.

The Week-1 live Full Slate mechanical incident is now **repaired and merged**. PR #523 was merged to `main` at `f84c6242da02b1804b4b9675c3a3a3e679838e10`. Post-merge Repo CI run `34656484043` and no-live Full Slate run `34656483980` both passed. The exact preserved paid artifact from run `34650067599` previously replayed through repaired Steps 29-31 with zero certification blockers and zero additional OddsAPI acquisition.

Do **not** restart the roster/event-scope repair, preserved-artifact replay, player-identity audit, rematch investigation, Knight alias investigation, downstream current-availability certification repair, or broad historical M107/M108 search. Do not spend another paid live-odds call merely to rediscover mechanical bugs.

Production model science remains frozen and unchanged by the repair. The active merged handoff contains the exact integrity verdict, paid-run lineage, replay evidence, merge SHA, and next authorized step.

Important continuity correction: prior chat notes carried an `M108 = 26/26 PASS` label, but repository search did not recover an authoritative M108 workflow/script/run/PR proving this was a canonical repository gate. Do not invent or require an M108 test by name unless concrete GitHub lineage is later recovered.

---

## PARKED SCIENCE CHECKPOINT

The QB/WR shared-opportunity / first-down pass-propensity / public pregame-intent V1B lane remains preserved at:

- `docs/handoffs/NFL_HANDOFF_2026-09-11_QB_WR_SHARED_OPPORTUNITY_CURRENT.md`

Resume that lane only after confirming no newer production incident supersedes the merged checkpoint.

No production-science change is authorized by the live-repair work itself.

---

## RESEARCH LEDGER — Vegas-line / game-script lane (2026-09-13)

User-initiated question (Issue #535): does modeling maximum pregame accuracy
beat Vegas, and specifically does Vegas's own spread/total/moneyline carry
usable signal for player props. Four PRs, in order, all research-only, no
production/model/threshold change from any of them:

- **PR #557** `STRONG_GATE_PROBABILITY_CALIBRATION_V1` — per-market isotonic
  recalibration of the STRONG/LEAN decision gate's probability. Merged
  (`24173a261ce620eab6f4347ca3d4990f468da5c8`). Result:
  `STRONG_GATE_CALIBRATION_NO_IMPROVEMENT` — mechanically collapses STRONG
  coverage as designed, but does not improve realized ROI in either holdout
  direction (2024->2025 or 2025->2024).
- **PR #558** `MARKET_IMPLIED_GAME_SCRIPT_V1` — does Vegas's spread/total add
  incremental value on top of our own historical team-plays/pass-rate
  baseline. Merged (`20f535047c263ae632409ce9a3fb796b9b2fae78`). Result:
  `MARKET_IMPLIED_GAME_SCRIPT_NO_INCREMENTAL_VALUE`.
- **PR #559** `VEGAS_LINE_GAMESCRIPT_CALIBRATION_V1` — is Vegas's own posted
  spread/total accurate against actual outcomes, unconditionally (no our-own-
  baseline involved). In review at last update. Real result (2023-2025, 816
  games): moderate correlation (0.3-0.5 margin/total), MAE ~10 points on
  both, monotonic bucket ordering (bigger spread -> more lopsided actual
  result and higher favorite win rate; bigger total -> higher actual score),
  and no exploitable linear bias survives out-of-sample recalibration
  (confirmed under both OLS and a median/L1 regression, added after a Codex
  finding that OLS alone can misjudge an MAE-based conclusion).
- **PR #560** `GAME_SCRIPT_CONFIRMED_PLAYER_USAGE_V1` (stacked on #559) — in
  games where Vegas's line was actually confirmed by the outcome, do RB rush
  volume (leading team) and WR/TE volume (high-total games) show a cleaner
  pattern than the raw unconditional pregame label, and how close does that
  get to the ground-truth (actual-outcome) ceiling. In review at last update.
  Real result (2023-2025, ~1630 team-games, after fixing a Codex-found bug
  that let sign-reversals/cutoff-crossings into the "confirmed" cohort):
  ground-truth effect is large and stable across all 3 seasons (margin->RB
  Cohen's d~1.0; total->WR/TE d~0.6-0.8). The raw unconditional Vegas label
  alone captures roughly a third of that (d~0.3-0.4). Restricting to
  Vegas-confirmed games meaningfully closes the gap toward the ceiling for
  both hypotheses, more so at a looser ±7-point confirmation threshold than
  ±3 (margin->RB confirmed d~0.55-0.78 by season; total->WR/TE confirmed
  d~0.38-0.93 by season). Important caveat: "confirmed" is defined from the
  actual outcome, so this measures a ceiling, not a pregame-actionable
  signal by itself — identifying which games will land in the confirmed
  bucket *before* kickoff remains open and unattempted.

Next open question (not yet attempted): can we predict, before kickoff,
which games are more likely to be "Vegas-confirmed" (e.g. line stability/
consensus across books, distance from key numbers) — since #560 shows that
knowing confirmation status in hindsight meaningfully sharpens the player-
usage signal.