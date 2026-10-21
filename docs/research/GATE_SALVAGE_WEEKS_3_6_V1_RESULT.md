# Pregame Decision-Gate Salvage, 2026 Weeks 3-6 — Frozen Result (V1)

Status: **frozen result**. Produced 2026-10-21.

## Why this was run

The canonical gate file covers 2026 weeks 1-2 only. I had reported weeks 3-5
as having permanently lost gate evidence along with their priced boards. The
control-room auditor corrected that: week 1's gate was reconstructed from the
**original Full Slate run logs**, not from the board, and week 2's from the
uploaded workbook — so weeks 3-5 were "not automatically unreconstructible" and
warranted a run-by-run salvage audit, with the standing condition that if the
surviving evidence does not preserve the decision state it must not be inferred.

That correction was right, and acting on it recovered four weeks of gate
evidence that would otherwise have been written off.

## Method

Actions **logs outlive artifacts**.
`scripts/validate_current_player_availability_timing_v1.py` prints its entire
certification meta to stdout, and that printed block is the authoritative
pregame record of which games had already kicked off and which were failed
closed at the T-75 boundary.

- reader: `scripts/operations/salvage_production_decision_gates_v1.py`
- pinned sources: `data/market_track_record/GATE_SALVAGE_SOURCES_V1.json`
- workflow: `.github/workflows/salvage-production-decision-gates-v1.yml`
- report: `data/market_track_record/GATE_SALVAGE_REPORT_V1.json`
- candidate: `data/market_track_record/GATE_SALVAGE_CANDIDATE_V1.json` (`AWAITING_AUDIT`)
- salvage run: `39620564121`
- 28 unit tests, mostly on the refusal paths

The reader is fail-closed per field. A week is `NOT_RECOVERABLE`, with its
missing fields named, when the log is absent, prints no certification block,
prints blocks that disagree, or fails an internal game-count check. Nothing is
completed from a neighbouring week, the schedule, or the board.

### The reader is validated, not asserted

Week 1 is a true control: its canonical gate was derived from this exact
printed block. The workflow fails if the reader cannot reproduce it.

| field | canonical | read back |
|---|---|---|
| `availability_asof_utc` | 2026-09-11T21:36:40.970365+00:00 | **identical** |
| `game_count` | 16 | **16** |
| `eligible_games` | 14 | **14** |
| `kicked_off_locked_games` | 2 | **2** |
| `kicked_off_locked_teams` | LAR, NE, SEA, SF | **LAR, NE, SEA, SF** |

A reader that could not replay a known week would not be trusted on an unknown
one. This one replays it exactly.

## Results

| week | status | asof (UTC) | games | eligible | withheld | states | withheld teams |
|---|---|---|---|---|---|---|---|
| 1 | RECOVERABLE (control) | 2026-09-11T21:36:40 | 16 | 14 | 2 | KICKED_OFF_LOCKED 2, NOT_YET_REQUIRED 14 | LAR, NE, SEA, SF |
| 2 | RECOVERABLE (control) | 2026-09-17T22:27:52 | 16 | 16 | 0 | NOT_YET_REQUIRED 16 | none |
| 3 | **RECOVERABLE** | 2026-09-27T04:07:03 | 16 | 15 | 1 | KICKED_OFF_LOCKED 1, NOT_YET_REQUIRED 15 | **ATL, GB** |
| 4 | **RECOVERABLE** | 2026-10-01T22:25:23 | 16 | 16 | 0 | NOT_YET_REQUIRED 16 | **none** |
| 5 | **NOT_RECOVERABLE** | — | — | — | — | two disagreeing blocks | ambiguous |
| 6 | **RECOVERABLE** | 2026-10-15T19:18:47 | 14 | 14 | 0 | NOT_YET_REQUIRED 14 | **none** |

Four of five salvage targets recovered. Weeks 1-2 read back consistently with
their canonical entries.

## Finding 1 — the Weeks 3-5 boards are gone beyond any doubt

Recovering a gate does not resurrect a week. The gate says which rows were
decidable; the **board** holds the rows. For weeks 3-5 the board is gone, and
this was established exhaustively rather than from the named artifact ids:

- the three pinned week 3/5 artifacts return hard 404, not "expired";
- **the oldest surviving artifact in the entire repository is 2026-10-11**,
  across all 1,910 artifacts;
- no surviving artifact before 2026-10-11 is board-sized (>1 MB).

So nothing from the Weeks 3-5 window (Sept 24 – Oct 10) survives anywhere.
Week 1's board was once rescued because an unrelated research capture happened
to hold it at 90 days; no such accident exists for weeks 3-5.

The ~10-day effective artifact window is the legacy of the old retention
settings. PR #692 raised all 63 uploads to the 90-day ceiling, but that binds
uploads made **after** it merged; everything older is still aging out on its
original 7/14/30-day clock.

**Logs are the more durable medium.** Week 1's log from Sept 11 is still fully
readable at 21,021 lines, 40 days on, while every artifact from that era is
gone. Anything that must survive should be printed as well as uploaded — or,
better, committed.

## Finding 2 — week 5 admitted a fail-closed game through a same-day override

Week 5 failed closed on disagreement, and the disagreement is substantive
rather than a glitch. Both blocks carry the **same** `asof`
(2026-10-08T23:41:14.413318+00:00) and the same 14+1 games. One game flips:

| candidate | games | eligible | withheld | states | withheld teams |
|---|---|---|---|---|---|
| 1 | 15 | 14 | 1 | NOT_YET_REQUIRED 14, **REQUIRED_MISSING_FAIL_CLOSED 1** | DAL, TB |
| 2 | 15 | 15 | 0 | NOT_YET_REQUIRED 14, **PRESERVED_PRE_T75_REPLAY 1** | none |

`PRESERVED_PRE_T75_REPLAY` is **not a state the T-75 validator can emit**. It
comes from a same-day P0, run `37860224986` at 23:34 ("preserve pinned pre-T75
game state during paid replay"), four minutes before the 23:38 dispatch.

Read charitably the mechanism is probably correct: during a replay that reuses
already-paid odds, the T-75 clock has moved on, so re-certifying would
spuriously fail games that were legitimately certified at original acquisition
time. Preserving the pinned state is the sane behaviour.

But the observable effect in this run is that a game the T-75 contract had
failed closed became eligible, and **that effect is not documented** in
`docs/operations/WEEK5_2026_10_08_FULL_SLATE_PRODUCTION_RECOVERY_RESULT.md`
(PR #677), which covers the bye-week guard and coverage-count repairs instead.
That doc is otherwise candid — `FULL_SLATE_DATA_QUALITY_NOT_CERTIFIED`, two
certification blockers, an explicit instruction not to treat the week as a
certified betting card, and a next step requiring a rerun to "honor a fresh
T-75 window".

This is recorded as an **open question for audit, not a verdict**. Whether the
DAL/TB game was legitimately certified at original acquisition time is not
answerable from the log, and it is not assumed either way here. Week 5's board
is gone, so there is no grading consequence; the consequence is forward, because
the mechanism is still live for future replays.

## Finding 3 — week 6's gate is recoverable and clean, which upgrades its result

This was the reason week 6 was added mid-audit: its board came from a
`PRESERVED_REPLAY`, so the week-5 override bore directly on the Week-6 result
already reported.

Week 6's log carries **one** certification block, no disagreement, and
**no `PRESERVED_PRE_T75_REPLAY` state at all**. 14 games, 14 eligible, 0
withheld, only `NOT_YET_REQUIRED`. The override did not touch week 6.

Two consequences for `docs/research/WEEK6_BACKTEST_V1_RESULT.md`:

1. **Week 6 had zero team-level blocks.** The `UNGATED_ALL_PRICED_ROWS`
   population I graded is therefore identical, *at the team layer*, to the
   production-decided population. The 308-bet result is not distorted by
   missing team gates, because there were none to miss.
2. **The "28/32 teams" falsification note is now explained.** Week 6 had 14
   games, so 28 teams played and 4 were on bye. That line was not evidence of
   selection bias. The separate finding — six teams with a board row but no
   graded `pass_yards` bet — is unaffected and still open.

### The limit on all of this

The gate has **two layers** and this salvage recovers only one.

- **Team layer** (`TEAM_ALL`: kicked off, or failed closed at T-75) — recovered
  from the log.
- **Player-market layer** (`PLAYER_MARKET`: the pregame workbook's
  Decision/Snapshot Signal `BLOCKED` rows) — **not recoverable from logs.**
  Week 2's canonical gate has three such blocks and zero team blocks, so a week
  reading "0 withheld teams" is *not* the same as a week with no blocks.

For weeks 3, 4 and 6 the player-market layer is **unknown**, and would need
that week's uploaded workbook. Week 6's result must not be upgraded to "fully
gated" on the strength of its clean team layer.

## What this authorises

**Authorised:**

- recording weeks 3, 4 and 6 team-layer gate facts as read-back evidence;
- softening the Week-6 `UNGATED` caveat at the team layer only, with the
  player-market layer still unknown;
- treating the printed certification block as the durable medium to target when
  building the runtime gate emitter for weeks 7-18;
- closing "are weeks 3-5 reconstructible?" with: gates partly yes, boards no.

**Not authorised:**

- promoting `GATE_SALVAGE_CANDIDATE_V1.json` into
  `PRODUCTION_DECISION_GATES_V1.json` — it is `AWAITING_AUDIT` and the grader
  does not consume it;
- calling weeks 3-5 gradeable; the boards are gone;
- declaring any week's player-market gate layer empty;
- choosing between week 5's two candidate blocks;
- treating the `PRESERVED_PRE_T75_REPLAY` override as either validated or
  improper without the audit.

## Open items

1. **Week 5's override needs an audit verdict** — was the DAL/TB game
   legitimately certified at original acquisition time? The mechanism is live
   for all future replays.
2. **Promotion of the candidate gates** for weeks 3, 4, 6 is a separate audited
   step.
3. **The player-market gate layer** for weeks 3, 4, 6 needs the uploaded
   workbooks, if they exist.
4. **The runtime emitter for weeks 7-18** is still unbuilt. It needs the
   provenance envelope the auditor specified (`source_run_id`,
   `source_git_sha`, `source_job_id`, artifact id/digest, workbook digests,
   `evidence_basis`, lineage version), should be hash-addressable, and should be
   written and validated **before** paid acquisition rather than after.
5. **Six teams with a board row but no graded `pass_yards` bet** in week 6
   (BAL, BUF, KC, SF, TB, WAS) remains unexplained.
