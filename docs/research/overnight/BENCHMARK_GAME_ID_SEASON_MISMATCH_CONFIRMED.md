STATUS: CONFIRMED BUG — CRITICAL. All ROI/win-rate/STRONG-gate/Vegas-comparison findings from this research thread (PRs #533, #534, #536, #537, #538) are suspended pending rebuild. Posted to Issue #535.

# Benchmark game_id/season mismatch — confirmed, traced, scoped

GPT-5.6 flagged (Issue #535 comment) that 16,040/16,973 rows in
`non_qb_detail_wr_r15_te_r5p_applied.csv` have a `game_id` whose season
prefix doesn't match the row's own `season` column. Independently reproduced
and traced.

## Confirmation

```
non_qb_detail_wr_r15_te_r5p_applied.csv: 16,040 / 16,973 mismatched (94.50%)
data/backtests/full_stack_vegas_benchmark_v1/non_qb_detail.csv (ORIGINAL,
  pre-dates this research thread, already on main): 16,040 / 16,973,
  IDENTICAL mismatch.
```

Same rows, same count, exact match. **This bug predates this research
thread entirely** — it's baked into the original committed benchmark, not
something introduced by V1/V2/the holdout scans. Week alignment (`game_id`
week component vs `week` column) is clean: 0 mismatches.

Example: row `team=NYJ, opponent=SF, season=2024, week=1` carries
`game_id=2023_01_BUF_NYJ` — not just a year off, a **different matchup
entirely** (2023 Week 1 was Buffalo @ NYJ; 2024 Week 1 was NYJ @ SF).
Verified against the schedule source (`/tmp/bt_combined_schedule.csv`,
which is itself correct): NYJ/week 1 across seasons is `2023_01_BUF_NYJ`,
`2024_01_NYJ_SF`, `2025_01_PIT_NYJ` respectively — three different, correct,
season-specific game_ids exist in the schedule data. The benchmark picked
the 2023 one for a 2024-labeled row.

## What's actually corrupted, and what probably isn't

Traced the two things `game_id` and `actual` depend on separately, since
`grade_full_stack_vegas_benchmark_v1.py::grade()` joins real historical
odds (`props`) to the projection on `["game_id", "player_clean_key",
"market"]` — meaning a wrong `game_id` doesn't just mislabel a row, it can
pull in **odds from a genuinely different game**.

- **`actual` (real outcome) is very likely NOT corrupted.** In
  `scripts/backtest/component_predictions.py::build_actual_rows()`, the
  outcome is filtered by `x["season"].eq(int(season)) & x["week"].eq(int(week))`
  directly against player logs — no `game_id` involved at all — and the
  caller (`build_market_vegas_benchmark_projections_v1.py`) explicitly
  overwrites `season`/`week` to the correct build-loop values immediately
  after computing `actual`. Model MAE (`|proj - actual|`) is probably safe.
- **`game_id` (and therefore the joined Vegas `line`/`over_odds`/`under_odds`)
  is the corrupted piece.** `build_market_vegas_benchmark_projections_v1.py`'s
  own internal schedule merge (filters `sk` to `a.season` before joining on
  `week`+`team`) *looks correct on inspection* — I could not find the bug in
  that function. The committed data's actual `game_id` values don't match
  what that logic should produce, which means the corruption most likely
  happened in a separate, ad-hoc post-processing step (the files carry an
  informal `_gid` suffix, e.g. `bt_proj_2024_gid.csv`, suggesting a
  not-committed, one-off script attached `game_id` after the fact, probably
  by joining a multi-season combined schedule on `week`+`team` without a
  season filter). I could not locate that step's source since it isn't
  checked into the repo.
- **Critically, this can fail silently, not loudly.** For any player who
  played in multiple seasons (nearly everyone), the wrong-season `game_id`
  still often matches a *real* row in the free props archive (that player's
  actual prop line from the wrong season's similar-week game) — so the join
  succeeds and produces plausible-looking, real, non-null odds. It does not
  error out or leave gaps that would have been easy to notice.

## Consequence for every finding produced through this pipeline tonight

Every ROI, win-rate, STRONG/LEAN-tier, `component_sd`, and `prob_edge`
result from PRs #533/#534/#536/#537/#538 was computed by comparing this
season's model projection against **another season's Vegas line** for 94.5%
of rows, joined via a plausible-looking but wrong `game_id`. Model-vs-actual
MAE (the "football mean gap" finding) is probably still directionally valid
since it never touches `game_id`, but "Vegas MAE" and everything
ROI/probability-derived is not trustworthy as reported. This also means the
already-flagged `component_sd`-overconfidence mechanism finding, while
independently reasoned and probably still real as a mechanism, was measured
on a benchmark now known to be misaligned — it needs to be re-measured after
a rebuild, not retracted, but not relied on either.

## Second bug confirmed: `home_away()` LA/LAR mismatch (full_market_holdout_scan_v1.py)

Independently reproduced GPT-5.6's second finding. My `home_away()` helper
compares the canonical `team` column directly against the raw team
abbreviation parsed out of `game_id`. `game_id` uses `LA` for the Rams;
canonical `team` uses `LAR`. Confirmed: **225 of 496 LAR rows (45%)** have
a `game_id` home-component of `LA`, which never string-matches `LAR` — so
every one of those genuinely-home LAR games gets misclassified `AWAY`. This
invalidates the `home_away` categorical slice from the full-market scan
(PR #537) independent of the season/game_id issue above.

## Status

Agree with GPT-5.6's proposed status: `HISTORICAL_BENCHMARK_IDENTITY_INTEGRITY_BLOCKER_OPEN`.
No production/model/threshold change. All prior ROI/STRONG findings from
this thread downgraded from `UNVERIFIED` to `INVALID_PENDING_REBUILD`
pending a `game_id`-clean rebuild with fail-closed identity assertions
before grading (season-prefix-of-game_id == season column, as a hard
`raise` not a silent pass).
