# RB Market Benchmark — Frozen Plan

Purpose: benchmark the frozen M94C 2025 RB rushing-yard projection against real archived sportsbook rushing-yard lines and actual outcomes. This is a downstream market audit, **not** retrospective model retuning and does not remove `AUTONOMOUS_RB_RESEARCH_STOP`.

## Source

Use the same free Action Network-derived player-prop archive audited in M60B: `gcampb41/nfl_data-`, 2025 player-prop parquet. Only DraftKings (`book_id=68`) and FanDuel (`book_id=69`) full-game player rushing-yard rows are eligible.

The archive does not preserve a trustworthy fixed pre-kick timestamp. Lines must be described as `archived_latest_per_book` / closing-like, never as a 30-minute-before-kickoff snapshot. If a player-week-book contains conflicting line values and no trustworthy timestamp resolves them, drop that book row rather than guess.

## Football side

- Frozen M94C 2025 `candidate_rush_yards` is the current football-only projection.
- Frozen actual `actual_rush_yards` is outcome truth.
- No sportsbook data enters M94C or any football model.
- No coefficient/threshold/feature/model changes are allowed in this audit.

## Primary comparison

On exact common market-covered RB player-games:

1. M94C projection vs actual.
2. DraftKings line vs actual.
3. FanDuel line vs actual.
4. Consensus line = median of available DK/FD lines vs actual.
5. Two-book consensus = mean/median of DK+FD only when both exist.

Report MAE, RMSE, bias, correlation, sample size, and head-to-head closer-to-actual percentage on identical consensus rows.

## Disagreement diagnostics

Predeclare absolute M94C-vs-consensus disagreement buckets: `<5`, `5-<10`, `10-<15`, `15+` rushing yards. Report:

- M94C MAE and market MAE within each bucket;
- which side is closer more often;
- directional market-side accuracy when M94C is >=5/10/15 yards above or below the market line (pushes excluded).

These diagnostics may motivate future **prospective** research questions but may not be used to retune the frozen 2025 RB architecture.

## Integrity

- No fake/synthetic lines.
- No sportsbook inputs upstream.
- Name/team joins audited; ambiguous/conflicting rows dropped.
- Persist source bet-type/book/coverage/conflict audit and full joined casebook.
