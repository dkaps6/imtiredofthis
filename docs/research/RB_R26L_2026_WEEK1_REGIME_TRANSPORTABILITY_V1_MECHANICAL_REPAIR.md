# RB R26L 2026 Week-1 Regime Transportability V1 — Mechanical Repair Note

Date: 2026-09-09
Status: MECHANICAL REPAIR ONLY; FROZEN SCIENCE UNCHANGED

Frozen plan:
`docs/research/RB_R26L_2026_WEEK1_REGIME_TRANSPORTABILITY_V1_FROZEN_PLAN.md`

Frozen evaluator:
`scripts/backtest/audit_rb_r26l_2026_week1_regime_transportability_v1.py`

Compatibility runner:
`scripts/backtest/run_rb_r26l_2026_week1_regime_transportability_v1.py`

## Failed attempt 1

Run `34378742255` failed before any scientific disposition because
`nflreadpy==0.1.5` rejected explicit season 2026 in `load_rosters_weekly` on
2026-09-09. Its generic current-season logic still returned 2025 until the
Thursday after Labor Day.

Repair: the isolated R26L workflow patches only the explicit weekly-roster
validation ceiling so the same nflverse source path
`weekly_rosters/roster_weekly_2026` can be requested. No roster source is
substituted.

## Failed attempt 2

Run `34388840174` passed the weekly-roster repair, frozen-plan/code checks,
production-boundary check, and immutable-parent digest checks. It then failed
before scientific scoring because the R9 identity runtime was asked to load
weekly player statistics through 2026. nflverse correctly returned 404 for
`stats_player_week_2026.parquet`; no 2026 games have yet occurred.

## Strict-prior repair

R26L targets 2026 Week 1. Therefore strictly-prior receiving identity has no
2026 game observations and must end at 2025. The compatibility runner leaves
the frozen evaluator unchanged and replaces only its call-time identity-atlas
boundary:

- requested target boundary: 2026;
- strict-prior historical endpoint: 2025;
- same production-safe R9 identity runtime;
- no 2026 outcomes or participation loaded;
- no sportsbook football inputs;
- no same-week depth;
- no prediction regeneration;
- no R9 refit;
- no production changes.

## Science unchanged

The following remain byte-for-byte governed by the frozen evaluator/plan:
- seven primary transportability features;
- source population semantics;
- feature floors;
- normalized-distance rules;
- 5-of-7 requirement;
- 0.75 distance-ratio rule;
- beyond-2020 anomalous-direction rule;
- disposition definitions;
- authority ceiling.

No 2026 aggregate feature value or R26L scientific disposition was available
before this repair was specified.
