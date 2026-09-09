# RB R26R — 2026 Week 1 Prospective Observation Snapshot V1 — Implementation Lock

Frozen plan commit:

`5ebdc456dfb58fee3cf9c2087401d332895908e9`

Frozen evaluator implementation commit:

`81aa92e2c74c46ea7191ba36444498f11219c4de`

Evaluator:

`scripts/backtest/observe_rb_r26r_2026_week1_receptions_market_snapshot_v1.py`

## Locked behavior

The evaluator may only:

- verify the immutable R26Q prospective seal and its R26O lineage;
- verify the exact sealed 107 reception arrays and their hashes;
- read current Ourlads roles for sportsbook identity classification only;
- read the hardened live OddsAPI market snapshot after the sealed candidate has been verified;
- compare exact `player_receptions` bookmaker lines/prices against sealed baseline/candidate receptions evidence;
- emit prospective observation artifacts.

It may not regenerate football values, refit R9, alter entitlement, change any sealed array, use sportsbook information upstream, tune a threshold, activate a live production shadow, or promote to production.

## Mechanical validation before lock

The evaluator was compiled successfully under Python 3.11 syntax.

It was executed locally against the exact R26Q artifact `10123251043` in two mechanical fixtures:

1. legitimate no-player-prop state -> 30/30 gates, disposition `R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_NO_RECEPTIONS_MARKET_YET`;
2. one synthetic exact Bam Knight ARI `player_receptions` book-line -> 30/30 gates, deterministic match to sealed member `rb_000`, disposition `R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_PASS_MARKET_CAPTURED`.

The synthetic fixture was only a mechanical path test and is not evidence in the real R26R study.

Any repair after workflow launch must be documented before implementation and must preserve the frozen plan, 30 gates, exact R26Q parent, and authority ceiling.
