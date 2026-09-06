# WR-ND4 — Role / Participation Source Audit Result

## Status

**COMPLETE — `SNAP_AND_DEPTH_SOURCES_RECOVERED`**

Canonical corrected run: `34053783657`

Canonical job: `101541889893`

Tested SHA: `d33e9f2b4d80df3eb5f98eea897d5bf8afc27442`

Artifact: `9995451798`

Artifact digest: `sha256:f718cdc3766a63eea6db803c7f5a9e0d8e009d5e30d5c5696dc37fad8f0b880f`

## Mechanical lineage

The first ND4 run (`34046811095`) failed before the source verdict because the canonical historical schedule artifact intentionally stores season/week/team/opponent but not the target-game date required for strict depth-chart timestamp filtering.

This was repaired mechanically only:

- exact game dates were restored from the same public nflverse schedule source already used elsewhere in the repository;
- no source gate, source family, casebook row, M38 reconstruction, model logic, or qualification threshold changed;
- corrected workflow commit: `d33e9f2b4d80df3eb5f98eea897d5bf8afc27442`.

## Parent integrity

Exact M38 reconstruction remained unchanged:

- `n = 4647`
- receiving-yard MC MAE = `17.099904733366`
- RMSE = `25.196099510685915`
- bias = `-5.238640833494836`
- correlation = `0.5679458508349821`

WR target-allocation evaluation casebook:

- rows = `2130`
- one known non-factorizable Isaiah Bond row excluded from evaluation only;
- sportsbook inputs used = `false`;
- model fitting used = `false`;
- production changed = `false`.

## Snap-count source result

Source status: **`ELIGIBLE_PRIOR_GAME`**

Loaded rows: `50,793`

Mapped rows: `50,701`

PFR-to-GSIS mapped row rate: `0.9981887268`

Mapped offense-percent non-null rate: `1.000000`

Coverage on the canonical WR casebook:

| Slice | Prior-1 snap % | Prior-2 mean snap % | Prior-4 mean snap % |
|---|---:|---:|---:|
| ALL | 0.960563 | 0.923005 | 0.852582 |
| W2-18 | 0.980529 | 0.940589 | 0.866201 |
| W13-18 | 0.980926 | 0.967302 | 0.934605 |
| WEEK1 | 0.645669 | 0.645669 | 0.637795 |

Frozen snap source gate: **PASS**.

## Depth-chart source result

Source status: **`ELIGIBLE_STRICT_PRE_GAME_DATE`**

Loaded rows: `554,215`

Depth coverage:

| Slice | Coverage | Rank usable among matched | Timestamp violations |
|---|---:|---:|---:|
| ALL | 0.978873 | 1.000000 | 0 |
| W2-18 | 0.981028 | 1.000000 | 0 |
| W13-18 | 0.974114 | 1.000000 | 0 |
| WEEK1 | 0.944882 | 1.000000 | 0 |

Frozen depth source gate: **PASS**.

GSIS-to-PFR casebook ID coverage: `0.9985915493` (`223 / 224` unique GSIS identities mapped).

## Participation source boundary

2025 nflverse play-level participation is **not canonical-gate eligible** for same-season walk-forward research because the 2023+ participation files are released after the postseason. It may not be used as though it were available before a 2025 target game.

Disposition for that source: `INELIGIBLE_POSTSEASON_RELEASE`.

## Scientific interpretation

ND3 showed that simple target-share recency and teammate-vacancy constructions did not clear the frozen dynamic-entitlement gate. ND4 establishes genuinely new information that was absent from ND3:

1. prior-game offensive participation / snap share;
2. strictly pregame depth-chart positional rank and slot.

These sources have enough historical coverage and timestamp integrity to support a new WR target-entitlement diagnostic without relaxing ND3's evidence standards.

## Authorized next step

A subsequent frozen branch may test whether **snap-role movement** and **strictly pregame depth-role movement/state** explain the post-M38 within-WR target-allocation residual and the established 10+ target / under-by-3 miss tail.

This authorization does **not** permit:

- M38 multiplier retuning;
- ND3 target-share window retuning;
- sportsbook inputs upstream;
- use of postseason-released participation as historical pregame truth;
- production changes without a separate predictive integration gate.
