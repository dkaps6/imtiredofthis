# GSIS RB Successor Lineup V1 — Week 4 Prospective Lock Checkpoint

**STATUS: FIRST PROSPECTIVE WEEK LOCKED PREGAME — OUTCOMES SEALED.**

Parent:
- `GSIS_RB_SUCCESSOR_LINEUP_V1_PLAN.md`
- `GSIS_RB_SUCCESSOR_LINEUP_V1_PREGAME_LOCK_CONTRACT.md`

## Public-source authority

No-odds current football source:
- Full Slate run `36800721395`
- artifact `11135626008`
- digest `sha256:ecc7231cd8fe033af026b9cd96520d49f12f1d407ba951a7bb58cdd5da289509`
- source SHA `2483c5a5b787089a120c6b5a258233d33600a670`
- `FETCH_LIVE_ODDS=false`

Frozen Vacancy-V1 / active-successor public-source build:
- run `36887277745` — SUCCESS
- artifact `11174114183`
- digest `sha256:26ca9658e7a25245e1e195d65f5b75cd06fe4af3f519472d83a7c16846410aae`
- vacancy state SHA256 `bf4c39298f025833000f68104fb1d01161a6bb3fa311faa52bea8d023a54dc81`
- active successor pool SHA256 `b6052594ff4f59aac7ae965e5e678cda199dd0d02098824f1a761136e45112a3`
- exact target-event manifest SHA256 `7823f7f74c3030a50e0417125f72bcad9d04f51ae50f708d1ff13405ac6cd8eb`

Strict-prior snap source validated:
- 2026 Week 3 rows: 1,494
- 2026 Week 3 teams: 32
- duplicate rate: 0

No sportsbook data or target outcomes were read.

## Private GSIS authority

Private point-in-time snapshot SHA256:
`779b1bd6c5d4dbd390c992d1e02e007edb80dac7d5e19b8b9045b161000c8a23`

The snapshot was captured after the relevant teams' Week-3 games and before
both target Week-4 kickoffs. Team-specific Lineup Detail capture timestamps
passed the frozen `capture < kickoff` gate.

Raw GSIS rows and player-level derived candidate rows remain private and were
not uploaded to the public repository.

## First prospective lock

Private disposition:
`GSIS_RB_SUCCESSOR_LINEUP_V1_WEEK4_PRIVATE_ALLOCATION_LOCKED`

Public-safe counts:
- prospective weeks locked: **1**
- qualifying vacancy team-games locked: **2**
- successor player-games locked: **5**
- event timing failures: **0**
- snap transfer conservation failures: **0**
- GSIS transfer conservation failures: **0**
- target outcomes read: **false**
- sportsbook inputs read: **false**

Private immutable allocation-lock SHA256:
`0a6c7b066152748313d89d4b52d8a522616cda3144d8220320812a440e1c9153`

Private event-audit SHA256:
`cd7832519d0c335ab44d9a29215e8fc8bf8e355c4e02ee3873e81c11d942d33e`

The private files are preserved outside the public repository.

## Availability-source semantics correction

During the pregame lock audit, a separate production correctness defect was
found:

`espn_official_inactives_v1.py` queries ESPN's team **injury log**, filters
latest player status == OUT, but the current pipeline labels those rows as if
they came from a complete official game-day inactive section.

For this scientific lock the underlying facts are normalized conservatively as
**reported OUT / UNAVAILABLE_REPORTED**, not as game-day official-inactive
certification.

A separate repair lane / PR handles that production semantics defect. The GSIS
candidate formula was not changed.

## Scientific support state

The frozen terminal support gate remains:
- >=6 distinct future weeks;
- >=10 qualifying vacancy team-games;
- >=20 successor player-games.

Current locked support, before grading:
- **1 / 6 weeks**
- **2 / 10 vacancy team-games**
- **5 / 20 successor player-games**

This is a valid prospective observation lock, not enough for a PASS/FAIL.

No Week-4 result may be attached until the target games are final. No formula,
threshold, source, blend, or cohort change is authorized after outcome exposure.
