# Player State Live Coverage V1 — Frozen Prospective Source/Architecture Audit

**STATUS: FROZEN BEFORE AUDIT RESULT. NO OUTCOME SCORING. NO PRODUCTION CHANGE.**

Target slate:
- season: 2026
- target week: 5

## Purpose

Test the user's player-centric hypothesis at the exact pregame information layer:

> Before Week 5, how much player-specific state is publicly observable for each QB/RB/WR/TE, and how much of that state is currently consumed by the promoted production lineage?

This is an architecture/source audit only.

It does not fit a model, alter a projection, grade Week-5 outcomes, use sportsbook data, or reopen any closed retrospective science.

The intended direction is complementary:

`existing validated production stack + any genuinely missing player-specific state`

not a replacement of the current stack.

## Standing science boundaries

Preserve all promoted authorities:
- QB M89/M90 and C2;
- WR M38 / WR-R15;
- TE-R5P;
- RB Week-1 P3/R26/R22;
- RB Rush+Receiving Conservation V2;
- Discrete Count Mean Alignment V1;
- current joint-MC / production orchestration.

The standing M96 prohibition remains binding:
- no new retrospective RB rushing feature/router test;
- no 2025 retuning;
- no threshold search;
- no depth-rank remap;
- no residual-calibration rescue;
- no historical YPC/YPT/YAC-family mean correction.

Any later RB work must be genuinely prospective 2026 or use a separately justified new source.

## Public pregame sources

Use only free nflreadpy/nflverse data available before Week 5:
- 2026 weekly player statistics through Week 4;
- 2025 weekly player statistics as prior-season identity/history;
- 2026 weekly offensive snap counts through Week 4;
- 2026 weekly rosters;
- 2026 Week-5 regular-season schedule;
- 2026 Week-5 injury report if published.

No Week-5 player result/stat row may be read.

## Week-5 universe

Start from weekly-roster skill players on teams scheduled in Week 5.

Eligible position groups:
- QB
- RB/HB/FB
- WR
- TE

Resolve stable identity using available GSIS/PFR IDs first and fail visibly to a normalized name/team key only when no stable ID exists.

Do not infer sportsbook-offered player universe.

## Player-specific state to audit

For every Week-5 player record, attempt to construct strictly-prior state:

### Individual history
- 2025 games played;
- 2026 games played through Week 4;
- current-season rush attempts / team RB-room rush share where applicable;
- current-season targets / team receiving target share where applicable;
- last-1 game individual carries / targets;
- last-3 game individual carries / targets;
- current-vs-prior-season usage direction where both exist.

### Participation
- latest strictly-prior offensive snap percentage;
- mean last-3 strictly-prior offensive snap percentage;
- number of prior 2026 snap games;
- same-team continuity into Week 5.

### Competition / room state
Within the target team and position room:
- number of rostered room competitors;
- player's latest-snap rank;
- player's last-3 snap-share fraction of room total;
- player's last-3 carry/target share fraction of room opportunity where applicable;
- concentration of the room using Herfindahl concentration from strictly-prior opportunity.

Depth rank alone is diagnostic context and may not become direct workload authority.

### Availability
- target-week injury status if publicly available;
- whether any same-room teammate is OUT/DOUBTFUL;
- number of unavailable same-room teammates.

No current game inactive list is assumed before publication.

### Environment identity
- target opponent;
- team identity;
- team/room continuity.

This audit does not score opponent matchup quality. Existing opponent/team science remains separate and protected.

## Production-consumption map

Classify whether each state family currently reaches the promoted lineage:

### QB
Current individual mean specialist:
- strict-prior individual attempts/YPA: CONSUMED by M89/M90.
Current snap/room state:
- not primary QB mean authority.

### WR
- WR1 M38 hierarchy: CONSUMED.
- WR2+ strict-prior participation/continuity: CONSUMED by WR-R15.
- individual receiving efficiency × environment response: NOT FULLY INDIVIDUALIZED.

### TE
- strict-prior participation/continuity entitlement: CONSUMED by TE-R5P.
- individual receiving efficiency × environment response: NOT FULLY INDIVIDUALIZED.

### RB
- Week-5 has no promoted P3/R26/R22 Week-1 specialist.
- individual strict-prior backfield participation/room allocation state: NOT represented by an equivalent promoted multiseason room specialist.
- canonical PlayerForm/Bayesian individual history remains consumed, but room-specific current-state allocation is the open seam.

This map is a code-lineage statement, not a predictive claim.

## Frozen audit outputs

Persist:
1. row-level Week-5 player state;
2. position coverage summary;
3. room-state summary;
4. production-consumption matrix;
5. RB-specific prospective readiness summary.

Required RB readiness facts:
- number of scheduled Week-5 RB/HB/FBs;
- stable-ID coverage;
- latest snap coverage;
- last-3 snap coverage;
- current carry-history coverage;
- teams with >=2 RB/FBs and enough strictly-prior state to distinguish their room;
- count of RB rooms where recent individual state materially differs between players.

For the final item, “materially differs” is diagnostic only and frozen as:
- max minus min last-3 room opportunity share >= 0.20, OR
- max minus min last-3 room snap fraction >= 0.20.

This threshold is only a descriptive coverage statistic. It is not a candidate selection threshold and may not be used to route production.

## Dispositions

`PLAYER_STATE_LIVE_READY_FOR_PROSPECTIVE_SHADOW`

requires:
- Week-5 schedule source present;
- >=90% stable-ID coverage among skill-player universe;
- >=90% latest-prior snap coverage among RB/WR/TE players with at least one 2026 game;
- >=90% current usage coverage among players with at least one 2026 game;
- zero Week-5 outcome rows read;
- zero chronology violations;
- zero sportsbook inputs;
- at least 20 scheduled RB/HB/FBs with both room opportunity and snap state;
- at least 10 multi-back rooms with distinguishable individual state.

If source coverage is adequate but one of the RB-specific support floors misses:
`PLAYER_STATE_LIVE_PARTIAL`.

If chronology/source identity is not safe:
`PLAYER_STATE_LIVE_NOT_READY`.

## What a READY result authorizes

Only a separately frozen **prospective shadow** design beginning Week 5 or later.

It does not authorize:
- a retrospective RB backtest;
- a production projection change;
- changing M89/M90, M38, WR-R15, TE-R5P, RB V2, or C2;
- using Week-5 outcomes to design the shadow after the fact.

If a prospective shadow is launched, its formula and row universe must be hash-locked before any target game starts, and evaluation must accumulate across future weeks.

Sportsbook inputs: **0**  
Week-5 outcomes authorized: **0**  
Candidate models fit: **0**  
Production mutations: **0**
