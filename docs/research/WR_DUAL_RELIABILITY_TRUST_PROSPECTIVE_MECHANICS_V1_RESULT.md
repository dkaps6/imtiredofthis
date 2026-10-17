# WR Dual Reliability Trust — Prospective Mechanics V1 Result

Date: 2026-10-17

Status: **MECHANICS READY / NO WEEK-7 SCIENTIFIC LOCK CREATED / RESEARCH ONLY**

Model:
**PRISM — Player Role & Individual Situation Model**

Frozen scientific parent:
`docs/research/WR_DUAL_RELIABILITY_TRUST_PROSPECTIVE_CONFIRMATION_V1_PLAN.md`

## Canonical mechanics execution

- run `39001244942` — **SUCCESS**
- exact head `0fbfd104cbea803d8afbbcc9c56b23414b309c88`
- job `117304434689`
- artifact `12088664427`
- digest `sha256:384850e42c28b53171a8f3062998cdf910dcc1714e36d1abea0a1e4acbee7418`
- 4 / 4 focused mechanics tests PASS
- strict repository audit PASS

Disposition:

`WR_DUAL_RELIABILITY_PROSPECTIVE_MECHANICS_READY__NO_WEEK7_LOCK`

## What was implemented

Generic pregame lock builder:

`scripts/research/lock_wr_dual_reliability_prospective_v1.py`

Focused guard tests:

`tests/test_wr_dual_reliability_prospective_v1.py`

Validation workflow:

`.github/workflows/research-wr-dual-reliability-prospective-mechanics-v1.yml`

The builder preserves the already-frozen Weeks 7-10 scientific plan:

- same named WR only;
- trailing maximum 8 qualifying games;
- minimum 4 prior games for both opportunity and efficiency histories;
- opportunity and efficiency same-player MAEs remain separate;
- percentile ranks use average ranking;
- `dual_fragility_score = (opportunity_fragility_score + efficiency_fragility_score) / 2`;
- frozen WR-R15 receiving-yard projection is retained on the lock row;
- DUAL_HIGH_TRUST <= 0.25;
- DUAL_LOW_TRUST >= 0.75;
- no fitted weight.

## Fail-closed chronology

The builder refuses to create a target-week lock unless strict-prior current-season
history is complete through exactly `target_week - 1`.

Therefore:
- a Week-7 lock requires history through completed Week 6;
- history stopping at Week 5 fails closed;
- any Week-7-or-later outcome row in a Week-7 input fails closed;
- Week 6 cannot be backfilled as a dual-composite target week.

This is stronger than merely requiring history `<= W-1`: it prevents a
scientific target lock from being created from stale/incomplete prior-week state.

## Ticket-layer interpretation

This mechanics milestone preserves the PRISM / Construct-NFL-Tickets separation:

`FOOTBALL PROJECTION -> RELIABILITY/TRUST -> MARKET COMPARISON -> APPROVED PLAY -> ALT-LINE/CORRELATION/TICKET CONSTRUCTION`.

Reliability is attached to the named player's unchanged football projection.
It is not:
- an over/under direction;
- a market edge;
- a sportsbook-conditioned selector;
- a new football mean;
- a new distribution width.

If prospectively confirmed after the fixed Weeks 7-10 horizon, it can later
inform how much confidence to place in an otherwise-valid named-player
projection before ticket construction.

## Integrity

Canonical successful mechanics run:
- Week-7 scientific rows frozen: **0**
- Week-6 outcomes read: **0**
- Week-7 outcomes read: **0**
- Week-6 dual composite backfilled: **false**
- sportsbook inputs: **0**
- OddsAPI calls: **0**
- models fit: **0**
- projection/distribution values changed: **0**
- production changed: **false**

The earlier run `39001205766` proved the same code/tests but failed the
repository-wide audit only because the focused workflow had not installed
PyYAML. That mechanical dependency was repaired without changing the science.
The canonical authority is the successful exact-head run above.

## Exact next action

Do **not** freeze Week 7 now.

Wait until Week 6 is legitimately complete and strict-prior Week-6 history plus
Week-7 roster/slate identity are ready. Then execute the pregame Week-7 dual
reliability lock under the already-frozen plan.

Week-6 component trust locks and the Week-6 Stage-2 shadow remain immutable and
must not be graded early.

No generic opponent/environment candidate is authorized at this checkpoint.
