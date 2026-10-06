# Week 4 Projection Authority Move-Direction Forward V1 — Observation 1

Status: **FORWARD_OBSERVATION_ONLY_INSUFFICIENT_SUPPORT**  
Date graded: 2026-10-06  
Parent plan: `docs/research/WEEK4_PLUS_PROJECTION_AUTHORITY_MOVE_DIRECTION_FORWARD_V1_PLAN.md`

## Authority

Week-4 graded board comes from the canonical recovered Week-4 artifact:
- recovery run: `36935917903`
- artifact: `11197776900`
- digest: `sha256:3f058570037ca016a5cbf1fa79e6e6abc4d845384de4dbdfaf477c8f0a3160a8`

Canonical Week-4 postmortem execution:
- run: `37482840038`
- head: `8c6225c4aae9876d6f3220d38424c57685613140`

No sportsbook re-fetch and no production change occurred.

## Frozen Week-4 observation

Among decided selected rows:
- STRENGTHENED: 111 rows, 48.65% win rate, -6.94u, model-closer rate 44.14%
- WEAKENED: 227 rows, 52.86% win rate, +1.95u, model-closer rate 47.14%
- UNCHANGED_DISTANCE: 41 rows, 51.22%, -1.58u
- CROSSED_OR_ON_LINE: 48 rows, 52.08%, +3.19u

Frozen primary differences, STRENGTHENED minus WEAKENED:
- win rate: **-4.21 percentage points**
- model-closer rate: **-2.99 percentage points**

10,000 game-cluster bootstrap, seed 42040:
- win-rate difference 95% CI: **[-15.51pp, +7.24pp]**
- model-closer difference 95% CI: **[-15.81pp, +9.86pp]**

Both intervals cross zero.

## Support state

Frozen confirmation floor:
- >=8 future eligible weeks
- >=400 strengthened rows
- >=400 weakened rows

Current:
- **1 / 8 weeks**
- **111 / 400 strengthened**
- **227 / 400 weakened**

The point estimate is directionally consistent with the Weeks 1-3 discovery pattern, but this is Observation 1 only and cannot authorize a rule.

Disposition:
`FORWARD_OBSERVATION_ONLY_INSUFFICIENT_SUPPORT`

No shrinking toward market, strengthened-row exclusion, edge threshold, or football-model change is authorized.
