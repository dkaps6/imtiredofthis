# Player State Live Coverage V1 — Result

**STATUS: COMPLETE / READY FOR PROSPECTIVE SHADOW**

Frozen plan:
`docs/research/PLAYER_STATE_LIVE_COVERAGE_V1_PLAN.md`

Authority:
- branch: `research-player-state-live-coverage-v1`
- run: `37560311001` — **SUCCESS**
- source SHA: `e8f90073f63a01c4457b6bb412114f4f5b9f0d82`
- artifact: `11456556226`
- digest: `sha256:a39b958e492a781e310de0f14d34153e21ca589a76cd226479b3b39e10f9328e`
- captured at: `2026-10-07T02:06:16.355584+00:00`
- target: 2026 Week 5

Final disposition:

`PLAYER_STATE_LIVE_READY_FOR_PROSPECTIVE_SHADOW`

## Prospective boundary

This result was created without:
- Week-5 player outcomes;
- Week-5 snap outcomes;
- sportsbook inputs;
- fitted candidate models;
- production mutations.

Chronology violations: **0**.

The audit script explicitly fails closed if target/future Week-5 player-stat or snap rows are already present.

## Coverage

Week-5 scheduled teams:
- **30**

Roster authority:
- Week-5 weekly roster was available.

Skill-player universe:
- **741** QB/RB/WR/TE rostered players.

Stable identity:
- **99.8650%** coverage.

Players with at least one 2026 game:
- **400**

Current 2026 usage coverage among played players:
- **99.75%**

Latest strictly-prior offensive-snap coverage among played RB/WR/TE:
- **99.7159%**

Week-5 injury rows in the nflreadpy injury source at capture:
- **0**

The absence of a published Week-5 injury row does not invalidate usage/snap state. Injury-created vacancy remains a separate source lane and is not silently inferred here.

## RB room evidence

Week-5 RB/HB/FB roster universe:
- **179** players.

RBs with both current room-opportunity state and snap state:
- **98**

Scheduled multi-back rooms with usable state:
- **30 / 30**

Rooms meeting the predeclared descriptive “distinguishable state” condition:
- **30 / 30**

The condition was frozen before the result:
- max-minus-min last-3 RB-room opportunity share >= 0.20, OR
- max-minus-min last-3 RB-room snap fraction >= 0.20.

This threshold is descriptive only. It is not a production router or candidate-selection rule.

Examples of current strictly-prior room separation:
- ARI: opportunity gap 0.6885 / snap gap 0.4806
- BAL: 0.7808 / 0.6067
- BUF: 0.9492 / 0.4777
- DET: 0.7586 / 0.7687
- IND: 0.6962 / 0.6667
- PIT: 0.7586 / 0.6022

Every scheduled RB room contained meaningful pregame differences among individual backs.

## Full-stack interpretation

This does **not** show that the current model treats all RBs identically.

The canonical stack already consumes player history through PlayerForm/Bayesian state.

What this result establishes is narrower and more important:

> A dense, leakage-safe, current individual RB-room state exists before Week 5, and the non-Week-1 production lineage does not currently have the same promoted multiseason room-allocation specialist that WR-R15 and TE-R5P provide for their rooms.

This is exactly the seam identified by the older frozen Joint Opportunity + Player Entitlement architecture:

`finite team opportunity -> finite room -> individual entitlement -> separate efficiency -> joint simulation`

No current production authority needs to be removed to investigate it.

## Production-consumption context

Protected specialist lineage remains:

- QB: M89/M90 individual history + team/opponent environment; C2 distribution.
- WR: M38 plus WR-R15 individual entitlement.
- TE: TE-R5P individual entitlement.
- RB Week 1: P3/R26/R22 specialist authorities.
- RB non-Week-1: canonical PlayerForm/Bayesian history + ensemble/joint MC, but no equivalent promoted multiseason RB room-entitlement specialist.
- RB Rush+Receiving Conservation V2 remains protected.

## Scientific authorization

This READY disposition authorizes only a separately frozen **prospective player-state shadow**.

It does **not** authorize:
- another retrospective RB rushing backtest;
- 2025 feature tuning;
- a production change;
- a depth-chart remap;
- a post-hoc residual correction;
- reopening the M96 chain;
- sportsbook-conditioned football generation.

Any Week-5 shadow must be formula-locked and row-universe-locked before target outcomes are read and must accumulate future untouched weeks before a promotion decision.

No paid OddsAPI pull is required.
