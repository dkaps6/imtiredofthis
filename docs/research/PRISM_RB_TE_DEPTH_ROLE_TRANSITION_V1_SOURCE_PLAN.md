# PRISM RB/TE Discrete Depth-Role Transition V1 — Frozen Source Audit Plan

Date: 2026-10-17

Status: **FROZEN SOURCE / LINEAGE AUDIT ONLY / NO PREDICTIVE SCORING**

Model:
**PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_RB_TE_DEPTH_ROLE_TRANSITION_V1_SOURCE_AUDIT`

## 1. Player-first question

The Week-4/5 all-row forensic replay elevated this unresolved PRISM question:

> When a named player's current football state differs from his longer identity/baseline, is that difference a real role transition or short-term noise?

This source audit tests one deliberately narrow observable:

> Can PRISM reconstruct a discrete, named-player **pregame depth-role transition** for RB/FB/HB and TE with historical/prospective semantic parity, without using target-game outcomes or recent opportunity share?

Examples of the state concept are RB1 -> RB2, RB2 -> RB1, TE2 -> TE1, or the reverse, when the source itself explicitly carries a week-tagged depth position/rank.

This source audit does **not** score whether the transition predicts carries, targets, yards, receptions or touchdowns.

## 2. Why this is distinct from protected science

This V1 is not a rescue or relabeling of:

- raw recent-share replacement — CLOSED;
- Opportunity Authority Priority V1 — CLOSED;
- opportunity-state disagreement uncertainty — CLOSED;
- Stage-1 active-role participation — QUALIFIED ALL;
- Stage-2 active-set conditional allocation — QUALIFIED for RB rushing / RB receiving / TE receiving;
- RB carry/snap 50/50 state — protected;
- RB receiving-room share — protected;
- WR/TE target-share trajectory — protected;
- WR-R15 / TE-R5P snap entitlement — protected;
- teammate-availability Role Elasticity V1 — SOURCE_BLOCKED;
- RB Vacancy / vacated-opportunity transfer — CLOSED;
- generic depth-chart/personnel discontinuity as a QB-model retest — exhausted in the QB lineage.

The active-role V1 anti-retest rule forbids using starter/depth subsets to rescue the already-answered **binary participation** question. This source audit asks a different Stage-2/state-arbitration question: whether a discrete role label can be observed cleanly before kickoff for a named RB/TE.

No outcome is scored in this audit.

## 3. Frozen source family

Primary historical/prospective candidate source:
- nflverse / nflreadpy depth charts;
- only rows carrying explicit `season` + `week` semantics are eligible.

Pregame roster identity:
- nflverse weekly rosters;
- same target season/week;
- RB room = RB/HB/FB;
- TE room = TE;
- deterministic team + normalized-player identity join.

The existing `scripts/backtest/historical_inputs.py::_weekly_depth_lookup` leakage guard is binding:
- if the depth source is date-only and does not expose a reliable target week, it is unavailable;
- do not select an arbitrary/full-season snapshot;
- do not infer depth from target-game participation.

## 4. Frozen state construction

For each eligible named player/team/week:

- `depth_position`;
- `depth_team`;
- canonical `depth_role = depth_position + depth_team` when rank exists;
- prior eligible same-team depth role;
- `role_transition = current depth_role != prior eligible same-team depth_role`;
- `rank_transition` where both prior/current numeric depth ranks are available;
- transition direction only from the source rank itself.

Minimum chronology:
- current row must be target-week pregame source state;
- prior comparison row must have week < target week;
- no target-game snaps, carries, targets, receptions, yards or TDs may define the transition.

This source audit does not decide what a transition should do to the projection.

## 5. Frozen audit seasons

Historical:
- 2024;
- 2025.

Prospective-readiness probe:
- 2026 source schema / week-tag availability only.

The 2026 probe may establish that a same-semantic source exists for future use. It may not use 2026 outcomes, nor may a current download be relabeled as proof of what was known before a completed 2026 game.

## 6. Frozen source gates

Adjudicate RB and TE separately for 2024 and 2025.

Require ALL per position-season:

1. weekly roster team-week coverage >= 95%;
2. depth source explicitly contains season + week;
3. roster -> depth identity coverage >= 90%;
4. nonblank discrete depth-role coverage >= 85% of roster player-games;
5. numeric depth-rank coverage >= 80% of roster player-games;
6. prior same-team role-state coverage >= 80% for Week 3+ rows;
7. at least 100 distinct named players with prior-state support for RB and >=80 for TE;
8. at least 100 observed role-transition player-games for RB and >=60 for TE across the season;
9. chronology violations = 0;
10. duplicate team/week/player role identities = 0;
11. target-game outcomes used = 0;
12. target-game snaps used as predictor/source state = 0;
13. sportsbook inputs = 0;
14. OddsAPI calls = 0;
15. predictive models fit = 0;
16. thresholds searched = 0;
17. production changes = 0.

Do not lower these gates after the audit runs.

## 7. Prospective parity gate

The 2026 source probe must report:

- whether `load_depth_charts(2026)` is available;
- whether explicit week tags exist;
- latest week represented;
- RB/TE roster -> depth coverage on available week-tagged rows;
- whether role/rank semantics match the historical fields used above.

If the current source is date-only, stale, or materially schema-incompatible, prospective parity fails closed even if 2024/2025 historical support is strong.

## 8. Dispositions

Per position:
- `RB_DEPTH_ROLE_TRANSITION_SOURCE_READY` or `RB_DEPTH_ROLE_TRANSITION_SOURCE_BLOCKED`;
- `TE_DEPTH_ROLE_TRANSITION_SOURCE_READY` or `TE_DEPTH_ROLE_TRANSITION_SOURCE_BLOCKED`.

Overall:
- both historical families + prospective parity ready:
  `PRISM_RB_TE_DEPTH_ROLE_TRANSITION_SOURCE_READY_ALL`;
- one family ready:
  `PRISM_RB_TE_DEPTH_ROLE_TRANSITION_SOURCE_READY_PARTIAL`;
- none / prospective parity invalid:
  `PRISM_RB_TE_DEPTH_ROLE_TRANSITION_SOURCE_BLOCKED`.

## 9. If source-ready

Do **not** change a projection.

Freeze a separate predictive mechanism before scoring. The future mechanism must:
- remain named-player + target-game + market specific;
- test incremental information beyond active-role probability, qualified Stage-2 allocation and the protected RB carry/snap / TE entitlement states;
- use 2024 development and 2025 independent confirmation without changing the source definition;
- never become a universal “RB1 gets a boost” rule.

## 10. If blocked

Fail closed.

Do not:
- infer role from target-game snaps/opportunities;
- substitute raw recent carries/targets;
- lower identity/coverage gates;
- use current Ourlads labels as if they were historically identical without a proven bridge;
- convert the result into vacancy transfer;
- use sportsbook listing presence as role evidence.

## 11. Integrity at freeze

- predictive candidates scored: **0**;
- target outcomes used: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- fitted coefficients: **0**;
- thresholds searched: **0**;
- production changes: **false**.

Disposition at freeze:
`PRISM_RB_TE_DEPTH_ROLE_TRANSITION_V1_SOURCE_AUDIT_FROZEN`.
