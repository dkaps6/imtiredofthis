# RB 2026 Week 1 Current-Role Carry-Authority Audit Result

## Canonical run

- Branch: `audit-rb-2026-w1-current-role-carry-authority`
- Mechanical repair commit: `b6b6a04591ca375a1172cb58bd12da0a61b7fed6`
- Run: `34060593280`
- Job: `101560300616`
- Artifact: `9997365508`
- Artifact digest: `sha256:8c856deab95fdb61358754eacbe6a2c0bd6a7aedd8a63b3f8109e2638425ffd9`

## Integrity

- Rows: 107
- Teams: 32
- Ourlads-to-P3 identity coverage: 100%
- P3 parent parity max absolute difference: 0.0
- `RB_P3_SYNTHESIS_V1`: confirmed
- `WEEK1_STACK_OVERRIDE`: confirmed
- Sportsbook inputs used: 0
- Model fitting performed: 0
- Production change: 0
- Integrity gate: PASS

## Result

Official disposition: `CURRENT_ROLE_PRESERVED_BUT_NOT_DIRECT_ALLOCATION_INPUT`

The current Ourlads role is retained in PlayerForm as `depth_role`, but PlayerForm's effective `role` is replaced by the historical-usage-ranked `model_role`. Neither `simulation_v2.py` nor `simulation_rules.py` directly references `depth_role` in the rushing allocation path.

Observed audit counts:

- Depth-role vs model-role mismatch rows: 26
- Depth-role vs projected carry-order mismatch rows: 45
- Teams where no current depth-rank-1 back was the projected carry leader: 7
- Inactive players with positive projection: 0

The seven teams flagged by the frozen team-level check were ARI, CLE, GB, HOU, KC, NE, and SEA. Several player-level examples show the same mechanism directly: Jeremiyah Love is current `RB1` but effective/model `RB3`; Marshawn Lloyd is current `RB1` but effective/model `RB3`; David Montgomery is current `RB1` but effective/model `RB2`; Rhamondre Stevenson is current `RB1` but effective/model `RB2`; Jadarian Price is current `RB1` but effective/model `RB3`.

## Interpretation

This is a football-architecture finding, not a market-fitting result. The promoted P3 Week-1 path is internally coherent and sportsbook-independent, but current pregame backfield hierarchy does not have direct authority over carry allocation. That is a plausible mechanism for some of the Week-1 role/allocation concerns and is now established independently of sportsbook lines.

This result does **not** authorize an ad-hoc 2026 override or sportsbook matching. The next legitimate step is a frozen leakage-safe historical integration test in which timestamp-safe pregame RB depth state is introduced at the carry-allocation layer and compared against the unchanged P3 baseline. No thresholds or allocation multipliers may be selected from the 2026 sportsbook discrepancies.
