# RB-R3 Dynamic Workload Allocation — Frozen Plan

## Why this is new
RB-R1 decomposed carry error into RB-room volume vs individual allocation and found 17 CARRIES-dominant players whose primary submechanism was `INDIVIDUAL_ALLOCATION`. RB-R2 then showed that static/current role-state indicators (depth mismatch, rookie, injury-created context, limited history, new-team state) did **not** explain those allocation errors strongly enough.

RB-R3 therefore moves to a different football mechanism rather than retrying static depth:

**Does recent realized workload behavior within the RB room predict which allocation-dominant backs the current model will under- or over-allocate in the next game?**

This is a leakage-safe 2025 discovery diagnostic. It does not change production.

## Frozen evidence
- RB-R1 run: `34070566812`, artifact `rb-r1-room-volume-vs-individual-allocation`.
- Expected RB-R1 casebook rows: **1393**.
- Expected qualifying player profiles: **86**.
- Expected CARRIES-dominant / `INDIVIDUAL_ALLOCATION` submechanism players: **17**.
- Expected conditioned rows before signal availability filtering: **212**.
- No sportsbook data.

## Target
Primary target:
- `individual_allocation_component` from the exact RB-R1 Shapley decomposition, in carry units.

Secondary target:
- `carry_residual_actual_minus_pred = actual_att - pred_att`.

Positive target values mean the current model allocated too few carries to the player relative to realized usage.

## Dynamic workload construction
All signals are built only from chronologically earlier rows in the exact RB-R1 casebook. RB-room totals for a historical game use only that already-completed game. Signals group by `player_key + team` so a player's workload with a prior team is not silently treated as current-team workload.

Exact signals:
1. `PRIOR1_ROOM_CARRY_SHARE`
   - previous observed same-team game: `actual_att / actual_room_att`.
2. `ROOM_CARRY_SHARE_ACCEL_1V4`
   - prior1 room carry share minus the mean of the previous up-to-4 same-team room carry shares;
   - requires at least 3 prior same-team observations.
3. `PRIOR1_CARRIES`
   - actual carries in the previous observed same-team game.
4. `CARRIES_ACCEL_1V4`
   - prior1 carries minus mean carries over previous up-to-4 same-team games;
   - requires at least 3 prior same-team observations.

No alternate windows, EWMA, snap signals, depth interactions, injury interactions, or combinations are authorized in RB-R3.

## Evaluation
For each signal, freeze quartiles on the conditioned cohort's valid rows and report:
- conditioned N and valid N;
- coverage;
- Spearman with `individual_allocation_component`;
- Q4-minus-Q1 allocation-component gap;
- Q4-minus-Q1 carry-residual gap;
- enrichment of the under-allocation tail `individual_allocation_component >= 3.0 carries` in Q4 vs all valid rows;
- W2-18 allocation-component gap;
- W13-18 allocation-component gap;
- players with >=6 valid games;
- positive within-player Spearman rate among players with a computable association.

## Frozen discovery gate
A signal passes only if every condition is true:
1. valid N >= **150**;
2. coverage >= **0.70**;
3. Spearman >= **0.10**;
4. Q4-Q1 allocation-component gap >= **1.00 carry**;
5. Q4-Q1 carry-residual gap >= **1.50 carries**;
6. Q4 under-allocation-tail enrichment >= **1.20x**;
7. W2-18 allocation-component gap > 0;
8. W13-18 allocation-component gap > 0;
9. at least **10** players have >=6 valid games;
10. positive within-player association rate >= **0.60**.

Disposition is `RB_DYNAMIC_WORKLOAD_ALLOCATION_DISCOVERY_PASS` if at least one exact signal passes all gates; otherwise `NO_ACTIONABLE_RB_DYNAMIC_WORKLOAD_ALLOCATION_SIGNAL`.

## Interpretation rules
- No threshold/window rescue.
- No signal combinations after seeing results.
- No static-depth retry inside this migration.
- A pass is discovery only. It would authorize a separately frozen multi-season replication/source-expansion and then a full-stack carry-allocation experiment.
- A failure means recent realized carry workload alone is insufficient; the next legitimate source family would be participation/rotation information such as offensive snaps, route/third-down work, personnel grouping, or coaching rotation sequence.
