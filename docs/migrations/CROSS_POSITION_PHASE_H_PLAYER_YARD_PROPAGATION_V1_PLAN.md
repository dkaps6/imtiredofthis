# CROSS-POSITION PHASE H — PLAYER YARD PROPAGATION V1

## Purpose
Phase G passed every frozen opportunity gate and established a prospective positive-only shared pass-state component for QB attempts and WR target-pool opportunity while preserving TE and RB exactly.

Phase H tests whether that exact opportunity improvement translates into better **individual-player yard projections**. It is not a new feature search and it does not alter efficiency models, individual WR hierarchy, TE-R5, RB-P3, or production.

## Frozen lineage
- Phase C all-player rows: run `34147777341`.
- Phase G positive-only opportunity component: run `34150685094`, disposition `POSITIVE_SHARED_PASS_STATE_ELIGIBLE`.
- QB mean authority remains M89/M90.
- WR individual hierarchy remains M38.
- TE authority remains TE-R5.
- RB rushing authority remains RB-P3.

## Cohort
Use the exact 2025 team-games scored by Phase G and the Phase-C player rows belonging to those team-games.

No new cohort filtering based on outcome is allowed.

## Frozen propagation
### QB
For each scored QB team-game:
- `attempt_ratio = candidate_qb_attempts / pred_qb_attempts`.
- `candidate_qb_yards = baseline_qb_yards * attempt_ratio`.

This holds the existing M89/M90 implicit yards-per-predicted-attempt mean fixed. No YPA or efficiency feature changes.

### WR
For each scored team-game:
- `wr_pool_ratio = candidate_wr_targets / pred_wr_targets`.
- each WR's candidate targets = `baseline_player_targets * wr_pool_ratio`.
- each WR's candidate receiving yards = `baseline_player_receiving_yards * wr_pool_ratio`.

This preserves M38 target hierarchy and every player's existing expected yards per target exactly. No individual entitlement reranking, catch-rate adjustment, YPR adjustment, or explosive adjustment is allowed.

### TE / RB
- TE target and receiving-yard projections remain unchanged exactly.
- RB rushing opportunity and rushing-yard projections remain unchanged exactly.

## Frozen scorecard
### QB
Report baseline vs candidate:
- yard MAE, RMSE, bias, correlation, p90 absolute error;
- 100+ yard catastrophic miss count;
- underprojection and overprojection catastrophic counts separately;
- PASS_STATE_HIGH and PASS_STATE_LOW descriptive slices.

### WR
Report baseline vs candidate:
- all-player receiving-yard MAE, RMSE, bias, correlation, p90 absolute error;
- 50+ yard catastrophic miss count;
- underprojection and overprojection catastrophic counts separately;
- opportunity quartiles Q1-Q4, with Q4 primary star-player slice;
- M38 within-team target rank parity.

### TE / RB guards
- max absolute player-yard change must be <=1e-9.

## Frozen eligibility gates
`PLAYER_YARD_PROPAGATION_ELIGIBLE` only if all are true:
1. QB yard MAE improves by >= **1.00 yard**.
2. QB p90 absolute error does not worsen.
3. QB 100+ yard catastrophic miss count does not increase.
4. WR receiving-yard MAE improves by >= **0.20 yard**.
5. WR Q4 receiving-yard MAE improves by >= **0.50 yard**.
6. WR p90 absolute error does not worsen.
7. WR 50+ yard catastrophic miss count does not increase.
8. M38 within-team target ordering is unchanged exactly.
9. TE player-yard max absolute change <=1e-9.
10. RB player-yard max absolute change <=1e-9.
11. sportsbook inputs = 0; same/future outcomes used as predictors = 0; production parameters changed = 0.

Otherwise disposition is `PLAYER_YARD_PROPAGATION_NOT_ELIGIBLE`.

No gate, cohort, ratio rule, hierarchy rule, or efficiency assumption may be changed after results.

## Next-step rule
If eligible, Phase H becomes evidence for assembling the deadline production candidate using:
- M89/M90 QB mean + Phase-G positive pass-state opportunity;
- M38 WR hierarchy + Phase-G positive WR-pool opportunity;
- TE-R5 unchanged;
- RB-P3 unchanged;
- C2 retained for its demonstrated QB distribution/conservation value only under an explicitly frozen integration test.

The next step after a Phase-H pass is therefore production-candidate integration / Week-1 player-by-player audit, not another broad diagnostic loop.
