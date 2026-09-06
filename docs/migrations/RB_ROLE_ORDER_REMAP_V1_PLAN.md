# RB Role-Order Remap V1 — Frozen Plan

## Purpose

The 2026 Week-1 current-role audit established a new football-architecture fact: current Ourlads depth state is preserved in `depth_role`, but the promoted Week-1 STACK1/P3 carry-allocation path does not give that current role direct authority. This plan tests one deterministic, no-fit correction to that specific mechanism. It is not a sportsbook-matching exercise and does not reopen previously failed RB feature hunts.

## Parent evidence and exact lineage

- Production parent: `754d0f4ed34a06a65d2db36f2fb2ca10c58264ed` (`RB_P3_SYNTHESIS_V1`).
- Current-role audit result: `CURRENT_ROLE_PRESERVED_BUT_NOT_DIRECT_ALLOCATION_INPUT`.
- Audit result commit: `f27eccbe3007ae25ffe55c3f44e22ccaa00961c4`.
- Exact historical STACK1 source: run `33535308110`, artifact `9811878828`, digest `sha256:b66c3e403f54a8948f63c66524bc5b404f2fe13f73f3b5228ae50f21239b220c`.
- Exact timestamp-safe depth/context source already built for STACK2: run `33538770934`, artifact `9812754276`, digest `sha256:793b90494e2ea5a562e53be79311cb59b07d9589d934fb4237ae03e014ea774a`.

The historical STACK1 parent must reproduce the frozen 2025 all-RB metrics before the candidate is scored:

- rush attempts: n=1393, MAE=3.482575936421331, RMSE=4.741739937918672, bias=-0.8627247216838725, corr=0.7353960735701257.
- rush yards: n=1393, MAE=20.4241632527228, RMSE=30.06980647091156, bias=-5.537127660834512, corr=0.6168472895165802.
- 2025 Week 1 STACK1 rushing-yard MAE=20.09082962816558 on n=85.

Any parent drift is a mechanical failure, not a scientific result.

## Leakage/source contract

The evaluation season is 2025 only. The current depth state comes only from the timestamp-safe depth rows already materialized by the canonical STACK2 source path. For 2025 that source selected the latest depth snapshot strictly earlier than kickoff. Same-day-at/after-kickoff and future snapshots are prohibited.

Sportsbook lines, odds, market consensus, target-game outcomes, future games, postseason participation, and realized target-game role are prohibited as candidate inputs.

The existing STACK2 evidence showed 2025 depth coverage of 0.949749 on the 1393-row evaluation casebook. No new source window or source substitution may be chosen after results.

## Single frozen candidate

Candidate name: `ROLE_ORDER_REMAP_V1`.

There is exactly one candidate. No blend strengths, rank multipliers, carry bonuses, thresholds, or alternative role transforms will be searched.

For each 2025 team-week:

1. Start from the exact STACK1 player rushing-attempt means (`stack_att`) and rushing-yard means (`stack_yards`).
2. Only true RB/HB rows with a finite, timestamp-safe current depth rank are eligible for remapping. Fullbacks are left unchanged so an `FB1` rank cannot outrank an `RB1`.
3. Preserve the exact multiset and sum of STACK1 rushing-attempt means among eligible backs. Sort the eligible backs by current depth rank ascending. Ties preserve baseline authority by sorting original STACK1 attempts descending, then stable player key ascending. Sort the same eligible STACK1 attempt values descending and reassign those values in that role order.
4. Players without a usable current RB/HB depth rank retain their original STACK1 rushing-attempt mean.
5. Team/player rushing-attempt mass must therefore be exactly preserved to floating-point tolerance. This candidate changes only which current RB receives which already-existing STACK1 opportunity amount; it creates no new team carries.
6. Preserve each player’s STACK1 implied efficiency: `stack_ypc = stack_yards / stack_att` when `stack_att > 0.20`. Candidate rushing yards are `role_order_att * stack_ypc`. If a row lacks usable implied efficiency, its original `stack_yards` is retained and the row is flagged; no learned/fitted fallback is introduced.

This is intentionally a strong, parameter-free test of the exact architectural hypothesis: **does aligning the already-calibrated STACK1 opportunity magnitudes with timestamp-safe current RB hierarchy improve football accuracy?**

## Outcomes

Primary outcome: player rushing-attempt MAE.

Secondary outcomes:

- rushing-attempt RMSE, bias, and correlation;
- rushing-yard MAE/RMSE/bias/correlation with efficiency unchanged;
- 2025 Week 1 performance, because that is the live production route under review;
- W2-18 and W13-18 stability;
- current depth-rank RB1, RB2, and RB3 slices;
- teams where the baseline carry leader disagrees with current RB1 versus teams already aligned.

## Frozen gates

`ROLE_ORDER_REMAP_V1` is actionable only if **all** of the following hold:

1. Integrity/source gate passes: exact 1393-row STACK1 parent parity, depth source coverage >= 0.90, zero timestamp violations in the inherited source evidence, sportsbook inputs=0, model fitting=0.
2. Opportunity mass gate: maximum absolute team-week change in summed player STACK1 carries <= 1e-10.
3. Overall rush-attempt MAE improves by at least **1.0% relative** to the exact STACK1 parent.
4. 2025 Week-1 rush-attempt MAE is strictly lower than STACK1.
5. W2-18 rush-attempt MAE is strictly lower than STACK1.
6. W13-18 rush-attempt MAE is strictly lower than STACK1.
7. Rush-attempt MAE is lower for at least **two of current RB1/RB2/RB3**, and no one of those three slices worsens by more than **0.10 carries MAE**.
8. Overall rushing-yard MAE is strictly lower than STACK1.
9. 2025 Week-1 rushing-yard MAE is not worse than STACK1 by more than **0.10 yards**.
10. Absolute overall rushing-attempt bias is not worse than STACK1 by more than **0.10 carries**.

These gates are frozen before candidate results. They may not be lowered, waived, reinterpreted, or replaced after results.

## Dispositions

- All gates pass: `ROLE_ORDER_REMAP_V1_ACTIONABLE`.
- Integrity/source gate fails: `MECHANICAL_OR_SOURCE_FAILURE` and no scientific disposition.
- Any scientific gate fails: `ROLE_ORDER_REMAP_V1_NOT_ACTIONABLE`.

A diagnostic pass does **not** authorize production. It authorizes one separately frozen full-stack integration test in the actual PlayerForm → Bayesian/ML → rules/context → `simulation_v2` Monte Carlo → P3 path, with current role introduced at the carry-allocation layer before simulation output is finalized.

## Anti-hunting restrictions

If the candidate fails, do not try 25/50/75 blends, RB-rank multipliers, alternate depth windows, nearby thresholds, sportsbook-informed role weights, or permutations of the same rank-remapping idea. A materially new football signal/source would be required to reopen the question.
