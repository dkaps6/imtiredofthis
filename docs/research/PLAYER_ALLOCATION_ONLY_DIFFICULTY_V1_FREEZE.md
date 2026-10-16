# Player Allocation-Only Difficulty V1 — Freeze

Date: 2026-10-14

Status: **FROZEN BEFORE ALLOCATION-ONLY SCORE**

Parents:
- `WR_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`
- `TE_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`
- `PLAYER_OPPORTUNITY_DIFFICULTY_TARGET_ALLOCATION_SOURCE_READY`
- WR `PLAYER_SHARE_EXCESS_VOLATILITY_MECHANISM_CONFIRMED`
- WR `JOINT_DIRICHLET_ALLOCATION_HISTORICAL_PASS_FREEZE_PROSPECTIVE_SHADOW`

## Question

After removing team pass-volume error, does THIS named WR/TE's own strictly-prior target-allocation difficulty still improve his next allocation distribution?

This is the bridge required before putting same-player opportunity difficulty into the finite team target allocator.

It is **not** a mean correction.

## Authority inputs

Recovered exact promoted authority:
- run `37782189538`
- artifact `11552447166`
- digest `sha256:94a4bee7011877b1d7a24708d5ba3e86ccc5d29b427bea650af574fc1f09fa24`

Share-volatility mechanism authority:
- run `38295515693`
- artifact `11768432238`
- digest `sha256:7b31dd29e4b94674d6948cf84af24fae7181d1a4f502a07beaa43919d4d7f3a6`

Target-allocation source authority:
- run `38655896001`
- artifact `11937475072`
- digest `sha256:4be927801792c646f859f9aa91918b8fa9817197f5bfab52769238f86cb541a2`

## Exact allocation-only construction

For each target player-game:

1. Recover the model's pregame team target total `T_hat`.
2. Preserve the promoted player target mean `mu_hat`.
3. Define promoted full-team target share:
   `q_hat = mu_hat / T_hat`.
4. Take the target game's realized player target share `s_actual` only as the outcome.
5. Define the counterfactual allocation-only realized target count under the model's own pregame team volume:
   `alloc_actual = T_hat * s_actual`.
6. Allocation-only residual:
   `alloc_error = mu_hat - alloc_actual`.

This removes team-volume error from the scored outcome. No realized team target total is used to construct `T_hat`, `mu_hat`, or `q_hat`.

## Player difficulty state

For each stable player identity:
- strictly-prior rows only;
- trailing max 8 allocation-error games;
- minimum 4 prior games;
- no target-game or same-week leakage.

State:
`player_alloc_mae = mean(abs(prior allocation errors))`.

Candidate zero-mean Normal width:
`sigma_player = player_alloc_mae * sqrt(pi/2)`.

Position-pool reference:
- contemporaneous season/week median `player_alloc_mae` across eligible same-position players;
- `sigma_pool = week_pool_mae * sqrt(pi/2)`.

No coefficient is fit.

## WR incremental reference

WR already has an independently qualified excess-share-volatility lineage. Therefore beating pooled WR uncertainty is not sufficient.

For the same WR row, use the frozen strictly-prior V2 `excess_share_var` and promoted `q_hat`:

`var_wr_v2 = T_hat*q_hat*(1-q_hat) + T_hat*(T_hat-1)*excess_share_var`

`sigma_wr_v2 = sqrt(max(eps, var_wr_v2))`.

This is a reference distribution only. It changes no mean and is not production promotion.

WR must show incremental proper-score value over this V2 reference on independent confirmation.

## Outcomes / proper score

Score `alloc_error` under zero-mean Normal distributions.

Primary proper score:
- Normal CRPS.

Coverage guards:
- central 80%;
- central 90%.

## Frozen seasons

WR:
- 2023 development;
- 2024 independent confirmation.

TE:
- 2024 development;
- 2025 independent confirmation.

No pooled season can rescue a failed independent confirmation.

## Frozen support

WR:
- 2023 >=700 rows / >=100 players;
- 2024 >=1,000 rows / >=140 players;
- pooled >=1,800 rows / >=150 players.

TE:
- 2024 >=600 rows / >=60 players;
- 2025 >=600 rows / >=60 players;
- pooled >=1,300 rows / >=90 players.

Stable authority-to-share-panel join is required. Any ambiguous normalized-name key fails closed.

## Inference

5,000 paired cluster bootstrap replicates.

WR candidate-vs-pool pooled gain:
- player cluster;
- game cluster.

WR candidate-vs-V2 **2024 confirmation** gain:
- player cluster;
- game cluster.

TE candidate-vs-pool pooled gain:
- player cluster;
- game cluster.

Seeds:
- player `2026101411`;
- game `2026101412`.

## WR hard gates

1. support passes;
2. candidate CRPS < pooled-position CRPS in 2023;
3. candidate CRPS < pooled-position CRPS in 2024;
4. pooled candidate-vs-pool CRPS gain >0;
5. pooled player-cluster CI low >0;
6. pooled game-cluster CI low >0;
7. 2024 candidate CRPS < WR V2 reference CRPS;
8. 2024 candidate-vs-V2 player-cluster CI low >0;
9. 2024 candidate-vs-V2 game-cluster CI low >0;
10. pooled 80% nominal gap no worse than pool;
11. pooled 90% nominal gap no worse than pool;
12. high-difficulty Q4 candidate CRPS improves vs pool.

If 1-6 and 10-12 pass but 7-9 fail:
`WR_PLAYER_ALLOCATION_DIFFICULTY_REDUNDANT_WITH_SHARE_VOLATILITY`.

If all pass:
`WR_PLAYER_ALLOCATION_DIFFICULTY_INCREMENTAL_CONFIRMED`.

Otherwise:
`NO_ACTIONABLE_WR_PLAYER_ALLOCATION_DIFFICULTY`.

## TE hard gates

1. support passes;
2. candidate CRPS < pool in 2024;
3. candidate CRPS < pool in 2025;
4. pooled gain >0;
5. pooled player-cluster CI low >0;
6. pooled game-cluster CI low >0;
7. pooled 80% nominal gap no worse;
8. pooled 90% nominal gap no worse;
9. high-difficulty Q4 CRPS improves.

If all pass:
`TE_PLAYER_ALLOCATION_DIFFICULTY_CONFIRMED`.

Otherwise:
`NO_ACTIONABLE_TE_PLAYER_ALLOCATION_DIFFICULTY`.

## Player-level reporting

For WR and TE separately:
- scored named players;
- rows by season;
- qualifying players with >=6 confirmation games;
- median per-player CRPS gain;
- fraction of qualifying players with positive CRPS gain.

Position aggregate alone cannot qualify the mechanism.

## Boundaries

- target means unchanged;
- team pregame target volume unchanged;
- no sportsbook;
- no OddsAPI;
- no 2026 outcomes;
- no production mutation;
- no support/window retune;
- no WR role/subgroup rescue;
- TE cannot be rescued by WR and vice versa.

A pass authorizes only a separately frozen joint team-conserving integration/shadow.
