# NFL HANDOFF — 2026-10-14 — TE INDIVIDUAL ALLOCATION UNCERTAINTY CONFIRMED / JOINT DIRICHLET TRANSMISSION CLOSED

GitHub is canonical. This supersedes the prior targeted-route/full-stack checkpoint for immediate research execution while preserving that lineage.

Repository:
- `dkaps6/imtiredofthis`

Production main:
- `812ac2cde79fe3846174613a2f2777e57b34b739`
- unchanged by this research.

Active research:
- branch `research-individual-opportunity-roadmap-2026-10-08`
- draft PR #672
- continuity Issue #535

No paid OddsAPI was used.
No sportsbook input entered research.
No 2026 outcomes were read.
No production model was mutated.

## 1. Non-negotiable model standard

This remains an **individual-player projection system**.

Position-wide aggregate metrics are guardrails, not the hypothesis grain.

Canonical state remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.

Do not use a failed shared/position-level candidate as evidence that named-player mechanisms are exhausted.

## 2. Targeted-route lane remains settled

Stage B canonical:
- `PLAYER_TARGETED_ROUTE_MATCHUP_CONFIRMED`
- run `38641891257`
- WR YPT and WR catch confirmed;
- WR direct receiving-yard residual closed;
- all TE route families closed.

Exact WR mean integration:
- `NO_ACTIONABLE_WR_ROUTE_MATCHUP_FULL_STACK_INTEGRATION`
- canonical run `38642241032`
- do not rescue coefficient/window/support/route-family/subgroup/cap.

The named-WR route-role mechanism remains scientifically real; the exact mean transmission is closed.

## 3. Opportunity uncertainty parents remain valid

WR:
- `WR_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`
- run `38595088817`.

TE:
- `TE_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`
- run `38594365393`.

RB:
- historical `RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED`.

Important recovered semantics:
`opportunity_error = (pred_targets - actual_targets) * pred_ypt`.

Therefore the earlier WR/TE uncertainty signal includes target-opportunity difficulty in yards-equivalent units and must not automatically be treated as a pure target-allocation width.

## 4. Target-allocation source lineage is now clean

Result:
`PLAYER_OPPORTUNITY_DIFFICULTY_TARGET_ALLOCATION_SOURCE_READY`

Canonical source audit:
- run `38655896001`
- job `116165448363`
- exact head `1cb9a582ef64305a5bfa7cdd47454b472c42e6fd`
- artifact `11937475072`
- digest `sha256:4be927801792c646f859f9aa91918b8fa9817197f5bfab52769238f86cb541a2`

Recovered exact fold authority:
- run `37782189538`
- artifact `11552447166`
- digest `sha256:94a4bee7011877b1d7a24708d5ba3e86ccc5d29b427bea650af574fc1f09fa24`
- manifest `WR_TE_FOLD_AUTHORITIES_RECOVERED_EXACT`
- provider rebuild false.

WR:
- 4,193 rows / 273 WRs;
- pregame team targets = `pred_targets / entitlement_tgt_share`;
- 100% recoverable.

TE:
- 3,214 rows / 167 TEs;
- pregame team targets = `b0_expected_targets / b0_target_probability`;
- TE-R5P full-team candidate share = `candidate_targets_r5p / pred_team_targets`;
- 100% recoverable.

No realized team target total is needed to manufacture a pregame player probability.

## 5. Allocation-only bridge — NEW KEY RESULT

Frozen result:
`docs/research/PLAYER_ALLOCATION_ONLY_DIFFICULTY_V1_RESULT.md`

Canonical:
- run `38656442803` SUCCESS
- job `116167234340`
- exact head `e936e3eb03bf085c7f2bce1249cfa97435eb4b37`
- artifact `11936407975`
- digest `sha256:7ded306cd0a049902130ccc5cab995231b192120c02679c3852757bc8df29cc0`

Construction explicitly removed team pass-volume error:
`alloc_error = promoted_player_targets - pregame_team_targets * actual_player_target_share`.

Same-player state:
- trailing max 8 strictly-prior allocation errors;
- minimum 4 games;
- no fitted coefficient;
- mean unchanged.

### TE — CONFIRMED

Disposition:
`TE_PLAYER_ALLOCATION_DIFFICULTY_CONFIRMED`

- 1,870 player-games / 121 TEs
- 2024 CRPS: `1.46973 -> 1.42466`
- 2025 independent confirmation: `1.37316 -> 1.32548`
- pooled: `1.43036 -> 1.38784` (~2.97% relative gain)
- player-cluster CI `[+0.02258,+0.06505]`
- game-cluster CI `[+0.02756,+0.05768]`
- 80% coverage `73.58% -> 77.33%`
- 90% coverage `82.09% -> 85.24%`
- high-difficulty Q4 CRPS `2.12865 -> 2.03949`
- 2025: 69 TEs >=6 games, median per-player gain `+0.01465`, 56.52% positive.

This proves the TE player-specific uncertainty signal is not merely team-volume noise.

### WR — exact allocation bridge CLOSED

Disposition:
`NO_ACTIONABLE_WR_PLAYER_ALLOCATION_DIFFICULTY`.

- 2023 candidate slightly worse than pool;
- 2024 candidate point estimate improves, but candidate-vs-existing WR V2 clustered CIs cross zero;
- pooled game-cluster CI crosses zero;
- coverage guards worsen slightly;
- 2024 median per-player gain is negative.

Do not rescue this exact WR bridge.

Existing WR excess-share-volatility / joint Dirichlet lineage remains WR allocation-uncertainty authority.

This WR closure does not erase `WR_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`.

## 6. TE joint Dirichlet transmission — CLOSED

Frozen result:
`docs/research/TE_PLAYER_ALLOCATION_JOINT_DIRICHLET_V1_RESULT.md`

Canonical:
- run `38656885725` SUCCESS
- job `116168645524`
- exact head `e34430812854588ade7d87b5fbfd5aed59c59f65`
- artifact `11937013082`
- digest `sha256:b6c8b885eb3185fc864bc49410dbe37c87f544ccbefc357447343f80ddfb04d4`

Disposition:
`NO_ACTIONABLE_TE_PLAYER_ALLOCATION_JOINT_DIRICHLET`.

Support/integrity:
- 735 candidate-applied rooms;
- 32 teams each season;
- 376 rooms / 204 games in 2024;
- 359 rooms / 202 games in 2025;
- zero N-inference failures;
- exact mean preservation;
- exact room count conservation.

But candidate fails:

2024:
- joint log-score gain `-0.00200`;
- marginal CRPS `1.52590 -> 1.64828` (worse).

2025:
- joint log-score gain `-0.07130`;
- marginal CRPS `1.42409 -> 1.54869` (worse);
- player-cluster marginal gain `-0.12460`;
- 95% CI `[-0.15259,-0.09771]`;
- 49 TEs >=6 games, median player gain `-0.12159`, 14.29% positive.

Interpretation:
- TE named-player marginal uncertainty remains confirmed;
- the exact assumption that the confirmed Normal/MAE width can be variance-matched into Beta-Binomial / Dirichlet-multinomial overdispersion is false.

Close exact:
`normal width -> count variance -> rho_i -> weighted room rho -> Dirichlet/Beta-Binomial`.

No rho scaling, coefficient fit, TE1 slice, window/support change, or pooled-season rescue.

## 7. Immediate legal frontier

Do NOT reopen the failed TE joint Dirichlet law.

Next step is a targeted anti-retest/integration audit of the **marginal TE distribution layer**:

1. inspect the actual historical TE-R5P / production-order distribution representation;
2. determine whether the confirmed named-TE marginal width is already represented, absent, or only position-pooled;
3. preserve TE-R5P target and receiving-yard means;
4. preserve the exact confirmed same-player marginal calibration semantics;
5. do not assume a different joint family is equivalent;
6. freeze any marginal calibration/shadow candidate before scoring;
7. require 2024 development + 2025 independent confirmation + player-cluster support;
8. no production promotion without a separate prospective/future-week lock.

WR:
- keep existing WR Dirichlet allocation lineage;
- no second WR allocation-width candidate from the closed bridge.

RB:
- preserve historical same-player width qualification; do not retune.

QB:
- no generic mean/YPA/bias reopening.

## 8. Mechanical-only failures to remember

Not scientific:
- source audit run `38655508713`: original Sept artifact expired;
- source audit run `38655625044`: shell formatting failure;
- allocation-only run `38656370089`: duplicate identity column merge failure.

All were repaired without changing frozen science.

## 9. Restart rule

Read:
1. `AGENTS.md`
2. newest TOP only in `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this handoff
4. `docs/research/PLAYER_ALLOCATION_ONLY_DIFFICULTY_V1_RESULT.md`
5. `docs/research/TE_PLAYER_ALLOCATION_JOINT_DIRICHLET_V1_RESULT.md`
6. newest Issue #535 comments
7. live PR #672 / branch / exact-head Actions

Do not recursively reread old handoffs.

No paid OddsAPI without explicit authorization.
No production mutation from research.
