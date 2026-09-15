# WR Phase 4B Authority-Exact Opportunity Attribution V1 — Frozen Plan

## Status

Research-only diagnostic. No challenger model, no production change, no sportsbook inputs, no paid Full Slate, no RB work.

This plan follows:
- canonical WR-R15 authority run `34238301577`, artifact `10061328722`, digest `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`;
- canonical Phase 4A source reconciliation run `34919390334`, artifact `10376694364`, digest `sha256:069dfe4d935d58e9fd232dea9c08ae07a306ade811e7382ccf5f5b23d4c89548`;
- Claude anchor-presence reconciliation run `34919926239`, which confirms `71` genuinely full graded WR rooms and `8` anchor-absent identity-set matches;
- Claude Phase 4B adversarial review comment `5673594282`.

The 2023 and 2024 WR-R15 authority seasons are diagnostic evidence only. They are not a fresh validation set for any later challenger.

## Population authorities

- Candidate WR-R15 graded rows: `4,193` = `2,076` in 2023 + `2,117` in 2024.
- Authority team-games: `1,088`.
- WR1-anchor-observable team-games: `1,026`.
- Canonical WR2+ feature identities: `5,321` rows from `wr_r15_confirmation_features.csv`, with `baseline_wr_rank >= 2`.
- Genuine full graded WR-room source-validation subset: `71` team-games, requiring WR1 anchor present plus exact WR2+ identity-set equality.

`player_clean_key` is the shared authority identity key for prediction/features reconciliation. nflverse weekly player stats / PlayerForm-normalized weekly data are the cleared actual-target source.

## Layer 1 — exact receiving-yard residual accounting

Population: all `4,193` candidate graded rows.

For each row:
- `T_p = pred_targets`
- `T_a = actual_targets`
- `Y_p = mc_rec_yards`
- `Y_a = actual_rec_yards`

Fail-closed consistency assertions:
- if `T_p == 0`, require `abs(Y_p) <= tolerance`;
- if `T_a == 0`, require `abs(Y_a) <= tolerance`.

Efficiency terms are assigned symmetrically:
1. If `T_p > 0` and `T_a > 0`:
   - `E_p = Y_p / T_p`
   - `E_a = Y_a / T_a`
2. If `T_p == 0` and `T_a > 0`:
   - `E_a = Y_a / T_a`
   - set `E_p := E_a`
   - therefore all residual is opportunity and efficiency contribution is exactly zero.
3. If `T_a == 0` and `T_p > 0`:
   - `E_p = Y_p / T_p`
   - set `E_a := E_p`
   - therefore all residual is opportunity and efficiency contribution is exactly zero.
4. If `T_p == 0` and `T_a == 0`:
   - require both yard values zero;
   - set both contributions to zero directly; no efficiency value is needed.

For all other rows use the exact symmetric identity:
- `opportunity_yards = (T_a - T_p) * (E_a + E_p) / 2`
- `efficiency_yards = (E_a - E_p) * (T_a + T_p) / 2`

Hard assertion row-by-row:
`opportunity_yards + efficiency_yards == Y_a - Y_p` within tolerance.

Report pooled, 2023, 2024, WR1, WR2+:
- mean and median absolute contribution by layer;
- signed mean contribution;
- share with `abs(opportunity) > abs(efficiency)`;
- same summaries for actual 100+ receiving-yard games and absolute residual >=30-yard games.

This is accounting, not causality.

## Layer 2 — team target-pool vs modeled WR-room share

Population: **only the 1,026 team-games with an observable WR1 anchor**. The 62 anchor-unobservable team-games are excluded from full-room attribution because a full modeled-room identity set cannot be constructed honestly from the authority artifact.

For each included team-game:
- predicted team target pool = the common `pred_targets / entitlement_tgt_share` ratio, verified constant within team-game and equal between baseline/R15 arms;
- actual team targets = nflverse weekly targets summed across all player positions;
- predicted WR-room share = conserved `candidate_wr_room_mass` from the R15 conservation audit;
- predicted WR-room targets = predicted team target pool × predicted WR-room share;
- actual canonical WR-room targets = weekly targets summed **only across the canonical modeled-room identities**: observable WR1 anchor + all canonical WR2+ feature identities for that team-game;
- actual canonical WR-room share = actual canonical WR-room targets / actual team targets.

Exact symmetric target-count decomposition:
- `team_pool_component = (actual_team_targets - pred_team_targets) * (actual_wr_share + pred_wr_share) / 2`
- `wr_room_share_component = (actual_wr_share - pred_wr_share) * (actual_team_targets + pred_team_targets) / 2`

Hard assertion:
`team_pool_component + wr_room_share_component == actual_canonical_wr_targets - predicted_wr_room_targets` within tolerance.

Report pooled + season:
- target-count MAE and signed bias for full modeled room;
- mean/median absolute team-pool and WR-room-share components;
- which component is larger in absolute magnitude by row;
- distributions of actual/predicted team target pool and WR-room share.

### Off-model target-mass disclosure

On the same 1,026 anchor-observable team-games, separately calculate weekly targets credited to raw-position WRs **outside** the canonical modeled-room identity set.

Report mean, median, P90, max, total targets, number/share of team-games with any off-model WR target mass, and the largest examples. These targets are **not** folded into Layer 2 or Layer 3 actual-room values.

For the 62 anchor-unobservable games, report coverage only; do not interpret raw-position WR mass as off-model because the missing anchor identity makes that classification ambiguous.

## Layer 3 — M38 WR1 vs secondary-pool responsibility

Population: same `1,026` anchor-observable team-games only.

- WR1 predicted targets = the immutable M38 anchor `pred_targets` from the candidate authority row.
- WR1 actual targets = cleared weekly actual targets for that exact anchor identity.
- predicted canonical secondary-pool targets = predicted full modeled WR-room targets − predicted WR1 targets.
- actual canonical secondary-pool targets = weekly targets summed directly across the canonical WR2+ feature identities for that team-game.

Report target MAE, RMSE, signed bias pooled + 2023 + 2024 for:
- WR1 anchor;
- canonical WR2+ secondary pool.

Do not assign the 62 anchor-unobservable games to M38 or R15 role layers.

## Layer 4 — R15's actual responsibility: within-WR2+ allocation

Population: all `5,321` canonical WR2+ feature rows. This is a diagnostic extension of the authority feature cohort, not a new holdout.

For each WR2+ identity:
- baseline predicted targets = `baseline_entitlement_tgt_share × implied_team_target_pool`;
- R15 predicted targets = `candidate_entitlement_tgt_share × same implied_team_target_pool`;
- actual targets = cleared weekly actual target count for that exact feature identity; a missing weekly row may be treated as zero only if the weekly source convention establishes that the player had zero targets rather than an unresolved identity.

Compare baseline vs R15 on the exact same 5,321 identities:
- target MAE;
- RMSE;
- signed bias;
- pooled, 2023, 2024;
- within-canonical-secondary-pool share MAE on team-games with positive actual canonical-secondary targets;
- count/share of player rows where R15 moves predicted targets toward vs away from actual;
- preserve the known 4,193 graded-cohort target-MAE comparison separately for lineage.

### Frozen R15 diagnostic interpretation

Established program-level meaningful target-MAE scale is `0.05` targets/player-game (same order used in prior WR target-model gating).

- If R15 improves full-5,321 target MAE pooled and does not regress either season by >=0.05, classify `R15_WR2PLUS_ALLOCATION_HEALTHY_OR_IMPROVED` and do not blame R15.
- If R15 worsens full-5,321 target MAE by >=0.05 pooled **and** worsens target MAE in both 2023 and 2024, classify `R15_WR2PLUS_ALLOCATION_STRUCTURED_ERROR` as a diagnostic finding only. A challenger still requires a separately frozen hypothesis and genuinely fresh validation route.
- Otherwise classify `R15_WR2PLUS_ALLOCATION_MIXED_OR_SMALL`; diagnose only, no challenger.

RMSE/share-MAE/toward-away counts are corroborating diagnostics and cannot override the frozen target-MAE disposition by themselves.

## Program-level interpretation tree

1. If opportunity contribution is not dominant relative to efficiency and upstream target diagnostics are healthy: `OPPORTUNITY_NOT_PRIMARY_ERROR_SOURCE`; preserve M38/R15.
2. If opportunity is material and Layer 2 shows team-pool or WR-room-share accounting dominates: upstream opportunity problem, not R15.
3. If Layer 3 shows WR1 error materially worse than canonical secondary-pool error: M38/WR1 layer is the candidate problem, not R15.
4. If Layer 4 reaches `R15_WR2PLUS_ALLOCATION_STRUCTURED_ERROR`: only then may one R15-specific hypothesis be proposed; no automatic redesign.
5. If Layer 4 reaches `R15_WR2PLUS_ALLOCATION_HEALTHY_OR_IMPROVED`: preserve R15 even if overall WR receiving-yard error remains large.
6. Mixed/unstable evidence: diagnosis only, no challenger.

No post-result threshold changes, cohort reshaping, role-only rescue, sportsbook feature selection, or model-zoo search are permitted.

## Pre-execution review gate

No Phase 4B attribution run is authorized until Claude independently reviews this exact revised contract, including:
- symmetric zero-target handling;
- the 1,026 anchor-observable scope for Layers 2–3;
- canonical-identity restriction for actual WR-room targets;
- off-model target-mass disclosure;
- all-5,321 WR2+ Layer 4 diagnostic extension;
- frozen `0.05` target-MAE materiality rule for the R15-specific disposition.
