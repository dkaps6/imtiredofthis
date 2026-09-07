# TE-R2 Team TE Pool vs Individual Allocation — Frozen Plan

## Status
Frozen before implementation/results. Diagnostic only. Production unchanged.

## Purpose
TE-R1 established that TARGETS are the largest TE receiving-yard error mechanism (45.23% of pooled absolute Shapley mass; 1,068 of the highest-error-quartile games were TARGETS-dominant). TE-R2 determines **where that target error lives** before fitting a predictive target model:

1. **TEAM_TE_POOL** — the model assigns the wrong total target opportunity to the team's TE room; or
2. **INDIVIDUAL_ALLOCATION** — the model assigns the wrong share of the TE-room opportunity to the individual TE.

This is explicitly player-centric. It prevents us from trying to repair an individual TE with a team-level coefficient, or trying to repair a team TE-volume problem with a player residual.

## Lineage
- Parent branch/head: `research-te-r1-individual-mechanism-decomposition` / `003ff7e7ec1a7e4d080f3ebaf182a281c5ab0aea`.
- TE-R1 canonical run: `34123413402`.
- TE-R1 artifact: `10019112418`, digest `sha256:bdf4cddce8477755bb407ffd88fed8b961808d2f4a0970dd15fb95882d572484`.
- Exact underlying source: Joint Pass/Receiving Conservation V1 run `34081764151`, artifact `10004223287`, digest `sha256:753aa191e6c80a059918553d8567499c9cdbce82b12a26b4d5b19b21225764ac`.
- Baseline: B0/current TE target architecture.
- Sportsbook inputs: 0.

## Cohort
Use `joint_v1_paired_player_casebook.csv`, regular-season TE player-games from 2020-2025, with finite `b0_expected_targets` and actual `targets`.

No target-game outcome may enter any projection-side quantity.

## Exact two-factor identity
For each team-game:

- `pred_te_pool = sum(b0_expected_targets)` over TEs on that team-game.
- `actual_te_pool = sum(actual targets)` over TEs on that team-game.

For each TE player-game:

- `pred_te_share = b0_expected_targets / pred_te_pool` when pred pool > 0, otherwise 0.
- `actual_te_share = targets / actual_te_pool` when actual pool > 0, otherwise 0.

Thus:

`pred_player_targets = pred_te_pool * pred_te_share`

`actual_player_targets = actual_te_pool * actual_te_share`

Use the exact two-factor Shapley decomposition of `actual_player_targets - pred_player_targets`:

- `TEAM_TE_POOL_component = (actual_te_pool - pred_te_pool) * (pred_te_share + actual_te_share) / 2`
- `INDIVIDUAL_ALLOCATION_component = (actual_te_share - pred_te_share) * (pred_te_pool + actual_te_pool) / 2`

The two components must reconstruct the exact player target residual within `1e-9` targets.

## Frozen outputs
Pooled and by season:
- target MAE/RMSE/bias/correlation;
- median/p75/p90 target absolute error;
- 2+/4+/6+ target miss rates;
- mean absolute TEAM_TE_POOL and INDIVIDUAL_ALLOCATION contributions;
- fraction of absolute mechanism mass for each component;
- same-direction rate of each component with the final target residual.

Team-game diagnostics:
- TE-room target-pool MAE/RMSE/bias/correlation;
- median/p90 pool absolute error;
- 3+/5+ pool miss rates.

Player profiles (>=20 scoreable games):
- player target MAE/p90 and 4+/6+ miss rates;
- mean absolute pool/allocation components;
- pool/allocation absolute-mass shares;
- dominant submechanism if one owns >=55% of player absolute mass, otherwise MIXED;
- seasons represented.

Highest player-target-error quartile:
- game-level dominant component counts;
- absolute-mass shares.

Player opportunity tiers based only on B0 expected targets within the evaluated row (Q1-Q4) will be reported descriptively to show whether allocation problems concentrate among high-opportunity TEs. This tiering is diagnostic only and is not a predictive feature.

## Frozen integrity gates
1. >=4,000 scoreable TE player-games.
2. all six seasons represented with >=500 player-games each.
3. max target identity reconstruction error <=1e-9.
4. max Shapley reconstruction error <=1e-9.
5. >=40 players with >=20 scoreable games.
6. no sportsbook inputs.

## Frozen routing rule
If integrity gates pass, TE-R2 is actionable and the next predictive lane is selected mechanically:

- pooled TEAM_TE_POOL absolute-mass share >=55% -> `TE_TARGET_POOL_FIRST`;
- pooled INDIVIDUAL_ALLOCATION absolute-mass share >=55% -> `TE_INDIVIDUAL_ALLOCATION_FIRST`;
- otherwise -> `TE_JOINT_POOL_ALLOCATION_REQUIRED`.

The same route must also be reported for the highest-error quartile. If pooled and high-error routes disagree, the next experiment must model the two layers separately rather than collapsing them into one residual.

If integrity fails -> `TE_R2_INTEGRITY_FAIL`; no science inference.

## What is not authorized
- no coefficient search;
- no post-result threshold changes;
- no copying WR M38 multipliers;
- no player-specific manual overrides;
- no target-game features;
- no production change.

## Authorized next step after TE-R2
Only after the route is known may TE-R3 freeze a predictive model. TE-R3 must use football information appropriate to the identified layer. For an individual-allocation route this means strict-prior player usage/role, TE-room competition, teammate availability/participation, and matchup context. For a pool route this means team pass environment, TE usage tendency, opponent TE allowance/coverage context, and expected game environment. A joint route must keep both layers explicit.
