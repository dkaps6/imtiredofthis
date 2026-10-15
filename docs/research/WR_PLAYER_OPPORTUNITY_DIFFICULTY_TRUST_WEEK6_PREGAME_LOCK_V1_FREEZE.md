# WR Player Opportunity-Difficulty Trust — Week 6 Pregame Lock V1 — Frozen Plan

Date: 2026-10-15

Status: **FROZEN BEFORE WEEK-6 WR TRUST LABEL CONSTRUCTION / RESEARCH ONLY**

Version:
`WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_WEEK6_PREGAME_LOCK_V1`

Parent:
`WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`

Parent authority:
- run `38823063793`
- artifact `12013153664`
- digest `sha256:12fd0de1dc3ebceee4d356669d59f30d96776d8c0800d1fb3ac6a293af457b7c`

## Objective

Freeze 2026 Week-6 WR trust labels before Week-6 outcomes using the exact
same-player opportunity-difficulty semantics that qualified historically.

Labels only. No WR-R15 projection or distribution changes.

## Current Week-6 cohort

Use the exact frozen Week-6 WR/TE trajectory parent:
- opportunity lock run `38644002109`
- artifact `11930748974`
- digest `sha256:8aae43d895599bcb516ae79ecc0045dd4fad7a15aea48a8fe24c407f0b62a0f9`
- file `wrte/player_target_share_trajectory_shadow_week6_lock.csv`.

Filter:
`position == WR`.

The file's `baseline_entitlement_tgt_share` is the unchanged WR-R15 parent
entitlement before the separate trajectory shadow.

Expected current cohort at freeze:
**152 WRs** across 28 scheduled teams.

Do not use the trajectory-transformed share to construct trust.

## Exact historical semantic parity

Before constructing 2026 history, independently prove the historical WR
opportunity-error definition against the exact recovered authority.

Sources:
- mechanism parent run `37634044900`, artifact `11487359185`;
- recovered WR-R15 run `37782189538`, artifact `11552447166`.

Filter recovered WR rows:
`variant == WR_R15_WR1_ANCHORED_PARTICIPATION`.

Join on:
`season, week, team, player_clean_key`.

For each row:

`pred_ypt = mc_rec_yards / pred_targets`

`opportunity_error = (pred_targets - actual_targets) * pred_ypt`

Frozen parity gates:
- exactly 3,224 matched rows;
- exactly 211 WRs;
- max absolute gap to parent `current_opportunity_error` <= `1e-9`;
- no duplicate identity.

No scientific score is taken from this parity check.

## Current 2026 W1-W5 history

Reconstruct each completed Week 1-5 with the current football pipeline and only
strictly-prior information for that target week:

1. build historical context;
2. build current canonical MC metric inputs;
3. materialize explicit target entitlement;
4. apply TE-R5P first;
5. apply frozen production WR-R15 second;
6. run the canonical explicit-entitlement simulation with 2,000 draws,
   seed `42 + week`;
7. no sportsbook input.

For each modeled WR player-week:

`pred_team_targets = clip(rules_plays_est,50,80) * clip(rules_pass_rate,.35,.75)`

`pred_targets = pred_team_targets * wr_r15_final_entitlement_share`

`pred_rec_yards = mean(canonical WR-R15 rec_yards draws)`

`pred_ypt = pred_rec_yards / pred_targets`

`opportunity_error = (pred_targets - actual_targets) * pred_ypt`

Target-week actual targets enter only after that week's pregame projection has
been generated and only as the completed-week label.

No Week-6 target outcomes may be read.

## Deliberate 2025 exclusion

Do **not** use 2025 WR errors to construct current Week-6 player difficulty.

The final production WR-R15 model is an all-history refit trained through 2025.
Using its 2025 in-sample residuals as current reliability history would violate
the clean prospective intent.

Current Week-6 trust therefore uses only completed 2026 W1-W5 player errors.

Players with fewer than four qualifying 2026 prior games remain
`UNSUPPORTED`.

No 2025 backfill.

## Frozen player state

For each Week-6 WR:
- same `player_clean_key` only;
- trailing max 8 errors (only W1-W5 can exist here);
- minimum 4 prior games;
- `player_mae = mean(abs(opportunity_error))`.

Among supported Week-6 WRs:
- `week_pool_mae = median(player_mae)`;
- `trust_fragility_score = rank(player_mae, method="average", pct=True)`;
- HIGH_TRUST <= .25;
- LOW_TRUST >= .75;
- otherwise MIDDLE.

Unsupported rows receive no score and bucket `UNSUPPORTED`.

## Integrity gates

Fail closed unless:
1. parent trust result is qualified;
2. Week-6 opportunity pack status is frozen;
3. exact roster source week = 6;
4. Week-6 WR cohort = 152;
5. historical semantic parity passes exactly;
6. 2026 history contains only W1-W5;
7. every reconstructed W uses only pre-W features;
8. Week-6 outcomes read = 0;
9. sportsbook inputs = 0;
10. OddsAPI calls = 0;
11. models fit = 0;
12. projection/distribution changes = 0;
13. production unchanged.

Report:
- supported/unsupported WR count;
- trust bucket counts;
- W1-W5 history rows;
- player prior-game counts;
- player MAE;
- pool MAE;
- row CSV SHA256.

## Boundaries

This lock does not:
- change WR-R15;
- change the trajectory shadow;
- choose an over/under;
- choose a bet;
- use live odds;
- use Week-6 outcomes.

Prospective scoring must occur only after Week 6 settles.
