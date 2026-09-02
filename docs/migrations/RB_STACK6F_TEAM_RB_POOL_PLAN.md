# RB STACK6F — Team RB Carry-Pool Qualification

Status: FROZEN BEFORE 2025 TEAM-POOL EVALUATION

## Why this experiment exists

STACK6B, STACK6C, and STACK6E failed to produce a retainable player-level secondary-back correction. A no-fit postmortem then localized the dominant miss at the team-backfield level:

- for secondary backs overprojected by 3+ carries, the whole team RB pool was overprojected by roughly 4.4 carries on average;
- the lead back was only modestly underprojected;
- the secondary player's carry error tracked team-pool error much more strongly than lead-back residual.

This means the remaining error may be **which games produce high or low team RB volume**, not primarily which RB receives a fixed pool.

STACK6F therefore qualifies team RB opportunity volume before any player-level recomposition is attempted.

## Frozen parent and population

- P3 remains champion and is not modified by this qualification run.
- 2025 evaluation universe is the frozen P3/STACK6 team-week universe.
- Team-pool scoring is Week 6–18, with Week 13–18 reported separately.
- P3 team RB pool = sum of frozen P3 `parent_att` within team-week.
- Actual team RB carries are reconstructed independently from nflverse weekly player stats using all RB/HB/FB rows for the team-week. The P3 casebook is **not** used to define actual team RB carry totals.

## Temporal split

- Feature-history seasons: 2023–2025 as needed for strictly prior observations.
- Fit target season: 2024 only.
- Evaluation season: 2025 only.
- Every feature row may use only games strictly before that row's season/week.
- No 2025 outcome may influence feature choice, feature transformations, model hyperparameters, gates, or arm construction.
- No sportsbook data are used anywhere in STACK6F qualification.

## Frozen learner

One fixed model only:

- Ridge regression
- `alpha = 10.0`
- median imputation
- standard scaling
- no hyperparameter search
- no feature search
- no threshold search

The 2024 target is actual team RB/FB carries.

## Exactly 12 features

### Offense history

1. `team_prior1_rb_carries`
2. `team_prior3_rb_carries`
3. `team_prior5_rb_carries`
4. `team_prior1_total_rush`
5. `team_prior3_total_rush`
6. `team_prior3_qb_rush_share`
7. `team_prior3_pass_att`
8. `team_prior3_play_proxy`

`play_proxy = team total rush attempts + team pass attempts` from weekly player statistics. This is not claimed to be official plays; it is a stable workload proxy used symmetrically in training/evaluation.

### Opponent defense history

9. `opp_prior1_rb_carries_allowed`
10. `opp_prior3_rb_carries_allowed`
11. `opp_prior3_total_rush_allowed`
12. `opp_prior3_pass_att_faced`

Opponent-allowed values are reconstructed from prior opponents' offensive weekly stats through the regular-season schedule. No target-game result is used.

No target-game injury, inactive, participation, weather, betting, or game-result feature is eligible in STACK6F.

## Frozen arms

### `HISTORY_POOL`

Use the Ridge team-RB carry prediction directly, clipped only at the physical lower bound of zero.

### `P3_HISTORY_50`

One fixed composition arm, not a weight search:

`0.50 * P3 team RB pool + 0.50 * HISTORY_POOL`

The 50/50 composition is frozen before outcome evaluation and mirrors the conservative integration pattern previously used in the RB allocation lineage. No other weights may be tested in STACK6F.

## Frozen team-level retention gates

An arm qualifies the team-pool family only if all are true versus P3 on 2025 Week 6–18 team-weeks:

- team RB carry MAE gain >= **0.30 carries**
- team RB carry RMSE gain > **0**
- correlation improvement >= **+0.05**
- absolute carry bias <= **0.50 carries**
- Week 13–18 team RB carry MAE gain > **0**

If both arms pass, choose the smaller-change `P3_HISTORY_50` arm when it is within 0.10 MAE gain of the better arm; otherwise choose the arm with the larger MAE gain.

## What a pass means

A pass qualifies a **team RB opportunity family**, not a player model and not production code.

Only after a pass may a separate frozen STACK6 composition experiment map the corrected team pool back to player carries. That later composition must preserve P3's validated allocation evidence and explicitly protect any previously protected populations unless its own protocol says otherwise.

## What a failure means

If neither arm passes, do not tune Ridge alpha, blend weight, feature windows, thresholds, or population slices on exposed 2025. The team-history pool hypothesis is rejected in this form and P3 remains unchanged.

## Possible dispositions

- `STACK6F_RETAIN_HISTORY_POOL`
- `STACK6F_RETAIN_P3_HISTORY_50`
- `STACK6F_NO_RETAINABLE_TEAM_POOL_MODEL`

Production change: **none**.
