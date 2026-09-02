# RB STACK6P — Rush-Event Component Shapley

## Purpose

STACK6H already established that the current P3 team-RB-pool bottleneck is overwhelmingly **total team rushing attempts**, not RB share of team rushing. STACK6I then localized the M94C total-rush error to effective rush rate rather than offensive play volume. STACK6K showed that most of that rate headroom is **within-state rushing tendency**, and STACK6L–O narrowed but did not convert the trailing/deep-late finding into a retainable pregame correction.

A remaining representation problem is that M94C's state `rush_rate` treats every nflverse `rush_attempt` as the same type of event. In particular, a QB scramble is counted as a rush attempt even though it begins as a pass/dropback decision. Thus a state-level total rush rate can conflate:

1. called/designed rushing plays,
2. QB scrambles generated from dropbacks,
3. QB kneels.

STACK6P asks, with no fitted model:

> Which rush-event component actually accounts for the frozen STACK6K within-state tendency headroom?

This is an attribution study only. It does not authorize a production correction.

## Frozen lineage

- Parent branch: `research-rb-stack6o-deep-late-urgency-shapley`
- Parent result commit: `3e927f9cc8d0c63aa57ab73b16a6c4030aed775b`
- M94C artifact: run `33353485070`
- STACK6H error-bin artifact: run `33632678179`
- STACK6J occupancy baseline MAE: `5.518381962346741`
- STACK6K full within-state tendency oracle MAE: `3.4503279031625445`
- Frozen within-state tendency MAE headroom: `2.0680540591841963`
- Sportsbook inputs: forbidden.

## Population / temporal contract

- PBP source: 2023, 2024, 2025 regular season.
- Evaluation: 2025 W6–18, expected `n=388` team-games.
- Target-game PBP may be used only for labels/oracle grading.
- Every component-mix estimate used to allocate the parent rate must use games strictly before the target `(season, week)`.
- No target-game injury, participation, box score, sportsbook, or outcome feature enters the pregame allocation mix.

## Frozen score states

Preserve the exact M94B/C state contract:

- `lead`: score differential > +3
- `neutral`: -3 through +3
- `trail`: score differential < -3

Target-game state occupancy is held at truth only as the same conditional grading scaffold used by STACK6J–O. It is not an allowed production input.

## Mutually exclusive rush-event components

Among plays with `rush_attempt == 1`:

### `DESIGNED`

`qb_scramble != 1` and `qb_kneel != 1`.

This includes RB/HB/FB carries **and designed QB rushing attempts**. It is intentionally a play-call / designed-rush component, not an RB-share component.

### `SCRAMBLE`

`qb_scramble == 1`.

### `KNEEL`

`qb_kneel == 1`.

Required source identity:

`DESIGNED + SCRAMBLE + KNEEL == all rush_attempts`

for every aggregated state/team-game. Scramble and kneel flags must not overlap. If this identity fails, STACK6P is invalid.

## Why this is not a repeat of STACK6G/H

- STACK6G tested target-week QB1 rushing-propensity regime changes and playcaller changes; it did not decompose M94C's `rush_attempt` label by event type.
- STACK6H decomposed **total team rushing × RB share of team rushing** and decisively found total rushing dominant. It did not separate the internal generation of total rush attempts into designed calls vs scrambles vs kneels.

STACK6P therefore operates one layer inside the already-validated total-rush bottleneck.

## Frozen parent reconstruction

Keep the frozen M94C 75/25 blend:

`candidate = 0.25 * baseline_team_rush_att + 0.75 * pred_off_plays * effective_rush_rate`

For the conditional occupancy baseline, the parent effective rate is:

`sum_s actual_state_share_s * parent_state_rush_rate_s`

for `s in {lead, neutral, trail}`.

The empty component-correction subset must reproduce the STACK6J occupancy oracle exactly.

When all three components are corrected to target truth, the effective rate becomes:

`actual_total_rush_attempts_pbp / actual_offensive_plays_pbp`

and must reproduce STACK6K's full within-state tendency oracle exactly.

## Frozen pregame component-allocation schemes

Attribution can depend on how the parent total state rush rate is split into components. STACK6P therefore uses **two precommitted, strict-prior allocation schemes** and requires a robust conclusion across both.

### Scheme A — `LEAGUE_STATE_MIX`

For each target game and each score state, use all strict-prior 2023–target-week league rush attempts in that state to estimate:

- designed share of state rush attempts,
- scramble share,
- kneel share.

The three shares must sum to 1.

### Scheme B — `TEAM8_SHRUNK_STATE_MIX`

For each target team/state:

- use exactly the team's last 8 prior games,
- count component rush attempts within that state,
- shrink to the strict-prior league-state mix with a fixed pseudo-sample of **24 state rush attempts**:

`mix_c = (team_component_count_c + 24 * league_mix_c) / (team_total_state_rushes + 24)`

No window, pseudo-sample, state definition, or shrinkage search is permitted after 2025 outcomes are observed.

If a team has no prior rush attempt in a state, the formula collapses to the league-state mix.

## Oracle substitution

For each allocation scheme and each subset of `{DESIGNED, SCRAMBLE, KNEEL}`:

- uncorrected component contribution in state `s`:
  `actual_state_share_s * parent_state_rush_rate_s * pregame_component_mix_{s,c}`
- corrected component contribution:
  `actual_target_component_rushes_{s,c} / actual_target_offensive_plays`

Sum all state/component contributions and apply the frozen 75/25 M94C blend.

All 8 component subsets are evaluated.

## Exact Shapley attribution

Compute three-player Shapley MAE recovery under each allocation scheme so component attribution is independent of correction order.

Frozen populations:

- `ALL_W6_18`
- `POOL_OVER_5`
- `POOL_UNDER_5`
- `POOL_ABS_5`
- `NON_EXTREME_ABS_LT3`

The P3 bins come only from the frozen STACK6H artifact and are grading populations.

## Required descriptive football audit

Report, for each of lead/neutral/trail and for deep-late (`score_diff <= -9`, Q4+):

- offensive plays,
- total rush attempts,
- designed attempts,
- scramble attempts,
- kneels,
- designed attempts / offensive plays,
- scrambles / offensive plays,
- kneels / offensive plays,
- each component's share of all rush attempts.

This determines whether the previously observed trailing/deep-late total-rush behavior is actually run-call abandonment, scramble generation, kneels, or a mixture.

## Required integrity gates

Before any attribution is interpreted:

1. W6–18 `n=388`.
2. Fresh PBP reproduces M94C target state shares within `1e-9`.
3. Fresh PBP reproduces M94C actual offensive plays within `1e-9`.
4. `DESIGNED + SCRAMBLE + KNEEL == rush_attempt` exactly at the target team-game/state level.
5. Scramble/kneel overlap count = 0.
6. Empty subset MAE under both allocation schemes equals `5.518381962346741` within `1e-9`.
7. All-component subset MAE under both schemes equals `3.4503279031625445` within `1e-9`.
8. Shapley sum under both schemes equals `2.0680540591841963` within `1e-9`.
9. Strict-prior allocation coverage = 100%.
10. Fitted models/search/sportsbook inputs = 0.

## Frozen diagnostic disposition

For each scheme and population, rank components by Shapley MAE recovery.

Call `DESIGNED_RUN_CALL_DOMINANT` only if:

- `DESIGNED` is the top Shapley component under **both** allocation schemes for `ALL_W6_18`, `POOL_OVER_5`, and `POOL_UNDER_5`;
- `DESIGNED` accounts for >= **60%** of total within-state tendency recovery overall under both schemes;
- `DESIGNED` accounts for >= **50%** of recovery in both `POOL_OVER_5` and `POOL_UNDER_5` under both schemes.

Call `SCRAMBLE_COMPONENT_DOMINANT` if the same conditions are met by `SCRAMBLE`.

Call `KNEEL_COMPONENT_DOMINANT` if the same conditions are met by `KNEEL`.

Otherwise disposition is `MIXED_RUSH_EVENT_COMPONENTS`.

No dominance gate may be waived after execution.

## Follow-up authorization

- `DESIGNED_RUN_CALL_DOMINANT` authorizes research into a separate **designed-run play-call/tendency** model, while keeping scrambles as a distinct process.
- `SCRAMBLE_COMPONENT_DOMINANT` authorizes a separate pressure/dropback-to-scramble process.
- `KNEEL_COMPONENT_DOMINANT` would require a late-win-state / possession-end representation audit.
- `MIXED_RUSH_EVENT_COMPONENTS` requires a joint component architecture rather than another single total-rush-rate correction.

Regardless of result, STACK6P itself authorizes no production change and no player recomposition. P3 remains champion.
