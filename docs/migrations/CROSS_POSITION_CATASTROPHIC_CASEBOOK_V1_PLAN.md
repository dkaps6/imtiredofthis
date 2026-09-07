# Cross-Position Catastrophic Player Casebook V1 — Frozen Plan

## Status

**FROZEN BEFORE RESULTS. DIAGNOSTIC ONLY. NO PRODUCTION MODEL CHANGE.**

Branch: `research-cross-position-catastrophic-casebook-v1`

## Purpose

Build one forensic casebook across QB, WR, TE and RB using the best currently retained football-model evidence, then determine whether the largest individual-player misses share repeatable pregame-identifiable mechanisms that can support materially better modeling.

The goal is not another generic feature hunt. The goal is to answer:

1. Why were the worst individual player projections wrong?
2. Was the miss primarily opportunity, player entitlement/share, conversion, efficiency, explosive-play variance, injury/participation, or game-state driven?
3. Which failures were plausibly identifiable before kickoff using legal football-only information?
4. Which failures are mostly irreducible/random and should be represented through wider/more realistic distributions instead of mean corrections?
5. Which recurring structural clusters are large enough to justify one narrowly targeted follow-up model change?

## Frozen source lineage

The casebook consumes existing frozen evidence; it does not refit the underlying models.

### QB
- M89 football-only synthesis run `33331073376`.
- M90 confirmation run `33333730480`.
- Production mean family: `QB_PASS_SYNTHESIS_V1`.
- Existing M89 catastrophic QB casebook is retained as supporting evidence.

### WR
- M38 WR target-share hierarchy run `32485770487`.
- Exact M38 parent `b98518d97b3038f471aee9ae3201009b2c70bb29`.
- Frozen multipliers `1.40 / 1.14 / 0.91 / 0.78`.

### TE
- TE-R5 run `34132127351`.
- Launch SHA `999c29d543e6854a903c5a0a4ee6fecbe69dce61`.
- Disposition `TE_PARTICIPATION_ENTITLEMENT_V1_PASS`.

### RB
- RB STACK3 / P3 composition run `33539468967`.
- Production family `RB_P3_SYNTHESIS_V1`.
- Production parent `754d0f4ed34a06a65d2db36f2fb2ca10c58264ed`.

### Joint receiving evidence
- C2 full-stack result run `34142510405` is diagnostic context only.
- Its QB distribution gains and receiver-tail tradeoff may be referenced, but C2 may not redefine the source player means in this casebook.

Sportsbook inputs are forbidden from cause classification.

## Historical scope

Use every row recoverable from the frozen artifacts, but prioritize the common recent OOS era.

Primary priority:
- QB: 2024-2025, current M89/M90 family where recoverable.
- WR: M38 OOS historical rows, with 2023-2025 highlighted.
- TE: TE-R5 OOS 2023-2025.
- RB: P3/STACK evidence, with 2023-2025 highlighted where available.

If an artifact lacks enough row-level information for one position, the workflow may mechanically supplement it with a leakage-safe historical trace from the same frozen model lineage. That is a data-recovery action only, not a model change.

## Frozen catastrophic cohorts

For each position/market, build:

1. **Top-50 absolute misses** overall where at least 50 rows exist.
2. **Top-25 underprojections** and **top-25 overprojections**.
3. Position-specific threshold cohorts:
   - QB passing yards: absolute miss >= `100` yards.
   - WR receiving yards: absolute miss >= `50` yards.
   - TE receiving yards: absolute miss >= `40` yards.
   - RB rushing yards: absolute miss >= `40` yards.
4. Also retain the top decile of absolute misses by position so conclusions do not depend on one fixed threshold.

No threshold may be changed after results are visible.

## Required pregame-side fields

Use artifact fields when available and strictly-prior football context when mechanically reconstructable:

- projection and actual;
- expected team opportunity state;
- projected attempts / carries / targets;
- projected player share / room rank / entitlement state;
- projected catch rate or YPA/YPR/YPC proxy;
- recent same-team usage;
- teammate competition / room size;
- injury/availability state if temporally valid;
- opponent defensive context;
- game environment / pace / pass-run tendency;
- indoor/weather availability status;
- model component disagreement where already frozen.

Sportsbook fields must not enter the football explanation.

## Required postgame forensic reconstruction

### All positions
- team offensive plays and drive count where available;
- score-state / leading-trailing-neutral usage;
- first-half vs second-half opportunity;
- actual participation and opportunity;
- actual efficiency;
- in-game injury/early exit flags where mechanically inferable;
- explosive-play contribution;
- largest-play contribution;
- outcome with largest play removed;
- teammate opportunity concentration.

### QB
- attempts and yards by quarter;
- actual YPA;
- sacks, interceptions, scrambles;
- longest completion;
- 20+/40+/60+ completions;
- total/max YAC;
- receiver concentration;
- passing yards after removing longest completion.

### WR / TE / RB receiving
- targets, receptions, receiving yards;
- catch rate;
- yards per target and yards per reception;
- longest reception;
- YAC where available;
- targets/receptions/yards after removing longest reception;
- team target share and position-room target share.

### RB rushing
- carries, rushing yards, YPC;
- carries by half and score state;
- longest rush;
- 10+/20+/40+ rush counts;
- rushing yards after removing longest rush;
- team RB-room carries and player carry share;
- QB rushing competition;
- teammate RB carry concentration.

## Frozen mathematical error decomposition

Where inputs are available, decompose each yardage miss into interpretable layers.

### QB passing yards
Use:
`projected_yards ~= projected_attempts * projected_ypa`

Report:
- attempt/opportunity contribution;
- efficiency/YPA contribution;
- interaction residual;
- largest-completion and YAC contribution.

### WR / TE receiving yards
Use:
`yards ~= team_pass_opportunity * player_target_share * catch_rate * yards_per_reception`

When exact team pass opportunity is not recoverable, use projected targets directly:
`yards ~= targets * catch_rate * yards_per_reception`

Report:
- opportunity/target contribution;
- player entitlement/share contribution where team pool is available;
- catch-conversion contribution;
- YPR/efficiency contribution;
- explosive-largest-play contribution;
- unexplained interaction residual.

### RB rushing yards
Use:
`yards ~= team_rush_opportunity * rb_room_share * ypc`

When team rush opportunity is not recoverable, use projected carries directly:
`yards ~= carries * ypc`

Report:
- room/team opportunity contribution;
- individual carry-share contribution where available;
- YPC/efficiency contribution;
- explosive-largest-run contribution;
- interaction residual.

## Frozen primary mechanism taxonomy

Each catastrophic case receives one primary label using deterministic rules and one secondary label when useful:

- `TEAM_OPPORTUNITY_MISS`
- `PLAYER_ENTITLEMENT_MISS`
- `CONVERSION_MISS`
- `EFFICIENCY_EXPLOSION`
- `EFFICIENCY_COLLAPSE`
- `SINGLE_EXPLOSIVE_PLAY`
- `YAC_DRIVEN_EXPLOSION`
- `GAME_SCRIPT_VOLUME_SHIFT`
- `TEAMMATE_ROLE_SHIFT`
- `INJURY_PARTICIPATION_DISTORTION`
- `TURNOVER_POSSESSION_DISTORTION`
- `PROTECTION_SUPPRESSION`
- `OVERTIME_OR_GARBAGE_TIME`
- `MIXED`

The script must report the deterministic rule inputs used for each label.

## Pregame predictability classification

Every case also receives one of:

- `STRUCTURAL_PREGAME_SIGNAL`
- `PARTIAL_PREGAME_SIGNAL`
- `LOW_PREDICTABILITY_GAME_EVENT`
- `SOURCE_OR_IDENTITY_ISSUE`
- `UNRESOLVED`

This classification is descriptive. It may use postgame evidence to say what happened, but any claim that a miss was structurally predictable must point to a feature/state that existed strictly before the target game.

## Required aggregate outputs

- one unified row-level catastrophic casebook;
- separate QB/WR/TE/RB casebooks;
- top-50 markdown summaries per position;
- mechanism counts and error-mass share by position;
- predictable-vs-low-predictability counts and error-mass share;
- largest-play sensitivity summary;
- opportunity-vs-efficiency error-mass summary;
- repeated player / repeated team / repeated role-state patterns;
- season stability of every major mechanism;
- candidate actionable clusters ranked by:
  1. total absolute error mass;
  2. number of distinct players;
  3. number of seasons;
  4. pregame signal availability;
  5. estimated reducible error if the structural layer were corrected.

## Actionability rule

A mechanism may justify a targeted next experiment only if all are true:

1. it accounts for at least `8%` of catastrophic absolute error mass within a position **or** at least 20 catastrophic cases;
2. it appears across at least 2 seasons where multi-season data exist;
3. it affects at least 8 distinct players unless the mechanism is explicitly team/system-level;
4. at least one legal pregame football signal is available at useful coverage;
5. the proposed fix targets the causal layer identified by the casebook rather than final-yardage residuals.

No model is promoted by this diagnostic alone.

## Stopping rule

This migration is a forensic map, not an optimization loop.

- If one or more actionable clusters exist, select the highest reducible-error cluster(s) for narrowly frozen follow-up work.
- If catastrophic error is dominated by low-predictability explosive/random events, improve distribution/tail realism rather than chasing the mean.
- Do not lower the actionability thresholds after seeing results.
- Do not create a generic residual model from postgame forensic labels.

`postgame_forensic_fields_used_for_prediction = false`

`sportsbook_features_used_for_cause_classification = false`
