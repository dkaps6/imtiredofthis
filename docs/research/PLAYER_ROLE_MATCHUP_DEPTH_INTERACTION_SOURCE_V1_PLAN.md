# PLAYER ROLE MATCHUP — DEPTH INTERACTION SOURCE V1 PLAN

Frozen: 2026-10-11

Status: **SOURCE / RECONSTRUCTION AUDIT ONLY. NO OUTCOME SCORING. NO MODEL CHANGE.**

Parent roadmap:
- `docs/research/INDIVIDUAL_OPPORTUNITY_ALLOCATION_ROADMAP_V1.md`
- opportunity allocation remains the primary active model-development layer;
- future matchup work must be conditional on individual player role/opportunity.

## Purpose

Audit whether one genuinely player-specific matchup state can be reconstructed
historically and prospectively without paid WR-CB assignment data:

> a receiver's own strictly-prior target-depth mix x the opponent defense's
> strictly-prior performance against those same target-depth bands.

This is **not** a generic defense-vs-position multiplier.

The source audit asks only whether the inputs are:
- public / reproducible;
- identity-safe;
- strictly pregame;
- dense enough for a later frozen predictive test;
- available for WR and TE separately.

No target-game receiving result is scored in this stage.

## Anti-retest boundary

This V1 is distinct from prior closed families.

### Football Matchup Transmission V1
Closed:
- generic RB opponent pass-rate-faced integration;
- generic WR offensive PROE integration;
- generic TE opponent pass-success integration;
- generic usage x matchup interactions;
- generic defense-vs-position boosts.

Do not reopen them.

### M72
M72 tested:
- team-level aggregate receiving-weapon explosive/YAC/air profile;
- aggregate defense explosive/YAC/air vulnerability;
- QB passing-efficiency residual as the target.

This V1 does **not** reuse that target or aggregation.

V1 is eligible only because it is:
- one individual WR/TE at a time;
- the player's own target-depth distribution;
- opponent defense split into the identical depth bands;
- intended for eventual player receiving projection evaluation, not QB residuals.

### WR-CB
No direct CB assignment, shadow flag, editorial matchup grade, or paid source is
used. Historical WR-CB assignment remains source-blocked.

## Frozen historical audit universe

Target seasons:
- 2024
- 2025

Target weeks:
- Weeks 2-18 regular season

History source:
- 2023-2025 nflverse / nflreadpy play-by-play;
- exact target-week nflverse weekly roster snapshots;
- official schedule identity.

Pregame universe:
- target-week rostered WR / TE only;
- ACT / INA roster statuses when status exists;
- exact stable GSIS identity required for a supported row;
- no target-week box-score participation requirement.

Target-game PBP must never be used to decide whether a row exists.

## Frozen target-depth bands

Use finite `air_yards` on targeted pass attempts:

1. `BEHIND`: air_yards < 0
2. `SHORT`: 0 <= air_yards < 10
3. `INTERMEDIATE`: 10 <= air_yards < 20
4. `DEEP`: air_yards >= 20

No band threshold may change after results are inspected.

## Frozen individual player state

For each target player-week:

- use target events strictly before target week;
- retain the latest up to 8 completed receiver games;
- require >=4 prior receiver games;
- require >=10 finite-air-yards targets.

Preserve:
- target count;
- mean air yards;
- target-depth population SD;
- target share in each frozen depth band.

Player band shares must sum to 1.0 within numerical tolerance.

No target-game target, catch, yardage, route, or snap is used.

## Frozen opponent-defense state

For each target player-week opponent:

- use pass targets faced strictly before target week;
- retain the latest up to 8 completed defensive games;
- require >=4 prior defensive games.

For each frozen depth band preserve:
- defensive target count;
- completion rate allowed;
- receiving yards per target allowed.

For source support, each of the four bands must have >=5 prior defensive targets.

No target-game defensive result is used.

## Descriptive player-specific matchup state

Source V1 may construct, for audit/descriptive purposes only:

`depth_mix_def_ypt = SUM(player_depth_share_band * defense_ypt_allowed_band)`

and

`depth_mix_def_completion = SUM(player_depth_share_band * defense_completion_allowed_band)`

These are **not** projection candidates in V1 and cannot be scored against
receiving outcomes in this stage.

They exist only to prove the player-specific interaction is mechanically
constructable.

## Frozen source-readiness gates

The source is
`PLAYER_DEPTH_MATCHUP_SOURCE_QUALIFIED`
only if all pass:

1. exact target seasons = 2024, 2025;
2. exact target weeks = 2-18 regular season;
3. target-week WR/TE roster universe is nonempty for every scheduled target week;
4. stable GSIS identity coverage among WR/TE universe rows >=95%;
5. >=1,500 WR/TE player-weeks have both qualified player and defense state;
6. each target season has >=500 supported player-weeks;
7. WR supported rows >=1,000;
8. TE supported rows >=300;
9. >=90% of player-profile-qualified rows have a four-band supported defense state;
10. player band-mass max absolute gap <=1e-12;
11. same/future player feature violations = 0;
12. same/future defense feature violations = 0;
13. sportsbook / market inputs used = 0;
14. football outcomes / model residuals used for scoring = 0;
15. fitted candidate models = 0;
16. production changed = false.

If any gate fails:
`PLAYER_DEPTH_MATCHUP_SOURCE_NOT_QUALIFIED`.

## Advancement rule

If source-qualified:
- freeze a separate predictive diagnostic before reading receiving outcomes;
- score WR and TE separately;
- preserve the already-frozen opportunity/entitlement baseline;
- require incremental replicated value in both 2024 and 2025;
- no coefficient / band / subgroup search after scoring;
- no immediate production promotion.

If not qualified:
- close this exact player-depth matchup source design;
- do not compensate with paid WR-CB data, generic defense-vs-position boosts, or
  post-hoc band thresholds.

## Integrity

Authorized:
- public schedule / weekly roster / PBP identity and strictly-prior features.

Forbidden:
- sportsbook data;
- OddsAPI;
- target-game outcomes for feature construction;
- target-game realized participation for universe selection;
- model residuals;
- predictive fitting;
- production changes.
