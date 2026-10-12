# RB PLAYER ROLE MATCHUP — INDIVIDUAL RUN-LOCATION SOURCE V1 PLAN

Frozen: 2026-10-11

Status: **SOURCE / RECONSTRUCTION AUDIT ONLY. NO RB OUTCOME SCORING. NO MODEL CHANGE.**

Parent roadmap:
- `docs/research/INDIVIDUAL_OPPORTUNITY_ALLOCATION_ROADMAP_V1.md`
- RB opportunity allocation remains primary and is already being accumulated
  prospectively.
- this source lane is a secondary player-specific matchup audit while those
  opportunity locks mature.

## Purpose

Audit whether one genuinely individual RB matchup state can be reconstructed
historically from free/public data:

> the individual RB's own strictly-prior left / middle / right rushing-location
> mix x the opponent defense's strictly-prior rushing efficiency allowed in
> those same locations.

No target-game rushing outcome is scored in V1.

## Anti-retest boundary

This is distinct from closed M95A/M95B.

M95A/M95B already tested:
- generic RB role x run-defense vulnerability;
- player/team efficiency and explosive profiles;
- box/front and defense strength families;
- a **team-level** directional rushing interaction:
  team left/middle/right rushing mix x defense left/middle/right YPA allowed.

This V1 does not use team directional mix.

V1 uses:
- one individual RB at a time;
- that player's own strict-prior run-location shares;
- opponent defense split across the same location bands.

Do not reopen:
- generic weak-defense => boost RB formulas;
- M95A/M95B model families;
- M96;
- box-rate retuning;
- sportsbook-conditioned inputs.

## Frozen historical universe

Target:
- seasons 2024 and 2025;
- regular-season Weeks 2-18;
- exact target-week weekly-roster RB / HB / FB identities;
- ACT / INA statuses when weekly roster status exists;
- stable GSIS identity required for a supported row.

History:
- public nflverse / nflreadpy PBP from 2023-2025;
- schedule identity only.

Target-game participation is not used to decide whether a roster row exists.

## Frozen rush-event definition

A located RB rush event requires:
- `rush_attempt == 1`;
- non-empty `rusher_player_id`;
- `qb_kneel != 1`;
- `qb_scramble != 1`;
- `two_point_attempt != 1`;
- finite `rushing_yards`;
- canonical offense / defense identity;
- `run_location` exactly reducible to:
  - LEFT
  - MIDDLE
  - RIGHT.

Events without a supported run location are excluded from location-profile
construction, not reassigned.

## Frozen individual RB state

For each target player-week:
- strictly prior located RB rushes only;
- latest up to 8 completed player rushing games;
- require >=4 prior rushing games;
- require >=20 located rush attempts.

Preserve:
- located attempts;
- mean rushing yards per located attempt;
- LEFT share;
- MIDDLE share;
- RIGHT share.

Player location shares must sum to 1.0 within numerical tolerance.

## Frozen opponent-defense state

For each target opponent:
- strictly prior located rushes faced only;
- latest up to 8 completed defensive games;
- require >=4 prior defensive games.

For each location preserve:
- located attempts faced;
- rushing yards per attempt allowed;
- rushing success rate allowed.

Each LEFT / MIDDLE / RIGHT band must contain >=10 prior located rush attempts.

## Descriptive interaction state

Source V1 may construct for audit only:

`run_mix_def_ypc = SUM(player_location_share * defense_location_ypc)`

`run_mix_def_success = SUM(player_location_share * defense_location_success)`

These are not projection candidates in V1.

## Frozen source-readiness gates

Disposition is
`RB_INDIVIDUAL_RUN_LOCATION_MATCHUP_SOURCE_QUALIFIED`
only if all pass:

1. exact target seasons 2024 and 2025;
2. exact target weeks 2-18;
3. target-week RB/HB/FB roster universe nonempty for every scheduled target week;
4. stable GSIS identity coverage >=95%;
5. >=1,200 RB player-weeks have both qualified player and defense state;
6. each target season has >=400 supported player-weeks;
7. >=90% of player-profile-qualified rows have supported three-location defense state;
8. player location-share max absolute mass gap <=1e-12;
9. same/future player feature violations = 0;
10. same/future defense feature violations = 0;
11. sportsbook inputs used = 0;
12. football outcomes / model residuals used for scoring = 0;
13. fitted candidate models = 0;
14. production changed = false.

Otherwise:
`RB_INDIVIDUAL_RUN_LOCATION_MATCHUP_SOURCE_NOT_QUALIFIED`.

## Advancement

If source-qualified:
- freeze a separate predictive diagnostic before reading RB rushing outcomes;
- preserve the frozen RB carry/snap opportunity mechanism;
- test incremental rushing-yard value against the canonical football-only
  historical baseline;
- require replication in both 2024 and 2025;
- no threshold / direction / subgroup search after scoring.

If source-not-qualified:
- close this exact run-location source design;
- do not replace it post-hoc with team directional shares already tested in
  M95B.

## Integrity

Authorized:
- schedule;
- weekly roster identity;
- strictly-prior nflverse PBP.

Forbidden:
- target-game outcomes for feature construction;
- target-game participation for universe selection;
- sportsbook / OddsAPI;
- model residuals;
- predictive fitting;
- production changes.
