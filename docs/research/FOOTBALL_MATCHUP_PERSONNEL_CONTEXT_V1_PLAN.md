# Football Matchup & Personnel Context V1 — Frozen Research / Engineering Plan

**Status:** PLAN + SOURCE ENGINEERING ONLY — NO PREDICTIVE RESULT, NO PRODUCTION CHANGE.

**Created:** 2026-09-17

**Branch:** `research-football-matchup-personnel-context-v1-plan`

## Purpose

Build the missing football-context layer needed to answer three practical pregame questions:

1. **What prior games are genuinely similar to this player's current matchup?**
2. **Which defender(s) is a receiver actually likely to face, and with what confidence?**
3. **How has current OL/DL personnel changed the trench environment relative to historical team averages?**

This program complements, but does not replace:

- `PLAYER_ROLE_ENVIRONMENT_REGIME_STATE_V1`;
- the frozen advanced-data feature dictionary;
- existing team opportunity / player entitlement / efficiency layers.

## Permanent boundaries

1. Sportsbook lines are downstream diagnostics only.
2. Target-game outcomes may never define a pregame similarity vector.
3. Nearest defender is geometry/proximity only unless an explicit responsibility label exists.
4. `pff_nflIdBlockedPlayer` remains a blocking-interaction identity, not universal assignment responsibility.
5. Personnel context is player-specific and time-aware; team names alone are not enough.
6. Similarity retrieval is evidence/context, not an automatic projection override.
7. Any learned predictive use must be separately frozen and walk-forward tested.
8. M72/M75 aggregate matchup families are not to be recycled under new names.
9. M84/M85 are source-blocked hypotheses, not scientific failures; new data may reopen only the information-acquisition layer honestly.

---

# Part I — Historical Matchup Analog Engine

## Question

For a current player-game, which completed historical player-games occurred under the most similar **pregame football state**?

The analog engine is not:

- same opponent only;
- same team only;
- same player only;
- similar Vegas line;
- similar final box score;
- an outcome-nearest-neighbor search.

It compares only information available before the target game's kickoff.

## Canonical analog grain

Primary grain:

`target_player_id + target_game_id -> ranked historical player-game analogs`

Every candidate analog must end before the target kickoff.

## Similarity state families

### A. Player role / regime

From `PLAYER_ROLE_ENVIRONMENT_REGIME_STATE_V1`:

- stable / expanded / contracted role;
- new-team status;
- room vacancy;
- hierarchy rank;
- current-season prior usage;
- role confidence.

### B. Team opportunity environment

Pregame football-only:

- projected plays / pace state;
- projected team pass/rush opportunity state;
- neutral pass/run tendency;
- QB rush competition;
- target/carry room size;
- expected player entitlement prior.

### C. Opponent structure

Strict-prior opponent descriptors:

- coverage man/zone family rates where qualified;
- pressure/blitz/front tendencies where qualified;
- defensive efficiency;
- rush/pass funnel tendencies;
- personnel continuity;
- current defender availability;
- trench continuity.

Do not use target-game realized coverage or pressure.

### D. Player archetype

Stable or strict-prior traits:

- position;
- alignment family;
- route-role family;
- target depth / usage archetype;
- rush style / receiving role for RB;
- QB mobility/style;
- TE route/block role.

### E. Advanced historical geometry profiles

Only frozen `hist_*` fields from the advanced feature dictionary:

- receiver historical route-conditioned release spacing;
- receiver historical crowding rates;
- route geometry history;
- blocker historical interaction geometry;
- other future strictly-prior advanced summaries after qualification.

No target-game BDB geometry may enter pregame similarity.

### F. Personnel / environment

- same-team vs new-team;
- QB continuity;
- play-caller continuity;
- OL continuity;
- opponent front continuity;
- meaningful starter additions/departures.

## Similarity computation

V1 engineering should compute a transparent component distance, not fit outcomes.

For each candidate historical player-game:

- categorical exact-match / mismatch indicators;
- standardized numeric distances using parameters frozen from the historical development set only;
- missingness indicators;
- evidence confidence.

Output separate component distances:

- `role_distance`
- `team_opportunity_distance`
- `opponent_structure_distance`
- `player_archetype_distance`
- `advanced_geometry_history_distance`
- `personnel_distance`

and:

- `analog_total_distance`
- `analog_rank`
- `analog_evidence_coverage`

No outcome is used to calculate distance.

## Analog abstention

Do not force analogs.

Abstain or flag low-confidence when:

- role regime is unresolved;
- less than a frozen minimum fraction of state dimensions are available;
- nearest historical examples remain far outside calibrated support;
- current player is a true novel state (e.g. rookie/new-team role with no close historical analogs).

Proposed states:

- `ANALOG_HIGH_SUPPORT`
- `ANALOG_MEDIUM_SUPPORT`
- `ANALOG_LOW_SUPPORT`
- `ANALOG_OUT_OF_DISTRIBUTION`
- `ANALOG_ABSTAIN`

## Future predictive question

Only after engineering is frozen:

> Does using outcomes from pregame-similar historical states improve a current baseline, especially in transition/matchup cohorts, without degrading stable/high-support cohorts?

Possible future uses:

- calibration prior;
- uncertainty width;
- regime-conditioned residual expectation;
- descriptive audit;
- nearest-neighbor sanity check.

Direct analog-outcome replacement is not authorized.

---

# Part II — WR/TE vs Defender Exposure / Assignment

## Historical evidence boundary

Existing repo result M84:

`HOLD_SOURCE_BLOCKED_NEW_INFORMATION`

This means exact receiver-defender responsibility was not scientifically rejected; the project lacked a complete historical + live source contract.

Current validated BDB laboratories materially improve geometry, but **do not create universal responsibility labels**.

## Three levels of defender evidence

### Level 1 — on-field / eligible defenders

Facts:

- which DB/LB defenders are active/on field;
- alignment / role if known;
- team coverage family.

This is not receiver assignment.

### Level 2 — geometric exposure

Derived from tracking:

- nearest defender;
- second-nearest defender;
- proximity duration;
- release proximity;
- throw proximity;
- leverage/spacing;
- crowding;
- zone-area proximity.

Name these fields with `exposure` or `proximity`, never `assignment`.

Examples:

- `receiver_defender_proximity_exposure_share`
- `receiver_primary_proximity_defender_id`
- `receiver_primary_proximity_share`
- `receiver_second_proximity_share`

These can describe who spent the most time geometrically close to a receiver.

They do **not** prove responsibility.

### Level 3 — explicit responsibility / matchup assignment

Requires an authoritative source with:

- receiver identity;
- defender identity;
- time/play segment;
- assignment/responsibility semantics.

Only Level 3 may populate:

- `primary_coverage_defender_id`
- `shadow_assignment_flag`
- `coverage_assignment_share`
- `responsibility_confidence`

Current BDB2021/BDB2026 validated labs do not by themselves qualify for Level 3.

## Geometry-based exposure reconstruction program

Engineering may build Level-2 exposure using BDB route/throw-window tracking.

Candidate deterministic fields:

- percent route frames defender is nearest;
- percent route frames defender is within 1/2/3 yd;
- continuity of nearest-defender identity;
- nearest-defender switches;
- release nearest defender;
- throw nearest defender;
- second-defender spacing;
- team man/zone source label where available;
- receiver alignment;
- route family.

Important:

`primary proximity defender` != `primary coverage defender`.

## Future responsibility classifier boundary

A learned assignment classifier is allowed only if a future labeled corpus contains actual defender-responsibility truth.

If such labels become available:

- train on explicit assignment truth;
- score precision/recall/calibration;
- require abstention for ambiguous zones/exchanges/brackets;
- never bootstrap labels from nearest defender and then claim validation.

Until then, geometry remains exposure.

## Current/live WR-CB reports

Current third-party matchup reports may be separately ingested as **current authoritative/reporting evidence** if source rights and stability are approved.

Without historical archive:

- useful for live diagnostics;
- useful for forward collection beginning now;
- not enough for retrospective walk-forward promotion.

Recommended engineering action:

Begin a timestamped **forward WR-CB assignment archive** whenever a stable approved current source is available.

Fields:

- target week/game/player;
- reported primary defender;
- reported shadow flag;
- source;
- publication time;
- pre-kickoff validity;
- source confidence;
- later postgame validation kept separately.

This creates proprietary history over time.

---

# Part III — OL vs DL / Pass-Rush Personnel Context

## Historical evidence boundary

M85 disposition:

`HOLD_SOURCE_BLOCKED_NEW_INFORMATION`

Exact blocker-rusher assignment was blocked by source availability, not rejected as a football mechanism.

BDB 2023 now gives a validated 2021 interaction-geometry laboratory:

- 46,396 reconstructed interactions;
- snap/min/terminal blocker-target distance;
- time-to-minimum distance;
- protection-window geometry;
- source roles;
- chip/release interactions.

This reopens **feature engineering / historical archetype development**, not live exact assignment.

## Current trench personnel state

Build player-level current OL and front-seven personnel artifacts.

### Offense

- expected starting LT/LG/C/RG/RT;
- primary swing tackle / interior backup;
- OL starter continuity;
- current injury/availability;
- new-team/new-starter flags;
- prior snap history;
- pass-block history where qualified;
- historical BDB-style blocker geometry summaries where identities can be bridged.

### Defense

- expected primary edge rushers;
- interior rushers;
- rotation depth;
- front personnel continuity;
- availability;
- strict-prior pressure/pass-rush history;
- historical interaction archetype where qualified.

## Matchup representation

Without exact live assignment, do not fabricate one fixed LT-vs-EDGE pairing.

Instead represent exposure probabilistically / structurally:

- likely edge-side exposure;
- interior-vs-interior exposure;
- front alignment tendencies where source supports;
- protection help / chip likelihood where supported;
- OL weak-link / DL strength distribution;
- personnel availability scenarios.

Potential context fields:

- `ol_returning_starters`
- `ol_current_starter_prior_snap_coverage`
- `ol_current_starter_passblock_history_coverage`
- `front_current_starter_prior_rush_history_coverage`
- `ol_personnel_change_count`
- `front_personnel_change_count`
- `ol_weak_link_state`
- `front_pressure_strength_state`
- `trench_matchup_evidence_confidence`

## BDB 2023 archetype library

Use the validated interaction geometry to build descriptive blocker/rusher archetypes:

Blocker side:

- typical snap separation;
- minimum interaction distance;
- time-to-engagement/minimum distance;
- terminal separation;
- chip/release interaction rates;
- position / alignment strata.

Defender side:

- source role;
- interaction geometry distributions;
- edge/interior strata where source position supports.

These are historical laboratory profiles. Live player usage requires a lawful current bridge and strict-prior history.

## Personnel turnover principle

Team historical averages should not be treated as identical when the relevant players changed.

For every target game, emit:

- offense continuity relative to prior history;
- defense/front continuity relative to prior history;
- percentage of historical snaps represented by current expected starters;
- missing/new starter count;
- regime state.

Potential states:

- `TRENCH_STABLE`
- `OL_MAJOR_TURNOVER`
- `DL_MAJOR_TURNOVER`
- `BOTH_TRENCHES_CHANGED`
- `KEY_OL_ABSENCE`
- `KEY_RUSHER_ABSENCE`
- `TRENCH_UNCERTAIN`

---

# Part IV — Matchup Story / Game Script Audit

Historical data describes previous teams and personnel. The current target game may have changed actors.

Build a pregame `FOOTBALL_MATCHUP_STORY_V1` audit containing:

- current player role/environment regime;
- team opportunity state;
- opponent structure;
- current personnel continuity;
- advanced historical profile;
- analog support;
- matchup exposure confidence;
- trench state;
- unresolved evidence.

The artifact is descriptive first.

It should be able to say things such as:

> This receiver's raw historical target share comes mostly from a different team/QB regime; current role is established primary, opponent is a high-man defense, but exact CB responsibility is unavailable. Historical analog support is medium.

or:

> Team pressure history is strong, but two current starting rushers were not part of the historical sample and opponent LT is new; team-level prior is low-continuity and should not be treated as a stable personnel prior.

No yardage adjustment is implied until separately tested.

---

# Part V — Source / Temporal Classes

Every matchup/personnel field must be classified as:

- `PREGAME_DIRECT_CURRENT`
- `PREGAME_STRICT_PRIOR_HISTORY`
- `PREGAME_DERIVED_ANALOG`
- `TARGET_GAME_POST_KICKOFF`
- `RETROSPECTIVE_VALIDATION_ONLY`

Target-game tracking, actual matchup switches, actual pressure, actual coverage responsibility and final outcomes are prohibited pregame.

---

# Part VI — Immediate Engineering Sequence

1. Finish the advanced-data materializer certification already in progress.
2. Freeze `PLAYER_ROLE_ENVIRONMENT_EVIDENCE_V1`.
3. Build a source inventory for current OL/DL starters, depth, availability, snaps and pass-rush history from already-qualified repo sources.
4. Build `CURRENT_TRENCH_PERSONNEL_STATE_V1` with no predictive coefficients.
5. Build deterministic BDB 2021 route-frame **proximity exposure** materializer; preserve assignment disclaimer.
6. Build deterministic BDB 2023 blocker-interaction archetype summaries.
7. Build historical pregame analog-state schema and distance audit.
8. Run only coverage/support/temporal QA — no outcome fitting yet.
9. Coordinate each position-specific predictive experiment separately after source engineering is frozen.

---

# Anti-loop / anti-overclaim rules

Do not:

- call nearest defender "the corner covering him";
- invent CB-WR assignments from on-field participation;
- invent exact OL-DL assignments from depth charts;
- treat old team-level defense/offense averages as stable when personnel turnover is high;
- use analog outcomes to define similarity;
- use sportsbook lines in analog distance;
- use target-game outcomes to select analogs;
- rerun M72/M75 aggregate matchup work under new labels;
- claim BDB competition slices are live feeds.

---

# Disposition

`FOOTBALL_MATCHUP_PERSONNEL_CONTEXT_V1_PLAN_FROZEN_PENDING_ENGINEERING`
