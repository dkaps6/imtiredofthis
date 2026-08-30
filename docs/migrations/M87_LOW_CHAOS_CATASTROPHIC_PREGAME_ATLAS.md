# Migration 87 — Low-Chaos Catastrophic Pregame Atlas

## Status

`PREREGISTERED / FORENSIC ONLY`

M87 follows the M86 error-floor audit by shrinking the research target to the 38 full-stack QB passing misses that exceeded 100 yards without tripping any of M86's frozen high-event-chaos markers.

M87 does not fit a predictive model and does not change production logic. Its purpose is to determine whether these low-chaos catastrophic misses show stable, strictly-pregame structure that justifies a later frozen predictive test.

## Authoritative upstream state

- M82 full-stack OOS ensemble benchmark MAE: `56.749517`
- M82 canonical QB games: `884`
- M82 full-stack 100+ misses: `123`
- M82 hindsight current-library oracle: `41.103131`
- M86 high-event-chaos 100+ misses: `85`
- M86 low-event-chaos 100+ misses: `38`
- M86 low-chaos component split: `16` volume-dominant, `17` efficiency-dominant, `5` mixed.

M87 must recover the exact authoritative M86 forensic trace and fail closed if these counts drift.

## Scientific questions

M87 asks separately what strictly-pregame characteristics distinguish low-chaos catastrophic volume failures and low-chaos catastrophic efficiency failures. It also asks whether the existing MC/ML/State library already contained a substantially better answer in these games and, if so, which representation won.

## Hard leakage boundary

All atlas features must be available before kickoff. Allowed inputs are frozen M82/M86 model predictions and disagreement, canonical predicted attempts/YPA, strictly-prior team/offense/defense PBP aggregates, strictly-prior opponent-offense aggregates, target-week historical injury reports, and deterministic venue architecture.

Target-game outcomes, target-game PBP events, realized receiver production, post-kickoff injury/participation information, observed kickoff weather, and all sportsbook variables are forbidden.

M86 target-game event variables are used only to define the already-frozen low-chaos subset and may not enter the pregame feature screen.

## Frozen controls

A control candidate must:

- come from the same season;
- also be `LOW_EVENT_CHAOS` under the already-frozen M86 postgame forensic classification;
- have M82 ensemble absolute error `< 50` yards;
- not be a 100+ tail;
- have complete matching variables.

For each target row select five nearest controls using standardized Euclidean distance over target week, ensemble projected passing yards, canonical predicted attempts, and canonical implied predicted YPA. Controls are selected separately for volume and efficiency atlases. Reuse is permitted because this is descriptive matching, not causal estimation.

The low-chaos requirement on controls prevents the atlas from merely rediscovering which pregame profiles later happened to produce M86's high-event-chaos games.

## Strictly-prior history

For every target/control game, history uses the latest `8` prior regular-season games, may cross into the immediately prior season, and must be strictly before `(season, week)`. If fewer than `4` prior games exist, that history feature is insufficient; future games are never borrowed.

## Frozen feature families

### A. Model-state diagnostics

- ensemble projection
- canonical predicted attempts
- canonical implied predicted YPA
- MC/ML/State prediction standard deviation
- MC/ML/State prediction range
- ML minus MC projection
- State minus MC projection
- ensemble minus canonical projection

These are existing-model representation diagnostics, not new football information.

### B. Target-offense recent structure

- pass rate
- neutral pass rate
- shotgun rate
- no-huddle rate
- plays per game
- pass EPA per dropback
- offensive success rate
- yards per pass attempt
- explosive 20+ completion rate per pass attempt
- deep attempt rate (`air_yards >= 15`)
- sack rate per dropback
- QB scramble rate per dropback

### C. Opponent-defense recent environment

- pass rate faced
- neutral pass rate faced
- pass EPA allowed per dropback
- offensive success rate allowed
- yards per pass attempt allowed
- explosive 20+ completion rate allowed
- deep attempt rate faced
- sack rate generated
- interception rate generated
- plays faced per game

### D. Opponent-offense game-script pressure

- pass EPA per dropback
- offensive success rate
- plays per game
- neutral pass rate
- yards per pass attempt

### E. Pregame availability / venue context

- target-team total injury-report count
- target-team Out/Doubtful count
- target-team Questionable count
- opponent total injury-report count
- opponent Out/Doubtful count
- opponent Questionable count
- target-team home indicator
- controlled-environment indicator

These variables are forensic context only. Generic injury burden has prior overlap with M78/M79 and may not be reopened merely because it characterizes the selected M87 subset.

## Frozen descriptive screen

For each primary target family (`VOLUME_DOMINANT`, `EFFICIENCY_DOMINANT`), report target N, matched-control N, feature coverage, target/control means, combined SMD, and separate 2024/2025 SMDs.

A feature is a `STABLE_FORENSIC_DIFFERENTIATOR` only if all are true:

1. target and control coverage each `>= 0.85`;
2. absolute combined SMD `>= 0.50`;
3. 2024 and 2025 SMD signs agree;
4. absolute SMD is at least `0.20` in each season.

Thresholds are frozen before results and cannot be changed.

### Directional secondary atlas

M87 must also describe `UNDERPROJECTED` and `OVERPROJECTED` subgroups within each primary family when a subgroup contains at least `5` rows. These directional rows use the same already-frozen matched controls and report descriptive SMDs only.

Directional subgroup results are explicitly `advancement_eligible = false`. They may explain whether explosions and collapses look different, but they cannot independently open a predictive migration or alter the primary stable-differentiator gate after results are seen.

## Existing-model rescue diagnostic

For each low-chaos family report:

- ensemble MAE;
- hindsight best MC/ML/State MAE;
- share with at least one component below 75 yards absolute error;
- share with at least one component below 50 yards absolute error;
- hindsight best-model distribution among MC, ML and State;
- mean component disagreement.

A family has a `MODEL_REPRESENTATION_CLUE` only if all are true:

1. one single component model is hindsight-best on at least `60%` of target rows;
2. the component-library hindsight MAE beats the ensemble MAE by at least `20` yards within that family;
3. at least `50%` of target rows have one component below `75` yards absolute error.

This remains post-hoc and nondeployable. M87 may not build or tune a model selector.

## Decision contract

Possible dispositions:

- `STABLE_PREGAME_DIFFERENTIATORS_FOUND`
- `MODEL_REPRESENTATION_CLUE_ONLY`
- `NO_STABLE_LOW_CHAOS_PREGAME_DIFFERENTIATOR`

A later predictive migration is allowed only if M87 identifies at least one stable differentiator or a frozen `MODEL_REPRESENTATION_CLUE` that can be translated into a genuinely pregame hypothesis without tuning on the M87 target outcomes.

## Anti-loop

M87 does not reopen previously failed families merely because one has a large descriptive SMD in the 38 selected tails. Any future test must distinguish already-tested information from genuinely new or architecturally unresolved information.

No Ridge/HGB/XGB/neural-network/selector search is permitted in M87.

`production_actionable = false`
