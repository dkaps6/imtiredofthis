# RB STACK6T — Fine-State Conditional-Call Context Oracle

## Status

**FROZEN BEFORE 2025 ATTRIBUTION.** No fitted model, no production change, no P3 recomposition, and no sportsbook input are authorized.

## Why this exists

STACK6R established that down/distance occupancy explains only about 4-5% of designed-run prediction error once coarse score state is known; roughly 95-96% remains in the conditional call decision. STACK6S then showed that conditional offense-vs-defense run/pass EPA and success-rate advantage does not qualify as a stable pregame explanation.

Before declaring this layer close to a pregame irreducible-noise floor, STACK6T asks whether the apparently unpredictable conditional-call residual is actually missing **fundamental in-game context** that was not represented by STACK6R: field position and game phase.

This is a structural oracle, not a fitted predictor.

## Parent decision cells

Use exactly the frozen STACK6R parent cells:

Score state:
- lead: score differential > +3
- neutral: -3 through +3
- trail: < -3

Down/distance context:
1. FIRST_DOWN
2. SECOND_SHORT_MED: down 2, ytg <=6
3. SECOND_LONG: down 2, ytg >=7
4. LATE_SHORT: down 3/4, ytg <=3
5. LATE_LONG: down 3/4, ytg >=4
6. OTHER identity bucket

Designed run: `rush_attempt == 1 AND qb_scramble != 1 AND qb_kneel != 1`.

## Fine-state subdivisions

Within each parent state × down/distance cell, add two football-natural dimensions.

### Field zone from nflverse `yardline_100`

- `RED_ZONE`: yardline_100 <=20
- `OPP_MID`: 21-50
- `OWN_MID`: 51-79
- `BACKED_UP`: >=80
- `FIELD_OTHER`: missing/unclassifiable only; report share explicitly.

### Game phase

- `EARLY`: quarter 1-3
- `LATE`: quarter >=4 (including OT)
- `PHASE_OTHER`: missing/unclassifiable only.

Fine cell = parent score state × parent down/distance context × field zone × game phase.

No threshold search is permitted.

## Strict-prior history schemes

Run both:
- `TEAM5_SHRUNK`
- `TEAM8_SHRUNK`

For every target team-week:

### Parent conditional rate

Team prior-N designed runs / decision opportunities in the parent state × down/distance cell, shrunk to the strict-prior league parent-cell rate with `24` pseudo opportunities.

### Fine conditional rate

Team prior-N designed runs / opportunities in the exact fine cell, shrunk to the strict-prior league fine-cell designed-run rate with `24` pseudo opportunities.

All priors use games strictly before the target week.

## Oracle arms

Actual target-game parent/fine-cell occupancy is grading scaffold only.

### PARENT_CONTEXT

`sum_target_parent_cells(actual_parent_plays * strict_prior_parent_run_rate)`

This must reproduce the corresponding STACK6R `ORACLE_CONTEXT_OCCUPANCY` MAE on W6-18:

- TEAM5 expected: `3.9636153118306288`
- TEAM8 expected: `4.015161235681258`

within `1e-9`.

### FINE_CONTEXT

`sum_target_fine_cells(actual_fine_plays * strict_prior_fine_run_rate)`

### ORACLE_BOTH

Actual target-game designed attempts; exact identity.

Fine-state recovery:

`PARENT_CONTEXT_MAE - FINE_CONTEXT_MAE`

Fine-state recovery fraction:

`recovery / PARENT_CONTEXT_MAE`

## Frozen populations

- ALL 2025 W6-18
- POOL_OVER_5
- POOL_UNDER_5
- W13-18

P3 bins come only from frozen STACK6H and are grading-only.

## Frozen dispositions

`FINE_STATE_CONTEXT_MATERIAL` only if, under **both** TEAM5 and TEAM8:

1. ALL W6-18 fine-state recovery fraction >= `0.20`;
2. POOL_OVER_5 fine-state recovery > `0`;
3. POOL_UNDER_5 fine-state recovery > `0`;
4. W13-18 fine-state recovery > `0`.

`FINE_STATE_CONTEXT_NOT_PRIMARY` if ALL W6-18 recovery fraction <= `0.10` under both schemes.

Otherwise `FINE_STATE_CONTEXT_MIXED`.

No gate may be waived.

## Integrity

Require:
- 2023-2025 regular-season PBP;
- `yardline_100` and `qtr` present;
- exhaustive parent and fine-cell identity;
- FIELD_OTHER + PHASE_OTHER share reported and <=1% of W6-18 decision plays;
- 544/544 2025 team-games joined to STACK6H;
- 388 W6-18 team-games;
- PARENT_CONTEXT W6-18 MAEs reproduce STACK6R expected values to `1e-9`;
- ORACLE_BOTH exact;
- strict-prior construction flag = 1;
- no fitted model/search/sportsbook input;
- target-game PBP used only as oracle/grading truth.

## Decision boundary

If fine-state context is material, the next architecture should be a probabilistic/generative game-state representation that can forecast field-position/game-phase exposure before kickoff rather than another scalar team rush-rate model.

If fine-state context is not primary, then down/distance, field position, game phase, conditional efficiency advantage, QB/scramble effects, and coarse game script have all failed to explain most of the remaining conditional-call error in a pregame-usable way. At that point the remaining team-rush layer should be treated as approaching an irreducible pregame component and RB work should pivot to final integration/benchmarking rather than endless context slicing.