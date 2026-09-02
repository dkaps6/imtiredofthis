# RB STACK6R — Designed-Run Context Occupancy vs Conditional Call Oracle

## Status

**FROZEN BEFORE 2025 ATTRIBUTION.** No fitted model, no production change, and no sportsbook input are authorized.

## Why this exists

STACK6P showed that designed-run calls account for roughly 90-92% of the remaining within-state rushing-tendency oracle headroom. STACK6Q then showed that one compact state-level team-week Ridge does not recover that headroom reliably: it helped P3 false-high pool games but worsened false-low games and failed four of seven frozen gates.

STACK6R changes representation rather than tuning STACK6Q. It asks whether target-game designed-run variation within the already-isolated lead/neutral/trail states is driven primarily by:

1. **context occupancy** — which down/distance situations the offense reaches; or
2. **conditional run calling** — whether the offense calls a designed run once in those situations.

## Football-natural contexts

Every offensive play (`rush_attempt == 1 OR qb_dropback == 1`) is assigned exactly one bucket:

1. `FIRST_DOWN`: down = 1;
2. `SECOND_SHORT_MED`: down = 2 and yards-to-go <= 6;
3. `SECOND_LONG`: down = 2 and yards-to-go >= 7;
4. `LATE_SHORT`: down in {3,4} and yards-to-go <= 3;
5. `LATE_LONG`: down in {3,4} and yards-to-go >= 4;
6. `OTHER`: any remaining or missing-down/distance offensive play, retained only so the identity is exhaustive.

No threshold search is allowed.

Score states remain the frozen M94B/M94C definition: lead > +3, neutral -3 through +3, trail < -3 from the offense perspective.

Designed rush remains `rush_attempt == 1 AND qb_scramble != 1 AND qb_kneel != 1`.

## Strict-prior baseline schemes

Run the decomposition under two precommitted history schemes:

- `TEAM5_SHRUNK`: prior five team games;
- `TEAM8_SHRUNK`: prior eight team games.

For every target team-week, state, and context:

- prior context occupancy share is the team's prior context plays / prior state plays, shrunk to the strict-prior league state-context share with 24 pseudo state plays;
- prior conditional designed-run rate is prior designed attempts / prior context plays, shrunk to the strict-prior league state-context designed rate with 24 pseudo context plays.

All history is strictly before the target week. No target-game outcome enters a prior estimate.

## Oracle sequence

Actual target-game **score-state play counts are supplied as an oracle scaffold** so STACK6R isolates only the representation inside state-conditioned designed-run behavior.

For each target team-game:

### BASE_CONTEXT

For every state:

`actual_state_plays * sum_context(prior_context_share * prior_context_designed_rate)`

Sum across states to obtain predicted designed attempts.

### ORACLE_CONTEXT_OCCUPANCY

Replace only the within-state context mix with the target game's realized context play counts while keeping strict-prior conditional designed-run rates:

`sum_state,context(actual_context_plays * prior_context_designed_rate)`

### ORACLE_BOTH

Actual target-game designed attempts. This must be an exact identity.

The sequential attribution is therefore:

- **context-occupancy recovery** = BASE_CONTEXT MAE - ORACLE_CONTEXT_OCCUPANCY MAE;
- **conditional-call remainder** = ORACLE_CONTEXT_OCCUPANCY MAE - ORACLE_BOTH MAE.

Because ORACLE_BOTH is the actual designed-attempt identity, the conditional remainder is simply the MAE still left after context occupancy is made perfect.

## Frozen populations

Primary: 2025 W6-18 team-games.

Also grade the inherited STACK6H/P3 bins:

- `POOL_OVER_5`;
- `POOL_UNDER_5`;
- W13-18 stability.

These bins are grading-only and never affect prior estimates.

## Frozen dispositions

For each prior-history scheme calculate:

`occupancy_fraction = occupancy_recovery / BASE_CONTEXT_MAE`

Then:

- `CONTEXT_OCCUPANCY_DOMINANT` only if occupancy_fraction >= 0.50 under **both** schemes and occupancy recovery is positive in both POOL_OVER_5 and POOL_UNDER_5 under both schemes;
- `CONDITIONAL_CALL_DOMINANT` only if occupancy_fraction <= 0.25 under **both** schemes (meaning at least 75% of baseline designed-attempt MAE remains after perfect context occupancy);
- otherwise `MIXED_DESIGNED_RUN_MECHANICS`.

No gate may be waived after seeing 2025.

## Integrity

Require:

- 544 2025 team-games and 388 W6-18 team-games after joining STACK6H;
- exact one-context assignment for every offensive play;
- `OTHER` share reported explicitly;
- ORACLE_BOTH max absolute error <= 1e-9;
- strict-prior construction flag = 1;
- no fitted models;
- no feature, threshold, model-family, or hyperparameter search;
- no sportsbook inputs;
- target-game PBP used only for oracle/grading truth.

## Decision boundary

STACK6R does not authorize a point-model change. It selects the next architecture:

- occupancy dominant -> investigate a pregame down/distance/drive context generator;
- conditional-call dominant -> investigate football predictors of run/pass choice *within* down/distance and score state (e.g. run-vs-pass efficiency advantage, short-yardage conversion environment, defensive front response) rather than another aggregate state rate;
- mixed -> a joint hierarchical context architecture is required.
