# Week 4+ Projection Authority Move-Direction Forward Confirmation V1 — Frozen Plan

Date frozen: 2026-09-29  
Status: **FROZEN BEFORE ANY WEEK-4+ OUTCOME — PROSPECTIVE SHADOW ONLY**  
Branch: `research-week3-postmortem-execution-v1`

## Why this exists

The preregistered Weeks 1-3 Projection Authority Line-Conflict V1 primary hypothesis failed:
`NO_CLEAR_CURRENT_SEASON_AUTHORITY_LINE_CONFLICT_SIGNAL`.

However, that same frozen diagnostic required predeclared state summaries. Those summaries exposed a secondary descriptive pattern:
- `SAME_SIDE_STRENGTHENED`: 250 rows, 44.0%, -38.15u, final MAE worse than MC;
- `SAME_SIDE_WEAKENED`: 471 rows, 55.2%, +24.44u, final MAE better than MC.

Because Weeks 1-3 outcomes were already known when that diagnostic was designed, this pattern is **discovery only**. It cannot authorize a rule.

This V1 is the clean prospective test.

## Prospective question

Among future production-selected props where upstream MC and final production means remain on the same side of the exact published sportsbook line:

> Do late authority moves that push the final mean **farther away from the market line** remain less reliable than late authority moves that pull the final mean **closer to the market line**?

This is a reliability / architecture diagnostic, not permission to feed sportsbook data upstream into football generation.

## Start boundary

Eligible observation begins with the first canonical 2026 **Week 4** production board archived after this plan commit.

No Week 1-3 row can count toward confirmation support.

## Immutable pregame inputs

Use only fields already present on the canonical archived production board:
- event/player/team identity;
- season/week;
- position;
- market;
- exact selected `vegas_line`;
- exact pregame `mc_proj`;
- exact pregame `model_proj`;
- selected side / price / decision lineage;
- source run ID / source git SHA.

No new paid OddsAPI acquisition is authorized by this plan.

The pregame board artifact is the authority. Postgame reconstruction of projections or lines is forbidden.

## Frozen state definitions

Tolerance: `1e-12`.

Let:
- `mc_gap = mc_proj - vegas_line`;
- `final_gap = model_proj - vegas_line`.

Only same-side nonzero rows enter the primary comparison:
- both gaps > tolerance; or
- both gaps < -tolerance.

Then:

### `STRENGTHENED`
`abs(final_gap) > abs(mc_gap) + 1e-12`

### `WEAKENED`
`abs(final_gap) < abs(mc_gap) - 1e-12`

### `UNCHANGED_DISTANCE`
remaining same-side rows.

Crossed-line / on-line rows are captured descriptively but excluded from the strengthened-vs-weakened primary contrast.

## Deduplication / scientific unit

The scientific row is one canonical published selected-bet row:
`(season, week, event_id, player_clean_key, market, book, side, vegas_line)`.

Exact duplicate rows must collapse only if every protected pregame field agrees. Conflicting duplicates fail closed.

Game clustering uses `(season, week, event_id)`.

## Outcome attachment

After each eligible week is final:
- attach verified actual only through the existing settlement authority;
- preserve VOID separately;
- never convert DNP / nonparticipation to a statistical zero unless the canonical settlement rules already do so.

No result may be scored from live/incomplete games.

## Frozen metrics

Primary:
1. selected-bet win-rate difference:
   `STRENGTHENED - WEAKENED`;
2. final model-closer-than-selected-line rate difference:
   `STRENGTHENED - WEAKENED`.

Secondary:
- units / ROI;
- raw final MAE by market;
- raw MC MAE by market;
- paired `abs(mc-actual) - abs(final-actual)`;
- strengthened/weakened support by market and position;
- unchanged-distance reference.

No new threshold search or subgroup rescue is allowed.

## Cluster-aware uncertainty

At the confirmation checkpoint:
- resample NFL games as clusters;
- 10,000 bootstrap replicates;
- seed `42040`;
- report 95% percentile intervals for the two primary differences.

## Minimum confirmation support

Do not call PASS or FAIL until **both**:
- at least 8 distinct eligible NFL weeks have been observed beginning Week 4;
- at least 400 unique eligible `STRENGTHENED` rows **and** 400 unique eligible `WEAKENED` rows exist.

Before both floors are met:
`FORWARD_OBSERVATION_ONLY_INSUFFICIENT_SUPPORT`.

## Frozen confirmation gate

### `AUTHORITY_MOVE_DIRECTION_FORWARD_CONFIRMED`
Requires all:
- support floors met;
- strengthened win rate at least 5 percentage points below weakened;
- strengthened model-closer rate at least 8 percentage points below weakened;
- both corresponding game-cluster bootstrap 95% intervals are fully below 0.

### `AUTHORITY_MOVE_DIRECTION_FORWARD_NOT_CONFIRMED`
If support floors are met and any confirmation condition fails.

No intermediate weekly result may be relabeled PASS/FAIL.

## If confirmed

Confirmation would authorize only a **mechanism audit** asking which late production authority layers generate harmful farther-from-line movement.

It would **not** directly authorize:
- shrinking projections toward Vegas;
- excluding strengthened bets;
- changing fair probability;
- changing EV / edge thresholds;
- changing football coefficients.

Any production candidate would need a separately frozen causal/mechanical test that preserves the sportsbook/football boundary.

## If not confirmed

Close this lane. Do not rescue with:
- market-only thresholds;
- position-only thresholds;
- QB/RB/WR/TE carveouts;
- nearby movement cutoffs;
- edge-size interactions;
- Week-4-only or Week-5-only rules.

## Production boundary

This is shadow research only. No production output changes.
