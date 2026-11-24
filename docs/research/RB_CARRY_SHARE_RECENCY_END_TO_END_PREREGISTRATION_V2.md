# RB carry-share recency, end to end: PREREGISTRATION V2 (the corrected test)

**Date:** 2026-11-24. Written before any V2 arm runs.

**Why V2 exists.** The PR #716 review found three gaps between the V1 test and what production
runs. All three were confirmed by reading the code:

1. **The V1 test changed every position.** The C1 arm replaced `rush_share_current` for QBs,
   WRs and TEs as well. Production changes only RB/FB rows. Those other shares compete in the
   same top-5 and 0.95 cap, so V1 measured a different intervention.
2. **Production dropped zero-touch games.** PlayerForm's weekly logs keep only games with a
   carry, target or pass attempt. So production's "last 3" skipped games an RB played without
   a touch, and reached back to older, busier games. The V1 test counted every snap game.
3. **No anomaly gate.** AGENTS.md requires that a game is explained before it becomes trend
   evidence. V1 gave an injury-shortened game or early exit 85% weight as if it were a role
   change.

A fourth gap, found while fixing these: V1's C0 computed season-to-date with the current team
only. Production uses every current-season game, with any team.

## What V2 tests

One shared module, `scripts/modeling/rb_carry_recency_v2.py`, runs in both production and this
test, so the two cannot drift apart again.

**C0 = production as it was before recency:**
- built with production's own `player_form_v2._normalize_weekly` and `_season_totals`;
- season-to-date share and `current_games` over every current-season game with a touch;
- prior from the previous season, the same way;
- Bayes through production's `_weighted_group_mean` and `_posterior_mean`, then top 5, the
  0.95 clip and the 0.95 scaling.

**C1u (recency, ungated):** for RB/FB only, `0.15 x C0 season-to-date + 0.85 x last-3`.
- Last 3 means his last three snap games with his slate team, carries filled as 0.
- Requires `current_games >= 3` and three snap games.
- This is reported to show what the gate costs or saves.

**C1g (the production candidate):** C1u, but recency applies only when all three window
games are labelled `HEALTHY_FULL` by the module. Otherwise C0's share is kept.

**Game-state labels** (the module docstring has the exact thresholds):
- each game is judged against the median snap share of the player's other games with the
  same team;
- inputs are nflverse snap counts and the official injury report;
- the labels are `RETURN_FROM_INJURY`, `INJURY_LIMITED`, `PARTIAL_GAME`,
  `UNRESOLVED_CONTEXT` and `HEALTHY_FULL`;
- the thresholds (0.75, 0.5, 2.0 and an absolute gap of 0.20) are fixed here, before the run,
  and are not tuned afterwards.

**Unchanged from V1:**
- universe;
- holdouts (2025 weeks 4-18 with 2024 priors, 2026 weeks 4-10 with 2025 priors);
- `a = 0.85`;
- actual team carries;
- scored rows (RBs in the universe who took a snap);
- the metrics.

## Kill and pass (identical to V1, applied to C1g vs C0)

- **K1:** the primary MAE gain CI includes 0 in either holdout → null.
- **K2:** lead-back MAE is worse in either holdout beyond its CI → null.
- **PASS:** the gain is positive with the CI excluding 0 in both holdouts, and there is no K2.

## Decision tied to the result

- **PASS:** `RB_CARRY_SHARE_RECENCY_V2` replaces V1 in production.
- **Null:** recency is removed from production. Neither V1 (it fails the anomaly gate) nor
  C1g has support, so the pre-recency behaviour returns.

**C1u is never a production option.** It breaks the anomaly gate whatever its score.
