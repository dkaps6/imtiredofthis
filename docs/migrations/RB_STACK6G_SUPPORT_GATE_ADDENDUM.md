# RB STACK6G — Frozen Descriptive Support-Gate Addendum

This addendum is frozen before the STACK6G 2025 forensic atlas is executed. It does not authorize a fitted model.

## QB1 regime support

The QB1 regime mechanism is `SUPPORTED` only if every item below passes on the frozen 2025 W6-18 P3 team-RB-pool population:

1. timestamp-safe target-QB1 coverage >= 0.90;
2. strictly-prior target-QB rushing-propensity delta coverage >= 0.75;
3. Pearson correlation between `qb_rush_propensity_delta` and `p3_pool_residual` >= +0.10, where positive residual means P3 predicted too many RB carries;
4. mean P3 pool residual in the top quartile of `qb_rush_propensity_delta` minus the bottom quartile >= +1.00 carry;
5. mean `qb_rush_propensity_delta` in `POOL_OVER_5` minus the mean in `NON_EXTREME_ABS_LT3` >= +0.50 rush attempt/game.

Football direction is frozen: a target QB who is more rush-prone than the QBs represented in the recent team-history window should, all else equal, leave fewer team carries for RB/HB/FB and therefore be associated with more-positive P3 RB-pool residuals.

QB1-change rate is reported but is not itself a support gate because a change from one pocket QB to another is not mechanically equivalent to a mobility-regime change.

## Playcaller regime support

The playcaller mechanism is `SUPPORTED` only if every item below passes on 2025 W6-18:

1. verified playcaller mapping coverage >= 0.95;
2. at least 8 team-games are in `playcaller_recent_change`, defined as the first three team-games under a newly documented in-season caller;
3. mean absolute P3 pool residual for `playcaller_recent_change` minus stable-caller games >= +1.00 carry;
4. `POOL_ABS_5` rate for `playcaller_recent_change` minus stable-caller games >= +0.10.

No direction is imposed on signed carry residual for playcaller changes; the hypothesis is that recent team tendency may become stale in either direction after a documented caller change.

## Source failure

A source family cannot be supported if the relevant coverage/integrity gate fails, regardless of descriptive outcome separation.

## No retuning

These thresholds are not to be changed after results are exposed. Failure is evidence and does not authorize a threshold search, feature search, or immediate derivative retune.