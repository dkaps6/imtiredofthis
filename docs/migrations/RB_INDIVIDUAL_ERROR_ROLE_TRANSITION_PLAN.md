# RB Individual Error / Role-Transition Audit — Frozen Plan

## Purpose

The prior production audit proved that current pregame depth state is preserved in the data but not directly used by the rushing allocation path. RB Role-Order Remap V1 then proved that blindly forcing existing carry magnitudes into current depth order is **not** a valid correction.

The next step is diagnosis, not another allocation tweak.

This audit asks **which individual RBs the current model repeatedly misses, in which direction, and whether those misses cluster around specific pregame role-transition states**.

No model is fit. No sportsbook data is read. No production value changes.

## Canonical evidence

- STACK1 production-equivalent baseline run: `33535308110`
- STACK2 enriched allocation casebook run: `33538770934`
- Role-order result commit: `c3ab51cebf0c4c6af29f8604fa852a6efe7c1b9b`
- Canonical 2025 evaluation rows: `1393`

The current audit retains the same 2025 leakage-safe casebook because that is the exact historical source carrying timestamp-safe depth, roster, injury, prior-usage, rookie and same-team metadata needed for the question.

## Individual-player error profile

For every RB/HB in the canonical casebook, report:

- games
- carry MAE
- carry signed bias (`projection - actual`)
- carry median absolute error
- carry 90th percentile absolute error
- rush-yard MAE
- rush-yard signed bias
- rush-yard 30+ / 50+ miss rates

These full-sample player profiles are diagnostic only. Any future player-specific calibration must be rebuilt walk-forward using only prior games.

## Frozen role-transition states

Evaluate the existing baseline errors by these pregame-available states already present in STACK2:

1. `DEPTH_VS_CARRY_ORDER_MISMATCH`
   - current depth rank differs from the player's rank by baseline projected carries within the same team-week.

2. `LIMITED_PRIOR_HISTORY`
   - `prior_games <= 2`.

3. `NO_PRIOR_SAME_TEAM_GAME`
   - `same_team_last_game != 1`.

4. `ROOKIE`
   - `rookie_flag == 1`.

5. `INJURY_CREATED_CONTEXT`
   - any of `injured_comp_count > 0`, `injury_out_doubtful == 1`, `injury_questionable == 1`, `practice_dnp == 1`, `practice_limited == 1`.

6. `DEPTH_PRESENT`
   - canonical timestamp-safe `depth_present == 1`.

The audit will also report combinations that are definitions of the above states only (for example mismatch + limited prior history). It will not search thresholds, fit interactions, or try correction weights.

## Frozen outputs

For each state:

- rows
- carry MAE / RMSE / bias
- rush-yard MAE / RMSE / bias
- 5+ carry absolute-miss rate
- 10+ carry absolute-miss rate
- 20+ rush-yard absolute-miss rate
- 40+ rush-yard absolute-miss rate
- mean signed actual-minus-projection carry residual
- mean signed actual-minus-projection yard residual

Also report the complementary state so we can distinguish a real concentration of error from the position-wide baseline.

## Integrity requirements

- exact 1393-row STACK1 parent parity
- exact 1393-row STACK2 metadata identity after canonical `JAC/JAX` and `LA/LAR` aliases
- inherited depth coverage `0.949748743718593`
- zero timestamp violations inherited from STACK2
- zero sportsbook inputs
- zero model fitting
- zero production changes

## Interpretation boundary

This audit does **not** authorize a correction by itself.

A later RB candidate is only justified if an error mechanism is both:

1. materially concentrated in a clearly pregame-identifiable state; and
2. football-mechanistically coherent.

Any candidate must then receive a separate frozen walk-forward/full-stack test. We will not rescue Role-Order Remap V1, search blend weights, or create a Week-1 exception from its observed results.
