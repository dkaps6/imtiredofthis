# RB Carry-Mechanism Role/Context Concentration Audit

## Status

`PREREGISTERED / DIAGNOSTIC ONLY`

## Why this exists

The RB individual-mechanism decomposition mapped 86 qualifying 2025 RBs into:

- 35 carry-dominant
- 25 YPC-dominant
- 26 mixed

This established that current rushing-yard error is not one homogeneous RB problem. For carry-dominant players, the next question is whether those errors are concentrated in specific pregame role-transition / availability contexts already present in the timestamp-safe RB casebook.

This audit does not change the RB model and does not use sportsbook lines.

## Frozen input

Canonical RB individual-mechanism artifact from run `34065409969`, including:

- `rb_mechanism_casebook.csv`
- `rb_individual_mechanisms.csv`

The casebook inherits the timestamp-safe role-transition states from the prior RB audit.

## Primary cohort

Player-games belonging to players whose frozen individual profile is `CARRIES` dominant.

Do not reclassify players after seeing this audit.

## Frozen role/context states

Evaluate exactly these five pre-existing states:

1. `state_depth_vs_carry_order_mismatch`
2. `state_limited_prior_history`
3. `state_no_prior_same_team_game`
4. `state_rookie`
5. `state_injury_created_context`

No additional state search after results.

## Frozen metrics per state

For state=1 and state=0 within the carry-dominant cohort report:

- rows
- carry MAE
- mean absolute carry-component yards
- mean signed carry residual (`actual_att - pred_att`)
- rushing-yard MAE
- mean absolute total rushing-yard residual

Derive:

- carry-MAE ratio = state1 / state0
- carry-component-absolute ratio = state1 / state0
- rushing-yard-MAE ratio = state1 / state0
- absolute difference in mean carry residual

## Frozen disposition gate

Call `ROLE_CONTEXT_CONCENTRATES_CARRY_MECHANISM_ERROR` only if:

- carry-dominant cohort has at least `250` player-games;
- depth-vs-carry-order mismatch carry-component-absolute ratio >= `1.20`;
- depth-vs-carry-order mismatch carry-MAE ratio >= `1.15`;
- at least `3 of 5` frozen states have carry-component-absolute ratio >= `1.15`;
- at least `3 of 5` frozen states have rushing-yard-MAE ratio >= `1.10`.

Otherwise call `NO_STRONG_ROLE_CONTEXT_CONCENTRATION_FOR_CARRY_ERRORS`.

Thresholds may not be changed after results.

## Secondary YPC check

For the `YPC`-dominant cohort, report the same role/context state ratios descriptively only.

This cannot rescue or overturn the primary carry-context disposition. The role-transition states are not designed as efficiency/matchup features. If YPC-dominant errors remain large without strong role-state concentration, the next legitimate RB lane is a separate frozen efficiency/matchup audit using the existing pregame defensive rushing / blocking / player-efficiency research lineage rather than forcing a carry solution onto YPC players.

## Interpretation

A pass would support targeted research on the *carry opportunity mechanism* in transition contexts. It would not authorize hard-coded depth-chart multipliers or player-specific carry corrections.

A fail would mean the previously observed RB1 / depth-chart concern is not sufficient to explain the carry-dominant error family on its own, and we should not force that narrative into production.

## Production rule

`production_changed = false`

`sportsbook_inputs_used = false`

No production change is authorized by this diagnostic.