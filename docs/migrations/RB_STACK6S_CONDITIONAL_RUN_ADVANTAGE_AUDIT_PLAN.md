# RB STACK6S — Conditional Run-vs-Pass Advantage Forensic Audit

## Status

**FROZEN BEFORE 2025 SIGNAL SCORING.** This is a no-fit forensic qualification only. No production change, P3 recomposition, or predictive model is authorized.

## Why this exists

STACK6P showed designed-run calls contain roughly 90-92% of the remaining within-state rushing-tendency oracle headroom. STACK6Q showed a compact state-level recent-tendency model does not recover that headroom reliably. STACK6R then showed that perfect down/distance occupancy removes only about 4-5% of designed-run MAE; roughly 95-96% remains in the conditional decision to call a designed run versus pass once score state and down/distance are known.

Broad run/pass EPA, PROE, opponent rushing context, pressure, and matchup variables were already used in M94/M94C/M95-family work. STACK6S does **not** relabel those broad signals as new. Its only novel question is whether the **relative run-versus-pass advantage conditional on the exact score-state × down/distance decision cell** explains target-game deviations from a team's strict-prior conditional run tendency.

## Decision cells

Use the frozen STACK6R score states:

- `lead`: score differential > +3;
- `neutral`: -3 through +3;
- `trail`: < -3.

Use the frozen STACK6R down/distance contexts:

1. `FIRST_DOWN`
2. `SECOND_SHORT_MED`: second down, yards-to-go <=6
3. `SECOND_LONG`: second down, yards-to-go >=7
4. `LATE_SHORT`: third/fourth down, yards-to-go <=3
5. `LATE_LONG`: third/fourth down, yards-to-go >=4
6. `OTHER`: exhaustive identity bucket; report separately but include in the all-play identity.

No threshold search is permitted.

## Play-call semantics

Offensive play: `rush_attempt == 1 OR qb_dropback == 1`.

- designed run = `rush_attempt == 1 AND qb_scramble != 1 AND qb_kneel != 1`;
- pass-intent = every offensive play that is not a designed run.

This intentionally treats QB scrambles and sacks as outcomes of pass/dropback intent rather than designed-run calls. Kneels remain non-designed nuisance outcomes.

## Strict-prior baseline conditional run probability

Run two precommitted history schemes:

- `TEAM5_SHRUNK`
- `TEAM8_SHRUNK`

For every target team-week × state × context, estimate the team's designed-run probability from only its prior N games in that exact cell, shrunk to the strict-prior league cell designed-run rate with `24` pseudo decision opportunities.

For every target play:

`call_residual = actual_designed_run(0/1) - prior_conditional_designed_probability`

Target-game play/state/context/outcome is grading truth only.

## Conditional football advantage signals

Compute two predeclared, no-fit matchup edges from only prior games in the same state × context cell.

For each offense and opponent defense, separately shrink designed-run and pass-intent efficiency to the strict-prior league cell mean with `24` pseudo plays per branch.

### EPA advantage

`offense_epa_diff = offense_designed_run_epa - offense_pass_intent_epa`

`defense_epa_diff = defense_designed_run_epa_allowed - defense_pass_intent_epa_allowed`

`EPA_RUN_ADVANTAGE = offense_epa_diff + defense_epa_diff`

Positive means the run branch has been relatively more efficient than the pass branch for both the offense and this defensive matchup.

### Success-rate advantage

Use nflverse `success == 1` when available; if `success` is absent, the run is an integrity/source failure rather than silently substituting `epa > 0`.

`offense_success_diff = offense_run_success - offense_pass_success`

`defense_success_diff = defense_run_success_allowed - defense_pass_success_allowed`

`SUCCESS_RUN_ADVANTAGE = offense_success_diff + defense_success_diff`

Positive again means relative run advantage.

No combined learned score and no coefficient fitting are allowed.

## Frozen populations

Primary play population: all 2025 W6-18 offensive decision plays with a valid team/opponent and exact cell assignment.

Also report independently:

- W6-12
- W13-18
- target team-games inherited from `POOL_OVER_5`
- target team-games inherited from `POOL_UNDER_5`
- each score state and each down/distance context descriptively only.

P3 error bins are grading-only and never influence signal construction.

## Frozen signal tests

For each history scheme and each signal (`EPA_RUN_ADVANTAGE`, `SUCCESS_RUN_ADVANTAGE`), calculate on play-level decision opportunities:

1. Pearson correlation with `call_residual`;
2. top-quartile minus bottom-quartile mean `call_residual` spread, where quartile cutpoints are determined from the full W6-18 signal distribution;
3. the same correlation separately in W6-12 and W13-18.

A signal is **qualified** only if under **both TEAM5_SHRUNK and TEAM8_SHRUNK**:

- full W6-18 correlation >= `+0.03`;
- top-minus-bottom residual spread >= `+0.03` (three percentage points of designed-run probability);
- W6-12 correlation > `0`;
- W13-18 correlation > `0`.

These thresholds are frozen before 2025 output. EPA and success are qualified independently; one may pass without the other.

## Dispositions

- `CONDITIONAL_ADVANTAGE_SIGNAL_QUALIFIED` — at least one of EPA or success passes all frozen conditions under both history schemes. Authorize a separate compact predictive test using only the prequalified signal(s), not production.
- `CONDITIONAL_ADVANTAGE_SIGNAL_NOT_QUALIFIED` — neither signal passes; do not fit a run-advantage model from these signals.
- `STACK6S_INTEGRITY_FAILURE_DO_NOT_INTERPRET` — source/identity/temporal failure.

No gate may be waived.

## Integrity

Require:

- 2023-2025 regular-season PBP loaded;
- exact one-context assignment for every offensive play;
- `success` source present;
- 544/544 2025 team-games join to frozen STACK6H grading bins;
- 388 W6-18 team-games represented;
- strict-prior construction flag = 1;
- both advantage signals finite after shrinkage for >=99% of W6-18 decision plays;
- no fitted model;
- no feature, model-family, hyperparameter, threshold, or coefficient search;
- no sportsbook inputs;
- target-game PBP used only for decision outcome / residual grading.

## Decision boundary

If a conditional advantage signal qualifies, the next model must remain compact and hierarchical: team prior conditional tendency plus only the qualified run-vs-pass advantage signal(s), evaluated temporally before any P3 recomposition.

If neither signal qualifies, do not keep adding generic efficiency features. Move to a genuinely different source/mechanism for conditional play calling.