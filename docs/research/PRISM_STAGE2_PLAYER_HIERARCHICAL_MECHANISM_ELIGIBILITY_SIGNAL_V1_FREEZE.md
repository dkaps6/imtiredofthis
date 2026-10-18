# PRISM Stage-2 Player Hierarchical Mechanism Eligibility Signal V1 — Frozen Before Scoring

Date: 2026-10-18

Status: **FROZEN BEFORE FIRST SIGNAL SCORE / NO PROJECTION CHANGE**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_STAGE2_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_V1`

Scientific parent:
`PRISM_STAGE2_PLAYER_MECHANISM_BENEFIT_HISTORY_SOURCE_READY_ALL`

Parent source run:
- `39098772787` — SUCCESS;
- exact source head `744b2e920592803937e8a8c94e7a0c89d6ebd4eb`;
- artifact `12126215015`;
- digest `sha256:5249f41f72a248443c8a596de2a3cf4211928369b8867b3f3a4d16e70601c78d`.

Underlying Stage-2 parent remains:
- run `38957575645`;
- artifact `12070774378`;
- digest `sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0`.

## 1. Frozen question

For an already-qualified Stage-2 market/cell:

> Does this named player's own strictly-prior OOS history of Stage-2 mechanism
> benefit add stable information beyond a strictly-prior current-role prior
> about whether Stage-2 will help that named player on the next OOS row?

This is the first direct test of the new PRISM principle:

> **The player should have the rules. The position should not have the player.**

It tests a meta-signal about **mechanism utility**.

It does not change the underlying football mechanism.

## 2. Cells

Score separately:

- RB rushing;
- RB receiving;
- TE receiving.

Do not pool cells for qualification.

RB rushing is an important near-universal-control case because the parent
Stage-2 mechanism improved 86.78% of evaluable 2025 players.

RB receiving and TE receiving are the primary heterogeneity cases:
- RB receiving parent improved 47.11%;
- TE receiving parent improved 57.89%.

No cell may rescue another.

## 3. Strict chronology

Target seasons:
- 2024 = development;
- 2025 = independent confirmation.

For every target row, all predictor state must come from rows with:
- the same qualified cell;
- a strictly earlier NFL week in the same season, or an earlier season;
- never the same target week;
- never a future week.

Same-week earlier kickoff results are intentionally excluded.

Target row `delta_crps` may only be the scored outcome.

2026 is excluded entirely.

## 4. Frozen role/archetype prior

Role state:

`authority_rank_state in {ZERO, RANK1, RANK2, RANK3_PLUS}`

derived only from the existing canonical pregame `authority_q`.

For target row `t`:

`role_prior_mean_t`

is the arithmetic mean of strictly-prior Stage-2 `delta_crps` rows with:
- same cell;
- same `authority_rank_state`.

If no same-role prior rows exist:
- fallback to the strictly-prior same-cell mean;
- if no same-cell prior rows exist, fallback to 0.

No outcome-selected archetype or clustering is allowed.

## 5. Frozen individual evidence

For target row `t`:

`player_prior_mean_t`

is the arithmetic mean of all strictly-prior same-player + same-cell
`delta_crps` rows.

Minimum support for individual deviation:

`n_player_prior >= 3`.

If fewer than 3:
- individual evidence receives zero authority;
- the hierarchical score equals the role prior.

No alternate support floor may be tried after scoring.

## 6. Frozen shrinkage

Exactly three role-prior pseudo-observations are used.

Reason:
- three is the already-frozen minimum player-history support floor;
- it gives the broader role prior equal authority to the minimum admissible
  amount of player evidence;
- it is fixed before scoring and is not fitted.

For `n >= 3`:

`hierarchical_expected_benefit = (n * player_prior_mean + 3 * role_prior_mean) / (n + 3)`.

For `n < 3`:

`hierarchical_expected_benefit = role_prior_mean`.

No alternate shrinkage constant, nonlinear transform, window, recency weighting,
or per-cell coefficient may be tried in V1.

## 7. Frozen target

Target:

`target_stage2_benefit = delta_crps = baseline_crps - candidate_crps`.

Positive:
Stage-2 helped this named player's finite-count CRPS on the target OOS row.

Negative:
Stage-2 hurt.

Zero:
tie.

This is not a sportsbook outcome and is not a new football target. It is the
already-completed OOS utility of the qualified Stage-2 mechanism.

## 8. Frozen baseline vs candidate

Baseline meta-prediction:

`role_prior_mean`.

Candidate meta-prediction:

`hierarchical_expected_benefit`.

Primary row score:

`meta_AE = abs(target_stage2_benefit - predicted_benefit)`.

Primary improvement:

`delta_meta_AE = baseline_meta_AE - candidate_meta_AE`.

Positive favors player-specific hierarchical evidence.

Secondary diagnostics:
- RMSE of benefit prediction;
- Spearman rank association with target benefit;
- sign accuracy for nonzero target rows;
- mean target benefit for rows where hierarchical expected benefit is >0 vs <=0.

No threshold other than the natural zero-benefit sign boundary is evaluated.

## 9. Support gates

Per cell / season require:

2024 development:
- >=900 evaluable target rows with >=3 same-player prior observations for RB;
- >=900 for TE;
- >=100 distinct evaluable players for RB;
- >=95 for TE;
- >=28 teams.

2025 confirmation:
- >=1,200 evaluable target rows for RB;
- >=1,100 for TE;
- >=120 distinct evaluable players for RB;
- >=110 for TE;
- >=28 teams.

Chronology violations = 0.
Duplicate player/cell/week identities = 0.

Fail closed if a support gate misses.

## 10. Player-aware inference

For `delta_meta_AE` use frozen 5,000-replicate bootstraps, seed `20261018`:

- player-cluster;
- game-cluster;
- equal-player.

Equal-player frame requires >=5 evaluated target rows for that named player in
the season.

Report:
- mean equal-player improvement;
- median equal-player improvement;
- share of evaluable named players with mean improvement >0;
- p10 / p25 / p50 / p75 / p90 named-player improvement;
- worst named-player mean change for disclosure.

## 11. Frozen qualification gates

A cell qualifies only if **all** pass.

### 2024 development

1. support gates pass;
2. mean `delta_meta_AE > 0`;
3. candidate benefit-prediction RMSE < role-prior baseline RMSE;
4. candidate Spearman with target benefit > role-prior Spearman.

### 2025 independent confirmation

5. support gates pass;
6. mean `delta_meta_AE > 0`;
7. player-cluster 95% CI lower bound > 0;
8. game-cluster 95% CI lower bound > 0;
9. equal-player mean improvement > 0;
10. equal-player 95% CI lower bound > 0;
11. median named-player improvement >= 0;
12. share of evaluable named players improved > 0.50;
13. candidate RMSE < role-prior baseline RMSE;
14. candidate Spearman > role-prior Spearman;
15. tail-harm guard:
    `p10_named_player_improvement >= -0.25 * baseline_equal_player_meta_AE`.

The tail guard is scale-relative: the lower decile may not give back more than
one quarter of the role-prior baseline's equal-player absolute-error scale.

No pooled result can override a failed cell gate.

## 12. If qualified

Qualification authorizes only the next research step:

freeze a **conservation-safe personalized Stage-2 authority bridge**.

The likely bridge family may alter the effective authority of a player's
Stage-1 active-role uncertainty while preserving:
- exact finite opportunity;
- teammate interaction;
- canonical entitlement;
- active-state conservation.

No projection change is authorized by this signal result alone.

## 13. If closed

Close that cell's V1 mechanism-benefit persistence signal.

Do not rescue with:
- alternate support floors;
- 6/10-game-only subsets;
- alternate pseudo-counts;
- recency windows;
- exponential weighting;
- player-name carveouts;
- rank1-only subsets;
- positive-benefit-only history;
- alternate target loss;
- matchup or sportsbook variables;
- 2026 outcomes.

A future attempt would need a materially distinct player-specific mechanism
family, not a parameter rescue.

## 14. Integrity at freeze

At this freeze:
- signal outcomes scored: **0**;
- personalization projection rules applied: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- Week-6 outcomes used: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_STAGE2_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_V1_FROZEN_BEFORE_SCORING`.
