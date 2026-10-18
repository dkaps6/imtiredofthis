# PRISM WR/TE Player Signed Component Bias V1 — Frozen Signal Before Scoring

Date: 2026-10-18

Status: **FROZEN BEFORE FIRST PREDICTIVE SCORE / NO COMPONENT MEAN CHANGE**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_V1_SIGNAL`

Source authority:
- source run `39100442923` — SUCCESS;
- exact source head `065be1eecde73c9a4cc5c46ad04b97e2063886e7`;
- source artifact `12126685270`;
- source digest `sha256:d29de06947a0b3aead1e20f05b66d5cdfacfa393d8b1c535808533ed3b0a0d7a`;
- source disposition `PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_SOURCE_READY_ALL`.

Immutable parent component source:
- run `39061606501`;
- artifact `12111278777`;
- digest `sha256:f64c6e7efde9deaf91b7bc19076693a5106cb632338c461acc4d2cfe2ae37ea9`.

## 1. Frozen question

For WR and TE, and for opportunity and efficiency separately:

> Does this named player's strictly-prior same-season signed component-error
> history add stable predictive information beyond a strictly-prior
> other-player position prior about the sign and magnitude of his next component
> error?

This is a player-specific **component-bias signal**.

It is not:
- a trust score;
- a volatility score;
- a mechanism-switch score;
- a total-error model;
- a projection correction.

No receiving-yard projection or component mean changes in this experiment.

## 2. Anti-retest boundary

This is materially distinct from CLOSED
`PRISM_PLAYER_MECHANISM_INSTABILITY_TRUST_V1`.

The closed instability lane predicted total-error fragility from:
- switch rate;
- dominant mechanism state;
- absolute component difficulty.

This V1 uses none of those fields.

Predictor:
- signed same-player prior component error.

Target:
- the same signed component error on the next target row.

Do not use:
- switch rate;
- dominant state;
- absolute-error volatility;
- total error as target;
- WR/TE reliability scores;
- sportsbook data.

## 3. Frozen cells

Score independently:
1. WR opportunity;
2. WR efficiency;
3. TE opportunity;
4. TE efficiency.

No pooling may rescue a failed cell.

TE catch-rate error remains excluded from V1.

## 4. Frozen historical windows

WR:
- development = 2023;
- independent confirmation = 2024.

TE:
- development = 2024;
- independent confirmation = 2025.

No 2026 outcomes.
No Week-6 outcomes.

## 5. Frozen target

For each cell, target is the source-defined current signed component error:

Opportunity:
`current_opportunity_error`.

Efficiency:
`current_efficiency_error`.

The source sign convention is preserved exactly. No sign inversion, clipping,
winsorization or transformation is permitted.

## 6. Frozen broader prior — target-player excluded

For each target row and component:

`position_prior_mean_t`

is the arithmetic mean of the same component's observed signed errors from:
- the same position;
- the same season;
- strictly earlier NFL weeks only;
- **different player identity from the target player**.

Same-week outcomes are never included.

If there are no prior other-player rows, fallback is exactly `0.0`.

No WR-rank, TE-role, team, opponent or archetype split is allowed in V1.
Those are not parallel source-certified across both families and adding them
after seeing results would create a rescue path.

## 7. Frozen individual evidence

Use the exact source-parent player state:

Opportunity:
`prior8_opportunity_error_mean`.

Efficiency:
`prior8_efficiency_error_mean`.

Support weight:
`prior_component_rows_available`.

The source contract already requires target eligibility at:
- minimum 4 strictly-prior same-player component rows;
- maximum 8 rows;
- same season;
- prior max week < target week.

No alternate support floor or recency window may be tried.

## 8. Frozen hierarchical shrinkage

Exactly four position-prior pseudo-observations.

For target support `n` in [4,8]:

`hierarchical_signed_bias = (n * player_prior8_signed_bias + 4 * position_prior_mean) / (n + 4)`.

Baseline meta-prediction:
`position_prior_mean`.

Candidate meta-prediction:
`hierarchical_signed_bias`.

No fitted coefficient.
No per-position pseudo-count.
No per-component pseudo-count.
No threshold search.

## 9. Primary scoring

For each target row:

`baseline_meta_AE = abs(target_error - position_prior_mean)`

`candidate_meta_AE = abs(target_error - hierarchical_signed_bias)`

`delta_meta_AE = baseline_meta_AE - candidate_meta_AE`.

Positive `delta_meta_AE` means the named-player hierarchical signed-bias signal
predicted the next component error more accurately.

Secondary diagnostics:
- baseline vs candidate RMSE;
- baseline vs candidate Spearman correlation with target error;
- sign accuracy on target rows with nonzero target error;
- mean signed residual of each meta-prediction.

## 10. Frozen support gates

Use the already-frozen source-ready gates.

WR 2023:
- >=700 target rows;
- >=120 distinct players;
- >=28 teams.

WR 2024:
- >=1,100 rows;
- >=135 players;
- >=28 teams.

TE 2024:
- >=575 rows;
- >=70 players;
- >=28 teams.

TE 2025:
- >=600 rows;
- >=75 players;
- >=28 teams.

Chronology violations must equal 0.
Maximum decomposition identity gap must remain <=1e-10.

## 11. Player-aware inference

Frozen:
- 5,000 bootstrap replicates;
- seed `20261018`;
- player-cluster bootstrap;
- team-week-cluster bootstrap;
- equal-player bootstrap.

The source does not certify an opponent/game key across both families, so
team-week is the frozen shared-environment cluster for V1 rather than inventing
a game mapping after the source audit.

Equal-player diagnostics require >=5 target rows for that named player in the
scored season/cell.

Report:
- equal-player mean and median `delta_meta_AE`;
- named-player improved share;
- p10/p25/p50/p75/p90 player improvement;
- worst named-player mean change.

## 12. Frozen qualification gates

Each of the four cells qualifies independently only if all gates pass.

### Development season
1. support passes;
2. mean `delta_meta_AE > 0`;
3. candidate RMSE < baseline RMSE;
4. candidate Spearman > baseline Spearman.

### Independent confirmation season
5. support passes;
6. mean `delta_meta_AE > 0`;
7. player-cluster 95% CI lower bound >0;
8. team-week-cluster 95% CI lower bound >0;
9. equal-player mean improvement >0;
10. equal-player 95% CI lower bound >0;
11. median named-player improvement >=0;
12. named-player improved share >0.50;
13. candidate RMSE < baseline RMSE;
14. candidate Spearman > baseline Spearman;
15. confirmation tail guard:
    `p10_player_improvement >= -0.25 * baseline_equal_player_meta_AE`.

Sign accuracy is diagnostic, not a separate qualification gate, because signed
continuous-error improvement is already governed by AE, RMSE, rank association
and player-aware inference.

## 13. If a cell qualifies

Qualification proves only that a named-player signed component bias is
predictable strictly prior.

It does **not** authorize changing the opportunity or efficiency component.

A later bridge must be separately frozen and must answer:
- whether to correct the component mean;
- correction direction under the parent sign convention;
- correction magnitude;
- interaction with finite opportunity conservation;
- player-level OOS harm gates.

Mechanism utility and projection action remain separate scientific objects.

## 14. If a cell closes

Do not rescue it with:
- alternate pseudo-counts;
- support floors;
- recency windows;
- player subsets;
- WR-rank or TE-role splits;
- team/opponent splits;
- sign-only thresholds;
- clipping/winsorization;
- nonlinear transforms;
- sportsbook/2026 outcomes.

A future hypothesis must be materially distinct, source-legal and frozen before
scoring.

## 15. Integrity at freeze

- predictive component-bias rows scored: **0**;
- component means changed: **0**;
- projection means changed: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- Week-6 outcomes used: **0**;
- sportsbook/OddsAPI: **0**;
- production changed: **false**.

Disposition:
`PRISM_WR_TE_PLAYER_SIGNED_COMPONENT_BIAS_V1_SIGNAL_FROZEN_BEFORE_SCORING`.
