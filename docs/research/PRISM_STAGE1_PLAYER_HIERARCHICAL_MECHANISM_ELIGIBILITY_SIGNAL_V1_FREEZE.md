# PRISM Stage-1 Player Hierarchical Mechanism Eligibility Signal V1 — Frozen Before Scoring

Date: 2026-10-18

Status: **FROZEN BEFORE FIRST SIGNAL SCORE / NO PARTICIPATION PROJECTION CHANGE**

Version:
`PRISM_STAGE1_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_V1`

Source authority:
- run `39099745136` — SUCCESS;
- exact head `a1ad78254926da3a6ccc135a6726cc05f5dcd8d4`;
- artifact `12125863397`;
- digest `sha256:1b81c160ac1f8421d5eed83bb6a6389b5d6872f5a39f468df01b3a98a53d0d38`.

Underlying Stage-1 parent:
- run `38907459996`;
- artifact `12049120604`;
- digest `sha256:41b1acd3f1b8d77abb53980db160ece32f8fc40351939d4827248b9caad0c64b`.

## 1. Frozen question

For each family separately:

> Does this named player's own strictly-prior OOS history of Stage-1 mechanism
> benefit add stable information beyond the broader family prior about whether
> Stage-1's player-identity participation rule will help him on the next OOS row?

This is a mechanism-utility meta-signal.

It does not change `p_candidate`.

## 2. Families

Score independently:
- QB;
- RB;
- WR;
- TE.

No pooling can rescue a failed family.

## 3. Strict chronology

Target seasons:
- 2024 development;
- 2025 independent confirmation.

All predictor history:
- earlier season, or same season with strictly smaller week;
- same-week outcomes excluded;
- target row excluded;
- 2026 excluded.

## 4. Frozen family prior

For target row `t`:

`family_prior_mean_t`

is the arithmetic mean of strictly-prior Stage-1 `delta_brier` rows:
- same family;
- **different player identity from the target player**.

If no other-player family rows exist, fallback = 0.

This is the broader population/role prior.

## 5. Frozen individual evidence

`player_prior_mean_t`

is the arithmetic mean of all strictly-prior same-player + same-family
`delta_brier`.

Minimum support for individual deviation:

`n_player_prior >= 3`.

If fewer than 3:
the meta-score equals the family prior.

No alternate support floor may be tried in V1.

## 6. Frozen shrinkage

Exactly three family-prior pseudo-observations.

For `n >= 3`:

`hierarchical_expected_benefit = (n * player_prior_mean + 3 * family_prior_mean) / (n + 3)`.

For `n < 3`:

`hierarchical_expected_benefit = family_prior_mean`.

No fitted coefficient.
No recency window.
No alternate pseudo-count.
No per-family tuning.

## 7. Frozen target

`target_stage1_benefit = delta_brier = (y-p_baseline)^2 - (y-p_candidate)^2`.

Positive:
Stage-1 player-identity participation helped.

Negative:
it hurt.

Zero:
tie.

## 8. Baseline vs candidate meta-prediction

Baseline:
`family_prior_mean`.

Candidate:
`hierarchical_expected_benefit`.

Primary:
absolute error predicting target Stage-1 benefit.

`delta_meta_AE = abs(target-family_prior) - abs(target-hierarchical_score)`.

Secondary:
- RMSE;
- Spearman;
- sign accuracy on nonzero target benefit;
- mean target benefit where hierarchical score >0 vs <=0.

The only sign boundary evaluated is natural zero benefit.

## 9. Primary evaluable population

Primary qualification rows require:

`n_player_prior >= 3`.

Low-support rows are fallback-only and are not used to claim individual signal
qualification.

## 10. Support floors

Use the already-frozen source-ready floors:

QB:
- 2024 >=550 rows / >=65 players / >=28 teams;
- 2025 >=700 / >=70 / >=28.

RB:
- 2024 >=900 / >=105 / >=28;
- 2025 >=1,100 / >=120 / >=28.

WR:
- 2024 >=1,300 / >=175 / >=28;
- 2025 >=1,700 / >=190 / >=28.

TE:
- 2024 >=850 / >=105 / >=28;
- 2025 >=1,100 / >=110 / >=28.

## 11. Player-aware inference

Frozen:
- 5,000 bootstrap replicates;
- seed `20261018`;
- player cluster;
- game cluster;
- equal-player.

Equal-player requires >=5 evaluated target rows in the season.

Report:
- mean / median player improvement;
- player improved share;
- p10/p25/p50/p75/p90;
- worst player mean change.

## 12. Frozen qualification gates

A family qualifies only if all pass.

### 2024 development
1. support;
2. mean `delta_meta_AE > 0`;
3. candidate RMSE < family-prior RMSE;
4. candidate Spearman > family-prior Spearman.

### 2025 confirmation
5. support;
6. mean `delta_meta_AE > 0`;
7. player-cluster CI lower bound >0;
8. game-cluster CI lower bound >0;
9. equal-player mean improvement >0;
10. equal-player CI lower bound >0;
11. median player improvement >=0;
12. named-player improved share >0.50;
13. candidate RMSE < family-prior RMSE;
14. candidate Spearman > family-prior Spearman;
15. tail guard:
`p10_player_improvement >= -0.25 * family_prior_equal_player_meta_AE`.

## 13. If qualified

Qualification authorizes only a separate Stage-1 projection-level authority
bridge.

The most direct later bridge candidate is:
- low support -> keep qualified Stage-1 player candidate;
- supported positive expected utility -> keep candidate;
- supported nonpositive expected utility -> revert to family-prior
  `p_baseline`.

That integration is **not** authorized until separately frozen.

## 14. If closed

Do not rescue with:
- support changes;
- alternate pseudo-counts;
- recency;
- player-name exclusions;
- family subgroups;
- alternate loss;
- threshold changes;
- sportsbook or 2026.

## 15. Integrity at freeze

- target utility rows scored: **0**;
- participation projections changed: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026/Week-6 outcomes: **0**;
- sportsbook/OddsAPI: **0**;
- production changed: **false**.

Disposition:
`PRISM_STAGE1_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_V1_FROZEN_BEFORE_SCORING`.
