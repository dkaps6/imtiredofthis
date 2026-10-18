# PRISM WR Opportunity-Bias Topology Eligibility V1 — Frozen Before Scoring

Date: 2026-10-18

Status: **FROZEN BEFORE TOPOLOGY-SUBGROUP SCORE / NO PROJECTION ACTION**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_WR_OPPORTUNITY_BIAS_TOPOLOGY_ELIGIBILITY_V1`

## 1. Purpose

The already-frozen player signed component-bias signal qualified **WR
opportunity** overall.

WR-R15, however, has a real pregame allocation hierarchy:
- one frozen anchor identity per team-game;
- a secondary-WR pool whose mass is conserved;
- R15 redistributes only that secondary pool.

Before any conservation bridge is allowed to move the frozen anchor, determine
whether the named-player signed opportunity-bias signal independently earns
deployment eligibility inside:
1. the frozen R15 anchor topology;
2. the frozen R15 secondary topology.

This is a **deployment/mechanism-eligibility** test, not a new signal fit and
not a projection correction.

## 2. Authorities

Signed-bias signal:
- run `39109423266`;
- artifact `12128983909`;
- digest `sha256:bb14d6474778d0afd53ca74bb60237b381aa4e973195bab3d09d54ed87ef2490`;
- WR opportunity overall is qualified.

WR topology:
- run `36203881629`;
- artifact `10893281689`;
- digest `sha256:33b12fd521196838054f4ee4240d8cea8f66c2fce5b5bfc9eb902bc67d730adb`;
- source commit `02c3dd1a681d4ab2953683039e39830554f9ec9f`;
- exact frozen full WR rows = 6,409;
- exact frozen team-game anchors = 1,088;
- anchor identity mismatches = 0;
- secondary bridge coverage = 100%;
- strict-prior future violations = 0.

This topology test is authorized to score only if the conservation-bridge source
recovery audit is SOURCE READY. If recovery blocks, this test is blocked
without scoring.

## 3. Frozen topology classes

Join signal to topology exactly on:
- season;
- week;
- team;
- player_clean_key.

Classes are source-defined and immutable:

### ANCHOR
`authority_row_type == FROZEN_ANCHOR_IDENTITY_COMPLEMENT_BRIDGE`.

### SECONDARY
`authority_row_type == FROZEN_SECONDARY`.

No rank approximation, fuzzy mapping, or reclassification is permitted.

## 4. Frozen sample counts known before scoring

These counts are source/topology facts only, not performance results.

2023:
- ANCHOR: 246 rows / 45 distinct players;
- SECONDARY: 502 rows / 107 distinct players.

2024:
- ANCHOR: 353 rows / 56 distinct players;
- SECONDARY: 837 rows / 128 distinct players.

Both classes therefore receive the same frozen minimum support gate:
- >=200 rows;
- >=40 distinct players.

No support threshold may be changed after topology-subgroup results are seen.

## 5. Frozen historical windows

Development:
- 2023.

Independent confirmation:
- 2024.

No 2025 or 2026 outcome may enter this test.
No Week-6 2026 outcome may enter this test.

## 6. Frozen predictor and target

No new predictor is created.

For every row:

Baseline meta-prediction:
`position_prior_mean`.

Candidate meta-prediction:
`hierarchical_signed_bias`.

Target:
`target_component_error`.

These columns come unchanged from the already-qualified WR opportunity signal
artifact.

No coefficient, pseudo-count, threshold, recency window, sign flip, clipping,
or transformation may be introduced.

## 7. Frozen scoring

For each topology class and season:

`baseline_meta_AE = abs(target_component_error - position_prior_mean)`

`candidate_meta_AE = abs(target_component_error - hierarchical_signed_bias)`

`delta_meta_AE = baseline_meta_AE - candidate_meta_AE`.

Positive means the player-specific signal predicts the next signed opportunity
component error more accurately than the target-player-excluded position prior.

Secondary diagnostics:
- baseline/candidate RMSE;
- baseline/candidate Spearman;
- baseline/candidate sign accuracy;
- mean signed residual.

## 8. Frozen player-aware inference

Exactly inherit the already-frozen signed-bias inference contract:
- 5,000 bootstrap replicates;
- seed `20261018`;
- player-cluster bootstrap;
- team-week-cluster bootstrap;
- equal-player bootstrap;
- equal-player diagnostics only for named players with >=5 scored rows.

Report:
- equal-player mean / median improvement;
- named-player improved share;
- p10 / p25 / p50 / p75 / p90 player improvement;
- worst named-player mean improvement.

## 9. Frozen qualification gates

ANCHOR and SECONDARY qualify independently.

Each class qualifies only if all of the same scientific gates used by the
already-qualified WR opportunity signal pass.

### Development 2023
1. support >=200 rows and >=40 players;
2. mean `delta_meta_AE > 0`;
3. candidate RMSE < baseline RMSE;
4. candidate Spearman > baseline Spearman.

### Independent confirmation 2024
5. support >=200 rows and >=40 players;
6. mean `delta_meta_AE > 0`;
7. player-cluster 95% CI lower bound >0;
8. team-week-cluster 95% CI lower bound >0;
9. equal-player mean improvement >0;
10. equal-player 95% CI lower bound >0;
11. median named-player improvement >=0;
12. named-player improved share >0.50;
13. candidate RMSE < baseline RMSE;
14. candidate Spearman > baseline Spearman;
15. tail guard:
    `p10_player_improvement >= -0.25 * baseline_equal_player_meta_AE`.

No pooled WR result can rescue either topology class.

## 10. Frozen deployment interpretation

This experiment does not alter projections.

If ANCHOR qualifies:
- the player-specific opportunity-bias mechanism has earned permission to be
  considered in a later separately frozen anchor-moving bridge.

If ANCHOR closes:
- WR anchor remains protected;
- do not move it using this signed-bias mechanism.

If SECONDARY qualifies:
- the player-specific opportunity-bias mechanism has earned permission to be
  considered in a later separately frozen secondary-pool bridge.

If SECONDARY closes:
- do not apply named-player signed-bias factors inside the secondary pool.

If both qualify:
- a later bridge may consider a whole-WR-room conserved action, but that action
  still requires a separate pre-result freeze.

If only ANCHOR qualifies:
- any later bridge must alter anchor mass and redistribute the conservation
  offset across secondary players **without** using failed secondary
  player-specific factors.

If only SECONDARY qualifies:
- anchor stays fixed and only the exact secondary pool may be redistributed.

## 11. Anti-retest / no rescue

Do not rescue a failed topology class with:
- rank thresholds;
- WR1/WR2 labels;
- alternate support floors;
- alternate pseudo-counts;
- alternate prior windows;
- positive/negative-bias subsets;
- player star/non-star splits;
- team/opponent splits;
- sportsbook information;
- 2026 outcomes.

## 12. Integrity at freeze

- topology-subgroup performance rows scored: **0**;
- projection means changed: **0**;
- opportunity entitlements changed: **0**;
- conservation rooms changed: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 / Week-6 outcomes used: **0**;
- sportsbook / OddsAPI used: **0**;
- production changed: **false**.

Disposition:
`PRISM_WR_OPPORTUNITY_BIAS_TOPOLOGY_ELIGIBILITY_V1_FROZEN_BEFORE_SCORING`.
