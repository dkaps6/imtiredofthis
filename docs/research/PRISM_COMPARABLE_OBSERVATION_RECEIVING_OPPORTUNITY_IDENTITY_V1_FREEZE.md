# PRISM Comparable-Observation Receiving Opportunity Identity V1 — Frozen Predictive Contract

Date: 2026-10-18

Status: **FROZEN BEFORE OUTCOME SCORING / NAMED-PLAYER ROLE IDENTITY ONLY**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_COMPARABLE_OBSERVATION_RECEIVING_OPPORTUNITY_IDENTITY_V1`

## 1. Core player-first question

PRISM's objective is the named player's projection.

The source program has now proved:
- positive-participation games can contain major same-player snap collapses;
- some collapses are injury-corroborated;
- many of those injury-corroborated partial-participation observations are
  admitted into current PlayerForm history as ordinary weekly evidence.

This V1 asks:

> For an affected named RB/WR/TE, is his next-game receiving-opportunity identity
> more faithful when an injury-corroborated partial-participation observation is
> prevented from updating ordinary current-season role history?

The object being predicted is the player's own next-game target share.

Position is not the candidate rule.

## 2. Distinction from closed trajectory V1

Do not rescue:

`PRISM_HEALTH_STATE_GATED_TARGET_SHARE_TRAJECTORY_V1_CLOSED`.

That test:
- operated on the separate Target Share Trajectory delta;
- neutralized the whole existing RECENT2 trajectory signal when a no-positive-
  participation row contaminated the window.

This V1:
- does not change the trajectory delta;
- does not use RECENT2;
- does not alter finite room allocation;
- changes only how the named player's **current-season PlayerForm target-share
  identity** is built.

No trajectory parameter is retuned.

## 3. Immutable source parents

### A. Participation-anomaly state

- run `39160554692`;
- artifact `12147549966`;
- name `prism-player-participation-anomaly-state-v1-source-39160554692`;
- digest
  `sha256:e7753e981f0ecaaf23289280830fd32b9d733402c02ab9aaa0116fc929450449`;
- disposition
  `PRISM_PLAYER_PARTICIPATION_ANOMALY_STATE_V1_SOURCE_READY`.

### B. Observation / participation universe

- run `39154145666`;
- artifact `12146625382`;
- name `prism-player-observation-health-state-v1-source-39154145666`;
- digest
  `sha256:d74f2d3c37a88aa639527aedf719e7e08001d464c722613b46291cda0f74cbeb`;
- disposition
  `PRISM_PLAYER_OBSERVATION_HEALTH_STATE_V1_SOURCE_READY`.

### C. PlayerForm implementation

- `scripts/player_form_v2.py`;
- blob SHA
  `9246e4daf19f3f01cf61de3ad036b07873fd9561`.

The experiment must rebuild historical PlayerForm identity using the exact
current formulas, not a new position model.

## 4. Historical provider data

Use:
`nflreadpy.load_player_stats(..., summary_level="week")`

Required seasons:
- 2023;
- 2024;
- 2025.

Historical evaluation:
- development season = 2024;
- independent confirmation season = 2025.

No 2026 outcomes.

## 5. Market / player scope

Market component:
**receiving opportunity / target share**.

Player families:
- RB;
- WR;
- TE.

QB excluded because target share is not the QB passing-opportunity market.

The same named-player state rule is used for every included player.

No position-specific coefficient or threshold.

## 6. Exact baseline identity

For each target player-game in 2024 or 2025:

Target player must:
- be in the certified positive-offensive-participation universe for that game;
- have family RB/WR/TE;
- have exact stable identity;
- be on a team with positive target attempts in the weekly-stat source.

### Comparable target-game gate

The primary role-fidelity population must not treat a target game that is itself
a major participation anomaly as an ordinary role outcome.

Using only the already-frozen, performance-free participation-state labels:
- exclude target games labeled
  `PARTICIPATION_COLLAPSE_INJURY_CORROBORATED`;
- exclude target games labeled
  `PARTICIPATION_COLLAPSE_UNRESOLVED`.

Target games labeled:
- `PARTICIPATION_NOT_COLLAPSED`;
- `PARTICIPATION_BASELINE_INSUFFICIENT`;

remain eligible.

Excluded abnormal target games must be counted and reported separately. Their
targets/yards may not be used to decide the state label.

This gate is frozen before candidate outcomes are scored.

Only information strictly before target week may enter the predictor.

### Prior-season prior

Build exact PlayerForm prior-season `tgt_share` using all exact PlayerForm-
admitted rows in season-1.

No anomaly correction is applied to the prior-season prior in V1.

### Current-season baseline

Using exact PlayerForm-admitted rows from the target season with week < target
week:

`baseline_current_tgt_share = sum(player targets) / sum(team_targets on the same admitted player rows)`

`baseline_current_games = count(unique admitted player weeks)`

Then reproduce exact PlayerForm blending:

`w_baseline = baseline_current_games / (baseline_current_games + 4)`

If both prior and current target share are finite:

`baseline_target_share = (1 - w_baseline) * prior_tgt_share + w_baseline * baseline_current_tgt_share`

If only one side is finite, use the finite side exactly as PlayerForm does.

## 7. Frozen candidate identity

The candidate differs only in the target player's current-season history.

A prior current-season PlayerForm row is excluded from candidate **role
identity** only if all are true:

1. exact same stable player identity;
2. exact same season/week/team anomaly row exists;
3. anomaly state is
   `PARTICIPATION_COLLAPSE_INJURY_CORROBORATED`;
4. its corroboration is legally available before the target:
   `corroboration_week < target_week`.

Rows with:
- `PARTICIPATION_COLLAPSE_UNRESOLVED`;
- insufficient anomaly baseline;
- ordinary/noncollapsed participation

remain unchanged.

No partial weighting.

No full-game extrapolation.

No performance-dependent exception.

Then:

`candidate_current_tgt_share = sum(targets on retained current rows) / sum(team_targets on retained current rows)`

`candidate_current_games = retained unique admitted player weeks`

`w_candidate = candidate_current_games / (candidate_current_games + 4)`

Use the exact same PlayerForm blend with the unchanged prior-season prior.

This V1 treats an injury-corroborated partial game as **not an ordinary role
update**.

It does not claim the game contained no useful football information for every
other component.

## 8. Target observation

For the target game:

`actual_target_share = player_targets / team_targets`

The target player comes from the positive-participation roster/snap universe.

If the weekly-stat source has no player target row:
- player_targets = 0.

Team targets come from the full team weekly-stat target total.

Target-game data are outcome only and may never enter baseline/candidate
construction.

## 9. Directly affected player-games

Primary evaluation includes only target player-games where:

`abs(candidate_target_share - baseline_target_share) > 1e-12`

This means the target player's legally available history actually changes under
the frozen state rule.

Clean/unaffected player-games are required to have candidate == baseline exactly.

## 10. Primary player-level scoring

For each directly affected player-game:

`player_game_improvement =
abs(baseline_target_share - actual_target_share)
-
abs(candidate_target_share - actual_target_share)`

Positive:
candidate is closer for that named player-game.

Negative:
candidate is worse for that named player-game.

For each named player:
- average the improvement across his directly affected target games.

The primary object is the distribution of **named-player improvements**.

Report:
- affected target games;
- affected named players;
- each player's target-game count;
- equal-player average improvement;
- median named-player improvement;
- improved-player share;
- tied-player share;
- worsened-player share;
- p10 / p25 / p75 / p90 named-player improvement;
- player-cluster bootstrap interval on equal-player improvement;
- target-game improvement distribution.

Do not select the result using a pooled position average.

## 11. Frozen development / confirmation

Development:
- 2024.

Independent confirmation:
- 2025.

The formula is identical.

No parameter changes after 2024.

## 12. Frozen qualification gates

The candidate qualifies only if all pass.

### Integrity
1. exact three parent authorities match;
2. PlayerForm blob SHA exact;
3. stable identity coverage >=99%;
4. target/predictor chronology violations = 0;
5. clean/unaffected candidate-baseline drift = 0 within 1e-12;
6. primary target games classified as major participation collapses = 0;
7. 2026 outcomes = 0;
8. sportsbook/OddsAPI = 0;
9. coefficients fit = 0;
10. thresholds searched = 0.

### Development — affected named players
11. affected target games > 0;
12. affected named players >= 20;
13. equal-player average improvement > 0;
14. median named-player improvement >= 0;
15. improved-player share > worsened-player share.

### Independent confirmation — affected named players
16. affected target games > 0;
17. affected named players >= 20;
18. equal-player average improvement > 0;
19. median named-player improvement >= 0;
20. improved-player share > worsened-player share;
21. player-cluster bootstrap 95% lower bound > 0.

Bootstrap:
- sample named players with replacement;
- preserve all target rows for sampled players;
- 5,000 replicates;
- seed `20261018`.

### Family guardrail only

For each RB/WR/TE family with >=10 affected named players in 2025:
- report equal-player improvement and improved/worsened shares.

Family summaries are **guardrails/diagnostics**, not the modeling objective and
do not create a different rule by position.

## 13. Anti-rescue

After scoring do not:
- change the participation-collapse thresholds;
- change the anomaly prior window;
- exclude unresolved collapses;
- partially weight contaminated rows;
- full-game extrapolate partial games;
- add a position coefficient;
- change PlayerForm's +4 prior pseudo-games;
- add recency weighting;
- add trajectory;
- add teammate/opponent variables;
- subset stars/WR1/lead backs;
- use sportsbook information;
- use 2026 outcomes.

A failure closes this exact comparable-observation identity action.

## 14. What qualification means

Qualification establishes:

> For affected named pass catchers, injury-corroborated partial-participation
> games should not update ordinary current receiving-role identity in the same
> way as comparable full-participation observations.

It does not directly authorize a production patch.

Next:
freeze a full PRISM integration that routes this corrected player identity into
the real finite opportunity projection while preserving teammate allocation.

## 15. What failure means

Failure means only:

> Excluding these rows from current-season PlayerForm target-share identity is
> not a reliable general solution.

It does not mean injury context is irrelevant.

The permanent anomaly-investigation rule remains binding.

## 16. Integrity at freeze

- target outcomes inspected for this candidate: **0**;
- candidate scores observed: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_COMPARABLE_OBSERVATION_RECEIVING_OPPORTUNITY_IDENTITY_V1_FROZEN`.


## 17. Pre-implementation comparable-target amendment

Before scorer implementation or candidate outcome inspection, the target-game
population was tightened so the evaluation itself obeys the permanent anomaly
rule.

A target player-game that is independently state-labeled as a major
participation collapse is not an ordinary role-fidelity observation and is
excluded from the primary qualification cohort.

This amendment uses only the already-frozen snap/injury state source. It does
not inspect target share, yards, projection error, or candidate performance.
