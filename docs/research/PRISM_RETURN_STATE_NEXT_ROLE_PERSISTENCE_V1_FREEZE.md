# PRISM Return-State Next-Role Persistence V1 — Frozen Predictive Contract

Date: 2026-10-18

Status: **FROZEN BEFORE PREDICTIVE SCORING / NAMED-PLAYER CURRENT-ROLE STATE ONLY**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_RETURN_STATE_NEXT_ROLE_PERSISTENCE_V1`

## 1. Player-first question

PRISM projects the named player in his actual current football state.

Return-from-Injury Role Restoration V1 is source-ready and can certify that a
named player's first positive-participation observation after an
injury-corroborated interruption is:

`RETURN_PARTICIPATION_RESTORED`

relative to that same player's own strictly-prior pre-injury participation
role.

This V1 asks one predictive state question:

> Once this named player has source-certified
> `RETURN_PARTICIPATION_RESTORED`, does his immediately following clean
> same-team participation state behave more like his own pre-injury role than
> like a continued-interruption / continued-decline participation state?

This is a current-role persistence test.

It is not a target model, receiving model, rushing model, yards model,
touchdown model, position correction, or production integration.

## 2. Anti-retest declaration

This mechanism is distinct from and may not rescue:

- Comparable-Observation Receiving Opportunity Identity V1 — CLOSED;
- Injury-Partial Receiving Exposure Decomposition V1 — CLOSED;
- any deletion / partial weighting / reweighting of contaminated target-share
  history;
- the closed exposure x earning formula;
- raw recent-share authority replacement;
- WR/TE target-share trajectory retuning;
- RB carry/snap 50/50 retuning;
- active-set conditional allocation;
- blocked teammate-vacancy / role-elasticity V1;
- position-specific multipliers.

No target-share history is deleted, reweighted or blended here.

## 3. Immutable source authority

### Return-state source

- run `39162437467`;
- artifact `12149417234`;
- artifact name
  `prism-return-from-injury-role-restoration-v1-source-39162437467`;
- digest
  `sha256:7d69111925b7ad48f16b07150f5e5fbfac5e592bd01430d03efd907ca778c51e`;
- file
  `return_from_injury_role_restoration_episodes.csv`;
- disposition
  `PRISM_RETURN_FROM_INJURY_ROLE_RESTORATION_V1_SOURCE_READY`.

### Observation-health source

- run `39154145666`;
- artifact `12146625382`;
- artifact name
  `prism-player-observation-health-state-v1-source-39154145666`;
- digest
  `sha256:d74f2d3c37a88aa639527aedf719e7e08001d464c722613b46291cda0f74cbeb`;
- file
  `player_observation_health_state_rows.csv`;
- disposition
  `PRISM_PLAYER_OBSERVATION_HEALTH_STATE_V1_SOURCE_READY`.

No provider rebuild is permitted in the scorer.

## 4. Scope

Seasons:
- development: 2024;
- independent confirmation: 2025.

Families:
- QB;
- RB;
- WR;
- TE.

The named player is the primary unit.

Family/position is diagnostic context only.

No 2026 outcomes.

## 5. Eligible restored-state episode

An episode is eligible only when all are true:

1. exact stable player ID is nonblank;
2. family is QB/RB/WR/TE;
3. return state is exactly
   `RETURN_PARTICIPATION_RESTORED`;
4. return week is finite and strictly after episode-start week;
5. pre-episode snap baseline is finite and > 0;
6. same-player / same-team / same-season episode identity is unique;
7. the exact health source contains the episode-start row and return row.

No still-reduced or no-return episode is scored as restored.

## 6. Frozen same-player role anchors

### A. Pre-injury role anchor

Use the source-certified value already frozen upstream:

`pre_role_anchor = pre_episode_snap_pct`

This value was built only from the same named player's latest up to four,
minimum three, strictly-prior positive-participation observations before the
injury interruption.

No position prior.

### B. Continued-decline participation anchor

This anchor represents the interruption state, not performance.

If the episode start type is
`INJURY_CORROBORATED_PARTICIPATION_COLLAPSE`:

`decline_anchor = episode_start_offense_pct`

taken by exact identity from the immutable observation-health source.

If the episode start type is
`REPORTED_INJURY_NO_OFFENSIVE_SNAPS`:

`decline_anchor = 0.0`

because the source-certified participation state is literally no offensive
snaps.

This zero is a participation-state value only.

It must never be interpreted as zero player performance, zero ability, zero
target-earning skill, or a DNP box-score outcome.

No alternate decline anchor is tested in V1.

## 7. Frozen target observation

For each eligible restored episode, locate the **immediately next scheduled
same-team player observation after the return week** in the immutable health
source.

Chronology:

`episode_start_week < return_week < target_week`

The target observation is scoreable only if its exact observation state is:

`NO_REPORT_PARTICIPATED`

and its offensive snap percentage is finite and > 0.

If the immediate next observation is any of:

- `REPORTED_INJURY_PARTICIPATED`;
- `REPORTED_INJURY_NO_OFFENSIVE_SNAPS`;
- `NO_REPORT_NO_OFFENSIVE_SNAPS_UNRESOLVED`;

the episode fails closed for predictive scoring and is counted by exclusion
reason.

The scorer may not skip that observation and search farther ahead for a clean
game.

This preserves the rule:

> anomalies are questions, not trends.

## 8. Frozen predictive comparison

Outcome:

`actual_next_offense_pct = target offense_pct`

Primary episode-level gain:

`pre_role_closeness_gain =
abs(decline_anchor - actual_next_offense_pct)
-
abs(pre_role_anchor - actual_next_offense_pct)`

Interpretation:

- positive: the player's own pre-injury role anchor was closer than the
  continued-decline participation anchor;
- negative: the continued-decline participation anchor was closer;
- zero: tied.

This is the only qualification formula.

The return game's own snap percentage may be persisted as context but is not a
second candidate and may not create a rescue path.

No fitted coefficient.

No searched threshold.

No blend.

No position coefficient.

## 9. Named-player scoring

Aggregate within named player first.

For every named player:
- average all his eligible episode-level `pre_role_closeness_gain` values.

Primary per-season reporting:
- scored episodes;
- named players;
- equal-player average gain;
- median named-player gain;
- improved / tied / worsened named-player shares;
- p10 / p25 / p75 / p90 named-player gain;
- per-player episode counts.

For 2025 independent confirmation:
- named-player bootstrap;
- 5,000 replicates;
- seed `20261018`;
- preserve all scored episodes belonging to each resampled player.

Family summaries are diagnostics only.

No family result can rescue a failed frozen overall contract.

## 10. Frozen support gates

Per season:

1. scored episodes >= 80;
2. scored named players >= 70;
3. chronology violations = 0;
4. duplicate episode identities = 0;
5. every scored target is the immediate next observation after return;
6. every scored target state is exactly `NO_REPORT_PARTICIPATED`;
7. every scored target offense_pct is finite and > 0;
8. every pre-role anchor is finite and > 0;
9. every decline anchor is finite and >= 0;
10. 2026 outcomes used = 0;
11. sportsbook/OddsAPI inputs used = 0;
12. coefficients fit = 0;
13. thresholds searched = 0;
14. production changed = false.

Do not lower these gates after scoring.

## 11. Frozen qualification gates

### 2024 development

15. equal-player average gain > 0;
16. median named-player gain >= 0;
17. improved-player share > worsened-player share.

### 2025 independent confirmation

18. equal-player average gain > 0;
19. median named-player gain >= 0;
20. improved-player share > worsened-player share;
21. named-player bootstrap 95% lower bound > 0.

All support, development and confirmation gates must pass.

Qualification disposition:

`PRISM_RETURN_STATE_NEXT_ROLE_PERSISTENCE_V1_QUALIFIED_STATE_ONLY`

Failure disposition:

`PRISM_RETURN_STATE_NEXT_ROLE_PERSISTENCE_V1_CLOSED`

No partial family rescue.

## 12. Teammate competition / finite opportunity boundary

This V1 changes **no** target share, carry share, reception rate, efficiency,
yards, touchdown probability, or team opportunity total.

A qualified result would authorize only this state interpretation:

> after source-certified restored participation, the named player's
> participation prior should no longer automatically remain anchored to the
> interruption/decline state.

Any later target/carry opportunity integration must be frozen separately and
must still pass through the existing teammate-competition / finite-opportunity
architecture.

Participation restoration is separate from target earning.

Target earning is separate from efficiency.

No automatic yards multiplier is authorized.

## 13. What qualification means

Qualification would mean that, for source-clean immediately subsequent games,
the named player's own pre-injury participation role is a more faithful
current-role state anchor than carrying forward the interruption state after a
source-certified restored return.

It would **not** mean:
- the player is medically 100%;
- route share is restored;
- target share is restored;
- carry share is restored;
- efficiency is restored;
- yards are restored;
- touchdown access is restored.

Snaps are snaps, never routes.

## 14. What failure means

Failure closes this exact state-persistence rule.

Do not rescue it with:
- a different snap window;
- a different return threshold;
- a different clean-target definition;
- a second-game / third-game lookahead;
- position-specific rules;
- player-star subsets;
- a blend of pre-role and return snaps;
- target-share deletion/reweighting;
- exposure x earning arithmetic;
- sportsbook inputs;
- 2026 outcomes.

A failure does not permit treating injury/DNP rows as ordinary decline evidence.

## 15. Integrity at freeze

Before this freeze:
- future target observations scored: **0**;
- predictive relationship values inspected: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- paid OddsAPI calls: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_RETURN_STATE_NEXT_ROLE_PERSISTENCE_V1_FROZEN`.
