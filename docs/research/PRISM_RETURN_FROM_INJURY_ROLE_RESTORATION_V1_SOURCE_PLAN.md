# PRISM Return-from-Injury Role Restoration V1 — Frozen Source Plan

Date: 2026-10-18

Status: **FROZEN PLAYER-STATE SOURCE ONLY / NO PERFORMANCE SCORING**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_RETURN_FROM_INJURY_ROLE_RESTORATION_V1_SOURCE_AUDIT`

## 1. Player-first question

A named player's injury interruption and his later return are not ordinary
consecutive box-score observations.

Before PRISM uses a return game as current role evidence, it needs an explicit
state:

> Did this named player restore his own pre-interruption offensive
> participation, return still materially reduced, or not return yet?

This V1 constructs that state from participation and injury chronology only.

No targets, carries, receptions, yards, touchdowns, fantasy points, projection
errors or sportsbook information may define the state.

## 2. Immutable source parents

### Observation-health state

- run `39154145666`;
- artifact `12146625382`;
- name `prism-player-observation-health-state-v1-source-39154145666`;
- digest
  `sha256:d74f2d3c37a88aa639527aedf719e7e08001d464c722613b46291cda0f74cbeb`;
- file `player_observation_health_state_rows.csv`.

### Participation-anomaly state

- run `39160554692`;
- artifact `12147549966`;
- name `prism-player-participation-anomaly-state-v1-source-39160554692`;
- digest
  `sha256:e7753e981f0ecaaf23289280830fd32b9d733402c02ab9aaa0116fc929450449`;
- file `player_participation_anomaly_state_rows.csv`.

No provider rebuild in V1.

## 3. Scope

Seasons:
- 2024;
- 2025.

Families:
- QB;
- RB;
- WR;
- TE.

Identity:
- exact stable player ID;
- same team;
- same season.

Position/family is reporting context only.

## 4. Injury-interruption episode start

A player can start an episode through either source-valid path.

### A. Injury-corroborated positive-participation collapse

Exact anomaly state:

`PARTICIPATION_COLLAPSE_INJURY_CORROBORATED`.

### B. Injury-reported no-offensive-snap observation

Exact health state:

`REPORTED_INJURY_NO_OFFENSIVE_SNAPS`.

No no-report/no-snap row is automatically called injury-caused.

## 5. Pre-interruption named-player role baseline

For every candidate episode start:

- same player;
- same team;
- same season;
- positive offensive participation only;
- strictly before episode-start week;
- latest up to four;
- require at least three.

Frozen pre-interruption role:

`pre_episode_snap_pct =
median(latest up to 4 prior positive-participation offense_pct)`

No position prior.

No performance stat.

For a positive-participation injury-collapse start that already carries
`player_prior_snap_pct`, V1 must also report the difference from the
reconstructed baseline as an integrity cross-check.

## 6. Episode grouping

Once an episode starts:

- later no-participation observations belong to the same episode;
- later disruption rows before a positive return do not create a second
  overlapping episode;
- the episode's return observation is the **first subsequent scheduled
  same-team positive-offensive-participation row**.

After that first return row, the episode ends.

A later independent interruption can start a new episode.

## 7. Frozen return classification

For a return row:

`return_retention_ratio =
return_offense_pct / pre_episode_snap_pct`

`return_absolute_drop =
pre_episode_snap_pct - return_offense_pct`

Reuse the already-frozen major participation-collapse boundary unchanged.

### `RETURN_PARTICIPATION_STILL_REDUCED`

Both are true:
- `return_retention_ratio <= 0.65`;
- `return_absolute_drop >= 0.20`.

### `RETURN_PARTICIPATION_RESTORED`

The return row does **not** satisfy both major-collapse conditions relative to
the player's own pre-interruption baseline.

This means participation has returned outside the previously defined major
collapse region.

It does not mean:
- medically 100%;
- identical route role;
- identical target role;
- identical efficiency.

### `NO_RETURN_OBSERVED`

No later positive-participation observation exists in the same-season,
same-team source after the episode start.

No new threshold is introduced.

## 8. Required episode context

Persist:
- season;
- team;
- stable ID;
- family;
- episode start week;
- start type;
- pre-episode positive-row count;
- pre-episode snap baseline;
- intervening scheduled no-participation rows;
- intervening injury-reported no-participation rows;
- return week when present;
- return injury-report flag;
- return offense snaps;
- return offense_pct;
- return retention ratio;
- return absolute drop;
- return state.

No performance outcome field is required.

## 9. Pre-score source support observed

Using only identity, chronology, participation and injury state:

2024:
- supported episodes: **309**;
- named players: **240**;
- restored returns: **162 episodes / 146 players**;
- still-reduced returns: **46 / 42**;
- no return observed: **101 / 100**.

2025:
- supported episodes: **342**;
- named players: **257**;
- restored returns: **187 episodes / 154 players**;
- still-reduced returns: **44 / 38**;
- no return observed: **111 / 111**.

No target/carry/yard relationship was inspected to obtain these counts.

## 10. Frozen source gates

Per season:

1. exact parent run/artifact/name/digest matches;
2. stable-ID coverage >=99%;
3. duplicate season/week/team/stable_id health identities = 0;
4. supported episodes >=200;
5. supported named players >=150;
6. restored-return episodes >=100;
7. restored-return named players >=100;
8. still-reduced return episodes >=25;
9. no-return episodes >0;
10. every episode baseline has >=3 strictly-prior positive rows;
11. every baseline snap pct finite and >0;
12. every return week > episode start week;
13. every return offense_pct finite and >0;
14. every return retention ratio finite and >=0;
15. overlapping episode identities = 0;
16. reconstructed anomaly baseline cross-check max absolute gap <=1e-10 where
    the frozen parent fields are semantically identical;
17. targets/carries/receptions/yards/TDs not used;
18. predictive relationships scored = 0;
19. coefficients fit = 0;
20. thresholds searched = 0;
21. 2026 outcomes = 0;
22. sportsbook/OddsAPI = 0;
23. production changed = false.

If the anomaly-parent baseline and the reconstructed baseline differ because
the exact source row histories have different eligibility semantics, fail the
cross-check rather than silently redefining either source.

## 11. Required reporting

By season and family:
- supported episodes;
- distinct players;
- restored returns;
- still-reduced returns;
- no-return episodes;
- median pre-episode snap pct;
- median restored-return snap pct;
- median still-reduced return snap pct;
- median no-participation rows between interruption and return;
- share of returns still injury-reported.

These are state/source summaries only.

## 12. If source-ready

The next experiment must remain player-first.

A legal next question is:

> Once this named player has a source-certified
> `RETURN_PARTICIPATION_RESTORED` observation, should PRISM's current-role
> interpretation treat the injury interruption as evidence of continuing
> decline, or treat the restored return as the new current participation state?

That predictive contract must be frozen separately.

It must not optimize a position-wide statistic.

## 13. Integrity at freeze

- target/carry/yard performance relationships scored: **0**;
- projections generated: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_RETURN_FROM_INJURY_ROLE_RESTORATION_V1_SOURCE_FROZEN`.
