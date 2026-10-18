# PRISM Player Participation-Anomaly State V1 — Frozen Source Plan

Date: 2026-10-18

Status: **FROZEN SOURCE / STATE CONSTRUCTION ONLY / NO PERFORMANCE SCORING**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_PLAYER_PARTICIPATION_ANOMALY_STATE_V1_SOURCE_AUDIT`

## 1. Player-first question

PRISM projects named players.

A completed player-game is not automatically a valid role observation merely
because the box score contains positive usage.

The source-ready health-state audit proved that PRISM can distinguish:
- ordinary positive participation with no weekly injury report;
- positive participation while injury-reported;
- injury-reported no-snap observations;
- unresolved no-snap observations;
- return after reported nonparticipation.

It did **not** yet identify a positive-participation game in which the player's
participation itself collapsed relative to his own established role.

This V1 asks only:

> Can PRISM source-validly identify a major same-player participation collapse,
> and distinguish an injury-corroborated partial-participation candidate from an
> unresolved participation anomaly, before that game is allowed to become later
> player-role evidence?

No targets, yards, receptions, carries, touchdowns, fantasy points, projection
error or sportsbook information may define this state.

## 2. Why this is distinct from the closed trajectory integration

`PRISM_HEALTH_STATE_GATED_TARGET_SHARE_TRAJECTORY_V1` is closed exactly as frozen.

That experiment changed one existing trajectory delta:
- if RECENT2 contained a no-positive-participation observation, set the existing
  trajectory delta to zero.

Do not rescue that action by changing its window, coefficients or gates.

This V1 is upstream and different:
- it does not score or alter target-share trajectory;
- it does not use the trajectory RECENT2 window;
- it does not change any projection;
- it constructs a named-player observation-quality state for **positive
  participation games** using the player's own snap history.

## 3. Immutable source parent

Use only the exact certified observation-state authority:

- run `39154145666` — SUCCESS;
- artifact `12146625382`;
- artifact name
  `prism-player-observation-health-state-v1-source-39154145666`;
- digest
  `sha256:d74f2d3c37a88aa639527aedf719e7e08001d464c722613b46291cda0f74cbeb`;
- disposition
  `PRISM_PLAYER_OBSERVATION_HEALTH_STATE_V1_SOURCE_READY`;
- file
  `player_observation_health_state_rows.csv`.

Required fields:
- season;
- week;
- team;
- history_key;
- stable_id;
- family;
- positive_participation;
- offense_snaps;
- offense_pct;
- injury_reported;
- observation_state.

Seasons:
- 2024 REG;
- 2025 REG.

No provider rebuild in V1.

## 4. Named-player comparison grain

A participation baseline is built only from the **same named player on the same
team in the same season**.

For each positive-participation target observation:
- look only at strictly earlier scheduled player-game observations;
- keep only prior rows with positive offensive participation;
- use the most recent 4;
- require at least 3.

Do not use a position baseline to decide whether a named player's participation
collapsed.

Position/family may be reported only as source-coverage inventory.

## 5. Frozen player-specific participation baseline

For each supported target row:

`player_prior_snap_pct = median(last up to 4 prior positive-participation offense_pct)`

The current row is excluded.

All prior rows must have week < target observation week.

No future row may enter this baseline.

## 6. Frozen participation-collapse definition

For a supported positive-participation target row:

`absolute_snap_drop = player_prior_snap_pct - current_offense_pct`

`snap_retention_ratio = current_offense_pct / player_prior_snap_pct`

A **major participation collapse** is frozen as BOTH:

1. `snap_retention_ratio <= 0.65`
2. `absolute_snap_drop >= 0.20`

The thresholds are frozen before any performance outcome is loaded.

Rationale:
- the signal must be materially different from the player's own established
  participation;
- the relative condition protects naturally rotational players;
- the absolute condition prevents tiny percentage changes from being promoted
  into an anomaly.

No alternative thresholds are scored in V1.

## 7. Injury corroboration

A major participation collapse is not automatically called injury-caused.

Corroboration is descriptive and source-bound.

### Same-week corroboration

`current_injury_corroborated = injury_reported == true`

If true, the corroborating information is available in the current observation
week.

### Next-scheduled-game corroboration

For the same season + team + history_key, identify the next scheduled roster
observation after the target row.

`next_game_injury_corroborated = next observation injury_reported == true`

This may support retrospective interpretation of the earlier partial game only
for a **later projection whose target week is strictly after the corroborating
week**.

Persist:
- `corroboration_week`;
- `corroboration_source = CURRENT_WEEK_INJURY | NEXT_SCHEDULED_GAME_INJURY | NONE`.

The source audit itself does not apply the label to any future prediction.

## 8. Frozen state vocabulary

For positive-participation rows with sufficient baseline:

### `PARTICIPATION_NOT_COLLAPSED`
The frozen collapse conditions are not both met.

This does **not** mean healthy/full-strength.

### `PARTICIPATION_COLLAPSE_INJURY_CORROBORATED`
Both collapse conditions are met and either:
- current-week injury report corroborates; or
- the next scheduled team-game injury report corroborates.

This is an injury-corroborated partial-participation **candidate**.

It does not claim exact injury timing or severity.

### `PARTICIPATION_COLLAPSE_UNRESOLVED`
Both collapse conditions are met but no permitted injury corroboration exists.

This row must not automatically be called role decline.

### `PARTICIPATION_BASELINE_INSUFFICIENT`
Positive participation exists but fewer than 3 strictly-prior same-player,
same-team positive-participation rows are available.

Existing no-participation observation states remain unchanged and are not
reclassified by this V1.

## 9. Leakage / as-of rule

Next-scheduled-game injury corroboration is **not** information available at the
end of the earlier game.

Therefore V1 must persist when that corroboration became available.

A later integration may use the retrospective injury-corroborated label only
when:

`future_target_week > corroboration_week`

Otherwise the earlier observation remains unresolved for that target.

This as-of rule is mandatory.

## 10. Frozen source gates

Adjudicate 2024 and 2025 separately.

1. exact parent run/artifact/name/digest match;
2. positive-participation offense_pct finite coverage >=99%;
3. duplicate season/week/team/history_key rows = 0;
4. supported positive-participation rows with >=3 prior same-player/same-team
   positive games > 0;
5. all prior baseline rows strictly earlier than target row;
6. player_prior_snap_pct > 0 on every supported row;
7. snap_retention_ratio finite and >=0;
8. major participation collapse rows > 0;
9. injury-corroborated collapse rows > 0;
10. unresolved collapse rows may be zero or positive but must be reported;
11. next-game corroboration always comes from a strictly later scheduled week;
12. corroboration_week is never earlier than the observation week;
13. no 2026 rows;
14. no performance outcome field is used to construct state;
15. no fitted coefficients;
16. no threshold search;
17. no sportsbook/OddsAPI;
18. production changed = false.

Do not weaken gates after execution.

## 11. Required reporting

Report by season and family:
- positive-participation rows;
- baseline-supported rows;
- insufficient-baseline rows;
- participation-not-collapsed rows;
- injury-corroborated collapse rows;
- unresolved collapse rows;
- current-week corroboration rows;
- next-scheduled-game corroboration rows;
- median prior snap pct;
- median current snap pct among collapse rows;
- median retention ratio among collapse rows.

These are state/source summaries only.

No output-performance metric is permitted.

## 12. If source-ready

Source readiness authorizes a separate **PlayerForm history bridge** asking:

> Which exact current/prior PlayerForm observations are being treated as
> ordinary player-history evidence even though the now-certified player-specific
> participation state says they are injury-corroborated partial-participation
> candidates or unresolved anomalies?

That bridge must be source-only before any projection comparison.

The eventual model question is not “does a position average improve?”

It is:

> For affected named players, does a state-correct historical identity produce a
> more faithful next-game player projection than treating contaminated games as
> ordinary role evidence?

## 13. Integrity at freeze

- performance outcomes loaded: **0**;
- predictive relationships scored: **0**;
- projection errors scored: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- sportsbook/OddsAPI used: **0**;
- production changed: **false**.

Disposition at freeze:

`PRISM_PLAYER_PARTICIPATION_ANOMALY_STATE_V1_SOURCE_FROZEN`.
