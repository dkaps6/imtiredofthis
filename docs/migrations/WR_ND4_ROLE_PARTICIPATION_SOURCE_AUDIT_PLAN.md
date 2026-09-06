# WR-ND4 — Historical Role / Participation Source Audit

## Status

Frozen source-recovery audit. No target model is fit, no projection is changed, and no WR-ND3 gate is reopened by this branch.

## Lineage

Parent result: WR-ND3 canonical result commit `680a8ef6f67425b0eea682d4cc0c985c6c4d3970`.

WR-ND3 disposition was `NO_ACTIONABLE_DYNAMIC_ENTITLEMENT_SIGNAL`.

The strongest candidate, `RECENCY_ACCEL_2V8`, was directionally coherent but failed the frozen action gate. Simple teammate-absence and vacated-target heuristics were rejected. Therefore the next legitimate branch must add genuinely new information rather than tune windows or thresholds on the same target-history family.

## Research question

Can the historical stack recover timestamp-safe, pregame-usable offensive role / participation information with enough coverage to support a later WR target-entitlement experiment?

This branch answers **source availability and integrity only**. It does not ask whether any recovered source improves target projections.

## Candidate sources

### 1. PFR snap counts via nflverse

`nflreadpy.load_snap_counts()` provides game-level offensive snap counts and offensive snap percentages. The source is updated repeatedly during the season, so completed prior-game snap information is eligible for subsequent-week pregame use.

Allowed future features from this source may include only strictly prior-game state, such as:

- last-game offensive snap percentage,
- last-2 / last-4 prior-game mean offensive snap percentage,
- snap-share acceleration or sustained participation level.

Target-week snap counts are outcomes and are prohibited upstream.

### 2. nflverse depth charts

For 2025 onward, depth-chart snapshots contain an ISO8601 `dt` timestamp plus player GSIS ID and positional rank. A target-week snapshot is eligible only when its timestamp is strictly before kickoff.

To avoid timezone ambiguity in this source audit, the canonical audit uses a conservative rule:

`depth_snapshot_date < target_game_gameday`

This intentionally discards same-day pregame snapshots rather than risk leakage.

Allowed future features may include pregame WR position rank / depth slot and changes versus earlier snapshots, but only after this source passes the audit.

### 3. nflverse participation

Participation data contains play-level offensive player lists and could retrospectively reconstruct on-field participation. However, 2023+ participation is supplied after the postseason is complete. That makes the nflverse participation release itself **source-time unsafe for same-season 2025 walk-forward use**.

Therefore participation may be described in the audit, but it is not eligible to pass the canonical source gate and may not be used to create WR-ND5 features unless a contemporaneous historical equivalent is separately established.

## Canonical population

The audit must separately check out exact M38 commit:

`b98518d97b3038f471aee9ae3201009b2c70bb29`

It must rebuild the 2025 Weeks 1-18 historical inputs and exact M38 component predictions with 2024 as the prior season.

Parent integrity must reproduce:

- M38 receiving-yards rows: `4647`
- M38 receiving-yards MAE: `17.099904733366`

The source audit population is the exact 2025 Full-Slate M38 WR receiving-yard prediction population with a recorded actual target result, excluding only the known mathematically non-factorizable Isaiah Bond CLE W16 row for comparability with ND1-ND3.

Expected WR evaluation population: `2130`.

## ID policy

Preferred joins:

- canonical player GSIS ID -> nflverse player map -> PFR ID -> snap counts,
- canonical player GSIS ID -> depth-chart GSIS ID.

Name-only fuzzy matching is prohibited in the canonical gate. Unmapped IDs must remain missing and be reported.

## Frozen snap-count audit

For every target WR player-game:

- use only snap-count games with `(season, week)` strictly before the target row,
- prefer same-team prior games,
- derive `prior1_offense_pct`, `prior2_mean_offense_pct`, `prior4_mean_offense_pct`,
- record number of prior same-team snap games available.

Source passes `SNAP_SOURCE_RECOVERED` only if all are true:

- W2-18 prior1 coverage >= `0.85`,
- W2-18 prior2-mean coverage >= `0.80`,
- W13-18 prior1 coverage >= `0.90`,
- mapped snap rows with non-null offense percentage >= `0.98`,
- no target-week snap row is used as a feature.

Week 1 is reported but is not required to pass because offseason team changes make same-team prior-game coverage structurally different.

## Frozen depth-chart audit

For each target WR player-game:

1. determine the target team's game date from the historical schedule,
2. select the latest 2025 depth-chart snapshot whose `dt` calendar date is strictly earlier than the game date,
3. join by team + GSIS ID,
4. audit `pos_rank`, `pos_slot`, and position-group fields.

Source passes `DEPTH_SOURCE_RECOVERED` only if all are true:

- all-weeks strict-pregame GSIS depth-row coverage >= `0.85`,
- W2-18 coverage >= `0.90`,
- W13-18 coverage >= `0.90`,
- rows with usable positional rank among matched WRs >= `0.95`,
- zero selected snapshots occur on or after target gameday.

## Frozen disposition

- snap and depth both pass -> `SNAP_AND_DEPTH_SOURCES_RECOVERED`
- snap only passes -> `SNAP_SOURCE_RECOVERED`
- depth only passes -> `DEPTH_SOURCE_RECOVERED`
- neither passes -> `ROLE_PARTICIPATION_SOURCE_BLOCKED`

No source threshold may be relaxed after results are visible.

## Required outputs

- source schema inventory,
- canonical WR population audit,
- snap coverage by week and phase,
- depth coverage by week and phase,
- ID mapping audit,
- strict timestamp audit,
- source disposition JSON.

## What a pass authorizes

A passing source authorizes a **new frozen predictive diagnostic branch** using the newly recovered information family. It does not authorize production or automatic target-share adjustment.

If snap counts pass, the next branch may test participation state independently of ND3 target-history acceleration.

If depth charts pass, the next branch may test actual pregame depth-role state.

If both pass, the next branch should freeze a small factorial comparison that keeps the new sources separate before testing any combination.

## Anti-duplication

Do not:

- retune `RECENCY_ACCEL_2V8`,
- test neighboring target-history windows because ND3 nearly passed,
- resurrect simple vacated-target / absent-WR heuristics,
- use target-week snap counts,
- use depth snapshots on or after game day,
- use postseason-released 2025 participation as if it were available during the 2025 regular season,
- fuzzy-match missing player IDs into the canonical pass gate,
- fit a target model in WR-ND4,
- use sportsbook inputs upstream.

## Separate explosive / QB lane

WR explosive-tail research and the eventual QB upper-tail bridge remain separate. M72 already rejected aggregate explosive-weapon × defense proxies; a future QB bridge requires materially new validated player-level WR ceiling information.
