# QB Pass-Opportunity Rate Designed-Run Source Audit V1 — Frozen Plan

## Purpose

Audit whether the existing player-specific **designed QB run** information can be reconstructed as a strict-prior, historically complete input for the exact M89/M90 QB cohort and whether its current production lineage is genuinely orphaned from the upstream 57/43 pass/run split.

This is a source/architecture audit only. It does **not** inspect target-game pass-opportunity rate, target-game QB attempts, passing yards, WR opportunity residuals, or any model-performance outcome. It cannot fit a predictive model and cannot change production.

The architecture question is:

> The production stack already builds `designed_run_rate` from PBP and joins only prior weeks into `metrics_v2`, while `project_game_script()` fixes the team pass share at `0.57`. Is designed-QB-run burden a clean, strict-prior player-level football signal that has never been consumed by the upstream team pass/run opportunity split?

A source pass would authorize exactly one separately preregistered predictive test of this linkage. It would not establish that designed-run history is predictive.

## Parent lineage

### Current pass-rate bottleneck
- Play/rate branch: `research-qb-team-pass-opportunity-play-rate-decomp-v1`
- Result commit: `7dbbc93e42eae68e6032ec6cb2299d357be6cb20`
- Run: `34535405829`
- Job: `103065737323`
- Artifact: `10175200512`
- Digest: `sha256:f2a79b330c7cc6f24d6b83479806478e5534afd2bd09961d37304a2c247b6bb4`
- Disposition: `PASS_OPPORTUNITY_RATE_PRIMARY_DIAGNOSTIC`

### Most recent blocked source family
- Directional personnel branch: `research-qb-pass-rate-directional-personnel-source-audit-v1`
- Result commit: `29c97573c6665134c9e8e5566e2fcf11fa5b00d1`
- Run: `34537299384`
- Job: `103071783186`
- Artifact: `10175912101`
- Digest: `sha256:fb2c646b1ba446a2e33bb6c1f5f94a89794b82a2ce68308753ac4ca503164da3`
- Disposition: `SOURCE_INELIGIBLE_IDENTITY_OR_USAGE_COVERAGE`

## Existing production source contract

The repository currently contains:

- `scripts/build/pbp_features.py::build_qb_run_metrics`
  - `designed_runs` = QB rush attempts excluding scrambles and kneels;
  - `designed_run_rate` = designed runs / (`dropbacks + qb_rushes`).
- `scripts/build/build_qb_run_metrics.py`
  - writes `data/qb_designed_runs.csv` and `data/qb_run_metrics.csv`.
- `scripts/metrics_v2.py`
  - reads `data/qb_run_metrics.csv`;
  - filters to `week < target_week`;
  - joins `designed_run_rate`, `designed_runs`, `scramble_rate`, `scrambles`, `dropbacks`, and `snaps` to the QB player row.
- `scripts/artifact_contracts.py`
  - recognizes `qb_run_metrics` and requires `designed_run_rate` when present.
- `scripts/modeling/rules_v2.py::project_game_script`
  - sets `pass_share = 0.57` regardless of player-specific designed-run burden.

Repo-wide code search at freeze time found no downstream use of `designed_run_rate` beyond source construction, metrics join, and artifact declaration.

## Anti-reinvention boundary

This family must not become a disguised retest of:

- generic team pass-rate history;
- PROE / score-state / lead-trail tendency;
- M40-M42 dynamic game-script formulations;
- M64-M65 state-occupancy/dropback-rate modeling;
- generic QB rushing or scramble effects on passing efficiency;
- M30 rushing-allocation calibration;
- another generic residual model.

The only potentially new linkage is **cross-layer conservation**: player-specific designed QB rushing entitlement may need to influence the upstream pass-vs-run opportunity split rather than existing only as a downstream QB rushing descriptor.

Scramble rate is not part of the candidate family because scrambles originate from called dropbacks and therefore belong on the pass-opportunity side of the play-choice split. The source audit may retain scramble counts only to verify the designed-run definition.

## Frozen target cohort

Use the exact 884 M89 corrected QB-game keys:

- 2024: `444`
- 2025: `440`

Target identity: `(season, week, team, player_clean_key)`.

The target artifact may be opened with **key columns only**. Target-game attempts, passing yards, pass-opportunity rate, residuals, and outcomes are forbidden in this source audit.

## Historical source universe

### PBP
Use nflverse/nflfastR regular-season PBP for 2023, 2024, and 2025.

Required fields or semantically exact equivalents:
- `season`
- `week`
- `posteam`
- `qb_dropback`
- `rush_attempt`
- `qb_scramble`
- `qb_kneel`
- `passer_player_id`
- `rusher_player_id`

Designed QB runs must be identified as rushing attempts by a QB player where:
- `rush_attempt == 1`
- `qb_scramble == 0`
- `qb_kneel == 0`

### QB identity
Use weekly rosters for 2023-2025 to identify QB-position player IDs and to bridge target M89 player keys to stable player IDs.

No fuzzy player match may silently resolve a target QB. Normalized-name matching must be unique within `(season, week, team)` or be reported unresolved/ambiguous.

## Frozen prior-history summaries

For each target QB, construct only from games strictly before target `(season, week)` and allow prior-season history for Week 1.

Report:

1. most recent eligible game:
   - prior designed runs;
   - prior designed-run rate;
   - prior QB rushes;
   - prior dropbacks;
   - prior snaps as defined by the production builder.
2. recent-8 aggregate history:
   - games available;
   - sum designed runs;
   - sum QB rushes;
   - sum dropbacks;
   - sum snaps;
   - aggregate designed-run rate = sum designed runs / sum snaps.

These are source-feasibility values only. No target label may be joined.

## Architecture audit

Statically verify at the tested repository head:

1. `build_qb_run_metrics()` defines designed runs by excluding scrambles and kneels;
2. `metrics_v2` enforces `qb_run_metrics.week < target_week` before joining the feature;
3. `rules_v2.project_game_script()` fixes `pass_share = 0.57`;
4. no production module other than construction/join/contracts references `designed_run_rate`.

If another downstream consumer is found, the audit must list it and the orphaned-signal claim fails.

## Frozen outputs

- `qb_designed_run_source_inventory.csv`
- `qb_designed_run_target_identity_audit.csv`
- `qb_designed_run_prior_history_audit.csv`
- `qb_designed_run_architecture_audit.csv`
- `qb_designed_run_source_result.json`

No output may contain target-game pass-opportunity rate, actual attempts, passing yards, actual designed runs for the target game, WR residuals, sportsbook data, or parent outcome/residual labels.

## Frozen source eligibility gates

The family is `QB_DESIGNED_RUN_LINKAGE_SOURCE_ELIGIBLE` only if all pass:

1. exact 884 target keys and 444/440 season counts;
2. 2023-2025 regular-season PBP loads successfully;
3. all required PBP fields exist in every season;
4. target QB identity resolution >= `98%` overall;
5. target QB identity resolution >= `97%` in each 2024 and 2025;
6. >= `92%` of resolved target QB-games have at least one strictly-prior NFL game with designed-run source history;
7. >= `85%` have at least three strictly-prior games;
8. Week-1 prior-season construction is mechanically supported and never reads same-season Week 1 outcomes;
9. historical designed-run definition exactly matches the current production builder semantics;
10. current production join is verified strict-prior (`week < target_week`);
11. current upstream pass share is verified fixed `0.57`;
12. no downstream `designed_run_rate` consumer exists beyond source construction, metrics join, and artifact contract;
13. sportsbook inputs used = `0`;
14. target-game outcomes read = `0`;
15. model fitting = `0`;
16. production changes = `0`.

## Allowed dispositions

- `QB_DESIGNED_RUN_LINKAGE_SOURCE_ELIGIBLE`
- `QB_DESIGNED_RUN_LINKAGE_SOURCE_INELIGIBLE_IDENTITY_OR_HISTORY`
- `QB_DESIGNED_RUN_LINKAGE_SOURCE_INELIGIBLE_SCHEMA`
- `QB_DESIGNED_RUN_LINKAGE_NOT_ORPHANED_ALREADY_CONSUMED`
- `MECHANICAL_SOURCE_AUDIT_FAIL`

## Stopping rule

- Do not correlate designed-run history with target pass-opportunity rate in V1.
- Do not inspect QB attempt or passing-yard performance.
- Do not inspect WR residuals.
- Do not fit any predictive model.
- Do not change the history window or eligibility gates after results are visible.
- If the source passes, freeze one separate predictive experiment before opening the target relationship.
- If the source fails, preserve the failure and do not rescue it by treating no-history QBs as zero-designed-run QBs.
