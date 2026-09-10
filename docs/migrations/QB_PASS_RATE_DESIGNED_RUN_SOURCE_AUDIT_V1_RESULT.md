# QB Pass-Opportunity Rate Designed-Run Source Audit V1 — Result

## Disposition

`QB_DESIGNED_RUN_LINKAGE_SOURCE_ELIGIBLE`

The designed-QB-run linkage source and architecture contract clears every frozen V1 gate. This is source eligibility only; it does not establish predictive value and does not change production.

## Canonical lineage

- Branch: `research-qb-pass-rate-designed-run-source-audit-v1`
- Frozen plan: `f9ad9dae8226c614609f504232e3c57995b9ff0a`
- Auditor commit: `5434607839d7086219a2579def936164723959d9`
- Tested/workflow head: `5044cc4be6e3b0eb0ddc209bee4ad151cfd82ec9`
- Run: `34537717639`
- Job: `103073096550`
- Artifact: `10176060151` (`qb-pass-rate-designed-run-source-v1`)
- Artifact digest: `sha256:1a3f5c908ce81731526904166fc334b6421229288dbaf09b29690d6dbaacf441`
- Parent play/rate result: `PASS_OPPORTUNITY_RATE_PRIMARY_DIAGNOSTIC`

## Frozen target cohort

- total M89/M90 target QB-games: `884`
- 2024: `444`
- 2025: `440`

Target-game outcomes were not read in this source audit.

## Identity and history coverage

- overall target-QB identity resolution: `99.0950%`
- 2024 identity resolution: `100.0000%`
- 2025 identity resolution: `98.1818%`
- resolved target QB-games with >=1 strict-prior game: `99.6575%`
- resolved target QB-games with >=3 strict-prior games: `97.6027%`
- Week-1 target rows with usable prior-season history: `50`
- Week-1 prior-season-only construction: PASS

All frozen identity/history thresholds passed.

## PBP source contract

Regular-season nflverse PBP loaded successfully for every required source season:

- 2023: `47,399` rows
- 2024: `47,274` rows
- 2025: `46,452` rows

All required fields were present in every season, including:

- `week`
- `posteam`
- `qb_dropback`
- `rush_attempt`
- `qb_scramble`
- `qb_kneel`
- `passer_player_id`
- `rusher_player_id`

The historical designed-run definition matches the current production builder: QB rush attempts excluding scrambles and kneels.

## Architecture result

Every frozen architecture check passed:

1. current builder excludes QB scrambles from designed runs: PASS
2. current builder excludes kneels: PASS
3. `metrics_v2` joins QB run metrics only from `week < target_week`: PASS
4. `rules_v2.project_game_script()` still fixes upstream `pass_share = 0.57`: PASS
5. artifact contract declares `designed_run_rate`: PASS
6. no unexpected production consumer of `designed_run_rate` exists: PASS

Therefore this is a genuine orphaned cross-layer football signal at the tested head: the stack constructs and carries player-specific designed-run tendency before kickoff, but the upstream team pass/run pool remains fixed at 57/43 and does not consume it.

## Leakage / production boundary

- sportsbook inputs: `0`
- target outcomes read: `0`
- model fitting: `0`
- production changes: `0`

## Scientific meaning

This result does **not** say that mobile/designed-run QBs should receive a lower pass-opportunity rate. It says the repository has a clean, high-coverage, strict-prior signal that is mechanically relevant to the pass-vs-run split and has not already been tested at that upstream layer.

That satisfies the M82 reopen requirement better than generic pass-rate history because the candidate is an architecture/conservation linkage, not a refit of prior team tendency features.

## Next authorized step

Freeze exactly one 2024-only development screen before opening the target relationship.

The screen must use a mechanistic, predeclared mapping from strict-prior designed QB runs into an adjustment to the 57% pass-opportunity-rate foundation. It may not run a model zoo, tune arbitrary coefficients, or inspect 2025 for candidate selection.
