# 2026 Repository Cleanup / Production Reachability Audit

**Audit branch:** `phase-a-repo-cleanup-audit`  
**Production baseline:** `main` at `a58af2e497a4da63724a6b58280800f4fd0f6ed8`  
**Canonical production authority:** `.github/workflows/full-slate.yml`

## Purpose

Reduce the repository to a clear production core plus intentionally retained research evidence before RB/WR/TE refinement resumes. Historical age alone is **not** a deletion criterion. A file is production-relevant only when it is:

1. invoked directly by canonical Full Slate;
2. imported by a production dependency;
3. a required static/frozen production artifact; or
4. required by a currently retained test/audit protecting production behavior.

Everything else must be classified as research, archive, generated runtime output, or dead/superseded code.

## Canonical production dependency contract

The strict repository audit currently protects the following production Python surface:

- `scripts/config.py`
- `scripts/runtime_context.py`
- `scripts/providers/ourlads_depth.py`
- `scripts/utils/build_team_week_map_v2.py`
- `scripts/utils/make_team_week_map.py`
- `scripts/fetch_props_oddsapi.py`
- `scripts/providers/sharpfootball_pull.py`
- `scripts/team_form_prior_bridge.py`
- `scripts/run_team_form_context.py`
- `scripts/make_team_form.py` (legacy builder still encapsulated by the canonical runtime wrapper)
- `scripts/run_qb_promoted_context.py`
- `scripts/build/build_weather_week_v2.py`
- `scripts/build/build_weather_week.py`
- `scripts/build/build_injuries_weekly.py`
- `scripts/run_coverage_v2.py`
- `scripts/build/build_coverage_v2.py`
- `scripts/build/pbp_features.py`
- `scripts/player_form_v2.py`
- `scripts/player_stats_loader_v2.py`
- `scripts/slate_universe_v2.py`
- `scripts/run_player_form_v2.py`
- `scripts/run_player_form_v2_loader.py`
- `scripts/enrich_player_scoring_v2.py`
- `scripts/modeling/context_bridge.py`
- `scripts/modeling/bayesian_v2.py`
- `scripts/modeling/ml_v2.py`
- `scripts/modeling/state_v2.py`
- `scripts/modeling/ensemble_v2.py`
- `scripts/modeling/rules_v2.py`
- `scripts/modeling/simulation_rules.py`
- `scripts/modeling/qb_pass_synthesis_v1.py`
- `scripts/run_model_context_bridge.py`
- `scripts/run_model_bayesian_bridge.py`
- `scripts/run_model_ml_bridge.py`
- `scripts/run_model_state_bridge.py`
- `scripts/run_model_ensemble_bridge.py`
- `scripts/run_model_rules_bridge.py`
- `scripts/metrics_v2.py`
- `scripts/metrics_enrichment_v2.py`
- `scripts/run_metrics_context.py`
- `scripts/metrics_ready.py`
- `scripts/pricing_v2.py`
- `scripts/simulation_v2.py`
- `scripts/run_pricing_v2.py`
- `scripts/validate_build_integrity.py`
- `scripts/artifact_contracts.py`

Protected production workflows:

- `.github/workflows/full-slate.yml`
- `.github/workflows/audit-only.yml`
- `.github/workflows/repo-ci.yml`

This is a conservative list: anything imported by these files remains subject to dependency review even if not named here.

## Static production artifacts that must remain committed

- `model/qb_pass_synthesis_v1.json` — promoted M89/M90 QB deployment fit.
- `data/model_ensemble_weights.csv` — frozen promoted QB pass-yards ensemble calibration.
- `data/stadiums.csv` — static stadium/weather metadata used by the weather path.
- configuration / source code / tests required by the production graph.

`data/model_ensemble_weights.csv` and `data/stadiums.csv` are explicitly unignored so a future cleanup cannot accidentally turn them into ephemeral runtime files.

## Wave 1 — safe removals

The first cleanup wave removes tracked files that are demonstrably runtime-shaped placeholders or retired-provider remnants, not source-of-truth inputs.

### Runtime `data/` placeholders

At the baseline, virtually every root `data/*.csv` was a 1-byte placeholder. Full Slate creates the active versions during each run. They are now removed from source control. Exceptions retained:

- `data/model_ensemble_weights.csv`
- `data/stadiums.csv`
- research material under `data/backtests/`

The stale tracked `data/opponent_map_from_props.csv` is also removed; live sportsbook mode must construct the active opponent map from the active slate rather than inherit a committed placeholder.

### `outputs/`

Tracked output placeholders are removed. Production outputs belong to GitHub Actions run artifacts and are regenerated, not committed.

### `logs/`

Tracked log placeholders are removed. Action logs are runtime evidence, not repository inputs.

### `external/`

The retired `external/api_sports`, `external/fanduel`, and `external/nflverse_bundle` content consisted only of 1-byte cache/output placeholders. Code search found no active repository references to these paths. They are removed and `external/` is ignored to prevent accidental revival as a hidden production input.

## Explicitly NOT deleted in Wave 1

### Historical research / migration evidence

`data/backtests/`, `docs/migrations/`, backtest scripts, and research tests are **not** deleted merely because they are old. They contain reproducibility/lineage information and are especially relevant while RB and WR research is about to resume.

### RB / WR trace workflows

Rushing and receiving trace workflows are retained until the RB/WR research restart determines which are still needed as baselines or regression harnesses.

### QB research workflows

Broad QB mean research is frozen after M90, so the many historical `backtest-qb-*` GitHub workflows are strong archive/removal candidates. They are intentionally deferred to Wave 2 so the underlying research scripts/results can first be separated from active GitHub Actions triggers. The goal is to stop obsolete workflows from appearing on every repository change without destroying research lineage.

### `engine/`

The old engine is retired and `engine/engine.py` fails closed, but the package is deferred to Wave 2 because current tests/audits may still assert that retirement behavior. We will remove the package only after tracing all imports/tests and replacing any retirement assertion with a production-authority test if needed.

## Wave 2 review queue

1. Retire `.github/workflows/2026-full-slate-smoke.yml` after cleanup validation; canonical Full Slate now has its own clean-main proof.
2. Remove/archive obsolete `backtest-qb-*` workflow YAML files while retaining authoritative QB migration docs/results.
3. Audit all scripts outside the production dependency contract for imports from production and tests.
4. Classify legacy provider/source implementations (API-Sports, GSIS, MySportsFeeds, obsolete ESPN wrappers) as delete vs research archive.
5. Audit `engine/`, `model/cli.py`, old `model/ingest`, old `model/pricing`, and any alternate entry points for accidental second-pipeline behavior.
6. Update `README.md`, which still documents the 2025-era pipeline and commands.
7. Update `docs/production/2026_MAIN_BRANCH_AUDIT.md` from baseline findings to a closed/current status ledger.
8. Run the no-credit canonical Full Slate after each destructive cleanup wave. A cleanup wave is not accepted until the same production path remains green.

## Exit criteria for Phase A

Phase A is complete only when:

- there is one obvious production entry point (`full-slate.yml`);
- generated runtime files are not tracked as source;
- retired provider caches/credentials/paths cannot silently feed production;
- historical research is clearly separated from production execution;
- obsolete QB workflows no longer pollute normal CI activity;
- README/production docs describe the 2026 system rather than the 2025 system;
- strict Repo CI passes;
- canonical Full Slate passes from a clean checkout after cleanup.

Only then should the repository move into RB, WR, and TE refinement.
