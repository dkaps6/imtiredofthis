# 2026 Repository Cleanup / Production Reachability Audit

**Canonical production authority:** `.github/workflows/full-slate.yml`

## Purpose

Reduce the repository to one clear 2026 production system plus intentionally retained research evidence before RB/WR/TE refinement resumes. File age is not a deletion criterion. A file is production-relevant only when it is invoked by Full Slate, imported by a production dependency, required as a frozen/static production artifact, or required by a current test/audit protecting production behavior.

Historical migration scripts/results can remain as research lineage without remaining active GitHub Actions entry points.

## Canonical production surface

The production path is the Full Slate dependency graph rooted at `.github/workflows/full-slate.yml`. Major protected components include runtime context, Ourlads, authoritative schedule, the live-odds gate, Sharp/TeamForm prior handling, promoted M89/M90 QB context, weather/injuries, Coverage v2, PlayerForm/Player Identity v3, Provider Readiness/Team Context v3, Bayesian/ML/State/rules/ensemble bridges, metrics/pricing v2, artifact contracts, and strict production audits.

Protected production workflows are:

- `.github/workflows/full-slate.yml`
- `.github/workflows/repo-ci.yml`
- `.github/workflows/audit-only.yml`

Production dependencies imported by those files remain protected even if not enumerated here.

## Static production artifacts

These must remain committed across cleanup:

- `model/qb_pass_synthesis_v1.json` — promoted M89/M90 QB deployment fit.
- `data/model_ensemble_weights.csv` — frozen promoted QB pass-yards ensemble calibration.
- `data/stadiums.csv` — static stadium/weather metadata.

Generated root `data/*.csv`, `outputs/**`, logs, and provider caches are runtime material unless explicitly allowlisted.

## Wave 1 — merged and validated

Wave 1 removed tracked runtime-shaped placeholders and retired provider remnants:

- root runtime `data/*.csv` placeholders, while retaining the ensemble calibration and stadium metadata;
- tracked `outputs/**` placeholders;
- tracked `logs/**` placeholders;
- retired `external/api_sports`, `external/fanduel`, and `external/nflverse_bundle` cache/output placeholders;
- stale committed sportsbook/opponent-map placeholders.

`.gitignore` and `tests/test_repository_hygiene.py` prevent those paths from quietly returning.

Wave 1 merged in PR #507 at `f3b4de379c537a81283ccd18626b8acff8762eed`. Canonical Full Slate Run #522 (`33345155252`) then passed from clean `main`, proving production does not depend on the removed placeholders.

## Wave 2 — execution-surface retirement

Wave 2 removes obsolete executable entry points while preserving research evidence.

### Removed standalone 2025-era model CLI

The isolated `model/cli.py` stack and its `model/features`, `model/ingest`, and `model/pricing` helpers had no active production references and represented an alternate pipeline. They are removed. The promoted `model/qb_pass_synthesis_v1.json` remains.

The obsolete `config.yaml` and stale Makefile tied to old output/file conventions are also removed.

### Removed redundant production smoke workflow

`.github/workflows/2026-full-slate-smoke.yml` is removed. It served the production migration and is now redundant because the actual canonical Full Slate runs from `main` and has already passed after production and cleanup merges.

### Removed frozen QB research from active GitHub Actions

Broad QB mean research is frozen after M90. All `.github/workflows/backtest-qb-*` files and `freeze-qb-frontier-canonical-v1.yml` are removed from the active Actions surface so ordinary repository changes no longer create dozens of irrelevant skipped QB migration runs.

This does **not** delete QB research history. The underlying migration documentation, research scripts, result lineage, and promoted artifacts remain available for audit/reproducibility.

### RB/WR research harnesses intentionally retained

Rushing and receiving trace/backtest workflows remain active because RB and WR refinement are next. Repository hygiene tests explicitly protect the canonical rushing and keyed rushing/receiving trace workflows from accidental deletion.

### Documentation corrected

`README.md` is rewritten around the actual 2026 Full Slate path, current artifact contracts, sportsbook separation, no-credit/live-odds behavior, and Week 1 operating procedure. It no longer instructs users to run the retired 2025 standalone model.

## Deferred Wave 3 review

The following require dependency tracing before deletion or edits:

1. `engine/` — retired/fail-closed, but current audits/tests deliberately assert retirement behavior.
2. legacy provider implementations such as API-Sports, GSIS, MySportsFeeds and redundant injury/schedule wrappers.
3. duplicate/legacy top-level scripts such as old PlayerForm/metrics/pricing/weather entry points.
4. unused legacy secret declarations still present in Full Slate; remove only after confirming no imported production dependency reads them.
5. `docs/production/2026_MAIN_BRANCH_AUDIT.md` — preserve the original audit as historical baseline but add a current closure/status record.
6. any remaining alternate entry point discovered by import/reference tracing.

Wave 3 must not delete `scripts/backtest/**`, `docs/migrations/**`, or RB/WR research evidence wholesale. Research lineage and production execution are separate concerns.

## Exit criteria for Phase A

Phase A completes only when there is one obvious production entry point, generated runtime files are not tracked, retired provider caches cannot silently feed production, frozen QB research no longer pollutes normal CI, stale alternate pipelines are gone, current docs describe the 2026 system, strict Repo CI passes, and canonical Full Slate passes from clean `main` after the final cleanup wave.

Only then do we resume RB, WR and dedicated TE refinement.
