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

## Wave 2 — merged and validated

Wave 2 removed obsolete executable entry points while preserving research evidence:

- removed the isolated `model/cli.py` stack plus old `model/features`, `model/ingest`, and `model/pricing` helpers;
- retained promoted `model/qb_pass_synthesis_v1.json`;
- removed obsolete `config.yaml` and stale Makefile;
- removed the redundant `2026-full-slate-smoke.yml` workflow;
- removed all active `.github/workflows/backtest-qb-*` workflows and the frozen-QB-frontier workflow;
- preserved QB migration documentation, backtest scripts/data, result lineage, and promoted production artifacts;
- retained RB/WR rushing/receiving research workflows for the next modeling phases;
- rewrote `README.md` around the actual 2026 production architecture.

Wave 2 merged in PR #508 at `17a0a2975e39f35a1c50dbcc5edcea99b6ffcfbc`. Canonical Full Slate Run #523 (`33345574794`) passed from clean `main`, proving the production path does not depend on the removed standalone model or historical QB workflow surfaces.

## Wave 3 — retired engine and legacy providers

Wave 3 removes another set of code that had no canonical production reachability and could create ambiguity about what supplies live 2026 data.

### Legacy engine package removed

The entire `engine/` package is removed. It had already been intentionally fail-closed and code search found no production imports. Full Slate is now not merely the preferred authority; the old engine implementation is absent from the active repository tree. The static 2026 readiness audit already treats an absent legacy engine as valid and continues to verify that `AGENTS.md` establishes Full Slate as sole production authority.

### Legacy provider implementations removed

The following provider files are removed:

- `scripts/providers/apisports_pull.py`
- `scripts/providers/gsis_pull.py`
- `scripts/providers/msf_pull.py`
- `scripts/providers/injuries.py`
- `scripts/providers/build_schedule.py`

Canonical production uses Ourlads and Sharp under `scripts/providers/`, schedule authority under `scripts/utils/build_team_week_map_v2.py`, and the v3 injury builder under `scripts/build/build_injuries_weekly.py`. These retired files were not used by Full Slate and conflict with the 2026 provider contract that forbids silently reactivating old API-Sports/GSIS/MySportsFeeds paths.

The empty one-byte `.gitmore` marker is also removed.

Repository hygiene tests now prevent the retired engine and provider paths from returning unnoticed.

## Remaining Phase A review queue

The remaining cleanup is narrower and should be handled by dependency tracing rather than broad deletion:

1. duplicate/legacy top-level scripts such as old PlayerForm/metrics/pricing/weather/game-line entry points;
2. unused legacy secret declarations in Full Slate — remove only after confirming no imported production dependency reads them;
3. legacy/experimental `scripts/models/**` and `scripts/model/rules_engine.py` — preserve anything required by RB/WR research or current tests;
4. standalone `scripts/fantasypoints_wr_cb_scraper.py` — currently not canonical Coverage v2, but retain until WR source/refinement work determines whether it is still useful;
5. `docs/production/2026_MAIN_BRANCH_AUDIT.md` — preserve it as the historical baseline and add a current closure/status record rather than rewriting history;
6. any alternate execution path discovered through import/reference tracing.

Do not delete `scripts/backtest/**`, `docs/migrations/**`, `data/backtests/**`, or RB/WR research evidence wholesale. Research lineage and production execution are separate concerns.

## Exit criteria for Phase A

Phase A completes only when there is one obvious production entry point, generated runtime files are not tracked, retired provider caches/implementations cannot silently feed production, frozen QB research no longer pollutes normal CI, stale alternate pipelines are gone, current docs describe the 2026 system, strict Repo CI passes, and canonical Full Slate passes from clean `main` after the final cleanup wave.

Only then do we resume RB, WR and dedicated TE refinement.
