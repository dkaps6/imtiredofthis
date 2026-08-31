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

## Wave 3 — merged and validated

Wave 3 removed code that had no canonical production reachability and could create ambiguity about live 2026 data:

- removed the entire retired `engine/` package;
- removed legacy live-provider implementations for API-Sports, GSIS, MySportsFeeds and the old injury path;
- retained only current Ourlads/Sharp providers plus `scripts/providers/build_schedule.py` as a historical backtest dependency;
- added hygiene guards preventing the historical schedule helper from entering Full Slate;
- removed the empty `.gitmore` marker.

An initial attempt to remove `scripts/providers/build_schedule.py` was rejected by CI because `scripts/backtest/historical_inputs.py` still imports it. The file was restored as research-only. This is why cleanup is accepted by dependency evidence rather than file age.

Wave 3 merged in PR #509 at `a9ea1d956ebd4e32c39bc9ccc7fae913c3bcfccb`. Repo CI Run #491 (`33346062432`) passed and canonical Full Slate Run #524 (`33346062433`) passed from clean `main`.

## Wave 4 — final legacy execution-surface cleanup

Wave 4 is the final narrow reachability pass before Phase A closes.

### Retired top-level compatibility/duplicate scripts

Dependency tracing found no canonical Full Slate or retained research dependency on the following stale alternate entry points, so Wave 4 removes them:

- `scripts/build_game_lines_from_schedule.py`
- `scripts/build_weather_week.py`
- `scripts/enrich_player_form.py`
- `scripts/enrich_team_form.py`
- `scripts/export_excel.py`
- `scripts/fetch_game_lines_oddsapi.py`
- `scripts/make_metrics.py`
- `scripts/make_player_form.py`
- `scripts/pricing.py`
- `scripts/run_all_builds.py`
- `scripts/validate_metrics.py`
- `scripts/volume.py`

Canonical v2/v3 replacements remain in the active production graph. `scripts/make_team_form.py` is **not** removed because the guarded production wrapper still imports it.

### Retired legacy model surfaces

The isolated `scripts/models/**` stack and `scripts/model/rules_engine.py` have no external production/research imports and are removed. Canonical modeling remains under `scripts/modeling/**`, `scripts/simulation_v2.py`, and the production bridge scripts.

### Retired provider credentials

Full Slate no longer declares unused credentials for API-Sports, ESPN-cookie, MySportsFeeds or GSIS paths. The active `ODDS_API_KEY` remains because the downstream sportsbook layer uses TheOddsAPI when explicitly enabled.

### Research surfaces intentionally retained

Wave 4 deliberately retains:

- `scripts/backtest/**`, `docs/migrations/**`, and `data/backtests/**`;
- RB/WR rushing and receiving workflows needed for the next research phase;
- `scripts/providers/build_schedule.py` as historical reconstruction support only;
- `scripts/fantasypoints_wr_cb_scraper.py` as WR research-only. It is explicitly barred from canonical Full Slate until its own season/week/archive contract is repaired and revalidated.

`tests/test_repository_hygiene.py` now prevents the retired Wave 4 surfaces and obsolete provider credential declarations from quietly returning.

The original `docs/production/2026_MAIN_BRANCH_AUDIT.md` remains unchanged as the historical baseline. `docs/production/2026_MAIN_BRANCH_AUDIT_CLOSURE.md` records current disposition and evidence.

## Wave 4 acceptance gate

Wave 4 is accepted only when:

1. Repo CI passes on the Wave 4 PR;
2. the PR is merged to `main`;
3. canonical no-credit Full Slate passes again from a clean `main` checkout.

If CI exposes a genuine historical/RB/WR dependency, restore only that required dependency and quarantine it from production rather than weakening the gate.

## Phase A exit criteria

Phase A completes only when there is one obvious production entry point, generated runtime files are not tracked, retired provider caches/implementations cannot silently feed production, frozen QB research no longer pollutes normal CI, stale alternate pipelines are gone, current docs describe the 2026 system, strict Repo CI passes, and canonical Full Slate passes from clean `main` after Wave 4.

After that final green run, Phase A closes and position-specific refinement resumes in order: RB, WR, then dedicated TE.