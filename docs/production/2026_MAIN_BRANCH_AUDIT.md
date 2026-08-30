# 2026 Main-Branch Production Audit

**Audit target:** `main` at/after QB M89/M90 production promotion (`b1e57d36afc4e688d66fd97386295a4da1771a8e`)

**Purpose:** make the repository safe for the 2026 season before resuming RB/WR/TE model refinement. This is a production/data audit, not a new QB research migration.

## Executive disposition

The repository is **not yet 2026-production-ready**. It is also **not true that every part of the repo is still 2025-only**. The modern Full Slate stack is substantially season-parameterized, but it coexists with legacy 2025 entry points and contains several high-impact hidden dependencies that can silently make a 2026 run use stale or non-promoted information.

The correct overhaul is therefore:

1. designate one canonical production entry point;
2. remove/guard hidden 2025 assumptions inside its dependency graph;
3. promote validated research artifacts into production explicitly;
4. add source-season/freshness provenance and fail-closed checks;
5. smoke-test providers in 2026 context;
6. only then resume position-specific research.

## Canonical production path

`.github/workflows/full-slate.yml` is the production entry point going forward.

The current intended order is:

`runtime/schedule -> roster -> optional live odds -> TeamForm -> promoted QB context -> weather/injuries -> coverage -> optional current PBP -> PlayerForm -> context/Bayes/ML/State/rules/ensemble -> metrics -> pricing -> audit`

`engine/engine.py` is a legacy 2025-era orchestration path and must not be treated as a second production authority.

## Severity legend

- **P0 blocker** — can make a 2026 run materially wrong while appearing successful.
- **P1 high** — provider/data contract can become stale, ambiguous, or fail early-season.
- **P2 medium** — duplicate/legacy code, misleading docs, or weak provenance that raises maintenance risk.
- **P3 cleanup** — does not currently alter canonical production results.

## Findings

### P0-1 — TeamForm silently reloads 2025 PBP inside a season-parameterized 2026 run

`full-slate.yml` invokes `run_team_form_context.py`, which imports and executes `scripts/make_team_form.py`.

`make_team_form.py` accepts the active season and uses it for most of TeamForm, but late in `main()` it hard-assigns `season = 2025` before deriving and merging:

- `success_rate_off`
- `success_rate_def`
- `success_rate_diff`
- `explosive_play_rate_allowed`

These fields feed `TeamContext` and game-script/rule logic. A 2026 artifact can therefore be stamped as 2026 while containing 2025-derived context.

**Required fix:** remove the hard-coded assignment and derive from the explicit active/current-or-prior context contract. Add source-season provenance and a regression test.

### P0-2 — Fresh production checkout does not contain the calibrated MC/ML/State ensemble used by M89/M90

`run_model_ensemble_bridge.py` only fits calibrated weights when `data/backtests/component_predictions.csv` exists. A clean `main` checkout does not contain that file and also does not contain `data/model_ensemble_weights.csv`.

The bridge therefore reports and uses `mc_weight=1`, `ml_weight=0`, `state_weight=0` unless a weights artifact is supplied externally.

Consequences:

- ML v2 and State v2 train every Full Slate run but are effectively zero-weight in final fresh-checkout production pricing.
- The promoted M89/M90 QB residual synthesis was validated on a calibrated canonical ensemble base, not an MC-only base.
- Merely having the QB synthesis JSON in production is not sufficient to reproduce the validated architecture.

**Required fix:** promote the validated/frozen OOS ensemble calibration artifact into production, with immutable lineage and diagnostics. For QB pass yards, the production base must use a previously validated frozen calibration rather than inventing a new unvalidated blend. Other markets remain explicit MC-only until their own promoted calibration is documented.

### P0-3 — Existing strict repo audit misses imported production dependencies

`scripts/utils/audit_repo.py` checks a manually maintained `PRODUCTION_SCRIPTS` list. It includes `run_team_form_context.py` but not the imported `make_team_form.py`, which is why the hidden `season = 2025` escaped strict CI.

It also omits recently promoted production dependencies such as the M89/M90 QB adapter/context builder and the current v2 weather wrapper.

**Required fix:** expand the production dependency contract immediately, then move toward deriving/checking the dependency graph rather than trusting a stale hand-written list.

### P0-4 — Two production authorities are documented

`engine/engine.py` still defaults to 2025, calls older builders/pricing, and labels validation as 2025 completeness. `AGENTS.md` also describes this legacy engine as canonical and gives 2025 commands.

This makes it easy for a human or automation to run a completely different model than Full Slate.

**Required fix:** Full Slate becomes the only canonical production authority. Deprecate/redirect the legacy engine and update `AGENTS.md`/README documentation.

### P1-1 — Sharp Football is mandatory in TeamForm but does not prove requested-season provenance

`sharpfootball_pull.py` accepts `--season`, first requests `?season=<season>`, then falls back to the same URL without a season parameter. It validates table shape/required columns, but it does not prove that the returned content actually belongs to the requested season.

If the provider ignores the query or has not rolled its page to 2026, a 2026 run can consume stale 2025 values under 2026 filenames.

**Required fix:** add provider provenance/freshness checks. If requested-season identity cannot be proven, label the data as prior-season/unavailable instead of silently presenting it as current.

### P1-2 — TeamForm has too many overlapping sources and fallback semantics

`make_team_form.py` combines PBP, participation, Sharp, and optional static files (`espn_team_form.csv`, `msf_team_form.csv`, `apisports_team_form.csv`, `nflgsis_team_form.csv`). These fallback files do not carry a strong common source-season contract.

M89 then overlays corrected semantics through a separate promoted context builder.

**Required direction:** replace the layered/monkey-patched TeamForm path with a TeamForm/Context v3 builder that natively implements:

- explicit active season / prior season / target week;
- completed games strictly before target week;
- M89 xPass PROE semantics;
- M89 neutral within-drive pace;
- explicit hit+sack pressure proxy labeling;
- official attempts / (attempts + sacks + QB scrambles);
- per-field source season, source provider, and freshness state.

### P1-3 — FantasyPoints WR/CB standalone scraper contains 2025 calendar logic and seasonless archive keys

The standalone scraper computes NFL week from a hard-coded September 4, 2025 start date. Its archive de-duplicates by player/week without a season key, so 2026 week numbers can collide with 2025.

Coverage v2 currently supplies the authoritative runtime week in the Full Slate path, so the hard-coded helper is not the primary production week authority today. It is still unsafe and must be corrected before relying on the archive/live scraper for 2026.

### P1-4 — Week 1 / new-player / roster-transition behavior needs explicit 2026 validation

PlayerForm v2 is correctly parameterized and uses 2025 as the prior season for 2026, with current evidence only from weeks before the target week. Ourlads + schedule can create a no-odds slate.

The remaining readiness cases are football/data cases rather than year literals:

- rookies with no 2025 NFL history;
- veterans changing teams;
- QB starter competitions;
- IR/PUP/inactive players;
- preseason depth-chart churn;
- players missing from prior nflverse statistics.

These need dedicated roster/identity smoke tests before Week 1.

### P1-5 — Provider availability is not the same as provider readiness

The schedule, nflverse weekly player stats, injuries, weather, Ourlads and live Odds API paths are season-parameterized. Each still needs a 2026 provider smoke test that records:

- requested season/week;
- returned season/week;
- row/team/player counts;
- fetched-at timestamp;
- provider/source name;
- fallback state;
- whether the source is current, prior, unavailable, or stale.

### P2-1 — Duplicate provider implementations remain

Examples include multiple injury paths, weather wrappers, schedule helpers and legacy GSIS/API-Sports/MSF paths. Some are no longer called by Full Slate but remain documented/configured as though they are active.

**Required fix:** maintain an explicit active-provider registry. Retire or quarantine unused paths rather than letting them drift into accidental reuse.

### P2-2 — Runtime `data/` contains placeholders and historical/debug artifacts

The repo tracks numerous runtime-shaped files and old 2025 debug artifacts. Modern readers often validate file size/schema, but legacy readers can still treat existence as availability.

**Required direction:** distinguish committed model/config artifacts from ephemeral run data. Runtime artifacts should carry manifests/provenance or be generated fresh.

### P2-3 — Sportsbook contamination must remain explicitly separated by market

The promoted QB synthesis itself is football-only. Metrics enrichment can derive `team_wp` from game odds, and some simulation paths may consume it in fallback/script logic (notably TD behavior). Production overhaul should document exactly which markets are independent-football projections versus market-assisted overlays.

## What from the research migrations is actually production logic?

The goal is **not** to copy all migration files into production. Failed or diagnostic migrations remain research evidence only.

Current canonical winners represented in production include:

- joint Monte Carlo game environment and finite opportunity allocation;
- empirical-Bayesian player baselines;
- fixed pass/rush opportunity framework;
- gentle pressure adjustment;
- top-five rushing allocation pool;
- WR hierarchy target-share sharpening with preserved target mass;
- supervised ML v2 and first-order State v2 component models;
- M89 corrected QB stat/context semantics;
- M89/M90 football-only QB passing-yards residual synthesis.

A validated component is not fully integrated until its **production artifact/calibration and exact feature semantics** are also present. P0-2 is the current example: ML/State code exists, but the promoted ensemble calibration does not yet survive a fresh checkout.

## 2026 production overhaul plan

### Phase 0 — correctness blockers

1. Fix TeamForm 2025 PBP contamination.
2. Promote validated QB ensemble calibration so M89/M90 receives the confirmed base projection.
3. Expand strict production dependency audit.
4. Make Full Slate the sole documented production authority; quarantine legacy engine.
5. Add a cross-artifact season/provenance readiness report.

**Exit gate:** a clean checkout cannot complete a 2026 production dry run while silently using an unapproved 2025 source or MC-only QB base.

### Phase 1 — Team Context v3

Build a single leakage-safe current/prior team context layer and retire the overlapping TeamForm bridge/overlay behavior.

**Exit gate:** one row/team, explicit target week, explicit source season/provider per critical field, M89 semantics native.

### Phase 2 — provider readiness matrix

Smoke-test schedule, Ourlads, nflverse weekly stats, current PBP, injuries, weather, Sharp, coverage/WR-CB, and Odds API.

**Exit gate:** each provider is classified `current`, `prior`, `unavailable`, or `stale`; no silent fallback.

### Phase 3 — roster/player readiness

Validate rookies, traded players, depth charts, QB starter selection, IR/PUP/inactives and no-history priors.

### Phase 4 — research-to-production promotion ledger

For every promoted model/rule:

- migration/result lineage;
- production file/function;
- artifact/version;
- frozen parameters;
- source semantics;
- markets affected;
- whether sportsbook information is excluded or explicitly allowed.

Rejected migrations are recorded as **do not integrate**.

### Phase 5 — 2026 dry run and controlled live run

Run Full Slate with live odds disabled first. Verify every produced artifact and provenance manifest. Then run a controlled live-odds slate and validate final model/component traces.

Only after these gates are green should RB/WR/TE model refinement resume against the new production foundation.

## Current audit status

| Area | Status | Priority |
|---|---|---|
| Runtime season/week context | Mostly ready | P1 smoke |
| 2026 schedule map | Parameterized/strict | P1 smoke |
| Ourlads roster/depth | Parameterized live source | P1 smoke |
| PlayerForm prior/current split | Structurally ready | P1 roster cases |
| Injuries v2 | Structurally ready | P1 smoke |
| Weather v2 | Structurally ready | P1 smoke |
| TeamForm | **Blocked: hidden 2025 PBP + overlapping fallbacks** | **P0** |
| Ensemble promotion | **Blocked: fresh checkout is MC-only** | **P0** |
| M89/M90 QB synthesis code | Promoted, but base ensemble must be fixed | **P0** |
| Sharp current-season provenance | Not proven | P1 |
| FantasyPoints WR/CB standalone/archive | 2025/calendar defect | P1 |
| Legacy engine/docs | Conflicting production authority | P0/P2 |
| Research promotion ledger | Incomplete | P1 |

This document is the baseline audit. Findings are closed only by code + automated validation, not by changing this table alone.
