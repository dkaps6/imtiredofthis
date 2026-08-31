# Sharp Edge — 2026 NFL Player-Prop Projection System

This repository contains the research, production data adapters, model components, simulations, audits, and sportsbook comparison layer for the 2026 NFL player-prop system.

## Canonical production authority

There is exactly one production orchestration path:

`.github/workflows/full-slate.yml`

Do not run a legacy engine, old standalone model CLI, or a historical backtest workflow as an alternate production pipeline. Historical migrations remain research evidence only unless a result was explicitly promoted into Full Slate.

The production operating contract is documented in `AGENTS.md`.

## 2026 Full Slate flow

The canonical workflow resolves the active season/week and then runs:

1. current Ourlads roster/depth roles;
2. authoritative NFL team/week schedule map;
3. optional active-slate Odds API fetch and validation;
4. team context with explicit active/prior-season provenance;
5. promoted M89/M90 QB context;
6. weather and injury context;
7. Coverage v2 / WR-CB context;
8. optional current-season PBP enrichment;
9. PlayerForm v2 and Player Identity v3;
10. Provider Readiness v3 and canonical Team Context v3;
11. Bayesian, ML, State and rule components;
12. promoted/frozen ensemble calibration;
13. deterministic metrics and simulation/pricing when an active sportsbook slate is available;
14. M89/M90 football-only QB passing-yards synthesis before sportsbook comparison;
15. strict repository and 2026 production-readiness audits;
16. workflow artifact upload.

The workflow fails closed when required identity, schedule, model, or provenance contracts are violated.

## Football model vs sportsbook data

The independent football projection is built before sportsbook comparison. Sportsbook player lines are downstream benchmark/decision information and must not construct the football-only QB synthesis or train the promoted MC/ML/State ensemble weights.

`FETCH_LIVE_ODDS=false` is the normal no-credit validation mode. When live odds are enabled, `scripts/run_live_odds_gate.py` clears stale sportsbook artifacts and accepts only events belonging to the authoritative active season/week slate. If player props have not been posted yet, sportsbook-dependent pricing is skipped cleanly while the football pipeline can continue.

## Promoted production artifacts

The repository intentionally commits a small number of static model/configuration artifacts that must survive a clean checkout:

- `model/qb_pass_synthesis_v1.json` — promoted M89/M90 QB passing-yards synthesis fit.
- `data/model_ensemble_weights.csv` — frozen promoted pass-yards MC/ML/State ensemble calibration.
- `data/stadiums.csv` — static stadium metadata used by weather.

Generated runtime CSVs under `data/` and `outputs/` are not source code. Full Slate creates them fresh and uploads run outputs as GitHub Actions artifacts.

## Key runtime artifacts

Important generated artifacts include:

- `data/team_week_map.csv` — authoritative active schedule/opponent map.
- `data/roles_ourlads.csv` — current roster and depth roles.
- `data/team_form.csv` — active/prior team evidence with provenance.
- `data/qb_promoted_team_context.csv` — M89/M90 corrected QB context.
- `data/player_game_logs.csv` — leakage-safe historical player evidence.
- `data/player_form.csv` / `data/player_form_consensus.csv` — current pregame player evidence.
- `data/provider_readiness_v3.csv` — provider health/readiness report.
- `data/team_context_v3.csv` — canonical team context.
- `data/model_ml_diagnostics.csv`, `data/model_state_diagnostics.csv`, and `data/model_ensemble_diagnostics.csv` — model component status.
- `data/metrics_ready.csv` — canonical pricing input when sportsbook rows are available.
- `outputs/props_raw.csv` / `outputs/odds_game.csv` — current sportsbook inputs when enabled.
- `outputs/props_priced_clean.csv` — final priced output when active props exist.

## Current QB production contract

Broad QB mean-projection research is frozen after M90. Production uses the corrected M89/M90 architecture and its promoted residual synthesis. The pass-yards ensemble weights are frozen in `data/model_ensemble_weights.csv`; the football-only synthesis is deployed from `model/qb_pass_synthesis_v1.json`.

New generic QB feature hunts should not be introduced without reopening the research contract explicitly. Production fixes, provider repairs, audits, and implementation corrections remain allowed.

## Research code

Historical backtests, migration documentation, and result lineage remain in the repository because they explain why a feature/rule was promoted or rejected. They are not alternate live-production entry points.

Active position refinement resumes from the corrected 2026 production foundation. RB and WR retain their relevant research harnesses; TE will receive a dedicated research program rather than being treated as an unversioned WR variant.

## Validation

Before merging production changes, run or require CI equivalents of:

```bash
python -m compileall scripts engine tests
python scripts/utils/audit_repo.py --strict
python scripts/audit_2026_production_readiness.py
pytest -q
```

A destructive repository-cleanup change is accepted only after canonical Full Slate also succeeds from a clean `main` checkout.

## Week 1 operation

Full Slate is manually dispatchable and also runs in default no-credit mode on production pushes. Production defaults are `SEASON=2026` and `PRIOR_SEASON=2025`; the authoritative NFL week comes from the schedule map, not calendar arithmetic.

Before the first regular-season sportsbook run, run Full Slate with live odds disabled. Once current Week 1 player props are posted, enable live odds for a controlled acceptance run and inspect final component/provenance traces before normal weekly use.

## Repository hygiene

Generated outputs, logs, retired external-provider caches, and root runtime CSV placeholders are intentionally not tracked. See `docs/production/2026_REPO_CLEANUP_AUDIT.md` for the cleanup/reachability ledger.

## License

MIT License. Sportsbook data use remains subject to the applicable provider terms.
