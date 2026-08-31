# 2026 Main-Branch Production Audit — Closure Ledger

This document records the disposition of findings in `docs/production/2026_MAIN_BRANCH_AUDIT.md`. The original audit remains unchanged as the historical baseline.

**Canonical production authority:** `.github/workflows/full-slate.yml`

**Current production baseline before Wave 4:** `main` at `a9ea1d956ebd4e32c39bc9ccc7fae913c3bcfccb` with Repo CI Run #491 and canonical no-credit Full Slate Run #524 both successful.

## Disposition by original finding

| Original finding | Disposition | Evidence / current contract |
|---|---|---|
| P0-1 TeamForm hidden 2025 PBP | Closed for production | `run_team_form_context.py` owns active/prior season semantics, redirects stale PBP requests, repairs guarded fields, stamps provenance, and Full Slate explicitly enables prior-season box backfill during preseason. The underlying legacy builder remains encapsulated rather than treated as an independent production entry point. |
| P0-2 fresh checkout missing calibrated ensemble | Closed | `data/model_ensemble_weights.csv` is a committed protected production artifact. Strict audit requires exactly one valid `pass_yards` calibration row and production pricing loads the frozen calibration before M89/M90 synthesis. |
| P0-3 strict audit missed imported dependencies | Closed | `scripts/utils/audit_repo.py` now includes material imported production dependencies and canonical workflow-contract checks; `scripts/audit_2026_production_readiness.py` separately guards v3/runtime wiring. Both strict audits pass in canonical Full Slate. |
| P0-4 two production authorities | Closed | Full Slate is the sole canonical production orchestration path. The legacy `engine/` package and obsolete standalone `model/` executable stack have been removed. |
| P1-1 Sharp requested-season provenance | Closed for current production contract | Sharp runs through `run_sharpfootball_v2.py` and the TeamForm prior bridge with explicit active/prior semantics and provider-readiness gating. Current/prior state is represented rather than silently treating an unproven source as current. |
| P1-2 overlapping TeamForm fallback semantics | Closed for 2026 production safety; modernization remains architectural debt | Team Context v3 is the canonical downstream authority with explicit provenance and promoted M89/M90 fields. `make_team_form.py` remains an encapsulated legacy dependency and is not an alternate production entry point. |
| P1-3 standalone FantasyPoints WR/CB 2025 logic | Quarantined, not production | The standalone scraper remains only for possible WR research. Canonical Coverage v2 does not invoke it, and repository hygiene explicitly prevents it from entering Full Slate. Its own calendar/archive logic must be fixed or replaced before any future production promotion. |
| P1-4 Week 1/new-player/transition behavior | Closed at structural smoke level; live Week 1 acceptance remains | Player Identity v3, current Ourlads roles, authoritative schedule joins, prior/current PlayerForm logic, and provider readiness all pass the 2026 no-credit Full Slate. Final acceptance with actual Week 1 roster/injury/prop data necessarily waits until those live inputs exist. |
| P1-5 provider availability vs readiness | Closed | Provider Readiness v3 plus Team Context v3 validate and materialize current/prior/unavailable states; canonical Full Slate has passed repeatedly from clean `main` in 2026 context. |
| P2-1 duplicate provider implementations | Closed | API-Sports, GSIS, MySportsFeeds, old injury provider, retired engine, external caches and other ambiguous live-provider surfaces are removed. `scripts/providers/build_schedule.py` is retained only for historical backtests and is explicitly barred from Full Slate. |
| P2-2 tracked runtime/debug artifacts | Closed | Root runtime CSVs, outputs, logs, and external caches are no longer tracked except explicit static production artifacts. Hygiene tests enforce the allowlist. |
| P2-3 sportsbook contamination/separation | Closed by architecture | Football projections are produced independently. Sportsbook data enters only through the downstream live-odds gate and pricing/comparison layer; stale/off-slate odds are cleared and scoped before use. Actual Week 1 live-market acceptance waits for posted regular-season props. |

## Phase A cleanup evidence

- Wave 1: PR #507, merged at `f3b4de379c537a81283ccd18626b8acff8762eed`; canonical Full Slate Run #522 passed.
- Wave 2: PR #508, merged at `17a0a2975e39f35a1c50dbcc5edcea99b6ffcfbc`; canonical Full Slate Run #523 passed.
- Wave 3: PR #509, merged at `a9ea1d956ebd4e32c39bc9ccc7fae913c3bcfccb`; Repo CI Run #491 and canonical Full Slate Run #524 passed.
- Wave 4: final legacy script/model-surface cleanup and retired credential removal. Acceptance requires Repo CI success and a post-merge canonical Full Slate success from clean `main`.

## Remaining known non-blocking items

1. A real live-odds Week 1 acceptance run cannot be completed until sportsbooks post the relevant regular-season player prop markets. The football pipeline is intentionally independent of that availability.
2. `scripts/fantasypoints_wr_cb_scraper.py` is research-only and must not be promoted without fixing/revalidating its season/week/archive contract.
3. `scripts/make_team_form.py` remains an encapsulated legacy builder beneath the guarded runtime wrapper. Replacing it with a fully native context builder may simplify maintenance later, but the current wrapper/v3 contract is the validated 2026 production path.

Phase A is considered complete only after Wave 4 Repo CI passes, Wave 4 is merged, and canonical Full Slate passes again from clean `main`.