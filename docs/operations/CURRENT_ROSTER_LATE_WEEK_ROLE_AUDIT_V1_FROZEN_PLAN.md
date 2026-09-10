# Current Roster / Late-Week Role Audit V1 — Frozen Plan

Status: `DIAGNOSTIC AUDIT ONLY / NO PRODUCTION MUTATION`

## Authority

- Parent main handoff commit: `99d0ae6f6e0c4d60458a919096ce5cec1dfe695e`
- Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- Canonical Full Slate: `.github/workflows/full-slate.yml`

This is an operational correctness audit. It must not rewrite historical science, reconstruct historical injury states, or promote a model.

## Frozen question

Can the current Full Slate carry a player with an unavailable/inactive late-week state into the authoritative role/opportunity universe, or fail to reallocate that unavailable player's role/opportunity correctly, because current roster/depth and injury/inactive authorities are not reconciled before modeling?

## Frozen audit surfaces

1. `scripts/providers/ourlads_depth.py`
   - inactive/status detection
   - default inclusion/exclusion behavior
   - canonical roles artifact schema
   - role/depth semantics and provenance/freshness fields
2. `.github/workflows/full-slate.yml`
   - order of roster, injury and modeling construction
3. `scripts/build/build_injuries_weekly.py`
   - source type, current-week contract, provider-outage behavior, official game-status semantics
4. `scripts/modeling/context_bridge.py`
   - injury-to-player-context merge
5. `scripts/modeling/simulation_rules.py`
   - unavailable-player treatment and redistribution
6. `scripts/run_rb_week1_no_odds.py`
   - construction of active RB universe and injury reconciliation
7. `scripts/validate_full_slate_pre_model_v1.py`, `scripts/validate_full_slate_data_quality_v1.py`, `scripts/artifact_contracts.py`
   - fail-closed gates for roster freshness/status and unavailable players

## Frozen questions / checks

A. Does Ourlads detect inactive status before writing roles?  
B. Does `roles_ourlads.csv` preserve status, source timestamp/fetch time and source identity?  
C. Does Full Slate default to including Ourlads inactive players?  
D. Can injury-provider state be `no_official_report` while Ourlads itself indicates an inactive player?  
E. Are `OUT/IR/PUP/inactive` players forced to zero opportunity before any model/simulation?  
F. Is removed opportunity conserved/reallocated to eligible teammates rather than left on the unavailable player?  
G. For promoted RB P3, is the active RB universe filtered/re-ranked using current unavailability before projection?  
H. Do pre-model/data-quality validators reject a roles artifact that lacks availability/freshness provenance or contains a known unavailable starter?  
I. Can a stale depth role survive after a late-week availability change?  
J. Are official game-day inactives a distinct authority from the weekly practice/game injury report, or is that information absent?

## Allowed dispositions

- `CURRENT_ROSTER_LATE_WEEK_ROLE_GAP_CONFIRMED_FIX_PLAN_REQUIRED`
- `CURRENT_ROSTER_LATE_WEEK_ROLE_CONTROLS_SUFFICIENT_NO_FIX`
- `CURRENT_ROSTER_LATE_WEEK_ROLE_PARTIAL_GAP_NEEDS_SOURCE_CLARIFICATION`

## Boundary

No production file, Full Slate workflow, R26, R22, P3, M89/M90, WR/TE production component, sportsbook path or historical model may change during this audit.

If a gap is confirmed, write a separate frozen operational fix plan defining authority precedence, fail-closed behavior, zeroing/reallocation semantics, provenance and verification before implementation.
