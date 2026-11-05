# RB Vacancy Opportunity V1 — Phase 0 Production Correctness Audit

Date: 2026-09-24
Branch: `research-rb-vacancy-opportunity-v1`
Parent frozen plan: `docs/research/RB_VACANCY_OPPORTUNITY_V1_PLAN.md`
Status: **PHASE 0 CLOSED — no inactive-RB production correctness defect found**

## Question

Can a confirmed unavailable RB survive into the final football simulation/pricing universe with nonzero rushing opportunity because `simulation_rules.py` only applies a 0.50 self-haircut?

## Finding

**No, not on the current canonical Full Slate path.** The 0.50 rule is legacy/fallback behavior for a player who is still present in the model context; definitive unavailability is resolved earlier and removes that player from the current opportunity universe before PlayerForm, rules, simulation, or pricing.

The exact production seam is:

1. `full-slate.yml` builds raw Ourlads roles, schedule, weather, and weekly injuries.
2. Before any current player opportunity is built, Full Slate runs `run_current_player_availability_candidate_prep_v1.py`.
3. That orchestrator runs `build_current_player_availability_v1.py`, which resolves official inactive / injury / depth-source availability and sets `definitive_unavailable=1` for `UNAVAILABLE_*` states. Its definitive reported set is `OUT`, `IR`, `PUP`, `RESERVE/INJURED`, `INJURED RESERVE`; a complete official inactive section is also authoritative. `DOUBTFUL` and `QUESTIONABLE` are intentionally **UNCERTAIN**, not definitive unavailable.
4. `build_reconciled_active_roles_v1.py` filters `definitive_unavailable == 1` rows out entirely and re-ranks active QB/RB/FB/TE roles. It fails if an unavailable player retains an active reconciled role.
5. `build_production_eligible_active_roles_v1.py` then removes timing-ineligible games, yielding `data/roles_current_production_eligible_v1.csv`.
6. Full Slate sets `ACTIVE_ROLES_CSV=data/roles_current_production_eligible_v1.csv` and runs PlayerForm from that exact current-role authority. Therefore definitive-unavailable players are absent from `player_form.csv` / `player_form_consensus.csv`.
7. The full-roster simulation universe is built from `player_form_consensus.csv`, requires exact identity equality with `model_context_bridge.csv`, applies Bayesian/rule layers, and simulates every player in that already-filtered universe. There is no later step that re-adds a removed player.
8. The existing football-stack certification independently intersects `current_player_availability.csv` definitive-unavailable identities with PlayerForm and then with generated simulation arrays; either intersection is a fatal error.
9. Sportsbook acquisition/matching is downstream of this football eligibility seam and cannot resurrect a removed player.

## Why the apparent 0.50 issue is not the current defect

`scripts/modeling/simulation_rules.py::_injury_limited()` recognizes `OUT`, `DOUBTFUL`, `IR`, and `PUP`, and a still-present injury-limited player can have `base_rush *= 0.50`. That source line alone looked dangerous. But current production resolves availability **before opportunity**. A definitive OUT/IR/PUP/official inactive is absent before the rule layer is built. A DOUBTFUL player is not automatically removed because the locked availability resolver classifies DOUBTFUL as uncertain rather than definitive unavailable. Thus the 0.50 behavior for a surviving doubtful player is not evidence that a confirmed inactive retains final carries.

## Phase 0 disposition

`CURRENT_PRODUCTION_DEFINITIVE_UNAVAILABLE_REMOVAL_CONFIRMED`

- production correctness defect: **NO**
- definitive unavailable player allowed into current PlayerForm: **NO**
- definitive unavailable player allowed simulation arrays: **NO**
- sportsbook can resurrect unavailable player: **NO**
- science hypothesis invalidated: **NO**

This actually sharpens the research question. Production already removes the unavailable backfield player correctly, but removing a player from the opportunity universe does not by itself estimate how much of his strict-prior rushing role should transfer to successor RB/FB teammates. Canonical simulation simply allocates from the shares that survive on the active players plus its residual bucket. V1 may therefore study the missing **vacated-share transfer** without conflating it with inactive-player eligibility.

## Phase 1 input inventory — source contract established

### Availability / injury

Canonical current production authority:
- `data/current_player_availability.csv`
- generated before opportunity by `run_current_player_availability_candidate_prep_v1.py`
- joins current Ourlads depth/status, weekly `data/injuries.csv`, and official inactive evidence when complete
- auditable fields include `final_availability_state`, `availability_authority`, `availability_reason`, `definitive_unavailable`, `eligible_for_opportunity`, raw depth role/index, injury status/designation/practice status where supplied, and generated timestamp
- no sportsbook input

Important semantic correction for V1: the unavailable set must follow the locked production resolver, not the broader legacy `_injury_limited()` token set. In particular, **DOUBTFUL is UNCERTAIN and must not be treated as a confirmed vacancy in the first deterministic candidate**. The first vacancy table should use `definitive_unavailable == 1` / `UNAVAILABLE_*` only.

### Successor current role

Canonical active-role authority:
- `data/roles_current_production_eligible_v1.csv`
- derived from the reconciled availability artifact before PlayerForm
- unavailable players removed; active RB/FB roles deterministically re-ranked
- preserves `depth_index`, `raw_depth_role`, `availability_authority`, `final_availability_state`, and source/as-of fields

### Strict-prior player evidence

Canonical current PlayerForm runner:
- `run_player_form_current_roles_v1.py`
- builds from `ACTIVE_ROLES_CSV`
- republishes `player_game_logs.csv` with only prior season plus current-season weeks `< target week`
- same-week/future target-game rows are explicitly rejected

Candidate construction must therefore source successor weighting from strict-prior fields already derived from this history, not target-game participation. The next implementation step is to identify the exact current snap-participation and rush-share columns/artifacts available at this seam and freeze a deterministic no-outcome vacancy table/formula before grading.

## Next action

Proceed to Phase 1/2 only:
1. inventory exact strict-prior snap participation and rush-share fields currently materialized for RB/FB;
2. build a research-only vacancy-state table from the full availability ledger plus active successor roles;
3. use only `definitive_unavailable == 1` for V1 vacancy events;
4. freeze successor weighting and conservation mechanics before attaching outcomes;
5. do not rerun M96, alter YPC, use sportsbook information upstream, or tune any coefficient after outcomes are visible.
