# Current Roster / Late-Week Role Fix V1 — Frozen Plan

Status: `FROZEN BEFORE PRODUCTION IMPLEMENTATION`

## Parent evidence

- Confirmed audit result commit: `810c344a437d411707185317033acc7004f1c7db`
- Audit disposition: `CURRENT_ROSTER_LATE_WEEK_ROLE_GAP_CONFIRMED_FIX_PLAN_REQUIRED`
- Audit plan commit: `848e5e44b1078c9d9dd7ab40ad0ccf16136f3754`
- Parent main handoff: `99d0ae6f6e0c4d60458a919096ce5cec1dfe695e`
- Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`

This is an operational correctness fix, not a predictive-model experiment. Historical research outputs must remain unchanged.

## Goal

Create one canonical current-player availability/role authority before PlayerForm and promoted opportunity construction so that definitively unavailable players cannot retain active roles or positive opportunity, while uncertain players are not falsely removed.

## Frozen authority layers and precedence

For each current scheduled player/team, maintain the raw source facts separately and derive one final availability state.

Precedence from strongest to weakest:

1. **Official NFL game-day inactive list**, when a complete validated team section exists for that game's pre-kickoff window.
   - listed player => `UNAVAILABLE_OFFICIAL_INACTIVE`
   - absence from a complete official team section is evidence of active eligibility for that game, subject to roster identity reconciliation.
2. **Official/weekly game-status injury report**:
   - `OUT`, `IR`, `PUP`, or equivalent definitive non-participation => `UNAVAILABLE_REPORTED`
   - `DOUBTFUL`, `QUESTIONABLE`, practice statuses => `UNCERTAIN`, not definitively unavailable.
3. **Ourlads current depth source**:
   - provider-marked inactive => `UNAVAILABLE_DEPTH_SOURCE` unless contradicted by stronger same-as-of official evidence;
   - otherwise supplies depth ordering/role, not definitive health certainty.
4. Missing/ambiguous state => `UNKNOWN`, never silently relabeled healthy.

Every source fact must retain source, source URL/type, fetched/generated timestamp, season/week, team, player identity and source-state/coverage metadata.

## Canonical new artifact

Create `data/current_player_availability.csv` with at least:
- season
- week
- team
- player
- player_clean_key
- position / position_group
- raw_depth_role
- raw_depth_index
- ourlads_status
- injury_status
- injury_designation
- official_inactive (0/1/NA)
- official_inactive_section_complete (0/1)
- final_availability_state
- definitive_unavailable (0/1)
- availability_authority
- availability_reason
- role_after_availability
- role_rank_after_availability
- source_asof_utc / generated_at_utc

A JSON sidecar must summarize source health, source timestamps, team coverage, official-inactive coverage by game window and any unresolved identities.

Do not silently mutate source fields during precedence resolution.

## Ourlads changes

Preserve Ourlads `status` in the raw/canonical depth artifact or an explicitly versioned raw sidecar. Do not discard the provider's own inactive signal.

Add generated/source timestamp provenance. Ourlads depth remains a depth source; it does not become the top availability authority.

## Official-inactive acquisition

Promote a production-safe adapter from the already-audited M78 NFL Inactives contract rather than creating a new source family.

Required semantics:
- source: NFL `/inactives/` live page;
- team sections must satisfy the hardened completeness/identity checks before absence-from-list can be treated as evidence;
- endpoint reachability alone is not availability evidence;
- acquisition must be timestamped before the relevant game's kickoff;
- game windows are evaluated independently; a valid early-window snapshot must not falsely certify later games if those lists are not yet published;
- before official inactive lists are expected/published, state is `NOT_YET_AVAILABLE`, not an error and not active evidence;
- once the source should be available for an imminent game, a missing/invalid section is fail-closed for production pricing of affected teams rather than silently using stale roles.

## Role reconciliation

For each team/position family:
1. remove rows with `definitive_unavailable == 1` from the eligible role pool;
2. preserve original depth ordering among remaining eligible players;
3. re-rank to `role_after_availability` deterministically (RB1/RB2..., TE1..., QB1..., and existing WR slot/perimeter semantics as applicable);
4. retain unavailable players in the audit artifact with no active role and zero eligibility; do not erase their evidence row.

Do not infer a new role from sportsbook listing or betting lines.

## Opportunity semantics

For `definitive_unavailable == 1`:
- target share = 0
- rush share = 0
- pass attempts/dropbacks = 0 when applicable
- receptions/carries/yardage/TD opportunity = 0
- player must not receive a positive promoted component mean or distribution

Removed team opportunity must be conserved/reallocated only through the position/component's existing qualified allocation logic or a separately frozen deterministic availability redistribution layer. No arbitrary 50% retention on an OUT/IR/PUP/official-inactive player is allowed.

`DOUBTFUL`/`QUESTIONABLE` remain eligible and may use existing uncertainty/limitation mechanics until definitive inactive evidence appears. The fix must not equate uncertainty with official unavailability.

## Production consumption order

Canonical Full Slate must become:
1. build timestamped raw Ourlads depth/status;
2. build schedule/kickoffs;
3. build weekly injury report/source state;
4. acquire timing-aware official game-day inactive evidence where available/required;
5. build and validate `current_player_availability.csv` + sidecar;
6. build reconciled active roles;
7. only then build PlayerForm/current contexts and promoted QB/RB/WR/TE opportunity paths;
8. price only teams whose required availability authority is certified for the requested as-of/game window.

Historical model training/backtests are unchanged.

## Required validation gates

All must pass before promotion:

1. 32 scheduled teams represented for Week1 (or exact scheduled-team set later weeks).
2. Every current role row has source/as-of provenance.
3. No duplicate team/player current identity.
4. Every definitive unavailable source row resolves to a canonical current player/team or is quarantined and causes affected-team fail-closed status.
5. Official-inactive absence is used only when the team section is complete.
6. Official-inactive snapshot time precedes relevant kickoff.
7. No definitively unavailable player has an active reconciled role.
8. No definitively unavailable player has positive target/rush/pass opportunity in model contexts.
9. No definitively unavailable player has positive promoted RB P3/QB/WR/TE mean or simulation distribution.
10. Remaining depth roles are deterministic and gap-free within the defined role family.
11. Team opportunity conservation checks pass after removals/reallocation.
12. `DOUBTFUL`/`QUESTIONABLE` are not automatically removed.
13. Provider outage/invalid official section fails closed only for teams/windows for which that source is required, and exposes the exact reason.
14. No sportsbook input defines availability or role.
15. R26/R22 model coefficients/mechanics unchanged; this fix only changes current eligibility/role inputs.
16. Existing qualified WR/TE/QB/RB component versions remain unchanged.
17. Full Slate no-odds football build passes.
18. Full Slate live-pricing build passes when live odds are available, without using market data to resolve availability.
19. Static production-readiness audit gains explicit checks for availability authority wiring.
20. Canonical handoff records production commit/run/job/artifact/digest and any teams withheld by timing-aware availability gates.

## Required fixture tests before live promotion

At minimum:
- RB1 OUT, RB2 active => RB1 removed; RB2 becomes active lead-role candidate; old RB1 has zero opportunity.
- WR1 official inactive => zero WR1 opportunity; eligible receivers receive conserved opportunity under the approved layer; inactive player stays in audit only.
- QB1 official inactive => QB1 zero; reconciled QB2 becomes starter candidate; no dual-QB starter authority.
- QUESTIONABLE starter => remains eligible.
- Ourlads inactive + injury source empty before official report => unavailable/depth-source or explicit uncertain state remains visible; status cannot disappear.
- official inactive endpoint reachable but incomplete => affected imminent teams fail closed; endpoint reachability cannot certify them.
- complete official team section where player is absent => player is not marked inactive from that source.

## Promotion boundary

This plan authorizes implementation on a dedicated branch only. Production may be promoted only after the exact implementation is locked and all required fixture/Full Slate gates pass. Any semantic change to precedence, zeroing, timing, re-ranking or conservation requires a new frozen plan before testing.
