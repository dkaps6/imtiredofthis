# PRISM TE Receiving Team-Context Stage-1 Participation Authority Reset V1 — Source Freeze

Date: 2026-10-19

Repository: `dkaps6/imtiredofthis`

Research branch: `research-individual-opportunity-roadmap-2026-10-08`

## Model principle

PRISM is an individual-player projection model.

This is another market-specific team-context arbitration seam. It does not rebuild PRISM and it does not reopen closed TE player-specific Stage-2 personalization.

## Preserved parent states

### Team-context boundary

`PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_READY`

- run `39182076127`
- artifact `12156204035`
- digest `sha256:abc2a5b9d3ea2a4d683e56626ea2d33495fa1e7cad2c36c56c01071ccc0339c8`

### Current TE receiving allocation baseline

`TE_RECEIVING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED`

Parent:
`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL`

- freeze `87400a9c9c1db465c9c4bd70b0cb241dba27c6c9`
- run `38957575645`
- artifact `12070774378`
- digest `sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0`

This qualified TE baseline remains the canonical baseline.

### Stage-1 named-player participation mechanism

- run `38907459996`
- artifact `12049120604`
- digest `sha256:41b1acd3f1b8d77abb53980db160ece32f8fc40351939d4827248b9caad0c64b`

Its frozen semantics provide the strictly-prior family/role participation prior `p_baseline` and, when supported, the same-player participation candidate `p_candidate`.

## Source-only scientific question

For a named TE whose team context changes, is there enough exact source support to test whether his old same-player Stage-1 participation identity should retain authority on the first canonical TE-receiving opportunity observation for the new team?

This is not a TE-wide reset rule and it does not reopen the previously CLOSED TE player-specific Stage-2 mechanism-utility family.

## Exact seam

Family: TE.

Market/opportunity cell: `TE_RECEIVING`.

Target seasons:
- 2024 development source cohort;
- 2025 independent-confirmation source cohort.

An eligible transition must:
1. be a canonical TE team-context transition;
2. be `OFFSEASON_TEAM_CHANGE` or `INSEASON_TEAM_CHANGE`;
3. have `prior_team_support_ge4 == true`;
4. have nonblank raw provider `stable_id`.

For each eligible transition select the first canonical qualified TE_RECEIVING parent row:
- exact same raw provider ID;
- exact new team;
- same season;
- week >= transition week;
- before any later same-player team-context transition in that season.

If no qualified TE_RECEIVING row exists in that boundary, report:
`NO_CANONICAL_TE_RECEIVING_BASELINE`.

No player name may be used.

## Frozen source-only reset

The current qualified TE Stage-2 allocation is the baseline.

For the transitioned target TE only:

- if an exact Stage-1 same-event named-player mechanism row exists, define
  `reset_p_active = p_baseline`;
- if no such row exists, the target is already on the Stage-1 family/role fallback and the reset is structurally a no-op.

The eventual candidate, only if SOURCE READY, will:
- replace only that target TE's `stage1_p_active` with source-certified `reset_p_active`;
- keep every player's canonical current-team `authority_q`;
- keep every teammate's baseline Stage-1 participation authority;
- keep room `N`, residual competition and exact finite-opportunity enumeration unchanged.

No TE player-specific Stage-2 personalization is introduced.

## Source gates frozen before outcomes

SOURCE READY requires all:

1. immutable parent run/artifact/digest contracts match;
2. eligible transition stable IDs are nonblank;
3. qualified TE_RECEIVING authority rows link to team-context state by exact raw provider ID at >=99%;
4. first-post-change target rows are unique by transition;
5. target chronology is at/after transition and before later same-player transition;
6. exact Stage-1 same-event rows attach only on season/week/team/family/raw player ID;
7. Stage-1 `p_candidate`, when present, matches parent `stage1_p_active` within 1e-12;
8. selected targets have positive canonical TE receiving entitlement, not canonical-zero-share;
9. reset probabilities are finite and strictly in (0,1);
10. 2024 selected rows >=30;
11. 2024 distinct stable IDs >=25;
12. 2025 selected rows >=30;
13. 2025 distinct stable IDs >=25;
14. 2024 actionable reset rows >=15;
15. 2024 actionable stable IDs >=15;
16. 2025 actionable reset rows >=15;
17. 2025 actionable stable IDs >=15;
18. names read/used = 0;
19. predictive outcomes read/scored = 0;
20. target-game opportunity counts read = 0;
21. sportsbook inputs = 0;
22. OddsAPI calls = 0;
23. coefficients fit = 0;
24. thresholds searched = 0;
25. production changed = false.

The support floors are frozen from source feasibility only.

## Dispositions

If all gates pass:

`PRISM_TE_RECEIVING_TEAM_CONTEXT_STAGE1_PARTICIPATION_AUTHORITY_RESET_V1_SOURCE_READY`

Otherwise:

`PRISM_TE_RECEIVING_TEAM_CONTEXT_STAGE1_PARTICIPATION_AUTHORITY_RESET_V1_SOURCE_BLOCKED`

A source block is not a predictive failure.

If SOURCE READY, a separate predictive freeze must be committed before any target outcome scoring.
