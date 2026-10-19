# PRISM RB Rushing Team-Context Stage-1 Participation Authority Reset V1 — Source Freeze

Date: 2026-10-19

Repository: `dkaps6/imtiredofthis`

Research branch: `research-individual-opportunity-roadmap-2026-10-08`

## Purpose

This is a distinct team-context seam from the already SOURCE-BLOCKED historical carry-count identity path.

It asks whether a named RB's **Stage-1 participation authority** has enough exact source support to test persistence across a team change inside the already-qualified RB_RUSHING Stage-2 finite-allocation baseline.

No historical carry-count identity is used.

## Preserved parents

### Team-context boundary

`PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_READY`

- run `39182076127`
- artifact `12156204035`
- digest `sha256:abc2a5b9d3ea2a4d683e56626ea2d33495fa1e7cad2c36c56c01071ccc0339c8`

### Qualified RB rushing allocation baseline

`RB_RUSHING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED`

Parent run:
- run `38957575645`
- artifact `12070774378`
- digest `sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0`

This is the universal qualified RB_RUSHING Stage-2 parent. The later RB-rushing player-specific hard Stage-2 authority bridge remains CLOSED and is not reopened.

### Stage-1 participation mechanism

- run `38907459996`
- artifact `12049120604`
- digest `sha256:41b1acd3f1b8d77abb53980db160ece32f8fc40351939d4827248b9caad0c64b`

## Source-only question

For an RB whose team context changes, is there enough exact source support to test whether his old same-player Stage-1 participation authority should persist on his first **positive-entitlement** canonical RB_RUSHING observation for the new team?

Positive entitlement is required because Stage-1 participation cannot change a finite-room rushing allocation when canonical `authority_q == 0`. Canonical-zero rows are structural no-action rows, not predictive observations.

## Exact source seam

Eligible team-context transition:
- family RB;
- season 2024 or 2025;
- transition kind OFFSEASON_TEAM_CHANGE or INSEASON_TEAM_CHANGE;
- `prior_team_support_ge4 == true`;
- nonblank raw provider stable ID.

For each transition, select the first parent RB_RUSHING row:
- exact same raw provider ID;
- exact new team;
- same season;
- week >= transition week;
- before a later same-player team transition;
- `authority_q > 0`;
- `canonical_zero_share == false`.

If none exists, report:
`NO_POSITIVE_CANONICAL_RB_RUSHING_BASELINE`.

No name matching.

## Source-only reset definition

If an exact same-event Stage-1 named-player mechanism row exists:

`reset_p_active = Stage-1 p_baseline`.

Otherwise the target is already on the Stage-1 role fallback and reset is a no-op.

`actionable_stage1_reset` is true only when `reset_p_active` differs from the qualified parent `stage1_p_active` by >1e-12.

No outcomes are read to define actionability.

## Frozen source gates

SOURCE READY requires all:

1. immutable parent contracts exact;
2. nonblank eligible transition IDs;
3. qualified RB_RUSHING parent -> team-context state exact raw-ID link rate >=99%;
4. selected target rows unique and chronology valid;
5. Stage-1 exact rows join only by season/week/team/family/raw player ID;
6. Stage-1 `p_candidate` matches parent `stage1_p_active` within 1e-12 when present;
7. Stage-1 support flag matches exact Stage-1 row availability;
8. every selected target has positive canonical rushing entitlement;
9. reset probability finite and strictly in (0,1);
10. 2024 selected rows >=30;
11. 2024 stable IDs >=25;
12. 2025 selected rows >=30;
13. 2025 stable IDs >=25;
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

The >=15 actionable support floor matches the immediately preceding TE team-context Stage-1 seam and is frozen before any RB_RUSHING outcome access.

## Dispositions

If all source gates pass:

`PRISM_RB_RUSHING_TEAM_CONTEXT_STAGE1_PARTICIPATION_AUTHORITY_RESET_V1_SOURCE_READY`

Otherwise:

`PRISM_RB_RUSHING_TEAM_CONTEXT_STAGE1_PARTICIPATION_AUTHORITY_RESET_V1_SOURCE_BLOCKED`

A source block is not a predictive failure and does not affect the qualified universal RB_RUSHING Stage-2 baseline.
