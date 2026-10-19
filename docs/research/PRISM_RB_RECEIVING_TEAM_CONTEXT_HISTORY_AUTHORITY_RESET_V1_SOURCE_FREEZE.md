# PRISM RB Receiving Team-Context History-Authority Reset V1 — Source Freeze

Date: 2026-10-19

Repository: `dkaps6/imtiredofthis`

Research branch: `research-individual-opportunity-roadmap-2026-10-08`

## Non-negotiable model principle

PRISM is an individual-player projection model.

The named player owns the rule. Position/family information is a prior and fallback.

This seam is additive. It does not replace, rebuild, or invalidate any qualified PRISM science.

## Preserved parent states

### Team-context boundary

`PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_READY`

- freeze: `bd254ec9435cd0a17e2c7d82d78532e67c2cd23e`
- run: `39182076127`
- artifact: `12156204035`
- digest: `sha256:abc2a5b9d3ea2a4d683e56626ea2d33495fa1e7cad2c36c56c01071ccc0339c8`

### Current qualified RB-receiving named-player authority

`RB_RECEIVING_PLAYER_SPECIFIC_STAGE2_AUTHORITY_BRIDGE_QUALIFIED`

Parent bridge:
`PRISM_RB_STAGE2_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_QUALIFIED_PARTIAL`

- freeze: `0831ed274161fa75a7a338041c365bda161041ee`
- run: `39099474655`
- artifact: `12125534698`
- digest: `sha256:3d1f8043dc7a2ded0ed7fd9705b359d31a094a272ecce6958823ce660c2898e4`

This current architecture remains the canonical baseline.

### Stage-1 named-player participation mechanism

- run: `38907459996`
- artifact: `12049120604`
- digest: `sha256:41b1acd3f1b8d77abb53980db160ece32f8fc40351939d4827248b9caad0c64b`

Its frozen semantics provide:
- a strictly-prior family/role Jeffreys participation prior `p_baseline`;
- a same-player supported participation candidate `p_candidate`;
- unsupported players fall back to the same strictly-prior family/role prior.

## Source-only scientific question

For a named RB whose team context has changed, is there enough exact source support to test whether his **old-history-derived player authorities** should retain authority on his first canonical RB-receiving market observation in the new team context?

This freeze certifies only source/identity/counterfactual availability. It scores no predictive outcome.

## Exact market seam

Family: RB.

Market/opportunity cell: `RB_RECEIVING`.

Target seasons:
- development source cohort: 2024;
- independent-confirmation source cohort: 2025.

A source transition is eligible when:
1. team-context family is RB;
2. season is 2024 or 2025;
3. `prior_team_support_ge4 == true`;
4. transition kind is exactly `OFFSEASON_TEAM_CHANGE` or `INSEASON_TEAM_CHANGE`;
5. raw provider `stable_id` is nonblank.

For each eligible transition, find the **first canonical qualified RB_RECEIVING authority row**:
- same raw provider ID;
- same new team;
- same season;
- row week >= transition week;
- row occurs before any later team-context transition for that same raw provider ID in that season.

If no such qualified market row exists, that transition is reported as `NO_CANONICAL_RB_RECEIVING_BASELINE` and is not scored later.

No name field may participate in this selection.

## Exact source-only candidate reset

The current qualified RB-receiving player-specific authority remains the BASELINE.

For the one transitioned named RB on the first canonical post-change market row, define a source-only reset participation probability:

- if the canonical Stage-1 mechanism has an exact same-event raw-ID row, use its frozen strictly-prior `p_baseline`;
- otherwise preserve that row's current `stage1_p_active`, because the frozen Stage-1 bridge defines unsupported named players as already using the canonical family/role fallback.

The eventual predictive candidate, if and only if this source audit passes, will also remove the target RB's same-player Stage-2 historical mechanism-authority override for that one row. It will therefore use the reset Stage-1 probability as the target player's candidate active probability.

Everything else remains unchanged:
- canonical current-team RB receiving entitlement `authority_q`;
- teammate identities;
- teammate current qualified `p_effective`;
- finite room size `N`;
- exact finite-opportunity enumeration/conservation;
- all downstream efficiency/yardage science;
- production.

This is not a post-hoc multiplier and not a position-wide reset.

## Source gates frozen before any new outcome scoring

The source disposition may be READY only if all of the following hold:

1. immutable parent run/artifact/digest contracts match exactly;
2. team-context transition raw IDs are nonblank and deterministic;
3. every qualified RB_RECEIVING authority source row used by the seam links to the canonical team-context state on exact raw provider ID;
4. exact linked state rate for the qualified RB_RECEIVING authority source is >= 0.99;
5. every selected first-post-change target row is unique by transition and raw provider identity;
6. selected row chronology is at/after the transition and before the next same-player transition;
7. no duplicate selected target identity exists for the same transition;
8. Stage-1 exact same-event rows attach only by raw provider ID + season/week/team/family;
9. whenever Stage-1 `p_candidate` is present, it exactly equals the qualified authority row's `stage1_p_active` within 1e-12;
10. reset participation probability is finite and strictly inside (0,1) for every selected target;
11. 2024 selected market rows >= 30;
12. 2024 distinct stable IDs >= 25;
13. 2025 selected market rows >= 30;
14. 2025 distinct stable IDs >= 25;
15. 2024 rows whose joint old-history-authority reset changes the target active probability >= 10;
16. 2024 distinct stable IDs with an actionable reset >= 10;
17. 2025 rows whose joint old-history-authority reset changes the target active probability >= 10;
18. 2025 distinct stable IDs with an actionable reset >= 10;
19. names read/used = 0;
20. predictive outcomes read/scored = 0;
21. target-game opportunity counts read = 0;
22. sportsbook inputs = 0;
23. OddsAPI calls = 0;
24. coefficients fit = 0;
25. thresholds searched = 0;
26. production changed = false.

The support floors are source-feasibility gates only and are frozen before outcome access.

## Source disposition vocabulary

If every source gate passes:

`PRISM_RB_RECEIVING_TEAM_CONTEXT_HISTORY_AUTHORITY_RESET_V1_SOURCE_READY`

Otherwise:

`PRISM_RB_RECEIVING_TEAM_CONTEXT_HISTORY_AUTHORITY_RESET_V1_SOURCE_BLOCKED`

A SOURCE BLOCK is not a predictive failure.

## If SOURCE READY

Only then may a separate predictive freeze be committed before opening target outcomes.

The later test must compare:
- baseline = current QUALIFIED RB-receiving player-specific PRISM authority;
- candidate = same exact architecture except the one transitioned named RB's old-history-derived Stage-1/Stage-2 player authority is reset to the canonical strictly-prior family/role participation fallback for the first canonical post-change market row.

2024 must be development.
2025 must remain independent confirmation.
No refit, threshold search, subgroup rescue, alternate lag cutoff, transition-kind carveout, or formula change after outcomes are visible.

## If SOURCE BLOCKED

Preserve all parent qualifications.
Close only this exact team-context RB-receiving reset seam.
Remain in team-context science.
