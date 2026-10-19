# PRISM RB Team-Context Carry Count Identity Bridge V1 — Frozen Source Contract

Date: 2026-10-19

Repository: `dkaps6/imtiredofthis`

Research branch: `research-individual-opportunity-roadmap-2026-10-08`

## Scientific status before this bridge

Preserve:

- `PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_READY`
- `PRISM_RB_TEAM_CONTEXT_CARRY_IDENTITY_PERSISTENCE_V1_SOURCE_BLOCKED`
- frozen RB predictive contract commit `668bd3266b6e6c131875ef245ce037b1a9925be7`
- canonical blocked RB run `39182998402`

The blocked RB run linked 0/39 eligible transition events to the historical count universe and therefore scored zero predictive outcomes. This bridge audit is source/identity lineage only and must not reinterpret that result as a predictive failure.

## Question

Is there a canonical, deterministic, leakage-safe provider-level identity bridge among:

1. team-context transition `stable_id`;
2. canonical Stage-2 RB rushing `player_id`;
3. historical carry-count `player_identity_key`;

for the frozen 2024/2025 RB team-context seam?

## Allowed identity evidence

Only provider-level stable identifiers and canonical repository identity transformations are allowed.

The audit may read only identity/source metadata needed to establish lineage from these immutable parents:

### Team-context parent

- run `39182076127`
- artifact `12156204035`
- digest `sha256:abc2a5b9d3ea2a4d683e56626ea2d33495fa1e7cad2c36c56c01071ccc0339c8`
- `player_team_context_transition_events.csv`
- `player_team_context_source_result.json`

### Stage-2 source parent

- run `38955368643`
- artifact `12070806828`
- digest `sha256:e91a108495d7a2b6eb289e0618ab6af09e3d6cdf6895e534742a07340e4a2a83`
- `active_role_conditional_allocation_source_join_detail.csv`

### Stage-1 identity parent used by Stage-2

- run `38895205962`
- artifact `12044155086`
- digest `sha256:a42322579aee682497223c7f53bce0b14a96dc6f466558e811770e681a035dad`
- `player_active_role_pregame_universe.csv`

### Canonical count identity code

- `scripts/utils/player_identity_v3.py::historical_identity_key`
- `scripts/research/audit_player_opportunity_share_stability_v1.py::_load_roster_universe`
- `scripts/research/audit_player_opportunity_share_stability_v1.py::build_weekly_universe`
- `scripts/research/audit_player_opportunity_allocation_uncertainty_betabinomial_v1.py::build_target_counts`

## Frozen bridge semantics

For a nonblank raw provider GSIS identifier `g`:

- team-context identity is raw `g`;
- Stage-1/Stage-2 identity is raw `g`;
- canonical historical count identity is exactly `historical_identity_key(g, "", "") == "gsis:" + g`.

This is a deterministic namespace representation bridge only. It is not a name join, alias map, fuzzy match, manual crosswalk, outcome-derived mapping, or fitted transformation.

Event-level absence from Stage-2 authority is NOT an identity mismatch. Stage-2 authority eligibility/coverage remains governed by the already-frozen RB predictive contract. The bridge audit must report such availability separately and must not manufacture missing authority rows.

## Frozen eligible transition scope

Use the already-frozen RB V1 source scope only:

- seasons 2024 and 2025;
- weeks 5 through 18;
- family RB;
- `transition_kind == INSEASON_TEAM_CHANGE`;
- `prior_team_support_ge4 == true`;
- nonblank `stable_id`.

The expected source population from the canonical parent is 39 event rows.

## Source-only gates

The bridge may be certified SOURCE READY only if ALL hold:

1. immutable parent run/artifact/digest contracts match exactly;
2. team-context 2024/2025 roster schema reports `stable_id == gsis_id`;
3. all frozen-scope transition `stable_id` values are nonblank provider IDs;
4. canonical `historical_identity_key` maps every frozen-scope raw ID to exactly `gsis:<raw_id>`;
5. round-trip from derived count key back to the raw ID is exact for every frozen-scope event;
6. exact derived count-identity link rate is >= 0.99; the frozen gate is not lowered;
7. no two distinct raw provider IDs map to the same derived count identity;
8. Stage-1 RB source rows use nonblank raw provider IDs and stable roster identity;
9. every Stage-2 RB_RUSHING row for the target scope has an exact same-event Stage-1 raw `player_id`;
10. Stage-2 raw IDs round-trip through the same canonical `gsis:` representation with zero ambiguity;
11. names are not read or used for bridge construction;
12. predictive outcomes are not read or scored;
13. target-game carries, targets, yards, TDs and sportsbook inputs are not read;
14. coefficients fit = 0, thresholds searched = 0, OddsAPI calls = 0, production changed = false.

## Disposition vocabulary

If all source gates pass:

`PRISM_RB_TEAM_CONTEXT_CARRY_COUNT_IDENTITY_BRIDGE_V1_SOURCE_READY`

This certifies only the provider-ID representation bridge.

If any source gate fails:

`PRISM_RB_TEAM_CONTEXT_CARRY_COUNT_IDENTITY_BRIDGE_V1_SOURCE_BLOCKED`

A block here is not a predictive failure.

## After SOURCE READY only

Only after this bridge is independently certified may the already-frozen RB V1 be executed again with the identity representation corrected on the count side:

`raw stable_id g -> historical count key gsis:g`.

No predictive formula, history window, finite-room construction, eligibility rule, development gate, confirmation gate, coefficient, threshold, or market definition may change.

The Stage-2 authority lookup remains raw-ID exact and unchanged.

No outcome may be inspected before the bridge certification is complete.
