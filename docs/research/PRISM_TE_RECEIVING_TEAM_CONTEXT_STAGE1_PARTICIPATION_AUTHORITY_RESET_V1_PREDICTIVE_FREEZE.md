# PRISM TE Receiving Team-Context Stage-1 Participation Authority Reset V1 — Predictive Freeze

Date: 2026-10-19

Repository: `dkaps6/imtiredofthis`

Research branch: `research-individual-opportunity-roadmap-2026-10-08`

## Source certification before outcomes

`PRISM_TE_RECEIVING_TEAM_CONTEXT_STAGE1_PARTICIPATION_AUTHORITY_RESET_V1_SOURCE_READY`

- source freeze: `37dc741d23fea0e030d03c33e8b63c094efb710f`
- run: `39255619282`
- exact head: `cf1e012369cbd873cf6aebdbe388437e709b1b17`
- artifact: `12187947053`
- digest: `sha256:885f3ec53f553f3d2aa1de04d87776497c2d76624bb6d35a600597b31064d266`

Certified:
- TE qualified parent -> team-context raw-ID lineage: 2639/2639;
- 2024 actionable resets: 17 / 17 stable IDs;
- 2025 actionable resets: 18 / 18 stable IDs;
- failed source gates: none.

No target outcomes were opened for source certification.

## Canonical baseline

`TE_RECEIVING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED`

Parent:
`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL`

- freeze: `87400a9c9c1db465c9c4bd70b0cb241dba27c6c9`
- run: `38957575645`
- artifact: `12070774378`
- digest: `sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0`

The previously CLOSED TE player-specific Stage-2 mechanism-utility signal remains closed and is not part of this experiment.

## Scientific question

On the first canonical TE-receiving opportunity observation after a source-certified team change, when a named TE's strictly-prior same-player Stage-1 participation history is exerting authority, should that old same-player participation identity persist, or should the target TE temporarily fall back to the canonical strictly-prior TE family/role participation prior while current-team entitlement and teammate competition remain intact?

This is a market-specific authority test, not a generic new-team penalty.

## Frozen predictive cohort

Use only immutable source rows with:

`actionable_stage1_reset == true`.

2024 development:
- exactly 17 events / 17 stable IDs expected.

2025 independent confirmation:
- exactly 18 events / 18 stable IDs expected.

Source-selected structural no-op rows are not predictive observations.

No post-outcome subgroup rescue is allowed by transition kind, lag, team, player, support tier or any other cut.

## Frozen candidate intervention

For each actionable transition event independently:

1. identify the exact qualified TE_RECEIVING room;
2. identify transitioned target TE by raw provider ID;
3. baseline participation vector = current qualified Stage-1 `stage1_p_active` for every positive-entitlement TE;
4. candidate vector = identical except target TE `p_active = reset_p_active`;
5. keep every `authority_q` unchanged;
6. keep all teammate participation authorities unchanged;
7. keep finite room total `N` unchanged;
8. re-enumerate the exact active-set room with the existing Stage-2 finite-allocation law;
9. preserve residual competition and exact mass conservation.

There is no TE player-specific Stage-2 override in either arm.

No coefficients or thresholds are fit.

## Baseline parity gate

Before interpreting any candidate delta, reconstruct the qualified TE Stage-2 baseline from parent `authority_q` + `stage1_p_active`.

Must match stored qualified baseline within 1e-10 for:
- target expected count;
- target CRPS;
- target AE;
- room allocation error.

State-weight and finite-room mass gaps must be <=1e-10.

Parity failure = INTEGRITY FAILURE.

## Frozen endpoints

Primary named-player endpoint:
- target CRPS gain = qualified baseline CRPS - reset candidate CRPS.

Secondary named-player endpoint:
- target expected-count AE gain = baseline AE - candidate AE.

Room safety:
- room allocation-error gain = baseline room error - candidate room error.

Player-first diagnostics:
- equal-player mean;
- median;
- p10;
- improved/tied/worsened shares.

Uncertainty:
- 5,000 player-cluster bootstrap;
- 5,000 game-cluster bootstrap;
- 5,000 equal-player bootstrap;
- fixed seed `20261019`.

## 2024 development gates

All must pass:
1. actionable events >=15;
2. stable IDs >=15;
3. baseline parity;
4. mean target CRPS gain >0;
5. mean target AE gain >=0;
6. equal-player mean CRPS gain >0;
7. median named-player CRPS gain >=0;
8. improved-player share > worsened-player share;
9. mean room allocation-error gain >=0;
10. exact conservation.

## 2025 independent confirmation gates

All must pass:
1. actionable events >=15;
2. stable IDs >=15;
3. baseline parity;
4. mean target CRPS gain >0;
5. player-cluster 95% CI low >0;
6. game-cluster 95% CI low >0;
7. equal-player mean CRPS gain >0;
8. equal-player 95% CI low >0;
9. median named-player CRPS gain >=0;
10. improved share > worsened share;
11. target AE gain >=0;
12. room allocation-error gain >=0;
13. p10 named-player CRPS gain >= -0.25 * baseline equal-player CRPS;
14. exact conservation.

## Dispositions

All gates pass:

`PRISM_TE_RECEIVING_TEAM_CONTEXT_STAGE1_PARTICIPATION_AUTHORITY_RESET_V1_QUALIFIED`

Source valid / parity valid / predictive gate fails:

`PRISM_TE_RECEIVING_TEAM_CONTEXT_STAGE1_PARTICIPATION_AUTHORITY_RESET_V1_PREDICTIVE_CLOSED`

Integrity failure:

`PRISM_TE_RECEIVING_TEAM_CONTEXT_STAGE1_PARTICIPATION_AUTHORITY_RESET_V1_INTEGRITY_FAILURE`

A predictive close affects only this exact TE team-context reset. It does not revoke the qualified universal TE Stage-2 allocation or the team-context source.

## Hard prohibitions after outcomes

No soft reset, lag cutoff, transition-kind carveout, support-floor change, TE subgroup, opponent add-on, efficiency/yardage/TD rescue, sportsbook input, OddsAPI, fitting, threshold search or production mutation.
