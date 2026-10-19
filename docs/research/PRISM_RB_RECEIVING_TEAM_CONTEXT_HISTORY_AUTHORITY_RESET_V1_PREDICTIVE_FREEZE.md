# PRISM RB Receiving Team-Context History-Authority Reset V1 — Predictive Freeze

Date: 2026-10-19

Repository: `dkaps6/imtiredofthis`

Research branch: `research-individual-opportunity-roadmap-2026-10-08`

## Non-negotiable principle

PRISM is an individual-player projection model.

This experiment asks which already-established authority should govern **this named RB** when his team context changes. It is an additive arbitration layer. It does not rebuild PRISM and it does not replace any qualified parent.

## Source certification completed before outcome scoring

Source parent:

`PRISM_RB_RECEIVING_TEAM_CONTEXT_HISTORY_AUTHORITY_RESET_V1_SOURCE_READY`

- source freeze: `821d43059c1feabb3c7de617698e4913dfee0e6b`
- run: `39254338930`
- exact head: `0b0f6e078f428b03a9d1cd0c104b7fd8c20e6420`
- artifact: `12187431307`
- digest: `sha256:7e7b51c75508b4e883db2934c2e892b377a3d284cb61b448e77e9b7c7eca96cd`

Certified source support:
- qualified RB_RECEIVING authority -> team-context exact raw-ID lineage: 2805/2805;
- first canonical post-change market rows: 71;
- 2024 actionable joint history-authority resets: 22 rows / 22 stable IDs;
- 2025 actionable joint history-authority resets: 17 rows / 17 stable IDs;
- source gates failed: none.

No outcomes were read or scored to obtain that certification.

## Canonical baseline

The baseline is the already-qualified named-player RB-receiving architecture:

`RB_RECEIVING_PLAYER_SPECIFIC_STAGE2_AUTHORITY_BRIDGE_QUALIFIED`

Parent:
`PRISM_RB_STAGE2_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_QUALIFIED_PARTIAL`

- parent freeze: `0831ed274161fa75a7a338041c365bda161041ee`
- run: `39099474655`
- artifact: `12125534698`
- digest: `sha256:3d1f8043dc7a2ded0ed7fd9705b359d31a094a272ecce6958823ce660c2898e4`

The baseline uses:
- current-team canonical RB receiving entitlement `authority_q`;
- qualified Stage-1 participation authority;
- qualified player-specific Stage-2 authority where earned;
- exact finite-opportunity active-set enumeration;
- residual competition mass;
- conserved room allocation.

The baseline is not refit or altered.

## Scientific question

For a named RB on the **first canonical RB-receiving opportunity observation after a source-certified team change**, when old-team-derived same-player history is actually exerting authority, is the current qualified PRISM distribution better if that old-history authority persists, or if it is reset to the canonical strictly-prior family/role participation fallback for that one boundary observation?

This is not a generic new-team boost or penalty.

It is a direct authority question:

> Does this named player's old-history-derived participation/mechanism authority still deserve control at the team-context boundary, or should current-team entitlement compete inside the room using the role prior until new-team player evidence accumulates?

## Frozen eligible predictive cohort

Use only rows already source-certified as:

`actionable_joint_history_reset == true`

from the immutable source artifact.

These rows were selected without outcome access.

Development:
- season 2024;
- exactly 22 actionable transition events / 22 stable IDs are expected.

Independent confirmation:
- season 2025;
- exactly 17 actionable transition events / 17 stable IDs are expected.

The 49 source rows where reset is structurally a no-op are not predictive observations for this mechanism. They remain reported as source context only.

No transition-kind, lag, support-tier, team, player, favorite/longshot, or outcome subgroup may be used to rescue or qualify the mechanism after scoring.

## Frozen candidate intervention

For each actionable transition event independently:

1. identify the exact baseline RB_RECEIVING finite room using season, target week, game key and team;
2. identify the transitioned target by exact raw provider player ID;
3. keep every player's canonical `authority_q` unchanged;
4. keep every non-target teammate's current qualified `p_effective` unchanged;
5. keep room total `N` unchanged;
6. for the transitioned target only, replace baseline `p_effective` with the source-certified `reset_p_active`;
7. re-enumerate the exact active-set room;
8. preserve residual competition and exact probability mass;
9. score the resulting target opportunity-count distribution and room allocation.

The target reset value was source-certified before outcomes:
- if exact Stage-1 named history existed, `reset_p_active = Stage-1 p_baseline`;
- otherwise the target was already on the canonical Stage-1 role fallback and the reset only removes any old-history Stage-2 override.

No new coefficient is fit.

No target share is directly multiplied.

No teammate share is manually redistributed.

No current-team entitlement is erased.

## Frozen baseline parity requirement

Before any candidate delta is accepted, reconstruct the current qualified baseline from the parent row-level `authority_q` + `p_effective` state.

Required parity:
- target expected count vs stored qualified baseline: absolute difference <= 1e-10;
- target CRPS vs stored qualified baseline: absolute difference <= 1e-10;
- target AE vs stored qualified baseline: absolute difference <= 1e-10;
- room allocation error vs stored qualified baseline room row: absolute difference <= 1e-10;
- baseline state-weight mass gap <= 1e-10;
- baseline finite-room mass gap <= 1e-10.

Any parity failure is INTEGRITY FAILURE, not a football result.

## Frozen endpoints

Primary named-player endpoint:
- CRPS gain = current qualified baseline target CRPS minus reset-candidate target CRPS.

Secondary named-player endpoint:
- expected target-count absolute-error gain = baseline AE minus candidate AE.

Room safety endpoint:
- finite-room allocation-error gain = baseline room allocation error minus candidate room allocation error.

Player-first diagnostics:
- equal-player mean CRPS gain;
- median named-player CRPS gain;
- p10 named-player CRPS gain;
- improved / tied / worsened named-player shares.

Uncertainty:
- 5,000-replicate player-cluster bootstrap;
- 5,000-replicate game-cluster bootstrap;
- 5,000-replicate equal-player bootstrap;
- fixed seed `20261019`.

A player with multiple actionable transitions contributes equally in the equal-player diagnostic through his own mean event delta.

## Development gates — 2024

All must pass:

1. actionable events >= 20;
2. distinct stable IDs >= 20;
3. baseline reconstruction parity passes;
4. mean target CRPS gain > 0;
5. mean target-count AE gain >= 0;
6. equal-player mean CRPS gain > 0;
7. median named-player CRPS gain >= 0;
8. improved named-player share > worsened named-player share;
9. mean room allocation-error gain >= 0;
10. state-weight and room-mass conservation gaps <= 1e-10.

If any development gate fails, disposition is PREDICTIVE CLOSED and 2025 is reported only if the scorer was already contractually configured to run both seasons in one immutable execution; no formula may be revised.

## Independent confirmation gates — 2025

All must pass:

1. actionable events >= 15;
2. distinct stable IDs >= 15;
3. baseline reconstruction parity passes;
4. mean target CRPS gain > 0;
5. player-cluster bootstrap 95% CI low > 0;
6. game-cluster bootstrap 95% CI low > 0;
7. equal-player mean CRPS gain > 0;
8. equal-player bootstrap 95% CI low > 0;
9. median named-player CRPS gain >= 0;
10. improved named-player share > worsened named-player share;
11. mean target-count AE gain >= 0;
12. mean room allocation-error gain >= 0;
13. named-player p10 CRPS gain >= -0.25 * baseline equal-player CRPS;
14. state-weight and room-mass conservation gaps <= 1e-10.

The p10 tail floor is fixed before scoring and mirrors the existing RB player-specific authority safety philosophy.

## Dispositions

If all development and confirmation gates pass:

`PRISM_RB_RECEIVING_TEAM_CONTEXT_HISTORY_AUTHORITY_RESET_V1_QUALIFIED`

If source is valid and baseline parity holds but any predictive gate fails:

`PRISM_RB_RECEIVING_TEAM_CONTEXT_HISTORY_AUTHORITY_RESET_V1_PREDICTIVE_CLOSED`

If immutable-source or baseline-parity integrity fails:

`PRISM_RB_RECEIVING_TEAM_CONTEXT_HISTORY_AUTHORITY_RESET_V1_INTEGRITY_FAILURE`

A PREDICTIVE CLOSED result closes only this exact reset mechanism. It does not revoke:
- `PRISM_PLAYER_TEAM_CONTEXT_TRANSITION_V1_SOURCE_READY`;
- universal RB_RECEIVING Stage-2 qualification;
- `RB_RECEIVING_PLAYER_SPECIFIC_STAGE2_AUTHORITY_BRIDGE_QUALIFIED`;
- any other qualified PRISM component.

## Hard prohibitions

After outcomes are opened:
- no alternate reset magnitude;
- no soft blend;
- no lag cutoff;
- no offseason/in-season carveout;
- no support-floor change;
- no top-RB subset;
- no player-name logic;
- no matchup add-on;
- no efficiency/yardage/TD scoring;
- no sportsbook input;
- no paid OddsAPI;
- no threshold search;
- no coefficient fitting;
- no production mutation.

The named player and his exact football state remain the unit of truth.
