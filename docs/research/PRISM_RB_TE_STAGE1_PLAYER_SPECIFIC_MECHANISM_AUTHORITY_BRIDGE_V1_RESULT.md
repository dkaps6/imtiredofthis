# PRISM RB/TE Stage-1 Player-Specific Mechanism Authority Bridge V1 — Result

Date: 2026-10-18

Status: **CLOSED ALL — UTILITY SIGNAL QUALIFIED, DIRECT FALLBACK INTERVENTION FAILED**

Model: **PRISM — Player Role & Individual Situation Model**

Frozen-before-scoring contract:
`docs/research/PRISM_RB_TE_STAGE1_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_V1_FREEZE.md`

Final pre-score freeze:
`ba405d3e67495af8a812362b3401250476fff5f1`

Canonical execution:
- run `39100106463` — **SUCCESS**;
- exact head `e575aacae34d402af41755735dd78eac679a731a`;
- job `117594464277` — **SUCCESS**;
- artifact `12126431332`;
- digest `sha256:d4046c51b47b6a0a17904f5c6331ffd35d51e8e66506310fbbc7545cbea0576f`.

Overall disposition:

`PRISM_RB_TE_STAGE1_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_CLOSED_ALL`

## 1. Frozen intervention

For each RB/TE row already supported by the universal qualified Stage-1
mechanism:

- meta-history <3 -> preserve `p_candidate`;
- supported + hierarchical expected Stage-1 benefit >0 -> preserve
  `p_candidate`;
- supported + expected benefit <=0 -> revert to family `p_baseline`.

No probability was fit or blended.

Comparator:
universal Stage-1 `p_candidate`.

## 2. RB — CLOSED

2024 development:
- Brier gain vs universal Stage-1: **+0.000302**;
- log-loss gain: **+0.000841**.

2025 independent confirmation:
- Brier gain: **-0.002897**;
- log-loss gain: **-0.009637**;
- player-cluster CI low **-0.005788**;
- game-cluster CI low **-0.005570**;
- equal-player CI low **-0.004812**;
- improved / tied / worsened:
  **7.02% / 82.46% / 10.53%**;
- median named-player change: 0;
- p10 -0.002771;
- 133 rows / 25 RBs reverted to family prior;
- low-meta-support fallback gain: exactly 0.

Pooled:
- Brier gain **-0.001298**;
- player-cluster CI low -0.002982;
- game-cluster CI low -0.002843.

Multiple mandatory confirmation and pooled gates failed.

Disposition:

`RB_STAGE1_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_CLOSED`.

## 3. TE — CLOSED

2024 development:
- Brier gain **-0.000528**;
- log-loss gain **-0.001920**.

2025:
- Brier gain **-0.000590**;
- log-loss gain **-0.004311**;
- player-cluster CI low **-0.002069**;
- game-cluster CI low **-0.002609**;
- equal-player CI low **-0.001273**;
- improved / tied / worsened:
  **5.56% / 82.41% / 12.04%**;
- median 0;
- p10 -0.001173;
- 146 rows / 24 TEs reverted to family prior;
- low-meta-support fallback gain exactly 0.

Pooled:
- Brier gain **-0.000559**;
- player-cluster CI low -0.001528;
- game-cluster CI low -0.001658.

Development, confirmation and pooled gates failed.

Disposition:

`TE_STAGE1_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_CLOSED`.

## 4. Scientific meaning

This is now the second strong demonstration that:

> **Predicting whether a mechanism is likely to help a named player is not the
> same as knowing the correct action to take when it is predicted to hurt.**

The Stage-1 RB and TE utility signals remain scientifically qualified.

The direct fallback action is wrong.

Do not reinterpret the utility signal as a failed player-specific architecture.
It contains real information; the intervention did not translate that
information into a better participation probability.

## 5. Anti-rescue

Do not rescue this exact bridge with:
- soft blends between `p_candidate` and `p_baseline`;
- alternate zero threshold;
- alternate meta support;
- alternate Stage-1 history window;
- player-name/depth subsets;
- recent-only utility;
- family subgroups;
- calibration after seeing outcomes.

A future Stage-1 personalization attempt must be materially distinct.

## 6. Current player-specific architecture

Validated actual projection-level personalization:
- **RB receiving Stage-2 mechanism authority bridge QUALIFIED**.

Qualified meta-signals without a validated direct intervention:
- RB rushing Stage-2 utility;
- RB Stage-1 utility;
- TE Stage-1 utility.

Closed player-specific signals/bridges:
- TE Stage-2 utility signal;
- RB rushing Stage-2 hard authority bridge;
- QB Stage-1 utility signal;
- WR Stage-1 utility signal;
- RB Stage-1 family-fallback bridge;
- TE Stage-1 family-fallback bridge.

This market/mechanism heterogeneity must remain explicit.

## 7. Next architecture boundary

Stop using a generic on/off/fallback action as the default personalization idea.

The next player-specific lane should test a **materially distinct feature
sensitivity / mechanism-routing object**, not another threshold around the same
utility score.

Promising structural question:

> Does a named player have a stable, strictly-prior **error-component
> sensitivity fingerprint** — opportunity-driven versus efficiency-driven —
> that predicts which already-existing component should carry more explanatory
> authority for that player's exact market?

This must be anti-retested against the CLOSED generic
mechanism-instability-trust V1.

The prior closed trust test asked whether generic mechanism *instability*
adds information beyond existing opportunity/efficiency difficulty.

A new lane is legal only if it instead measures a persistent named-player
**component sensitivity / dominance** state and does not simply rename
instability/volatility or rescore the same trust target.

Perform source/anti-retest mapping before any score.

## 8. Integrity

Canonical bridge:
- production changed: **false**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- Week-6 outcomes used: **0**;
- sportsbook/OddsAPI: **0**.

Final disposition:

`PRISM_RB_TE_STAGE1_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_CLOSED_ALL`.
