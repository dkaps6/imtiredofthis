# PRISM RB Stage-2 Player-Specific Mechanism Authority Bridge V1 — Result

Date: 2026-10-18

Status: **QUALIFIED PARTIAL — RB RECEIVING QUALIFIED / RB RUSHING CLOSED / RESEARCH ONLY**

Model: **PRISM — Player Role & Individual Situation Model**

Frozen-before-scoring contract:
`docs/research/PRISM_RB_STAGE2_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_V1_FREEZE.md`

Freeze commit:
`0831ed274161fa75a7a338041c365bda161041ee`

Canonical execution:
- run `39099474655` — **SUCCESS**;
- exact head `b0bd239e2e5ea3cf9dd6abd939e1d2b65ff1c9b3`;
- job `117592600989` — **SUCCESS**;
- artifact `12125534698`;
- digest `sha256:3d1f8043dc7a2ded0ed7fd9705b359d31a094a272ecce6958823ce660c2898e4`.

Overall disposition:

`PRISM_RB_STAGE2_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_QUALIFIED_PARTIAL`

## 1. Frozen integration

For supported RBs with the qualified hierarchical expected Stage-2 utility:

- prior positive-entitlement mechanism rows <3:
  preserve universal Stage-2 `p_active`;
- >=3 prior rows and expected benefit >0:
  preserve universal Stage-2 `p_active`;
- >=3 prior rows and expected benefit <=0:
  set `p_effective = 1.0` for Stage-2 active-set purposes only.

Canonical entitlement `q` remained unchanged.

The room was then re-enumerated under the exact finite-opportunity state law.

This was not a post-hoc player multiplier.

## 2. RB rushing — CLOSED

Disposition:

`RB_RUSHING_PLAYER_SPECIFIC_STAGE2_AUTHORITY_BRIDGE_CLOSED`

2024 development:
- personalized CRPS gain vs universal Stage-2: **-0.003521**;
- expected-count AE gain: **-0.002994**;
- room-allocation error gain: **-0.000072**.

All three central development endpoints worsened.

2025 confirmation:
- personalized CRPS gain: **-0.000780**;
- player-cluster CI low: **-0.004255**;
- game-cluster CI low: **-0.003749**;
- equal-player CI low: **-0.003188**;
- player improved / tied / worsened:
  **6.36% / 87.27% / 6.36%**;
- median named-player change: 0;
- own authority disabled: 23 rows / 5 players;
- low-support fallback-row CRPS gain: +0.001499.

Binding failed gates included:
- development CRPS;
- development AE;
- development room allocation;
- 2025 CRPS;
- player cluster;
- game cluster;
- equal-player CI;
- improved-share > worsened-share.

Interpretation:

The RB-rushing **meta-signal** is real and remains qualified, but this exact
hard mechanism-authority integration is wrong.

Knowing that the universal mechanism is more/less likely to help a named RB is
not sufficient evidence that the correct action is to make a negative-signal RB
deterministically active inside the rushing room.

This is an important distinction between:
- predicting mechanism utility; and
- knowing the correct projection intervention.

Do not rescue this bridge with:
- a soft authority coefficient;
- a different sign threshold;
- top-RB carveouts;
- alternate support floors;
- recent-only history;
- room-size subsets.

A future RB-rushing personalization hypothesis must be materially distinct.

## 3. RB receiving — QUALIFIED

Disposition:

`RB_RECEIVING_PLAYER_SPECIFIC_STAGE2_AUTHORITY_BRIDGE_QUALIFIED`

2024 development:
- personalized CRPS gain vs universal Stage-2: **+0.006563**;
- expected-count AE gain: **+0.009567**;
- room-allocation error gain: **+0.000814**.

2025 independent confirmation:
- personalized CRPS gain: **+0.008520**;
- player-cluster CI low: **+0.005441**;
- game-cluster CI low: **+0.006117**;
- equal-player CI low: **+0.005582**;
- player improved / tied / worsened:
  **78.51% / 0.00% / 21.49%**;
- median named-player CRPS improvement: **+0.000952**;
- p10 named-player improvement: **-0.000719**;
- own authority disabled:
  **496 rows / 59 named RBs**;
- low-support fallback-row CRPS gain:
  **+0.000946**.

Every frozen qualification gate passed.

Finite opportunity remained exact:
- state-weight mass conserved;
- candidate room mass conserved;
- canonical entitlement unchanged;
- low-support players retained the universal parent as their own rule.

## 4. What this proves

PRISM now has a historically validated example of a **named-player mechanism
authority rule** improving an actual player opportunity distribution beyond a
qualified position-wide mechanism.

For RB receiving:

`league/market parent -> current role prior -> named-player prior mechanism utility -> current player mechanism authority -> conserved room allocation`

worked in development and independent confirmation.

That is materially stronger than merely showing a player feature correlates
with an outcome.

It demonstrates that a named player can earn the right for the model to apply a
different rule while still using population information and preserving team
constraints.

## 5. What it does not prove

It does not authorize:
- production mutation;
- 2026 retrospective grading;
- Week-6 lock grading;
- automatic deployment to Week 7 while the prior-state source is incomplete;
- TE personalization;
- RB rushing personalization under this bridge;
- free per-player coefficients;
- sportsbook-driven rules.

The Week-7 Stage-2 readiness probe remains fail-closed until source maturity.

## 6. Architecture consequences

Canonical player-specific state now separates by exact market:

- **RB receiving**
  - mechanism-utility signal QUALIFIED;
  - conservation-safe player-specific Stage-2 authority bridge QUALIFIED.
- **RB rushing**
  - mechanism-utility signal QUALIFIED;
  - hard authority bridge CLOSED.
- **TE receiving**
  - universal Stage-2 remains QUALIFIED;
  - player-specific utility signal CLOSED on tail harm.
- **QB passing**
  - Stage-2 CLOSED.
- **WR receiving**
  - Stage-2 source-blocked.

This is the intended PRISM behavior:
same position does not imply same rule, and even the same named player need not
have the same personalization mechanism across markets.

## 7. Next research boundary

Do not rescue the closed RB-rushing bridge.

The next distinct player-specific parent should be evaluated independently.

A strong candidate is the already-qualified **Stage-1 active-role participation
mechanism**, because:
- it has legal historical OOS player rows;
- it is upstream of allocation;
- it spans QB/RB/WR/TE;
- it offers a distinct player-specific rule family for WR/TE without reopening
  source-blocked Stage-2;
- it can first be audited for strictly-prior named-player mechanism-utility
  history before any personalization score.

## 8. Integrity

Canonical bridge run:
- research projection values changed: **yes**;
- production changed: **false**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- Week-6 outcomes used: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**.

Final disposition:

`PRISM_RB_STAGE2_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_QUALIFIED_PARTIAL`.
