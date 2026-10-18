# NFL PRISM Handoff — 2026-10-18 — Player-Specific Rule Learning First Breakthrough

## Canonical model principle

PRISM is an **individual-player projection model**.

> **The player should have the rules. The position should not have the player.**

Hierarchy:

`league prior -> role/archetype prior -> individual player evidence -> current player state`.

Individualization is hierarchical/shrunk, strictly prior, market-specific and
support-gated. Position/role is a prior/fallback, not the permanent final rule.

## Live research branch

Branch:
`research-individual-opportunity-roadmap-2026-10-08`

Branch head at this handoff write:
`0e014fd66eaaa3d70a4a7011e168388143acdc4e`

PR #672:
OPEN / DRAFT.

No paid OddsAPI was used.
No production model was changed.

## Player-specific architecture audit

Canonical map:
`docs/research/PRISM_PLAYER_SPECIFIC_RULE_LEARNING_V1_ANTI_RETEST_SOURCE_SUPPORT_MAP.md`

Key decision:
prefer **player-specific mechanism eligibility / authority** before free
player-specific coefficients.

Do not reopen closed mechanisms under a player-specific wrapper.

## Stage-2 player mechanism-benefit history — SOURCE READY ALL

Source audit:
- run `39098772787` — SUCCESS;
- exact head `744b2e920592803937e8a8c94e7a0c89d6ebd4eb`;
- artifact `12126215015`;
- digest `sha256:5249f41f72a248443c8a596de2a3cf4211928369b8867b3f3a4d16e70601c78d`;
- result:
  `docs/research/PRISM_STAGE2_PLAYER_MECHANISM_BENEFIT_HISTORY_V1_SOURCE_RESULT.md`.

Disposition:
`PRISM_STAGE2_PLAYER_MECHANISM_BENEFIT_HISTORY_SOURCE_READY_ALL`.

2025 support:
- RB rushing: 1,278 rows / 142 players with >=3 prior OOS mechanism rows;
- RB receiving: 1,278 / 142;
- TE receiving: 1,215 / 127.

Chronology violations = 0.
Stable identity = 100%.

## Hierarchical future mechanism-utility signal — QUALIFIED PARTIAL

Canonical signal:
- run `39099164620` — SUCCESS;
- exact head `36b34a1b392d4d6196b72116b0ef5b88273906bd`;
- artifact `12125359950`;
- digest `sha256:efa919e2590e7409e1d989b59cafe293d6d4831dfb434635df8fb5d15143e95b`;
- result:
  `docs/research/PRISM_STAGE2_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_V1_RESULT.md`.

Frozen hierarchy:
- role prior = strictly-prior other-player benefit in current canonical
  entitlement-rank state;
- player evidence = strictly-prior same-player same-cell Stage-2 benefit;
- canonical-zero rows excluded as structural no-action ties;
- minimum 3 positive-entitlement prior mechanism rows;
- exactly 3 role-prior pseudo-observations;
- no fit / no threshold search;
- natural zero-benefit sign boundary only.

Dispositions:
- RB rushing: **QUALIFIED**;
- RB receiving: **QUALIFIED**;
- TE receiving: **CLOSED** on the frozen 2025 named-player p10 tail-harm gate.

Do not rescue TE.
Universal TE Stage-2 remains qualified.

Important 2025 signal results:
- RB rushing meta-AE gain +0.099254; player CI low +0.066250;
- RB receiving +0.019477; player CI low +0.013832;
- TE +0.006943 centrally, but p10 -0.012228 missed tail floor -0.010402.

## First projection-level player-specific authority bridge — QUALIFIED PARTIAL

Canonical bridge:
- run `39099474655` — SUCCESS;
- exact head `b0bd239e2e5ea3cf9dd6abd939e1d2b65ff1c9b3`;
- artifact `12125534698`;
- digest `sha256:3d1f8043dc7a2ded0ed7fd9705b359d31a094a272ecce6958823ce660c2898e4`;
- result:
  `docs/research/PRISM_RB_STAGE2_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_V1_RESULT.md`.

Frozen intervention:
- low support (<3 prior positive-entitlement mechanism rows): preserve universal Stage-2;
- supported + predicted Stage-2 benefit >0: preserve universal Stage-2;
- supported + predicted benefit <=0: set that RB's Stage-2
  `p_effective = 1.0`;
- keep canonical `q_i`;
- re-enumerate the whole room under exact finite-opportunity conservation.

This acts inside the room system, not as a post-hoc player multiplier.

### RB receiving — QUALIFIED

2025:
- CRPS gain vs universal Stage-2 **+0.008520**;
- player-cluster CI low **+0.005441**;
- game-cluster CI low **+0.006117**;
- equal-player CI low **+0.005582**;
- named-player improved / tied / worsened:
  **78.51% / 0.00% / 21.49%**;
- median player improvement +0.000952;
- 496 rows / 59 RBs had their own Stage-2 authority disabled;
- low-support fallback rows also improved +0.000946.

All frozen gates passed.

This is the first historically validated PRISM example where a named player
earns a different projection rule than the universal position-level mechanism
while finite opportunity remains exact.

### RB rushing — CLOSED

The utility signal was qualified, but the exact hard authority intervention was
not.

2024 development:
- CRPS -0.003521;
- AE -0.002994;
- room allocation -0.000072.

2025:
- CRPS -0.000780;
- player CI low -0.004255;
- game CI low -0.003749;
- equal-player CI low -0.003188.

Do **not** rescue with:
- soft authority;
- different zero threshold;
- support changes;
- top-RB subsets;
- recent-only benefit;
- room-size subsets.

A future RB-rushing player rule must be materially distinct.

## Current market-specific player architecture

- RB receiving:
  - universal Stage-2 QUALIFIED;
  - player mechanism-utility signal QUALIFIED;
  - player-specific conservation-safe authority bridge **QUALIFIED**.
- RB rushing:
  - universal Stage-2 QUALIFIED;
  - mechanism-utility signal QUALIFIED;
  - hard authority bridge **CLOSED**.
- TE receiving:
  - universal Stage-2 QUALIFIED;
  - player-specific mechanism-utility signal **CLOSED** on tail harm.
- QB passing:
  - Stage-2 CLOSED.
- WR receiving:
  - Stage-2 source-blocked.

This heterogeneity is a feature of the new architecture, not a problem to smooth
away.

## Protected prior state

Still preserve:
- extended Week-4/5 player-first replay COMPLETE — do not rerun;
- component cancellation proven;
- finite opportunity conservation mandatory;
- Week-7 Stage-2 readiness run `39093471632` fail-closed until legal Week-6
  prior-state source matures;
- same-QB allocation CLOSED;
- QB-regime target-depth CLOSED;
- QB-regime target-CPOE CLOSED;
- RB NGS efficiency SOURCE BLOCKED;
- player high-value conversion identity CLOSED ALL;
- RB short-yardage carry-type identity CLOSED;
- raw recent-share replacement CLOSED;
- Room Competition State SOURCE BLOCKED;
- RB/TE depth-role transition SOURCE BLOCKED;
- generic DvP CLOSED;
- RB run-location matchup CLOSED;
- WR downstream target-quality chain CLOSED;
- conventional RB receiving efficiency transforms CLOSED;
- TE target-quality efficiency CLOSED;
- mechanism-instability / efficiency-volatility trust rescues CLOSED;
- Week-6 locks immutable/ungraded;
- Weeks 7-10 WR dual prospective protected;
- no sportsbook upstream;
- no paid OddsAPI without explicit user authorization;
- no production mutation from research unless separately authorized.

## Exact next research action

Do **not** rescue RB rushing.

Open the next materially distinct player-specific parent:
**Stage-1 Active-Role Participation**.

Why:
- Stage-1 is already qualified across QB/RB/WR/TE;
- it has immutable row-level OOS player authority;
- it is upstream of Stage-2;
- it offers a distinct individualization path for WR/TE without reopening the
  blocked/closed Stage-2 families.

First perform a source-only mechanism-benefit-history audit from canonical
Stage-1 run `38907459996` / artifact `12049120604`.

The artifact contains 11,166 OOS player rows with:
- player ID;
- season/week/game/team/family;
- binary participation outcome;
- position/family prior;
- player prior support;
- baseline participation probability;
- player-specific candidate participation probability.

Row-level mechanism utility can be reconstructed as:
`delta_brier = (y-p_baseline)^2 - (y-p_candidate)^2`.

Source-only support inspection already shows >=3 prior mechanism rows in 2025:
- QB 781 rows / 76 players;
- RB 1,244 / 129;
- WR 1,889 / 209;
- TE 1,207 / 119;
- chronology violations 0.

Freeze source gates before any predictive personalization score.

If source-ready, test whether prior same-player Stage-1 mechanism utility
predicts future utility beyond the family/role prior, with named-player tail
protection.

Do not assume the eventual integration is the same hard Stage-2 authority rule.

## Continuity rule

Next chat must:
1. read `AGENTS.md`;
2. read only the newest TOP checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`;
3. read this handoff;
4. reconcile newest Issue #535 comments, branch/PR #672 and exact-head Actions;
5. continue Stage-1 source audit if nothing newer supersedes it.

Do not restart completed work.
