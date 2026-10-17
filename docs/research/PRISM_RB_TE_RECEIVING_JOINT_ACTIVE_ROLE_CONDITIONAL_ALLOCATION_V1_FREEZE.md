# PRISM Joint RB+TE Receiving Active-Role Conditional Allocation V1 — Mechanism Freeze

Frozen: 2026-10-17

Status: **FROZEN BEFORE ANY JOINT OUTCOME SCORING. RESEARCH ONLY. NO PRODUCTION CHANGE.**

Version:
`PRISM_RB_TE_RECEIVING_JOINT_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_V1_FREEZE`

## Why this experiment exists

The separately-qualified Stage-2 receiving cells cannot simply be applied one after another in one real team target pool.

Existing qualified parents:
- `RB_RECEIVING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED`
- `TE_RECEIVING_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED`

Each parent was scored as its own finite-allocation cell, with all non-cell target opportunity represented by a complement. Sequentially applying the two cell laws would make the result order-dependent and could adjust the same finite target mass twice.

This experiment is therefore a **composition test**, not a post-Week-5 rescue and not a new target-share mean model.

It is motivated by deployment algebra and the governing PRISM architecture:

`PLAYER IDENTITY/HISTORY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> FINITE OPPORTUNITY ALLOCATION -> OPPORTUNITY TYPE/QUALITY -> TEAM/GAME ENVIRONMENT -> PLAYER-CONDITIONED OPPONENT MATCHUP -> INDIVIDUAL EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

No Week-4/5 outcome is used to define this law.

## Immutable scientific parents

Stage 1:
- `PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`
- run `38907459996`
- artifact `12049120604`
- exact scored head `1c2c92ca498e8dce1eee19d39b2f276dfd37d8f7`

Stage-2 source:
- `PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY_PARTIAL`
- run `38955368643`
- artifact `12070806828`
- digest `sha256:e91a108495d7a2b6eb289e0618ab6af09e3d6cdf6895e534742a07340e4a2a83`
- exact head `cb65a634a0e09296bfeabc94ffe1e77363a30f8a`

Separate Stage-2 scoring parent:
- `PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL`
- run `38957575645`
- artifact `12070774378`
- digest `sha256:e07edbad1bf97a4917ba95c58156c8941448e7b2668bd0eae2e9bcad5d4945b0`
- frozen mechanism commit `87400a9c9c1db465c9c4bd70b0cb241dba27c6c9`

The separate parent results remain valid regardless of this composition result.

## Frozen chronology

Development:
- 2024 regular season, Weeks 5-18.

Mandatory independent confirmation:
- 2025 regular season, Weeks 5-18.

No 2026 outcome may be read or scored.

## Frozen player universe

For each team-game, the explicit joint receiving set contains:
- every source-qualified RB-family receiving row with finite canonical `q_i`;
- every source-qualified TE receiving row with finite canonical `q_i`.

RB family means the exact parent RB receiving family.

WRs, other positions, source-uncovered RB/TE rows, canonical residual mass, and every target not assigned to an explicit joint RB/TE row remain in **one shared complement**.

WR Stage-2 remains source-blocked and is not smuggled into this experiment.

## Canonical entitlement q

For every explicit RB/TE player i, preserve the exact source-qualified canonical target entitlement already reconstructed by the parent source audit.

RB:
- canonical target entitlement from the projection-neutral finite team target state.

TE:
- exact recovered OOS TE-R5P entitlement overlay.

No q is refit.
No q is replaced by recent target share.
No q is inferred from target-game participation.
No target-game outcome changes q.

Define one team-wide joint complement:

`q_0 = 1 - SUM_i q_i`

where i ranges over all explicit covered RB+TE receiving players in the team-game.

Required before any outcome attach:
- every q finite and >= 0;
- `q_0 >= -1e-12`;
- baseline probability mass gap <= `1e-12`.

Source-uncovered RB/TE identities stay inside the complement. Their missing q is not imputed.

## Stage-1 active-role probability

For every explicit player, use the exact already-qualified Stage-1 semantics:
- named-player `p_candidate` when support exists;
- otherwise the exact frozen strictly-prior same-season position Jeffreys fallback.

No threshold.
No calibration.
No new history window.
No depth-chart carveout.
No target-game snap/usage input.

## Frozen joint active-set law

Across **all explicit positive-q RB and TE players together** in one team-game:

`A_i ~ Bernoulli(p_active_i)`

with the same product-Bernoulli independence law as the qualified parent V1.

The law is jointly enumerated across RB and TE. This is the only new scientific object being tested.

No pairwise correlation.
No position-specific probability transform.
No mutual-exclusion starter rule.
No vacancy-transfer rule.

## Joint conditional allocation

For active-state vector a:

`Z(a) = q_0 + SUM_i a_i q_i`.

If `Z(a) > 0`:

`q_i(a) = a_i q_i / Z(a)`

and:

`q_0(a) = q_0 / Z(a)`.

Therefore:
- inactive RB/TE players receive zero;
- active RB/TE players preserve their exact canonical relative entitlement;
- WR/other/residual mass remains represented by the single complement;
- finite target opportunity is conserved exactly;
- RB and TE are composed in one state, not applied sequentially.

If `Z(a)=0`, all explicit named-player probability is zero and the complement receives probability 1.

## Exact enumeration

Enumerate all `2^K` states exactly.

Source-only preflight has already established that the preserved parent authority has at most 9 explicit positive-q RB+TE players in a team room, so exact enumeration is computationally tractable.

Required:
- state weights sum to 1 within `1e-12`;
- every conditional vector sums to 1 within `1e-12`;
- no negative/non-finite probability;
- no finite team target volume change.

## Outcomes — scoring only after candidate construction

After the complete joint candidate mixture is constructed:
- N = official team targets;
- y_i = official player targets;
- complement count `y_0 = N - SUM_i y_i`.

Target-game outcomes are scoring labels only.

## Primary scoring

For every authority-covered named RB/TE row:

Baseline:
`Y_i ~ Binomial(N, q_i)`.

Candidate:
the exact active-state mixture of Binomial count distributions.

Primary metric:
named-player discrete count CRPS.

`delta_crps = CRPS_baseline - CRPS_candidate`.

Positive favors the joint composition.

## Mean protection

Per row:
- baseline expected targets = `N*q_i`;
- candidate expected targets = `N*E[q_i(A)]`;
- `delta_ae = |y-mu_base| - |y-mu_candidate|`.

Candidate mean expected-target AE must be non-worse separately for RB and TE in both seasons.

## Joint room protection

For each team-game, compare the full vector:
- explicit covered RB targets;
- explicit covered TE targets;
- shared complement.

Room error is one-half L1 distance between expected and observed allocation vectors.

Joint room error must be non-worse in both 2024 and 2025.

## No-family-rescue qualification

This is critical.

The mechanism qualifies only if **RB receiving AND TE receiving independently pass** their frozen family gates.

RB cannot rescue TE.
TE cannot rescue RB.
A pooled joint mean cannot rescue either family.

For each family separately:

2024 development:
1. parent source support passes;
2. mean individual CRPS improvement > 0;
3. mean expected-target AE is non-worse.

2025 independent confirmation:
1. parent source support passes;
2. mean individual CRPS improvement > 0;
3. player-cluster 95% bootstrap CI lower bound > 0;
4. game-cluster 95% bootstrap CI lower bound > 0;
5. equal-player mean CRPS improvement > 0;
6. equal-player 95% bootstrap CI lower bound > 0;
7. mean expected-target AE is non-worse.

Joint room:
- mean room error non-worse in 2024;
- mean room error non-worse in 2025.

Bootstrap:
- 5,000 replicates;
- seed `20261017`.

Support floors remain the parent Stage-2 floors and may not be lowered:
- RB receiving: >=1,200 player-games, >=100 players, >=400 team rooms, >=75 evaluable players with >=5 rows per season;
- TE receiving: >=1,200 player-games, >=100 players, >=400 team rooms, >=75 evaluable players with >=5 rows per season.

## Possible dispositions

- `PRISM_RB_TE_RECEIVING_JOINT_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_QUALIFIED`
- `PRISM_RB_TE_RECEIVING_JOINT_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_CLOSED`
- `PRISM_RB_TE_RECEIVING_JOINT_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_INTEGRITY_FAILURE`

There is no partial production qualification from this V1. If either family fails, the joint composition closes.

Separate parent RB/TE Stage-2 qualifications remain unchanged.

## Anti-rescue

After scoring, do not:
- change p_active;
- threshold p_active;
- change q;
- use recent target share as replacement q;
- fit a share multiplier;
- alter the complement definition;
- add RB/TE correlation;
- add starter/depth carveouts;
- assign complement mass to a chosen successor;
- drop players or rooms because they hurt the result;
- lower support or CI gates;
- pool RB and TE to rescue a failed family;
- use 2024-only evidence to rescue 2025;
- use Week-4/5 forensic outcomes to tune this law;
- use any 2026 outcome;
- use sportsbook information or OddsAPI;
- mutate production.

A failure closes this exact joint-composition law.

## Integrity at freeze

At this freeze:
- joint outcomes scored: **0**;
- target-game target counts attached to joint candidate: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- Week-4/5 forensic outcomes used to define the law: **0**;
- 2026 outcomes: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- production changed: **false**.

This freeze does not modify or grade the protected Week-6 shadow.
