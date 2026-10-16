# Player Active-Role -> Conditional Allocation Bridge V1 — Mechanism Freeze

Frozen: 2026-10-16

Status: **FROZEN BEFORE ANY BRIDGE OUTCOME SCORING. RESEARCH ONLY. NO PRODUCTION CHANGE.**

Version:
`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_MECHANISM_FREEZE`

Scientific parents:
- `PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`
- `PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY_PARTIAL`

Canonical source authority:
- run `38955368643` — SUCCESS
- exact source head `cb65a634a0e09296bfeabc94ffe1e77363a30f8a`
- artifact `12070806828`
- digest `sha256:e91a108495d7a2b6eb289e0618ab6af09e3d6cdf6895e534742a07340e4a2a83`
- result `docs/research/PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_SOURCE_RESULT.md`

PRISM principle remains binding:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

This experiment asks one narrow question:

> After estimating whether each named player will actually participate, does conditioning the already-existing canonical opportunity entitlement on that probabilistic active set improve the named player's finite opportunity distribution?

This is not a new opportunity-share model.

## Eligible cells

Only source-ready cells enter V1 scoring:
1. `QB_PASSING`
2. `RB_RUSHING`
3. `RB_RECEIVING`
4. `TE_RECEIVING`

`WR_RECEIVING` is excluded because 2025 WR-R15 OOS authority is unavailable/forbidden by its frozen contract.

A 2024 WR result may not be scored as a rescue.

## Historical chronology

Development:
- 2024 regular-season Weeks 5-18.

Independent confirmation:
- 2025 regular-season Weeks 5-18.

No 2026 outcomes.

All candidate inputs must be frozen strictly before the target game.

Target-game opportunity counts are permitted **only after candidate construction** for scoring.

## Stage-1 probability

For every true pregame player row, use the exact qualified Stage-1 probability semantics.

If the named player met the Stage-1 named-player support contract:
- use the frozen Stage-1 `p_candidate`.

Otherwise:
- use the exact strictly-prior same-season position Jeffreys fallback
  `(K_pos + 0.5) / (N_pos + 1)`,
- with the same `N_pos >= 100` rule.

No thresholding.
No calibration.
No exponent.
No shrinkage change.
No conversion of `p_active` into a direct opportunity share.

## Canonical conditional entitlement

For player i in one team-game/cell, let:

`q_i`

be the exact source-qualified canonical pregame entitlement already present in the historical model.

The bridge may not refit or replace `q_i`.

Canonical authorities remain:
- QB passing: existing pregame QB pass-attempt-share authority;
- RB rushing: canonical rushing share after the existing top-five + residual semantics;
- RB receiving: canonical target entitlement;
- TE receiving: exact recovered OOS TE-R5P entitlement overlay.

Rows with canonical `q_i = 0` remain zero-entitlement named rows in V1. The bridge is not allowed to invent positive named-player share for them.

That limitation is deliberate and must be disclosed.

## Evaluation categories and complement

Within each team-game/cell:

- explicit modeled named categories are players with finite `q_i > 0`;
- zero-entitlement named rows are not promoted to positive share;
- define the complement category as every opportunity not assigned to an explicit positive-entitlement named category.

Therefore:

`q_0 = 1 - SUM_i q_i`.

The complement can contain:
- zero-entitlement players in the same family;
- players outside the tested family;
- existing canonical residual/other-player mass.

Required:
- `q_i >= 0`;
- `q_0 >= -1e-12`;
- baseline mass gap <= `1e-12`.

Pure floating-point values inside `1e-12` of zero may be set to zero.

## Frozen active-set law

For every explicit positive-entitlement player i:

`A_i ~ Bernoulli(p_active_i)`.

The V1 joint active-state law is the product of those Bernoulli marginals:

`P(A=a) = PRODUCT_i p_i^(a_i) * (1-p_i)^(1-a_i)`.

This independence law is frozen for V1.

Do not add:
- pairwise correlation;
- depth-chart correlation;
- mutually-exclusive starter logic;
- injury buckets;
- active thresholds;
- position-specific probability transforms.

Those would be different mechanisms.

## Conditional renormalization

For one active-state vector a, define:

`Z(a) = q_0 + SUM_i a_i q_i`.

If `Z(a) > 0`:

`q_i(a) = a_i q_i / Z(a)`

and:

`q_0(a) = q_0 / Z(a)`.

Thus:
- inactive named players receive zero;
- active named players preserve their **relative canonical entitlement**;
- no new share coefficient is fit;
- finite team opportunity remains exactly conserved.

### Degenerate all-off fallback

If `Z(a) = 0`:
- every explicit named-player probability is 0;
- complement probability is exactly 1.

Interpretation:
all canonical-positive modeled players are inactive and the finite opportunity is assigned to the anonymous complement/other-player bucket.

V1 does **not** invent which zero-entitlement backup receives it.

This fallback is frozen before scoring.

## Exact mixture — no Monte Carlo

The candidate distribution must be computed by exact enumeration of active states.

No stochastic Monte Carlo is needed for the scientific score.

For a room with K explicit positive-entitlement players, enumerate all `2^K` states and their exact weights.

Required numerical checks:
- active-state weights sum to 1 within `1e-12`;
- every state probability vector sums to 1 within `1e-12`;
- no negative or non-finite probability;
- finite opportunity count N is unchanged in every state.

## Target-game scoring counts

Attach outcomes only after the complete candidate mixture is constructed.

### QB passing
- N = official team pass attempts;
- y_i = official player pass attempts.

### RB rushing
- N = official team rushing attempts;
- y_i = player rushing attempts.

### RB receiving
- N = official team targets;
- y_i = player targets.

### TE receiving
- N = official team targets;
- y_i = player targets.

For the room complement:

`y_0 = N - SUM_i y_i`.

Require:
- integer `N > 0`;
- integer `0 <= y_i <= N`;
- integer `y_0 >= 0`.

Any count identity violation is an integrity failure, not a row to silently discard.

## Primary individual-player proper score

The primary endpoint is **named-player count CRPS**.

This is intentional: PRISM is an individual-player projection model.

For each authority-covered named-player row, including canonical zero-share rows:

Baseline marginal:
`Y_i ~ Binomial(N, q_i)`.

Candidate for `q_i > 0`:
a finite mixture of Binomial count laws induced by the exact active-state mixture.

Candidate for `q_i = 0`:
identical to baseline.

Discrete CRPS:

`CRPS = SUM_{k=0}^{N} (F(k) - 1[y <= k])^2`.

Per-row improvement:

`delta_crps = CRPS_baseline - CRPS_candidate`.

Positive favors the active-role bridge.

Zero-entitlement rows remain in the primary population with exactly zero candidate delta unless the existing canonical entitlement itself changes in a future separately-qualified model.

## Mean-projection protection

For each player row calculate:

Baseline expected opportunities:
`mu_base = N * q_i`.

Candidate expected opportunities:
`mu_candidate = N * SUM_a P(A=a) * q_i(a)`.

Absolute-error improvement:

`delta_ae = |y_i - mu_base| - |y_i - mu_candidate|`.

The candidate may not earn qualification by improving distribution width while worsening the player opportunity mean.

Required mean-AE protection:
- candidate mean absolute error <= baseline in 2024;
- candidate mean absolute error <= baseline in 2025.

## Room-level allocation protection

For each team-game/cell compare expected allocation shares with realized shares.

Baseline vector:
`q`.

Candidate expected vector:
`E_A[q(A)]`.

Observed vector:
`y / N`.

Room error:
one-half L1 distance between expected and observed share vectors.

Candidate mean room error must be <= baseline in both 2024 and 2025.

This is a protection gate, not the primary proper score.

## Frozen support

Per season, after source integrity and finite target counts:

QB passing:
- >= 350 player-games;
- >= 45 players;
- >= 300 team-games;
- >= 30 evaluable players with at least 5 rows.

RB rushing:
- >= 1,200 player-games;
- >= 100 players;
- >= 400 team-games;
- >= 75 evaluable players with at least 5 rows.

RB receiving:
- >= 1,200 player-games;
- >= 100 players;
- >= 400 team-games;
- >= 75 evaluable players with at least 5 rows.

TE receiving:
- >= 1,200 player-games;
- >= 100 players;
- >= 400 team-games;
- >= 75 evaluable players with at least 5 rows.

Do not lower a failed support gate.

## Frozen inference

Bootstrap:
- 5,000 replicates;
- seed `20261016`.

2025 confirmation requires:
- mean individual `delta_crps > 0`;
- player-cluster 95% bootstrap CI lower bound > 0;
- game-cluster 95% bootstrap CI lower bound > 0;
- equal-player mean CRPS improvement > 0;
- equal-player 95% bootstrap CI lower bound > 0.

2024 development requires:
- mean individual `delta_crps > 0`.

Pooled 2024+2025 is corroboration only:
- report pooled individual CRPS improvement;
- report player-cluster and game-cluster intervals;
- pooled evidence cannot rescue a failed 2025 confirmation.

## Per-cell qualification gate

A cell earns:

`<CELL>_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED`

only if ALL are true:

1. source-ready in both 2024 and 2025;
2. frozen support passes in both seasons;
3. 2024 mean individual CRPS improvement > 0;
4. 2024 mean player expected-count AE is non-worse;
5. 2024 mean room allocation error is non-worse;
6. 2025 mean individual CRPS improvement > 0;
7. 2025 player-cluster CRPS 95% CI lower bound > 0;
8. 2025 game-cluster CRPS 95% CI lower bound > 0;
9. 2025 equal-player mean CRPS improvement > 0;
10. 2025 equal-player CRPS 95% CI lower bound > 0;
11. 2025 mean player expected-count AE is non-worse;
12. 2025 mean room allocation error is non-worse;
13. every probability/conservation/count integrity check passes;
14. no sportsbook input;
15. no OddsAPI call;
16. no 2026 outcome;
17. no fitted share coefficient;
18. no production mutation.

Otherwise the cell receives either:
- `<CELL>_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_CLOSED`, or
- `<CELL>_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_INTEGRITY_FAILURE`.

No cell may rescue another.

## Overall disposition

Possible results:
- `PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_ALL_ELIGIBLE`
- `PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_QUALIFIED_PARTIAL`
- `PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_CLOSED`
- `PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_INTEGRITY_FAILURE`

"ALL_ELIGIBLE" refers only to the four source-ready V1 cells. WR is outside this experiment.

## Mandatory diagnostics

Report, but do not use for post-result rescue:
- share of rows with named-player Stage-1 support versus position fallback;
- canonical-zero entitlement row counts;
- candidate expected opportunity shift by player;
- per-player CRPS and AE deltas;
- share of evaluable players improved;
- room-size / active-set-size distribution;
- probability mass of degenerate all-off states;
- complement expected-share shift;
- largest positive and negative named-player changes.

These diagnostics exist to understand **which individual players and role states drive the result**, not to create after-the-fact subgroups.

## Anti-rescue

After scoring, do not:
- tune `p_active`;
- threshold `p_active`;
- change prior strength or history window;
- add active-state correlation;
- add starter/depth-chart carveouts;
- change all-off fallback;
- assign complement opportunity to a chosen backup;
- fit a share multiplier;
- change canonical entitlements;
- change team opportunity volume;
- drop difficult players because they hurt the score;
- score WR using 2024 only;
- use sportsbook or market information;
- pool positions to rescue a failed cell.

A failure of this exact bridge closes this exact mechanism. A different mechanism requires genuinely new pregame information and a new pre-score freeze.

## Integrity at freeze time

At this freeze:
- bridge outcomes scored: **0**
- target-game counts attached to bridge candidate: **0**
- bridge coefficients fit: **0**
- thresholds searched: **0**
- sportsbook inputs: **0**
- OddsAPI calls: **0**
- 2026 outcomes: **0**
- production changed: **false**
