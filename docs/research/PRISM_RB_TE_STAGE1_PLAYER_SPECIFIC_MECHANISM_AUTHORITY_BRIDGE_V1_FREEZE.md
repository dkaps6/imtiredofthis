# PRISM RB/TE Stage-1 Player-Specific Mechanism Authority Bridge V1 — Frozen Before Scoring

Date: 2026-10-18

Status: **FROZEN BEFORE BRIDGE SCORING / RB + TE ONLY / RESEARCH ONLY**

Model: **PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_RB_TE_STAGE1_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_V1`

Scientific parent:
`PRISM_STAGE1_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_QUALIFIED_PARTIAL`

Signal authority:
- run `39099930673` — SUCCESS;
- exact head `2e61f48ceace967c2533765eeb631c0f421617e9`;
- artifact `12126251820`;
- digest `sha256:9ef8c1c7e07fd4be381a745b5eb96d5ed976809d88aff478e623a5d74794f519`.

Universal Stage-1 parent:
- run `38907459996` — SUCCESS;
- artifact `12049120604`;
- digest `sha256:41b1acd3f1b8d77abb53980db160ece32f8fc40351939d4827248b9caad0c64b`.

## 1. Frozen question

For RB and TE separately:

> Can a supported named player's prior OOS Stage-1 mechanism utility determine
> whether PRISM should use his qualified player-identity participation
> probability or fall back to the broader family participation prior?

This is a direct player-specific authority decision over an already-qualified
Stage-1 mechanism.

It does not refit participation probabilities.

## 2. Eligible families

Score:
- RB;
- TE.

Do not score:
- QB — player utility signal V1 CLOSED;
- WR — player utility signal V1 CLOSED.

No pooling may rescue a failed eligible family.

## 3. Frozen authority rule

For each target Stage-1-supported row:

### Low mechanism-history support

If:

`player_prior_mechanism_rows < 3`

then:

`p_personalized = p_candidate`.

The player has not earned a deviation from the qualified universal Stage-1
player rule.

### Supported player with positive expected Stage-1 utility

If:

`player_prior_mechanism_rows >= 3`

and:

`hierarchical_expected_benefit > 0`

then:

`p_personalized = p_candidate`.

The player retains the qualified player-identity participation rule.

### Supported player with nonpositive expected Stage-1 utility

If:

`player_prior_mechanism_rows >= 3`

and:

`hierarchical_expected_benefit <= 0`

then:

`p_personalized = p_baseline`.

The named player falls back to the broader family participation prior for that
target row.

No player is removed from the pregame universe.

No probability is set to zero or one.

No alternate boundary is tested.

## 4. Predictor authority

Use only immutable pregame meta-state from the canonical signal artifact:

- player identity / family / target game;
- `player_prior_mechanism_rows`;
- `hierarchical_expected_benefit`;
- parent `p_baseline`;
- parent `p_candidate`.

The target row's `delta_brier` may not determine its authority state.

## 5. Comparator

Baseline for bridge evaluation:

**universal qualified Stage-1 `p_candidate`**.

Candidate:

`p_personalized` from the frozen authority rule.

The original family `p_baseline` is only the fallback probability for a
supported negative-utility player.

## 6. Evaluation population

Score every RB/TE row in the immutable Stage-1 mechanism artifact.

These are already Stage-1-supported rows under the parent mechanism.

Low meta-support rows remain in the evaluation as explicit universal Stage-1
fallback rows.

Seasons:
- 2024 development;
- 2025 independent confirmation.

## 7. Frozen endpoints

Primary:

`delta_personalized_brier = Brier_universal_stage1 - Brier_personalized`.

Positive is better.

Guardrail:

`delta_personalized_logloss = LogLoss_universal_stage1 - LogLoss_personalized`.

Positive is better.

Report:
- changed-authority rows;
- affected named players;
- unchanged low-support rows;
- observed participation rate;
- mean universal probability;
- mean personalized probability.

## 8. Support floors

Use the original Stage-1 parent support population.

RB, each season:
- >=800 scored rows;
- >=80 distinct RBs;
- >=500 positives;
- >=75 zeros;
- >=55 evaluable named players with >=5 rows.

TE, each season:
- >=700 scored rows;
- >=75 distinct TEs;
- >=550 positives;
- >=25 zeros;
- >=50 evaluable named players with >=5 rows.

At least 28 teams.

Changed-authority counts are diagnostics, not tunable support gates.

## 9. Player-aware inference

Frozen:
- 5,000 replicates;
- seed `20261018`;
- player cluster;
- game cluster;
- equal-player.

Equal-player requires >=5 scored target rows.

Report:
- mean player Brier improvement;
- median;
- improved / tied / worsened player shares;
- p10/p25/p50/p75/p90;
- worst named-player mean change;
- changed-authority-player-only diagnostics;
- low-meta-support fallback diagnostics.

## 10. Frozen qualification gates

A family qualifies only if all pass.

### 2024 development
1. support passes;
2. mean `delta_personalized_brier > 0`;
3. mean `delta_personalized_logloss >= 0`.

### 2025 independent confirmation
4. support passes;
5. mean Brier improvement >0;
6. mean log-loss improvement >=0;
7. player-cluster Brier-improvement CI lower bound >0;
8. game-cluster CI lower bound >0;
9. equal-player mean Brier improvement >0;
10. equal-player CI lower bound >0;
11. median named-player improvement >=0;
12. improved-player share > worsened-player share;
13. low-meta-support fallback mean Brier improvement >=0;
14. tail guard:
`p10_named_player_improvement >= -0.25 * universal_stage1_equal_player_brier`.

### pooled corroboration
15. pooled 2024+2025 mean Brier improvement >0;
16. pooled player-cluster CI lower bound >0;
17. pooled game-cluster CI lower bound >0.

No pooled result may rescue a failed 2025 gate.

## 11. If qualified

Qualification establishes a player-specific Stage-1 authority rule for that
exact family in historical OOS participation prediction.

It still does not authorize production or automatic downstream prop mutation.

A separate downstream bridge must test how the personalized Stage-1 probability
interacts with existing finite opportunity allocation.

For RB receiving, that later downstream test may combine:
- personalized Stage-1 authority;
- already-qualified RB-receiving player-specific Stage-2 authority.

That combined end-to-end test must be frozen separately.

## 12. If closed

Do not rescue with:
- alternate sign threshold;
- alternate meta support;
- soft blends;
- recent-only utility;
- player-name or depth subsets;
- alternate Stage-1 history window;
- sportsbook or 2026 outcomes.

A new attempt must be materially distinct.

## 13. Protected boundaries

- QB remains universal Stage-1.
- WR remains universal Stage-1.
- RB-rushing Stage-2 hard authority bridge remains CLOSED.
- TE Stage-2 player-specific utility signal remains CLOSED.
- universal Stage-2 parent statuses remain unchanged.
- Week-6 locks remain ungraded.
- no paid OddsAPI.
- no production mutation.

## 14. Integrity at freeze

- personalized Stage-1 rows scored: **0**;
- participation probabilities changed: **0**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- Week-6 outcomes used: **0**;
- sportsbook/OddsAPI: **0**;
- production changed: **false**.

Disposition:

`PRISM_RB_TE_STAGE1_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_V1_FROZEN_BEFORE_SCORING`.
