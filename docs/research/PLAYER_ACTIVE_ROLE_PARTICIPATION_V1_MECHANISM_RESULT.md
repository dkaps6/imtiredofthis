# Player Active-Role Participation V1 — Mechanism Result

Date: 2026-10-16

Status: **QUALIFIED ALL POSITIONS / RESEARCH ONLY / NO PRODUCTION MUTATION**

Version:
`PLAYER_ACTIVE_ROLE_PARTICIPATION_V1_MECHANISM`

Frozen-before-scoring contract:
`docs/research/PLAYER_ACTIVE_ROLE_PARTICIPATION_V1_MECHANISM_FREEZE.md`

Pre-score freeze commit:
`16df6e95388a5dd7d682a619ea0b6f49d50f37b4`

Canonical scoring authority:
- run `38907459996` — SUCCESS;
- job `116994537005`;
- exact scored head `1c2c92ca498e8dce1eee19d39b2f276dfd37d8f7`;
- artifact `12049120604`;
- artifact digest `sha256:41b1acd3f1b8d77abb53980db160ece32f8fc40351939d4827248b9caad0c64b`.

Parent source authority:
- `PLAYER_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY_ALL`;
- run `38895205962`;
- job `116953914726`;
- exact source head `28ebf8d7159fdd1fec8c2e4a0fe1c9e2c691537c`;
- artifact `12044155086`;
- digest `sha256:a42322579aee682497223c7f53bce0b14a96dc6f466558e811770e681a035dad`.

## Overall disposition

`PLAYER_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED_ALL`

Per position:
- QB: `QB_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED`
- RB: `RB_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED`
- WR: `WR_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED`
- TE: `TE_ACTIVE_ROLE_PARTICIPATION_MECHANISM_QUALIFIED`

Every frozen qualification gate passed for every position.

## Frozen mechanism that was actually scored

Target:
- any offensive snap in the target game = 1;
- valid pregame-universe player with zero offensive snaps = 0.

Pregame universe:
weekly roster ACT + stable identity - exact official inactives, with a valid
team-week snap source.

Baseline:
strictly-prior same-season position-family participation rate with Jeffreys
smoothing.

Named-player state:
last up to four eligible same-team strictly-prior player-games, minimum three.

Candidate probability:
`(k_player + 2 * p_baseline) / (n_player + 2)`.

No parameter was fit or retuned after outcomes were seen.

2024 was development evidence.
2025 was the mandatory independent confirmation season.

Primary metric:
Brier score improvement, defined as baseline Brier minus candidate Brier.

Dependence-aware inference:
- 5,000 player-cluster bootstrap replicates;
- 5,000 game-cluster bootstrap replicates;
- 5,000 equal-player bootstrap replicates for named-player confirmation;
- seed `20261016`;
- frozen 95% percentile intervals.

## Exact 2025 independent confirmation

### QB

Support:
- 835 scored player-games;
- 78 QBs;
- 537 positive participation outcomes;
- 298 zero-snap outcomes;
- 69 named QBs with at least five scored target rows.

Performance:
- baseline Brier: 0.230613;
- candidate Brier: 0.137893;
- Brier improvement: **+0.092720**;
- log-loss improvement: **+0.220043**.

Dependence-aware Brier-improvement intervals:
- player-cluster 95% CI: **[+0.077584, +0.107311]**;
- game-cluster 95% CI: **[+0.079565, +0.105855]**.

Named-player confirmation:
- equal-player mean Brier improvement: **+0.091251**;
- equal-player 95% CI: **[+0.074804, +0.108387]**;
- 88.41% of evaluable named QBs improved.

### RB

Support:
- 1,374 scored player-games;
- 138 RBs;
- 1,242 positive outcomes;
- 132 zero-snap outcomes;
- 114 named RBs with at least five scored rows.

Performance:
- baseline Brier: 0.088063;
- candidate Brier: 0.049461;
- Brier improvement: **+0.038602**;
- log-loss improvement: **+0.143050**.

Dependence-aware intervals:
- player-cluster 95% CI: **[+0.023118, +0.057427]**;
- game-cluster 95% CI: **[+0.030158, +0.047372]**.

Named-player confirmation:
- equal-player mean Brier improvement: **+0.043869**;
- equal-player 95% CI: **[+0.023471, +0.068245]**;
- 89.47% of evaluable named RBs improved.

### WR

Support:
- 2,065 scored player-games;
- 215 WRs;
- 1,948 positive outcomes;
- 117 zero-snap outcomes;
- 180 named WRs with at least five scored rows.

Performance:
- baseline Brier: 0.053660;
- candidate Brier: 0.038094;
- Brier improvement: **+0.015566**;
- log-loss improvement: **+0.073242**.

Dependence-aware intervals:
- player-cluster 95% CI: **[+0.006899, +0.026141]**;
- game-cluster 95% CI: **[+0.010167, +0.021234]**.

Named-player confirmation:
- equal-player mean Brier improvement: **+0.015997**;
- equal-player 95% CI: **[+0.005099, +0.029972]**;
- 82.22% of evaluable named WRs improved.

### TE

Support:
- 1,286 scored player-games;
- 126 TEs;
- 1,252 positive outcomes;
- 34 zero-snap outcomes;
- 108 named TEs with at least five scored rows.

Performance:
- baseline Brier: 0.026392;
- candidate Brier: 0.020096;
- Brier improvement: **+0.006296**;
- log-loss improvement: **+0.045717**.

Dependence-aware intervals:
- player-cluster 95% CI: **[+0.001550, +0.012415]**;
- game-cluster 95% CI: **[+0.002302, +0.010758]**.

Named-player confirmation:
- equal-player mean Brier improvement: **+0.004324**;
- equal-player 95% CI: **[+0.000587, +0.009706]**;
- 88.89% of evaluable named TEs improved.

## 2024 development robustness

Every 2024 support and score gate also passed.

Brier improvement:
- QB: +0.088031;
- RB: +0.029402;
- WR: +0.013872;
- TE: +0.006523.

All 2024 player-cluster, game-cluster and equal-player 95% Brier-improvement
intervals had lower bounds above zero.

## Pooled corroboration

2024+2025 pooled Brier improvement:
- QB: +0.090366;
- RB: +0.034004;
- WR: +0.014716;
- TE: +0.006411.

Pooled player-cluster 95% CI lower bounds:
- QB: +0.078345;
- RB: +0.024024;
- WR: +0.008052;
- TE: +0.002409.

Pooled game-cluster 95% CI lower bounds:
- QB: +0.080355;
- RB: +0.028262;
- WR: +0.010900;
- TE: +0.003471.

## Named-player interpretation

This result supports the Player Projection First Principle.

A player's own recent same-team participation state contains useful pregame
information about whether **that named player** will have any offensive role at
all. The gain is not merely a position-average artifact: the mandatory 2025
equal-player test passed in all four position families.

The effect is largest for QB and RB, where the pregame roster contains more
legitimate zero-role / backup states. It remains smaller but statistically
qualified for WR and TE despite their high unconditional participation rates.

The mechanism is not perfect for every player. Some named players are worse
under the fixed V1 candidate. That heterogeneity is preserved rather than
hidden. Qualification means the player-specific state is a robust probability
signal at the population and named-player levels under the frozen test; it does
not mean every player must be forced through the same downstream action.

## Scientific meaning

Stage 1 is now scientifically qualified:

`P(any offensive snap | named player's strictly-prior same-team participation state)`

This is a separate upstream state variable.

It does **not** answer:
- how many carries the player receives;
- how many targets the player receives;
- how many routes/snaps he receives conditional on participating;
- his yards/efficiency;
- his TD probability;
- whether he should be removed from the player universe.

Those remain downstream questions.

Stage 2 must remain separate:

`conditional opportunity allocation given participation`.

## What is NOT authorized

This result does not authorize:
- production mutation;
- automatic zeroing/removal of players;
- multiplying all player projections by `p_active_role` without a separate
  bridge test;
- reopening RB Vacancy or teammate-availability role-elasticity V1;
- changing WR-R15 / TE-R5P;
- changing frozen Week-6 WR/TE trust labels;
- changing generic ATD;
- sportsbook-assisted state;
- any OddsAPI acquisition.

Any mapping from the qualified Stage-1 probability into the projection engine
must be frozen separately before scoring/deployment.

## Anti-retest

Do not rescue/retest this exact V1 with:
- alternative last-N windows;
- snap-percentage thresholds;
- starter/depth subsets;
- alternate prior strengths;
- calibrated logistic/isotonic transforms;
- position pooling;
- TD outcomes;
- target-game stat presence;
- sportsbook inputs.

The V1 question is answered.

## Integrity

Canonical scoring run:
- target-game used for pregame-universe membership: 0;
- target-game snap used as predictor: 0;
- Stage-2 opportunity inputs used: 0;
- sportsbook inputs used: 0;
- OddsAPI calls: 0;
- 2026 outcomes read: 0;
- projection values changed: 0;
- production changed: false.

## Next authorized scientific step

Preserve Stage 1 as qualified.

The repo reconciliation shows that a new generic recent-share allocator would
duplicate existing RB/WR/TE allocation science. The next distinct seam is the
bridge between the qualified probabilistic active-role state and the existing
canonical conditional allocation authorities.

Frozen next source plan:
`docs/research/PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_BRIDGE_V1_SOURCE_PLAN.md`.

Before any Stage-2 bridge scoring:
- prove historical as-of/OOS-compatible canonical entitlement authority by
  cell;
- preserve current QB/RB/WR/TE allocation science rather than fitting a new
  share mean;
- fail a cell closed if its canonical historical parent cannot be proven;
- only after source readiness freeze the active-set sampling,
  renormalization, all-off fallback, scoring and clustered inference.

Do not collapse Stage 1 and Stage 2.
