# Player Individualization Audit V1 — Result

**STATUS: COMPLETE / PLAYER_INDIVIDUALIZATION_PARTIAL**

Frozen plan:
`docs/research/PLAYER_INDIVIDUALIZATION_AUDIT_V1_PLAN.md`

Authority:
- branch: `research-player-individualization-audit-v1`
- run: `37558838148` — **SUCCESS**
- source SHA: `8aa62c1947caa3f45781f94857d5335eeb373b06`
- artifact: `11455757608`
- digest: `sha256:0d60c03a6d3aebdaf1cce47472c82988defd6320e0106a07418b567ea8071a56`
- audit universe: 2025 Weeks 2-18
- distinct players traced: 600
- player-week-metric rows: 40,464
- sportsbook inputs: 0
- candidate models fit: 0
- 2026 outcomes read: 0
- production changed: false

Final disposition:

`PLAYER_INDIVIDUALIZATION_PARTIAL`

## What is already genuinely player-specific

The current stack is not a positional-average-only model.

PlayerForm uses stable `player_identity_key` history. Prior-season and current-season evidence are separately retained for:
- target share;
- rush share;
- receptions per target;
- YPT;
- YPC;
- YPA;
- YPRR / route rate when a real source exists.

Bayesian v2 creates one posterior row per active player and production rules preferentially consume those player-specific posterior fields.

Therefore existing player identity/history infrastructure should be **preserved**.

## Where position pooling remains material

Bayesian v2 explicitly combines:

`position population prior + prior-player evidence + current-player evidence`

using fixed population prior strengths and capped prior-player evidence.

For established players around Week 5, the historical audit shows typical player-specific weight is already a majority, but positional pooling remains material.

Representative Week-5 medians:

### WR
- target share: group 23.1%, player-specific 76.9%, current-season 30.8%
- receptions/target: group 25.0%, player-specific 75.0%, current-season 25.0%
- YPT: group 29.4%, player-specific 70.6%, current-season 23.5%

### RB
- rush share: group 23.1%, player-specific 76.9%, current-season 30.8%
- target share: group 23.1%, player-specific 76.9%, current-season 30.8%
- receptions/target: group 25.0%, player-specific 75.0%, current-season 25.0%
- YPC: group 29.4%, player-specific 70.6%, current-season 23.5%
- YPT: group 29.4%, player-specific 70.6%, current-season 23.5%

### TE
- target share: group 25.0%, player-specific 75.0%, current-season 25.0%
- receptions/target: group 26.7%, player-specific 73.3%, current-season 20.0%
- YPT: group 31.25%, player-specific 68.75%, current-season 18.75%

The exact percentages vary with available player history and games played.

Important implication: the stack is player-specific, but **early-season current-player evidence is not dominant**. A large fraction of the individual component is still prior-season history, while 23-31% of common metrics can remain direct position-population shrinkage around Week 5.

For an established player, current-season evidence alone does not exceed the population-prior weight until approximately:
- 4 games for target share / rush share / route rate if available;
- 5 games for receptions per target / YPRR if available;
- 6 games for YPT / YPC;
- 7 games for YPA.

For YPT/YPC, the population prior does not mechanically fall below 25% until about 8 current games. None of the current fixed-strength established-player curves reaches <10% position pooling during a normal 17-game current-season evidence range.

## Missing player-specific source fields

The public weekly player source does not supply a real routes field in this historical authority.

Accordingly route-rate/YPRR-related rows can fall back entirely to population information rather than genuine individual current/prior route evidence.

This is a source limitation, not evidence that the rest of PlayerForm is non-individualized. It should not be interpreted as a reason to invent routes.

## Where individualization is lost downstream

The audit classified the current production transformations.

Genuine player-specific foundations:
- stable historical identity;
- player prior/current opportunity and efficiency metrics;
- injury-vacancy redistribution uses individual posterior target shares.

But several major weekly environment transformations are shared:

1. team plays/pass/rush environment is team-specific and shared by teammates;
2. target matchup adjustment is a WR1 / WR1.5 / SLOT / TE / RB receiving bucket;
3. receiving efficiency gets the same team/opponent `pass_eff_mult` for teammates;
4. rushing efficiency gets the same team/opponent `rb_rush_eff_mult` for teammates;
5. matchup volatility is shared from the team/opponent context.

In particular, if two TEs have different personal YPT baselines, they begin differently, but the opponent receiving-efficiency response is still largely the same multiplier for both. The model does not generally learn that Player A historically responds differently than Player B to the same type of defensive environment.

The main existing true player-by-environment interaction found was injury-vacancy redistribution, where player-specific posterior shares determine redistribution inside a team availability event.

## Aggregate evaluation hides substantial player heterogeneity

Historical projections are scored at the player-game level, but promotion decisions are often summarized by position/market cohorts.

For players with at least eight scored 2025 games:

### WR receiving yards
- aggregate MAE: 21.61
- per-player MAE 10th percentile: 12.87
- median: 19.67
- 90th percentile: 35.53
- 38.8% of qualified WRs had signed bias opposite the aggregate WR bias direction.

### RB rushing yards
- aggregate MAE: 20.57
- player MAE P10: 8.96
- median: 20.29
- P90: 33.65
- 26.2% of qualified RBs had bias opposite the aggregate direction.

### RB receiving yards
- aggregate MAE: 10.97
- player MAE P10: 6.90
- median: 9.62
- P90: 16.60
- 46.4% of qualified RBs had bias opposite the aggregate direction.

### RB receptions
- aggregate MAE: 1.116
- player MAE P10: 0.659
- median: 1.025
- P90: 1.776
- 54.8% of qualified RBs had bias opposite the aggregate direction.

### TE receiving yards
- aggregate MAE: 15.64
- player MAE P10: 8.28
- median: 16.25
- P90: 25.04
- 26.8% of qualified TEs had bias opposite the aggregate direction.

### QB passing yards
- aggregate MAE: 61.60
- player MAE P10: 46.28
- median: 55.43
- P90: 72.32
- 42.3% of qualified QBs had bias opposite the aggregate direction.

This confirms that a position-level average can conceal materially different persistent player error profiles.

## Scientific interpretation

The current system is best described as:

`player-level outputs + meaningful individual history + positional shrinkage + mostly shared weekly environment transforms`

not as either extreme:
- not “all players in a position are treated the same”;
- not “every player has a fully individualized response model.”

That distinction matters.

The audit supports the user's proposed direction: stronger player-level modeling should be built as a **complementary layer**, preserving the existing validated stack rather than replacing it.

A future player-state candidate should therefore begin from the current production baseline and ask whether strictly-prior individualized information can explain residual differences among players exposed to similar team/opponent/role conditions.

Position remains useful as a prior / descriptor, especially for sparse-history players, but should not be assumed to define how every established player responds to the same weekly environment.

## Next authorized step

Freeze a separate complementary Player State V1 candidate plan.

Required design principles:
1. current production projection remains the baseline;
2. existing team, opponent, usage, injury, matchup, conservation, Monte Carlo, and distribution science remains intact;
3. player identity is the modeling grain;
4. use strictly-prior individual history;
5. learn player-specific state / conditional response rather than another universal position correction;
6. position can act as regularization/fallback, not erase player identity;
7. evaluate out of sample at both player-game and player-level heterogeneity grains;
8. no sportsbook input upstream;
9. no production change unless independently confirmed.

No current production authority is invalidated by this audit.
