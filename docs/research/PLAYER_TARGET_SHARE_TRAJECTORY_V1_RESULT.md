# Player Target Share Trajectory V1 — Result

**STATUS: COMPLETE / PLAYER_TARGET_SHARE_TRAJECTORY_SIGNAL_CONFIRMED**

Frozen plan:
`docs/research/PLAYER_TARGET_SHARE_TRAJECTORY_V1_PLAN.md`

Authority:
- branch `research-player-target-share-trajectory-v1`
- run `37638269235` — SUCCESS
- source SHA `2ab9dea0e0260cb652c5c13d336e3a5ad3738ddc`
- artifact `11490707699`
- digest `sha256:2f032cb5e68f5e097e30ca56a2ebe30e505f90233bc60774e4318158fad3fafd`

Final disposition:

`PLAYER_TARGET_SHARE_TRAJECTORY_SIGNAL_CONFIRMED`

## Design

For each individual WR/TE target player-game:

- use only same-season, same-team games strictly before the target game;
- require at least four prior team games;
- RECENT2 = latest two completed team games;
- EARLIER = every earlier completed same-season team game;
- `recent2_share = player recent2 targets / team recent2 targets`;
- `earlier_share = player earlier targets / team earlier targets`;
- `trajectory_delta = recent2_share - earlier_share`.

The feature is raw football usage state. It does not contain model residuals, sportsbook data, target-game plays, or fitted coefficients.

Frozen hypothesis:

> A rising individual target-share trajectory should expose lag in a season-average/participation-based entitlement system and therefore associate with more negative opportunity error (predicted minus actual).

Expected sign: **negative**.

## Integrity and support

Identity mapping:
- WR 100%
- TE 100%

Panel:
- 5,706 scoreable player-games
- 265 WR identities
- 165 TE identities
- same/future feature violations: 0
- target-game feature rows: 0
- sportsbook inputs: 0
- fitted candidate models: 0
- 2026 outcomes: 0
- production changes: 0

## WR replication

### 2023
- 1,591 rows / 196 WRs
- rho = **-0.03999**

### 2024
- 1,619 rows / 213 WRs
- rho = **-0.10773**

### Pooled WR
- 3,210 rows / 265 WRs
- rho = **-0.07646**
- player-cluster bootstrap P(rho < 0) = **1.000**
- 95% CI = **[-0.1163, -0.0345]**

WR trajectory is materially live:
- 60.34% of rows differ by >=3 target-share points between recent2 and earlier state;
- 41.03% differ by >=5 points.

Pooled mean opportunity error:
- rising players: **-9.36 yards-equivalent**
- falling players: **-7.05**
- exact-flat players: **+0.76**

The important scientific evidence is the continuous replicated negative association, not these descriptive buckets.

## TE replication

### 2024
- 844 rows / 115 TEs
- rho = **-0.10032**

### 2025
- 865 rows / 123 TEs
- rho = **-0.06948**

### Pooled TE
- 2,496 rows / 165 TEs
- rho = **-0.09876**
- player-cluster bootstrap P(rho < 0) = **0.9998**
- 95% CI = **[-0.1472, -0.0490]**

TE trajectory is also common:
- 48.80% of rows differ by >=3 target-share points;
- 30.29% differ by >=5 points.

Pooled mean opportunity error:
- rising players: **-8.32 yards-equivalent**
- falling players: **-6.83**
- exact-flat players: **+1.63**

Again, the confirmation comes from the continuous replicated relationship.

## Combined WR + TE

- 5,706 rows / 430 position-player identities
- rho = **-0.08496**
- cluster-bootstrap P(rho < 0) = **1.000**
- 95% CI = **[-0.1174, -0.0536]**

Absolute trajectory >=3 points:
- **55.29%** of rows

Absolute trajectory >=5 points:
- **36.33%**

All frozen confirmation gates passed.

## Scientific interpretation

This is direct support for the player-centric hypothesis.

The promoted model already knows:
- the player's identity/history;
- season-level/current target-share state upstream;
- M38 WR hierarchy;
- WR-R15 / TE-R5P participation and snap continuity.

But it does not explicitly represent the **direction of recent individual target-share movement**.

Two players can have similar season-to-date target shares while having very different current role trajectories:
- one gaining target share over his latest two games;
- one losing it.

The confirmed signal says that compression matters. Rising trajectory is associated with the promoted entitlement system lagging low on individual opportunity.

This is not a generic WR or TE adjustment. It is an individual-player current-role signal, and it replicated across both promoted position architectures.

## What this does NOT authorize

Do not directly add `trajectory_delta` to target share yet.

No coefficient, window, cap, or threshold has been validated for integration.

Do not:
- search recent1/recent3/recent4;
- create rising-only or falling-only routing;
- rescue position subgroups;
- use residual history as a feature;
- change team pass volume;
- break M38 / WR-R15 / TE-R5P room conservation.

## Next authorized step

Freeze a separate integration candidate using the **exact production entitlement trace**.

That candidate must:
1. preserve total modeled team target mass exactly;
2. preserve all protected QB/team-volume science;
3. operate only inside the existing WR/TE entitlement structure;
4. use only strictly-prior trajectory state;
5. predeclare one transformation before scoring;
6. compare against exact promoted M38/WR-R15/TE-R5P baseline;
7. prove out-of-sample target and receiving-yard non-harm/improvement before any production promotion.

No production change is authorized by this result.
