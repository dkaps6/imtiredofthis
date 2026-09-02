# RB STACK6N Deep-Late Persistence Qualification — Results

## Canonical execution

- Branch: `research-rb-stack6n-deep-late-persistence`
- Frozen implementation SHA: `0d324bee698b925b2f1f12bee296d62ea2cb5581`
- Workflow run: `33643086329`
- Job: `100290830131`
- Artifact: `9851638018`
- Artifact SHA256: `970c9d21995a515adf9bc602cef28ea050e49b918a14447b6ab3c1921bc9a338`
- Disposition: `DEEP_LATE_HISTORY_NOT_RETAINABLE`
- Production change: **No**
- Player recomposition authorized: **No**

## Integrity / temporal safety

- M94C rows: 544
- STACK6H rows: 544
- 2023–2025 regular-season PBP team-games: 1,632
- 2025 joined rows: 544
- W6–18 rows: 388
- Frozen occupancy MAE reproduced exactly: `5.518381962346741`
- Strict-prior coverage: 100%
- Fitted models: 0
- Feature / hyperparameter / threshold / window search: 0
- Sportsbook inputs: 0
- Target-game PBP used for candidate construction: 0
- Target-game PBP used only for labels + conditional occupancy scaffold.

## Coverage

The failure is not caused by missing historical deep-late samples:

- 150 / 388 target games (38.66%) contained deep-late offensive plays.
- 357 / 388 target rows (92.01%) had at least one deep-late play in the team's frozen prior-8-game window.
- Mean prior-8 deep-late plays: `44.55`
- Median prior-8 deep-late plays: `42`
- Mean team shrinkage weight: `0.5675`

## Conditional rate accuracy

On the 150 target games containing deep-late plays:

| Rate estimator | Game-weighted MAE | Play-weighted MAE |
|---|---:|---:|
| Parent generic trailing rate | 0.1710 | 0.1650 |
| **League deep-late context** | **0.1398** | **0.1250** |
| Team-shrunk deep-late | 0.1449 | 0.1319 |

The universal deep-late football effect is real. Adding a recent team-specific deviation does not improve it.

## Team-rush reconstruction

### All W6–18

| Arm | MAE | RMSE | Bias | Corr | MAE gain vs occupancy base |
|---|---:|---:|---:|---:|---:|
| Occupancy base | 5.5184 | 6.9688 | +0.1198 | 0.4777 | — |
| **League deep-late context** | **5.3267** | **6.7742** | -0.4400 | 0.5232 | **+0.1917** |
| Team-shrunk deep-late | 5.3694 | 6.8163 | -0.3747 | 0.5136 | +0.1490 |
| Perfect deep-late oracle | 5.1211 | 6.5450 | -0.3613 | 0.5675 | +0.3973 |

League context recovers **48.25%** of the exact deep-late oracle headroom. Team-shrunk history recovers 37.50% and is `0.0427` MAE worse than the simpler league arm.

## Original error-bin guardrails

### P3 pool overprojection >= 5

- League context gain: **+1.0540** attempts.
- Team-shrunk gain: **+0.8675**.
- Perfect deep-late oracle gain: **+1.5483**.

This confirms that late multi-score run abandonment is highly relevant to the original false-high secondary-back failure.

### P3 pool underprojection >= 5

- League context: **-0.2029 regression**.
- Team-shrunk: **-0.2003 regression**.
- Perfect deep-late oracle: only **+0.0207** gain.

This is decisive. Deep-late suppression is not the missing mechanism in the false-low games, and a blanket deep-late penalty damages them.

### Non-extreme games

- League context: `-0.0620` regression.
- Team-shrunk: `-0.0741` regression.
- Oracle itself: `-0.0108` regression.

Again, broad contraction is not safe.

## Frozen gate result

Both candidate arms passed:

- overall MAE materiality,
- oracle-headroom recovery,
- false-high improvement,
- RMSE,
- bias.

Both failed:

- false-low protection,
- non-extreme protection.

Therefore neither is retainable.

## Interpretation

STACK6N rules out a tempting but overly simple conclusion from STACK6M:

> `deep_late` cannot be converted into a blanket pregame run-abandonment penalty, and recent team-specific deep-late history does not solve the heterogeneity.

The next unresolved question is whether the frozen `deep_late` state is itself too broad. A team down 9 early in Q4 retains a materially different decision set than a team down 17+ late in Q4. Before adding richer learned predictors, the next investigation should decompose deep-late tendency by **deficit severity × remaining game time** and determine whether one urgency cell explains the false-high benefit without the false-low damage.
