# RB STACK6M Trail-Context Shapley — Results

## Canonical execution

- Branch: `research-rb-stack6m-trail-context-shapley`
- Frozen implementation SHA: `145ff331d8077d850b215b1b477e9802f2a9f74e`
- Workflow run: `33639765082`
- Job: `100279590883`
- Artifact: `9850307719`
- Artifact SHA256: `42feec62a30ac3256b8008dc2efcd49a61ee6c638aba7b4872c1bea91ca54871`
- Disposition: `DEEP_LATE_DOMINANT`
- Production change authorized: **No**
- Predictive model authorized: **No**

## Integrity

The frozen attribution reproduced all required parent identities before interpretation:

- M94C rows: 544
- 2025 PBP team-games: 544
- Joined rows: 544
- W6–18 rows: 388
- Reconstructed trailing share max absolute difference: `1.11e-16`
- Sum of four trailing contexts vs parent trailing share max absolute difference: `1.11e-16`
- Empty-context MAE expected/observed: `5.518381962346741`
- All-context MAE expected/observed: `4.503012474954635`
- Direct trailing-tendency recovery expected/observed: `1.015369487392106`
- Shapley sum: `1.015369487392106`
- Shapley sum absolute error: `0.0`
- Fitted models: 0
- Feature / hyperparameter / threshold search: 0
- Sportsbook inputs: 0
- Target-game PBP: oracle grading only

## Frozen context definitions

Trailing plays were partitioned into four football-natural contexts:

1. `close_early`: one-possession deficit, Q1–Q3
2. `deep_early`: multi-score deficit, Q1–Q3
3. `close_late`: one-possession deficit, Q4+
4. `deep_late`: multi-score deficit, Q4+

## Observed context behavior

| Context | Team-games with plays | Mean plays / team-game | Total plays | Total rushes | Aggregate rush rate |
|---|---:|---:|---:|---:|---:|
| close_early | 215 | 7.6186 | 2,956 | 1,292 | 0.4371 |
| deep_early | 151 | 7.2345 | 2,807 | 1,103 | 0.3929 |
| close_late | 95 | 2.3763 | 922 | 293 | 0.3178 |
| deep_late | 150 | 5.4974 | 2,133 | 519 | 0.2433 |

The realized run rate falls monotonically as trailing context becomes more urgent, with deep-late multi-score deficits producing the strongest run abandonment.

## Order-independent Shapley attribution

### All W6–18

| Context | Shapley MAE recovery | Share of trailing recovery |
|---|---:|---:|
| close_early | 0.233298 | 22.98% |
| deep_early | 0.253060 | 24.92% |
| close_late | 0.106825 | 10.52% |
| **deep_late** | **0.422186** | **41.58%** |

### P3 team-pool overprojection >= 5 carries

| Context | Shapley MAE recovery | Share |
|---|---:|---:|
| close_early | 0.075149 | 3.22% |
| deep_early | 0.486954 | 20.90% |
| close_late | 0.223960 | 9.61% |
| **deep_late** | **1.544327** | **66.27%** |

This is the central STACK6M finding. The original false-high secondary-back problem is strongly connected to games where the offense ultimately reaches a **late multi-score deficit and abandons the run much more aggressively than M94C's coarse trailing-state tendency anticipates**.

### P3 team-pool underprojection >= 5 carries

The asymmetry is important:

| Context | Shapley MAE recovery | Share |
|---|---:|---:|
| **close_early** | **0.743903** | **63.05%** |
| deep_early | 0.358807 | 30.41% |
| close_late | 0.054574 | 4.63% |
| deep_late | 0.022534 | 1.91% |

Therefore the evidence does **not** support a blanket trailing-state contraction. False-high and false-low games arise from different trailing-context failures.

## Interpretation

The validated lineage is now:

`P3 RB carry-pool error -> total team rushing error -> effective rush-rate error -> within-state rushing-tendency error -> trailing-state tendency -> late multi-score-deficit run abandonment for false-high pools.`

STACK6M identifies a high-value football mechanism but does not establish a usable pregame predictor. The next research question must therefore be whether the deep-late exposure/tendency mechanism can be recognized with timestamp-safe pregame information. No production correction is authorized from target-game context labels themselves.

## Next required qualification

Before fitting another player-level correction, decompose the deep-late mechanism into two pregame-relevant questions:

1. **Exposure:** can we anticipate which offenses are at meaningful risk of spending offensive plays in a late multi-score deficit?
2. **Conditional tendency:** conditional on that context, can we anticipate how aggressively a team continues or abandons the run using only prior games and current pregame football state?

The next protocol must be frozen before evaluating any 2025 target outcomes, keep sportsbook data downstream-only, and qualify the team-level mechanism before any RB recomposition.
