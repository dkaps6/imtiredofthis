# RB STACK6O Deep-Late Urgency Shapley — Results

## Canonical execution

- Branch: `research-rb-stack6o-deep-late-urgency-shapley`
- Frozen implementation SHA: `b66b1b39fed6668927008007131df71516b3de6c`
- Workflow run: `33643481084`
- Job: `100292168045`
- Artifact: `9851790773`
- Artifact SHA256: `5419ec7e697bb22c418bd58115b954d7568777623d3071fe87caa11eb95b4719`
- Disposition: `DEEP_LATE_URGENCY_DISTRIBUTED`
- Production change: **No**
- Player recomposition authorized: **No**
- Predictive model authorized: **No**

## Integrity

The frozen oracle identities passed exactly before interpretation:

- W6–18 rows: 388
- Occupancy baseline MAE: `5.518381962346741`
- Full deep-late four-cell oracle MAE: `5.121110810459461`
- Deep-late recoverable MAE headroom: `0.39727115188728046`
- Shapley sum: exactly `0.39727115188728046`
- Four urgency-cell shares reproduced the parent deep-late share within numerical tolerance.
- Fitted models / feature search / hyperparameter search / threshold search: 0
- Sportsbook inputs: 0
- Target-game PBP used only for oracle grading.

## Frozen urgency cells

1. `two_score_early_q4`: deficit 9–16, >7:30 remaining
2. `three_plus_early_q4`: deficit >=17, >7:30 remaining
3. `two_score_late_q4`: deficit 9–16, <=7:30 remaining
4. `three_plus_late_q4`: deficit >=17, <=7:30 remaining

The clock split is the fixed midpoint of Q4 and the deficit split is the fixed three-possession boundary; neither was searched on 2025 outcomes.

## Observed rushing behavior

| Urgency cell | Plays | Rushes | Aggregate rush rate |
|---|---:|---:|---:|
| two_score_early_q4 | 354 | 104 | 0.293785 |
| three_plus_early_q4 | 464 | 123 | 0.265086 |
| two_score_late_q4 | 373 | 51 | 0.136729 |
| three_plus_late_q4 | 511 | 145 | 0.283757 |

The two-score late-Q4 cell shows the lowest realized rushing rate, but the total deep-late error is not concentrated enough in that single cell to pass the frozen dominance rule.

## Exact Shapley attribution

### All W6–18

| Cell | Shapley MAE recovery | Share of deep-late headroom |
|---|---:|---:|
| two_score_early_q4 | 0.099388 | 25.02% |
| three_plus_early_q4 | 0.086866 | 21.86% |
| **two_score_late_q4** | **0.169713** | **42.72%** |
| three_plus_late_q4 | 0.041304 | 10.40% |

### P3 team-pool overprojection >= 5

| Cell | Shapley MAE recovery | Share of false-high deep-late recovery |
|---|---:|---:|
| two_score_early_q4 | 0.471326 | 30.44% |
| three_plus_early_q4 | 0.309760 | 20.00% |
| **two_score_late_q4** | **0.603496** | **38.98%** |
| three_plus_late_q4 | 0.164108 | 10.60% |

## Frozen dominance gate

`URGENCY_CELL_DOMINANT` required the same top overall cell to satisfy, among other conditions, at least **40%** of the `POOL_OVER_5` deep-late Shapley recovery.

The top cell, `two_score_late_q4`, reached only **38.98%**.

That gate is not waived. The disposition remains:

`DEEP_LATE_URGENCY_DISTRIBUTED`

## Interpretation

STACK6O closes the score-state slicing loop. There is football structure inside late deficits, but the remaining error is distributed across urgency cells rather than isolated cleanly enough to justify another threshold/sub-state tuning cycle.

Combined with STACK6N:

- the deep-late effect is real,
- a universal historical penalty is too blunt,
- recent team-specific deep-late tendency is not sufficient,
- and further deficit/time slicing does not produce a retainable dominant cell.

The next research move should therefore change the representation rather than continue slicing game state. In particular, because the downstream object is the **RB carry pool**, the next audit should verify whether total team rushing obscures different rush types — RB/HB/FB carries versus QB scrambles/designed QB runs and other non-RB rushing — before any new predictive model is frozen.
