# RB STACK6P — Rush-Event Component Shapley Results

## Status

Authoritative execution completed successfully. No production change. No predictive model or player recomposition authorized.

## Authoritative run

- Workflow: `RB STACK6P Rush-Event Component Shapley`
- Run: `33645364513`
- Job: `100298582222`
- Branch: `research-rb-stack6p-rush-event-component-shapley`
- Tested SHA: `a467ef9c0b6f0e307ba50c3e350b9fa74dc56113`
- Artifact: `rb-stack6p-rush-event-component-shapley`
- Artifact ID: `9852591537`
- Artifact SHA256: `49359e5f6bb6d457f0ebacf51f7d25eb9720e64f31a6ebf65820bc333b4ce590`
- Disposition: **`DESIGNED_RUN_CALL_DOMINANT`**

## Integrity

- M94C rows: `544`
- STACK6H rows: `544`
- PBP rows: `101,636`
- joined rows: `544`
- W6-18 population: `388`
- state-share max absolute reconstruction diff: `1.11e-16`
- offensive-play max absolute reconstruction diff: `0`
- PBP rush max absolute reconstruction diff: `0`
- component-mix sum max absolute error: `2.22e-16`
- scramble/kneel overlap: `0`
- strict-prior coverage: `1.000`
- fitted models: `0`
- feature search: `0`
- hyperparameter search: `0`
- threshold search: `0`
- sportsbook inputs: `0`
- target-game PBP used only as oracle grading truth: `1`

Both attribution schemes reproduce the frozen STACK6K identities exactly:

| Scheme | Empty MAE | Full MAE | Recovery | Shapley sum | Identity |
|---|---:|---:|---:|---:|---|
| league_state_mix | 5.5183819623 | 3.4503279032 | 2.0680540592 | 2.0680540592 | PASS |
| team8_shrunk_state_mix | 5.5183819623 | 3.4503279032 | 2.0680540592 | 2.0680540592 | PASS |

## Rush-event context

Events are partitioned into mutually exclusive rushing mechanisms:

- **DESIGNED**: designed rushing plays, including RB/HB/FB carries and designed QB runs.
- **SCRAMBLE**: QB scrambles arising from pass/dropback intent.
- **KNEEL**: QB kneels.

Observed 2025 context:

| Context | Designed / off play | Designed share of rushes | Scramble / off play | Scramble share | Kneel share |
|---|---:|---:|---:|---:|---:|
| Lead | 0.470939 | 0.879134 | 0.028316 | 0.052859 | 0.068006 |
| Neutral | 0.419391 | 0.919296 | 0.029566 | 0.064808 | 0.015896 |
| Trail | 0.325017 | 0.893670 | 0.035609 | 0.097911 | 0.008419 |
| Deep late | 0.202532 | 0.832370 | 0.037975 | 0.156069 | 0.011561 |

Deep-late rushing is therefore structurally different: designed runs collapse to roughly 20.3% of offensive plays while scrambles become a materially larger fraction of the remaining official rushing attempts.

## Shapley attribution — W6-18

### League state mix

- designed: `1.855535` MAE recovery = **89.72%** of within-state tendency recovery
- scramble: `0.190881` = `9.23%`
- kneel: `0.021637` = `1.05%`

### Team-last-8 shrunk state mix

- designed: `1.895138` MAE recovery = **91.64%**
- scramble: `0.160369` = `7.75%`
- kneel: `0.012547` = `0.61%`

The conclusion is robust to the frozen allocation method.

## Large-error bins

Designed-run behavior also dominates the large P3 team-RB-pool misses rather than merely improving the aggregate mean.

### P3 pool over by 5+

- league mix designed share of tendency recovery: **93.35%**
- team8-shrunk designed share: **95.00%**

### P3 pool under by 5+

- league mix designed share: **99.48%**
- team8-shrunk designed share: **100.35%**; the small excess above 100% occurs because the kneel contribution is negative.

### Absolute pool miss 5+

- league mix designed share: **96.35%**
- team8-shrunk designed share: **97.62%**

## Durable conclusion

STACK6P materially narrows the remaining RB opportunity frontier.

The unresolved within-state team-rushing problem is **not primarily QB scramble generation or kneel accounting**. It is the offense's **designed-run-call tendency**.

This is distinct from RB-room share or RB-vs-QB allocation: designed QB runs remain inside the designed-run component. The result therefore points upstream to the decision to call/execute a designed rushing play in the expected game environment.

Do not spend the next cycle on:

- generic RB-share correction;
- QB scramble correction as the primary team-rush fix;
- kneel correction;
- further score/clock threshold slicing without new predictive information.

The next predictive qualification should preserve M94C's frozen pregame play-volume and state-occupancy architecture and test whether a strictly-prior **designed-run-call model** can improve the state-conditioned rushing tendency. Scramble and kneel components should remain simple strict-prior nuisance components unless later evidence says otherwise.

P3 remains the RB point champion until a predictive arm passes frozen temporal gates and downstream RB-pool recomposition is separately validated.
