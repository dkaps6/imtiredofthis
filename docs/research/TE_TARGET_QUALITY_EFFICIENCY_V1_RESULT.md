# TE Target-Quality Efficiency V1 — Blind Predictive Result

Date: 2026-09-24

Status: `TE_TARGET_QUALITY_EFFICIENCY_V1_FAIL`

## Canonical execution

- frozen plan commit: `5221b2f60aca96326883a88513140e246c117534`
- scientific head: `93c9d831cc29ab6cb5aa48b2a879d2bffe636fd8`
- run: `36002008896`
- job: `107640697270`
- artifact: `10808388506`
- digest: `sha256:4d46dee81c107bb6b4ae5aa091ff04de0a7d9c5db97313054b76d5a0cb721f63`

Integrity:
- exact frozen PR #549 TE rec-yard football rows used;
- 0 unresolved identities;
- sportsbook inputs: 0;
- 2026 outcomes used: 0;
- production changed: false;
- no hyperparameter/feature/cap search.

## Blind results

### Train 2024 -> test 2025

n = 1,123

- baseline MAE: **15.4123**
- candidate MAE: **16.0075**
- MAE change: **-0.5953 yd** (worse)
- baseline RMSE: 22.7156
- candidate RMSE: **21.5992**
- absolute bias: 6.4119 -> **0.6357**
- p90 absolute error: 35.0496 -> **32.6404**
- 30+ yard misses: 136 -> **142**
- correction vs realized residual correlation: **+0.1713**

Primary gate failed because MAE worsened and 30+ misses increased.

### Train 2025 -> test 2024

n = 1,088

- baseline MAE: **16.1443**
- candidate MAE: **16.3328**
- MAE change: **-0.1885 yd** (worse)
- baseline RMSE: 23.1746
- candidate RMSE: **21.8085**
- absolute bias: 6.2009 -> **0.1332**
- p90 absolute error: 36.8976 -> **34.3717**
- 30+ yard misses: 158 -> **147**
- correction vs realized residual correlation: **+0.2153**

Primary gate failed because MAE worsened.

### Pooled blind rows

n = 2,211

- baseline MAE: **15.7725**
- candidate MAE: **16.1676**
- baseline RMSE: 22.9426
- candidate RMSE: **21.7024**
- absolute bias: 6.3081 -> **0.2573**
- p90 absolute error: 36.0039 -> **33.6786**
- 30+ yard misses: 294 -> **289**
- correction-residual correlation: **+0.1875**

## Interpretation

The NGS target-quality family is **not** a validated global TE receiving-yard mean correction under the frozen V1 architecture. It worsened MAE in both blind directions, so it must not be promoted or rescued with a cap/coefficient/feature search.

There is still genuine structure in the family:
- positive blind residual correlation in both directions;
- materially lower RMSE;
- dramatically lower signed bias;
- lower p90 error in both directions;
- fewer pooled 30+ yard misses.

That pattern says the information is related to TE receiving-yard error but the fixed global residual correction trades improvement on larger/signed errors for worse median/absolute performance. This may be relevant to a future reliability/tail/distribution study, but V1 does not authorize that study or any production change.

## Disposition

`TE_TARGET_QUALITY_EFFICIENCY_V1_FAIL`

Do not:
- search Ridge alpha;
- add a correction cap;
- subset feature families after seeing these results;
- rescue only high-error rows;
- call the RMSE/tail improvements a mean-model win.

Any continuation must either use genuinely untouched evidence/prospective 2026 outcomes or be a separately justified distribution/reliability question with a new frozen validation authority.
