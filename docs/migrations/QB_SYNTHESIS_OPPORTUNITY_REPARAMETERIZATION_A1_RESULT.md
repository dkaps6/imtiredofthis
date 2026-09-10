# QB Synthesis Opportunity Reparameterization A1 — Result

## Disposition

`M89_SYNTHESIS_OPPORTUNITY_REALLOCATION_NOT_SUPPORTED`

The deterministic mean-preserving reparameterization is **not** authorized for shared QB/receiver opportunity integration. Production remains unchanged.

## Canonical lineage

- Branch: `research-qb-synthesis-opportunity-reparameterization-a1`
- Frozen plan: `e7d930a73a7153e0dcd01bb84c209095172be56f`
- Evaluator commit: `28162b56b4f94f83793023a8681da65fe8cc97ef`
- Tested/workflow head: `6d3eedbdc26b124152fe97b69f99be30ce975da0`
- Run: `34534125445`
- Job: `103061631053`
- Artifact: `10174698244` (`qb-synthesis-opportunity-reparameterization-a1`)
- Artifact digest: `sha256:25ee96a6c35b426cc4312fe9bb2d34df3d526099534958b02fa5ce013ebe3643`
- Parent opportunity-chain run: `34523313743`
- Shared QB/WR source run: `34066549394`

## Integrity

All frozen integrity gates passed.

- M89 QB rows: `884`
- primary WR target-mass rows: `440`
- secondary WR reception-mass rows: `884`
- sportsbook inputs: `0`
- model fitting: `0`
- production change: `false`
- current attempt identity max gap: `1.7763568394002505e-14`
- implied QB mean identity max gap: `2.842170943040401e-14`
- implied team-opportunity identity max gap: `7.105427357601002e-15`
- shared source residual alignment max gaps: `3.552713678800501e-15`

## Deterministic candidate

The frozen candidate was exactly:

`implied_attempts = football_synthesis / pred_ypa`

`implied_team_pass_opportunity = implied_attempts / (pred_C * pred_S)`

No coefficient, blend, clipping, or model was fit.

## QB attempt result

The reparameterization materially improved official QB attempts:

### 2024
- MAE: `7.339680 -> 6.185636` (**+1.154045**)
- RMSE: `9.535052 -> 7.978533`
- bias: `-4.366022 -> +1.171542`
- correlation: `-0.014592 -> 0.205129`
- 10+ attempt miss rate: `26.8018% -> 20.7207%`

### 2025
- MAE: `6.841074 -> 6.235193` (**+0.605881**)
- RMSE: `8.846363 -> 7.639741`
- bias: `-3.932360 -> +1.049577`
- correlation: `0.162534 -> 0.280364`
- 10+ attempt miss rate: `24.3182% -> 19.7727%`

### Pooled
- MAE: `7.091506 -> 6.210302` (**+0.881203**)
- RMSE: `9.198713 -> 7.811740`
- bias: `-4.150172 -> +1.110836`
- correlation: `0.072907 -> 0.240322`
- 10+ attempt miss rate: `25.5656% -> 20.2489%`

The implied attempt correction had real game-to-game information against actual attempt residual:

- 2024 Spearman: `0.356723`
- 2025 Spearman: `0.264867`
- pooled Spearman: `0.313999`

## Team pass-opportunity result

The same reparameterization did **not** robustly recover the physical shared team-pass-opportunity state.

### 2024
- MAE: `7.403439 -> 7.159113` (**+0.244326**, below frozen >=0.25 pooled-style support threshold)
- correlation: `0.006234 -> 0.222537`
- p90 absolute error: `14.944770 -> 15.534936` (worse)

### 2025
- MAE: `7.027862 -> 7.161781` (**0.133919 worse**)
- correlation: `0.114736 -> 0.162983`
- p90 absolute error: `14.309074 -> 13.992483`

### Pooled
- MAE: `7.216500 -> 7.160441` (**+0.056059 only**)
- correlation: `0.057777 -> 0.193710`

The frozen pooled team-pass-opportunity MAE support gate failed, and 2025 team-pass-opportunity MAE was worse.

## Shared receiver result

The critical shared-opportunity gates also failed.

### 2025 WR target mass (`n=440`)

Implied attempt correction vs WR target residual:

- Pearson: `0.133880`
- Spearman: `0.108443`
- same-sign: `75.6818%`
- Q4-minus-Q1 WR target residual gap: `1.840024`

Frozen required Spearman: `>=0.20` — **FAIL**.

### 2024-2025 WR reception mass (`n=884`)

- pooled Pearson: `0.097183`
- pooled Spearman: `0.103957`
- same-sign: `73.3032%`
- 2024 Spearman: `0.118975`
- 2025 Spearman: `0.079664`

Frozen required pooled Spearman: `>=0.15` — **FAIL**.

## Scientific meaning

M89/M90's promoted synthesis adjustment contains meaningful information that can improve **QB official attempts** if the final yard mean is mechanically converted back into attempts at fixed predicted YPA.

However, that signal is not the same thing as the physical shared `TEAM_PASS_OPPORTUNITY` state isolated by the parent diagnostic. It does not improve corrected team pass opportunities robustly across seasons and does not carry enough of the independent WR opportunity residual.

Therefore the project must not treat the M89 synthesis correction as a shared receiver target-pool signal merely because it improves QB attempts.

The parent conclusion remains intact:

- shared upstream team pass opportunity is the primary missing mechanism;
- M89's downstream synthesis correction partially compensates QB attempt/yards errors but does not solve the shared team-volume state.

## Stopping rule honored

- no alternate YPA anchor;
- no A0/A1 blend;
- no coefficient search;
- no cap search;
- no schedule/rest combination;
- no production change.

## Next research direction

Return to genuinely new pregame information for the physical team-pass-opportunity state. Do not reopen generic M89 synthesis reallocation or generic residual modeling.
