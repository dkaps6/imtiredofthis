# RB STACK6R — Designed-Run Context Occupancy vs Conditional Call Results

## Status

Authoritative no-fit oracle completed successfully. No production change and no predictive model are authorized.

## Authoritative run

- Workflow: `RB STACK6R Designed Run Context Oracle`
- Run: `33654134942`
- Job: `100328316397`
- Branch: `research-rb-stack6r-designed-run-context-oracle`
- Tested SHA: `3018def749533b52264d254821394ac368427b5d`
- Artifact: `rb-stack6r-designed-run-context-oracle`
- Artifact ID: `9856011628`
- Artifact SHA256: `9059f9a779269c8ef5cfd33eb5cc575a98cf1d487d42f7c1b0955ae898a5efe8`
- Disposition: **`CONDITIONAL_CALL_DOMINANT`**

## Integrity

- PBP rows: `101,636`
- completed team-games: `1,632`
- 2025 joined team-games: `544`
- W6-18 team-games: `388`
- context assignment identity max abs error: `0.0`
- `OTHER` context share, 2025 W6-18: `0.003736`
- full oracle identity max abs error: `0.0`
- strict-prior construction: PASS
- fitted models / feature search / threshold search / model-family search / hyperparameter search: `0`
- sportsbook inputs: `0`

## Attribution

### TEAM5_SHRUNK

| Population | Base MAE | Perfect context-occupancy MAE | Occupancy recovery | Occupancy fraction | Conditional-call remainder |
|---|---:|---:|---:|---:|---:|
| ALL W6-18 | 4.154667 | 3.963615 | 0.191052 | **4.60%** | **95.40%** |
| POOL_OVER_5 | 4.553853 | 4.256946 | 0.296907 | 6.52% | 93.48% |
| POOL_UNDER_5 | 5.815597 | 5.609702 | 0.205894 | 3.54% | 96.46% |
| W13-18 | 4.224555 | 4.154094 | 0.070461 | 1.67% | 98.33% |

### TEAM8_SHRUNK

| Population | Base MAE | Perfect context-occupancy MAE | Occupancy recovery | Occupancy fraction | Conditional-call remainder |
|---|---:|---:|---:|---:|---:|
| ALL W6-18 | 4.180903 | 4.015161 | 0.165741 | **3.96%** | **96.04%** |
| POOL_OVER_5 | 4.616482 | 4.401110 | 0.215372 | 4.67% | 95.33% |
| POOL_UNDER_5 | 5.711566 | 5.565274 | 0.146292 | 2.56% | 97.44% |
| W13-18 | 4.209962 | 4.154795 | 0.055167 | 1.31% | 98.69% |

## Durable conclusion

The designed-run error exposed by STACK6P and not recovered by STACK6Q is **not primarily caused by forecasting the wrong down/distance mix**. Even with perfect target-game down/distance occupancy inside each score state, only about 4-5% of baseline designed-attempt MAE disappears. Roughly 95-96% remains in the conditional decision to call a designed run versus pass once the situation is known.

Do not spend the next research cycle building a drive/down/distance occupancy generator as the primary RB fix.

The next investigation should remain no-fit and test genuinely conditional football information: whether a strict-prior offense-vs-defense **run-versus-pass advantage within the same score-state and down/distance context** predicts deviations from a team's strict-prior conditional run tendency. Broad unconditioned run/pass EPA, PROE, matchup, and pressure features were already tested in earlier migrations; the novelty must be the conditional decision surface itself.