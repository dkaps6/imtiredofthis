# RB STACK6G — Target-Week Regime-Change Source / Forensic Audit Results

## Status

Authoritative corrected execution completed successfully. No production change. No fitted model authorized.

## Authoritative run

- Workflow: `RB STACK6G Regime Change Source Audit`
- Run: `33631937776`
- Job: `100253253450`
- Branch: `research-rb-stack6g-regime-change-audit`
- Tested SHA: `97f1e5a1c54dd4331a139f9b50f24d11d9c70bfe`
- Artifact: `rb-stack6g-regime-change-audit`
- Artifact ID: `9847176112`
- Artifact SHA256: `c91c25577ea9bf187fa366cfb7a3a0c033118f779f56ae3fb7c96f5324b32ac7`
- Disposition: `STACK6G_SOURCE_USABLE_BUT_NO_MATERIAL_FORENSIC_SIGNAL`

The prior run `33631495970` failed mechanically before output because an empty historical playcaller string was passed to `int()`. The correction is documented in `RB_STACK6G_IMPLEMENTATION_CORRECTION.md`; no frozen hypothesis or support gate changed.

## Source coverage

| Season | Scheduled team-games | Timestamped QB depth? | Pregame QB1 coverage | Timestamp-safe QB1? |
|---|---:|---:|---:|---:|
| 2020 | 512 | No | 0% | No |
| 2021 | 544 | No | 0% | No |
| 2022 | 542 | No | 0% | No |
| 2023 | 544 | No | 0% | No |
| 2024 | 544 | No | 0% | No |
| 2025 | 544 | Yes | 100% | Yes |

The older nflverse depth source contains QB rows but lacks the target-time timestamp necessary to prove the selected QB1 state existed before kickoff. STACK6G therefore does not backfill 2020-2024 expected-QB1 identity from target-game participation or postgame outcomes.

For 2025, timestamp parse rate, pregame snapshot coverage, and QB1 coverage were all 100%. Median selected snapshot age was `10.7703` hours before kickoff; p90 was `17.8939` hours.

## 2025 QB regime forensic results

Frozen W6-18 population: `n=388` team-games.

- QB1 coverage: `1.000`
- QB rushing-propensity-delta coverage: `1.000`
- observed QB1 change rate: `4.64%`
- corr(`qb_rush_propensity_delta`, P3 team-RB-pool residual): **`-0.0300`**
- top-quartile minus bottom-quartile QB-delta pool-residual spread: **`-0.8607` carries**
- mean QB-delta in `POOL_OVER_5` minus `NON_EXTREME_ABS_LT3`: **`-0.1061` rush attempts/game**

All three directional signal gates failed. The observed sign was opposite the frozen hypothesis rather than merely too small in the expected direction.

Selected descriptive bins:

| Subset | n | Mean P3 pool residual | Mean QB rush-propensity delta |
|---|---:|---:|---:|
| All W6-18 | 388 | -0.4690 | -0.3028 |
| P3 over by 3+ | 133 | +7.0391 | -0.3747 |
| P3 over by 5+ | 95 | +8.2477 | -0.3825 |
| P3 under by 3+ | 138 | -8.0959 | -0.2560 |
| P3 under by 5+ | 96 | -9.9015 | -0.2396 |
| Non-extreme abs <3 | 117 | -0.0080 | -0.2764 |

Conclusion: target-week QB rushing-regime discontinuity does not explain the current P3 team-RB-pool ranking failure strongly enough to justify a point-model follow-up.

## Playcaller source and forensic results

M68's embedded verified playcaller contract provides full mappings for 2023-2025 and no mapping for 2020-2022:

- 2023: 544/544 team-games mapped; 6 documented change team-games
- 2024: 544/544; 4 documented changes
- 2025: 544/544; 4 documented changes

On the frozen 2025 W6-18 population:

- mapping coverage: `1.000`
- recent-change observations (first three games under documented in-season caller): `10`
- recent-change minus stable mean absolute P3-pool-error spread: **`-1.1736` carries**
- recent-change minus stable `POOL_ABS_5` rate spread: **`+0.00794`**

The frozen support gate required at least `+1.00` carry absolute-error enrichment and `+0.10` absolute-5 rate enrichment. Both failed; recent caller-change games were actually less error-prone on average.

Conclusion: do not fit a playcaller-based RB-pool correction from this evidence.

## Integrity

- fitted models: `0`
- hyperparameter search: `0`
- feature search: `0`
- threshold search: `0`
- sportsbook inputs: `0`
- target-game QB rushing used upstream: `0`
- target-game participation used upstream: `0`
- target-game injury used upstream: `0`
- P3 residual used for grading only: `1`
- source seasons attempted: `2020;2021;2022;2023;2024;2025`

## Durable conclusion

STACK6G closes two plausible but unsupported target-week discontinuity directions:

1. Do not retest generic or target-QB mobility/rushing-regime adjustments against the current P3 team-RB-pool problem without genuinely new source information.
2. Do not add a playcaller-change correction to RB opportunity from the current evidence.

P3 remains the RB point champion. STACK6G does not authorize a follow-up fitted model.

The next RB investigation should return to the current team-RB-pool error itself and identify, with a no-fit oracle/decomposition, whether the remaining P3 pool error is primarily **total team rushing opportunity** or **RB share of team rushing** on the current 2025 frontier before another source/model family is attempted.