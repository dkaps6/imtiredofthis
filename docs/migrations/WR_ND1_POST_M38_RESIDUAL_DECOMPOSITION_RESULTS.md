# WR-ND1 — Post-M38 Residual Decomposition Results

**Branch:** `research-wr-nd1-post-m38-decomposition`

**Base:** `main` at `754d0f4ed34a06a65d2db36f2fb2ca10c58264ed`

**Frozen plan:** `docs/migrations/WR_ND1_POST_M38_RESIDUAL_DECOMPOSITION_PLAN.md`

## Canonical run

- run: `34001285325`
- job: `101400491931`
- tested SHA: `4c0d8b9b0ed08c98fdb41d2334757698f326d7d5`
- artifact: `9979591713`
- artifact digest: `sha256:9940c040052ffaf04ed01a0ba4bb164fa5aef7117a7d5d9ef6351cc1bbc95353`
- conclusion: **success**

The initial run `34001097633` failed mechanically because one WR target-game row recorded nonzero receiving yards with zero recorded targets. The frozen factorization and routing gate were not changed. A wrapper was added that excludes mathematically non-factorizable target-result rows from the target-game evaluation view only while preserving the original historical logs for all pregame priors.

The canonical run audited two such source rows:

- Isaiah Bond, CLE, 2025 W16: 0 targets, +21 receiving yards;
- Riley Leonard, IND, 2025 W18, QB: 0 targets, -3 receiving yards.

Only the Isaiah Bond row intersects the WR target-result population. Historical pregame evidence is untouched.

## Population and integrity

- evaluation WR player-games: `2130`
- teams: `32`
- weeks: `1-18`
- M38 multipliers: `(1.40, 1.14, 0.91, 0.78)`
- max M38 WR-share mass drift: `2.220446049250313e-16`
- sportsbook inputs used: **false**
- model fitting used: **false**
- production changed: **false**

All exact Shapley identities passed. The all-components-corrected prediction reproduced target-game truth to floating-point tolerance for targets, receptions, and receiving yards.

## Post-M38 reference metrics

| Outcome | Stage | N | MAE | RMSE | Bias | Corr |
|---|---|---:|---:|---:|---:|---:|
| targets | deterministic | 2130 | 2.076010 | 2.845266 | -1.049869 | 0.600832 |
| receptions | deterministic | 2130 | 1.473930 | 2.015936 | -0.589800 | 0.565897 |
| receptions | canonical MC | 2130 | 1.474730 | 2.016232 | -0.591010 | 0.565931 |
| receiving yards | deterministic | 2130 | 22.022545 | 30.788707 | -7.550397 | 0.501740 |
| receiving yards | canonical MC | 2130 | 22.016387 | 30.734172 | -7.268089 | 0.501101 |

## Exact Shapley attribution

### Targets

| Component | MAE attribution |
|---|---:|
| TEAM_TARGET_VOLUME | 0.419369 |
| WR_TARGET_MASS | 0.354203 |
| WITHIN_WR_ALLOCATION | **1.302438** |

Within-WR allocation remains the dominant target-count error after M38, accounting for ~62.7% of target MAE attribution. M38 improved the static hierarchy but did not eliminate dynamic entitlement/role misses.

### Receptions

| Component | MAE attribution |
|---|---:|
| TEAM_TARGET_VOLUME | 0.219192 |
| WR_TARGET_MASS | 0.181641 |
| WITHIN_WR_ALLOCATION | **0.675392** |
| CATCH_RATE | 0.397705 |

Opportunity remains the largest single reception component, with catch-rate error meaningful but secondary.

### Receiving yards

| Component | MAE attribution |
|---|---:|
| TEAM_TARGET_VOLUME | 2.762802 |
| WR_TARGET_MASS | 2.042554 |
| WITHIN_WR_ALLOCATION | 7.651051 |
| YARDS_PER_TARGET | **9.566137** |

Frozen dominance gate:

- overall top component: `YARDS_PER_TARGET`
- top component positive-attribution share: `0.434379`
- required threshold: `>= 0.40`
- WR1/WR2/WR3 same-top count: `3 / 3`
- required count: `>= 2 / 3`
- gate: **PASS**

**Disposition: `YARDS_PER_TARGET_DOMINANT`.**

## Role slices — receiving yards

| Role | N | Base MAE | Team volume | WR mass | Within-WR allocation | YPT | Top |
|---|---:|---:|---:|---:|---:|---:|---|
| WR1 | 523 | 29.302270 | 5.046294 | 3.597791 | 8.246423 | **12.411762** | YPT |
| WR2 | 421 | 24.849220 | 2.880675 | 2.475230 | 8.935658 | **10.557656** | YPT |
| WR3 | 328 | 20.392752 | 2.070752 | 1.737858 | 8.031196 | **8.552946** | YPT |
| WR4+ | 858 | 16.821199 | 1.577606 | 0.998726 | 6.512488 | **7.732380** | YPT |

YPT is not a WR1-only phenomenon; it is the largest component in every WR hierarchy tier.

## Error-direction asymmetry

| Base direction | N | Base MAE | Team volume | WR mass | Within-WR allocation | YPT | Top |
|---|---:|---:|---:|---:|---:|---:|---|
| overprojection | 1038 | 14.848591 | 2.734609 | -0.355057 | 2.198323 | **10.270716** | YPT |
| underprojection | 1092 | 28.841742 | 2.789601 | 4.321602 | **12.834139** | 8.896400 | allocation |

This is a durable architectural clue. Ordinary false-high WR-yard outcomes are primarily efficiency/YPT errors. False-low outcomes are more opportunity-driven, especially within-WR allocation.

A blanket efficiency correction would therefore risk worsening the already-large false-low opportunity problem.

## Week-phase slices

YPT is the largest receiving-yard component in every phase:

- W1-4: YPT `9.315269` vs allocation `7.457418`
- W5-9: YPT `9.771950` vs allocation `9.064795`
- W10-13: YPT `10.220511` vs allocation `6.978111`
- W14-18: YPT `9.100937` vs allocation `7.036614`

The YPT conclusion is not a Week-1/preseason artifact.

## Actual target-tier slices

| Actual targets | N | Base MAE | Team volume | WR mass | Within-WR allocation | YPT | Top |
|---|---:|---:|---:|---:|---:|---:|---|
| 0-3 | 997 | 14.226899 | 2.111531 | 0.415209 | 4.677840 | **7.022320** | YPT |
| 4-6 | 601 | 21.746719 | 2.889849 | 1.905668 | 5.980128 | **10.971074** | YPT |
| 7-9 | 340 | 29.616724 | 2.843514 | 3.412919 | 10.882654 | **12.477637** | YPT |
| 10+ | 192 | 49.918427 | 5.604054 | 8.494682 | **22.597776** | 13.221916 | allocation |

The main exception to broad YPT dominance is the extreme-volume tail. For 10+ target games, dynamic within-WR allocation is overwhelmingly the largest remaining miss.

## Durable conclusions

1. M35's old conclusion remains partly true: within-WR target allocation is still the largest target-count error after M38.
2. But after M38, **receiving-yard error is no longer primarily an opportunity problem overall**. YPT is now the largest recoverable component and clears the frozen structural-dominance gate.
3. This result is stable across WR1/WR2/WR3/WR4+ and all season phases.
4. The receiving-yard error is asymmetric:
   - false-high yards -> predominantly YPT/efficiency;
   - false-low yards -> predominantly within-WR opportunity allocation.
5. Extreme 10+ target games remain a separate dynamic-entitlement ceiling problem.
6. Therefore the next broad WR investigation should decompose/model YPT mechanics, while preserving the 10+ target / false-low opportunity tail as a separate unresolved lane.

## Explicit anti-duplication implications

Do **not** respond to `YARDS_PER_TARGET_DOMINANT` by rerunning M75 under a new algorithm. NGS separation/cushion/aDOT/YACOE + PFR secondary aggregate interactions already failed on canonical-v3.

The next step must first decompose current YPT into more fundamental football mechanics (catch/conversion vs yards-per-reception, and then route-depth/YAC if supported) before choosing new predictive information.

Do not retune M38 hierarchy multipliers. Do not reopen generic target-pool pruning. Do not infer fake WR-CB assignments from participation. Do not use sportsbook inputs upstream.
