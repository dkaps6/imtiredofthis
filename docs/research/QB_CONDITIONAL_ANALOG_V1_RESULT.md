# QB Conditional Historical-Analog Reliability V1 — Terminal Result

## Final disposition

`NO_ACTIONABLE_QB_CONDITIONAL_ANALOG_RELIABILITY`

The exact preregistered V1 architecture failed its blind 2025 confirmation. Per the frozen stop rule, this architecture is closed. No nearby-`k`, distance-metric, density-threshold, feature, source, evidence-rule, or gate rescue is authorized.

No production model, projection, probability, EV, signal, eligibility, pricing, or Full Slate decision logic is changed by this result.

## Frozen architecture tested

- market: QB `pass_yards` only
- authority/grading population: `CURRENT_PRODUCTION_ORDER` authority-exact Vegas rows
- feature source: exact QB-PD3 authority casebook, documented in `QB_CONDITIONAL_ANALOG_V1_PRE_OUTCOME_LINEAGE_AMENDMENT.md`
- frozen features: 17 pregame variables
- reference year: 2024
- blind confirmation year: 2025
- scaler: `StandardScaler`, fit on 2024 only
- metric: Euclidean
- neighbors: `k=15`
- density gate: 90th percentile of 2024 leave-one-out k-NN distance
- frozen density threshold: `4.221606664273808`
- missingness: exclude-only, no imputation
- clean rows: 768 (2024=404, 2025=364)

## Blind confirmation result

The preregistered candidate-supported bucket contained:
- 2024 supported rows: **217**
- 2025 supported rows: **180**

ROI:
- 2024 supported-bucket ROI: **-7.0473%**
- 2025 supported-bucket ROI: **-9.5602%**
- unconditional 2025 QB pass-yards ROI baseline: **-3.2288%**

Frozen gates:
1. 2025 supported N >= 40: **PASS** (`180`)
2. 2025 supported ROI > 0: **FAIL** (`-9.5602%`)
3. 2025 supported ROI > unconditional 2025 baseline: **FAIL** (`-9.5602%` vs `-3.2288%`)
4. 2024 supported-bucket ROI > 0: **FAIL** (`-7.0473%`)

Because three of four hard gates failed, the terminal disposition is `NO_ACTIONABLE_QB_CONDITIONAL_ANALOG_RELIABILITY`.

This is not an underpowered or borderline failure: the 2025 supported bucket contained 180 rows and materially underperformed the unconditional baseline.

## Evidence-class consequence

Because the architecture-level gate failed, no row may carry a final `SUPPORTED` label from this V1 architecture. On the 2025 evaluation cohort after the terminal architecture gate:
- `DESCRIPTIVE_ONLY`: 328
- `NO_ANALOG_SUPPORT`: 36
- `SUPPORTED`: 0

The failed V1 analog retrieval may remain a preserved research diagnostic, but it must not be integrated as a live bet-selection/trust signal.

## Reproducibility / integrity

Canonical branch: `research-qb-analog-v1`
Draft PR: #619
Pre-confirmation head / CI-tested package: `057081cc83f4cddeb6636a31765d6cb3e5187082`
Repo CI: `35274896938` — SUCCESS before confirmation.

Canonical upstream artifacts:
- authority-exact Vegas: run `34843204550`, artifact `10346639168`, digest `sha256:e0041386c7f6600c6e8a8781d0d91d7603c4d1fde75d9aa3d0f0055aaa746225`
- QB-PD3 exact authority casebook: run `34122984048`, artifact `10018942911`, digest `sha256:cde910939dba0631a2a68e9acfbcafcdb90af1d1b2128627442073854bf3f849`

Local confirmation output digests:
- `QB_CONDITIONAL_ANALOG_V1_RESULT.json`: `sha256:eeb5aeebf1736e68baf89a2010f340d08791944d22b78f6bd3496ce41b69cd45`
- `reference_2024_confirmed.csv`: `sha256:7d0e28e3c0667511bf623f91d0af37145634d56d48a74217b8fd2d6d19e916ec`
- `evaluation_2025_confirmed.csv`: `sha256:fa11fab895183d5c86f66545278c7cf4ab01ce055e1731ed64bfbedf21200e50`

The CONFIRM execution was run exactly once after the PREPARE geometry and code were frozen and CI-green. No rescue execution was performed.

## Interpretation

The tested hypothesis was that continuous similarity across the frozen player-state, opponent-state, game-script, and model-disagreement variables could identify a subset of current-authority QB pass-yards bets whose historical neighbors made the model more trustworthy against the market.

That hypothesis did not hold under the exact V1 design. The analog-selected subset was worse than the unconditional QB baseline in both the 2024 reference sample and the untouched 2025 confirmation sample.

This result does **not** imply that all forms of historical context are useless. It closes this specific 17-feature / Euclidean / k=15 / density-gated nearest-neighbor trust architecture. Any future analog research must be a genuinely different preregistered family rather than a parameter rescue of V1.