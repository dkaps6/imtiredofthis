# TE-R5P Receiving-Yards Width V2 — Frozen Result

Status: **FAILED CLOSED**
Disposition: `TE_R5P_REC_YARDS_WIDTH_V2_FAILED_CLOSED`

Research only. No production change.

## Canonical run authority

- workflow run: `36040911515`
- job: `107772560075`
- branch: `research-te-live-entitlement-efficiency-v1`
- exact head: `a9c7a93ae375f75e926139a0a29675aab89d2fdc`
- result artifact: `10827222034`
- result artifact digest: `sha256:f128cdf5b5d41c18ff1a08f1a9c23f346f911294d6e8e6afe678c8c64f526cc2`
- reconstructed distribution artifact: `10826839080`
- distribution artifact digest: `sha256:37fffe9d2b176ac4db98e18a96ebe4eb7bc418a1c67b2e501ffc5c69f45501be`
- workflow conclusion: **SUCCESS**
- production changed: **false**
- sportsbook inputs used to fit k: **0**

The workflow completed exact PR #549 baseline replay, specialist-array reconstruction, specialist trace parity, blind science, and scope-invariant verification before uploading the result.

## Frozen factors

| Fit season | n | mean row MC SD | residual SD | k |
|---|---:|---:|---:|---:|
| 2024 | 1,088 | 12.9717 | 22.3399 | 1.7222 |
| 2025 | 1,123 | 12.9946 | 21.8015 | 1.6777 |

The predeclared pooled future-only factor would have been `1.6993560688492668` **only if the candidate qualified**. It did not qualify and this factor is not authorized for production.

## Blind results

### Fit 2024 -> blind 2025

- n: `1,123`
- point MAE: `15.4122763680 -> 15.4122763680` (invariant)
- max absolute mean shift: `2.13e-14`
- CRPS: `11.5174400858 -> 11.6625534053`
- CRPS change: **-1.26%** (worse)
- 80% coverage: `0.715049 -> 0.887801`
- 80% coverage gap: `0.084951 -> 0.087801` (worse)
- 90% coverage: `0.813001 -> 0.935886`
- 90% coverage gap: `0.086999 -> 0.035886` (better)

### Fit 2025 -> blind 2024

- n: `1,088`
- point MAE: `16.1443318217 -> 16.1443318217` (invariant)
- max absolute mean shift: `1.42e-14`
- CRPS: `12.0157779990 -> 11.9659837865`
- CRPS change: **+0.41%** (better)
- 80% coverage: `0.697610 -> 0.866728`
- 80% coverage gap: `0.102390 -> 0.066728` (better)
- 90% coverage: `0.796875 -> 0.930147`
- 90% coverage gap: `0.103125 -> 0.030147` (better)

## Secondary historical-line calibration

These were downstream-only diagnostics after k was frozen; they did not fit the factor.

- matched rows: `1,353`
- pooled Brier: `0.284289 -> 0.265652`
- pooled log loss: `0.790993 -> 0.730611`

## Frozen gate disposition

- point MAE invariant both directions: **PASS**
- max mean shift <= 1e-8 both directions: **PASS**
- CRPS strictly improves both directions: **FAIL**
- 80% coverage gap improves both directions: **FAIL**
- 90% coverage gap improves both directions: **PASS**
- pooled Brier non-worse: **PASS**
- pooled log loss non-worse: **PASS**
- sportsbook inputs used to fit k = 0: **PASS**
- exact replay/fold/conservation authority: **PASS**
- overall qualified: **FALSE**

The candidate therefore **fails closed** under the preregistered contract.

## Stopping rule

Do not:
- search a different k;
- add caps;
- search subgroups;
- condition width on sportsbook inputs;
- apply a global TE receiving-yard SD multiplier;
- use 2026 outcomes to rescue the exposed candidate;
- alter the gates after seeing the result.

The useful signal is descriptive only: the historical TE-R5P distributions are often too narrow, but one global season-fitted width factor is not stable enough across blind seasons under the frozen primary scoring rules.

Next work must be genuinely different model-improvement science.
