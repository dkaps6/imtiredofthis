# Historical Analog State Experiment V1 — Result

**Status:** CLOSED — ALL FAMILIES FAILED 2024 PRIMARY GATE  
**Frozen plan:** `docs/research/HISTORICAL_ANALOG_STATE_EXPERIMENT_V1.md`  
**Execution commit:** `9a15d082a4c7da793f098e565d12ac6a7ce56f40`  
**Workflow run:** `35512335301`  
**Job:** `106082294476`  
**Artifact:** `10605238933`  
**Artifact digest:** `sha256:70a188482f627fc74dc7ddf5ae16d26d32cd9ddb3a3f13e956130bc3e5d85c1d`  
**Sportsbook read:** false  
**Production mean changed:** false  
**Analog geometry changed:** false

## Frozen-policy verdict

`HISTORICAL_ANALOG_STATE_EXPERIMENT_V1` failed closed in the 2024 primary holdout.

Every family failed the preregistered minimum of **100 high-novelty rows** at the frozen
`analog_novelty_risk >= 0.80` threshold. Because no family passed every 2024 gate,
**no 2025 predictive result was exposed**. The manifest records
`replication_families: []`.

The result may not be rescued by lowering the novelty cutoff, pooling positions,
changing the composite, changing the row floor, selecting favorable seasons, or
reading 2025.

## Exact 2024 results

| Family | High novelty n | Comparison n | MAE ratio | RMSE ratio | p90 AE ratio | Catastrophic ratio | 2-of-3 tail gate | Integrity | Final |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| RB rush opportunity | 34 | 977 | 1.285802 | 1.261885 | 1.382000 | 1.874041 | PASS (3/3) | PASS | `FAILED_CLOSED_PRIMARY` |
| RB target opportunity | 34 | 977 | 0.949612 | 0.943206 | 0.972376 | 1.138031 | FAIL (1/3) | PASS | `FAILED_CLOSED_PRIMARY` |
| WR target opportunity | 96 | 1,457 | 1.112817 | 1.091417 | 1.044078 | 1.248747 | PASS (3/3) | PASS | `FAILED_CLOSED_PRIMARY` |
| TE target opportunity | 63 | 790 | 1.206016 | 1.244562 | 1.205434 | 1.567460 | PASS (3/3) | PASS | `FAILED_CLOSED_PRIMARY` |
| QB rush opportunity | 31 | 466 | 1.078874 | 1.214056 | 1.372333 | 1.840685 | PASS (3/3) | PASS | `FAILED_CLOSED_PRIMARY` |

The comparison-row floor of 250 passed for every family. The high-novelty row floor
failed for every family.

### Absolute 2024 metrics

| Family / cohort | n | MAE | RMSE | bias | p90 AE | catastrophic miss rate |
|---|---:|---:|---:|---:|---:|---:|
| RB rush — high novelty | 34 | 0.157759 | 0.199018 | 0.038080 | 0.355552 | 0.176471 |
| RB rush — comparison | 977 | 0.122693 | 0.157715 | 0.002816 | 0.257274 | 0.094166 |
| RB targets — high novelty | 34 | 0.041769 | 0.053013 | 0.012708 | 0.087596 | 0.117647 |
| RB targets — comparison | 977 | 0.043985 | 0.056205 | 0.002931 | 0.090084 | 0.103378 |
| WR targets — high novelty | 96 | 0.067569 | 0.084170 | -0.003614 | 0.132653 | 0.135417 |
| WR targets — comparison | 1,457 | 0.060719 | 0.077120 | -0.001207 | 0.127053 | 0.108442 |
| TE targets — high novelty | 63 | 0.057720 | 0.078116 | -0.001450 | 0.122022 | 0.190476 |
| TE targets — comparison | 790 | 0.047860 | 0.062766 | -0.002617 | 0.101227 | 0.121519 |
| QB rush — high novelty | 31 | 0.067418 | 0.099695 | -0.005883 | 0.183876 | 0.193548 |
| QB rush — comparison | 466 | 0.062489 | 0.082118 | 0.000447 | 0.133988 | 0.105150 |

Training rows and frozen catastrophic thresholds:
- RB rush: 3,923 rows; threshold 0.263345
- RB targets: 3,923 rows; threshold 0.089463
- WR targets: 6,140 rows; threshold 0.123401
- TE targets: 3,228 rows; threshold 0.094479
- QB rush: 1,909 rows; threshold 0.131225

## Integrity / leakage result

All hard integrity checks were clean:

- stable identity coverage: **1.0000**
- history duplicate keys: **0**
- analog-state duplicate keys: **0**
- missing target identities: **0**
- missing analog identities: **0**
- chronology violations: **0**
- declared strict-prior mismatches: **0**
- neighbor-rank duplicates: **0**
- state/history unmatched rows: **0**
- position mismatches: **0**
- forbidden target/outcome columns in analog state: **0**
- leakage violations: **0**

Input / artifact fingerprints:
- canonical player-game history SHA-256:
  `33e9ba1e98e7d9057642cca6707925f4fd3ee3b3e7cd3a246e46ced15c7020e8`
- analog states SHA-256:
  `724df507fe30dbc9baaaf31655fedc07ce15e4b2639507c5ee6e44be88cdadb9`
- analog neighbors SHA-256:
  `4fbf210fccf28a61aacbc0346048e6c9aa0507a593497c0a8266ee725daf1de9`

Explicit abstentions:
- `NO_ANALOG_SUPPORT` rows: **306**
- `NO_PERCENTILE_SUPPORT` rows: **308**

## Interpretation

The result is scientifically useful but not promotable.

RB rushing, WR targets, TE targets and QB rushing all showed materially worse 2024
error in the high-novelty cohort on MAE and all three preregistered tail/dispersion
metrics. However, the preregistered event cohort was too sparse to satisfy the
minimum support gate. That support gate was frozen precisely to prevent promotion
from a small, selectively extreme cohort.

RB target opportunity did not show the same degradation and independently failed
the error-direction gates.

Therefore V1 establishes only a **descriptive sparse-regime pattern**, not a
replicated reliability signal.

## Governance

- No 2025 rescue.
- No alternative `analog_novelty_risk` cutoff may be inspected in V1.
- No position pooling may rescue support.
- No descriptor substitution or post-hoc combination may rescue V1.
- No production confidence/abstention/uncertainty change is authorized.
- No production mean change is authorized.
- No sportsbook or paid odds input was used.
- Issue #535 remains untouched.

Final experiment disposition:

`HISTORICAL_ANALOG_STATE_EXPERIMENT_V1_FAILED_CLOSED_PRIMARY_SUPPORT`

## Next distinct mechanism

The active Football Context sequence now leaves historical analog reliability and
moves to the already-certified **advanced geometry** frontier. The first candidate
family is BDB2026 targeted-receiver release geometry, whose strict-prior
materializers and reconnaissance evidence already exist.

Before any predictive outcome test, the next lane must perform a formal,
outcome-free qualification/redundancy audit against canonical production/context
state and preserve the WR Issue #535 boundary. No target-game geometry, sportsbook
data, or production change is authorized.
