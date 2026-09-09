# RB R25 — Receptions Specialist V1 Result

Status: **SCIENTIFICALLY RESOLVED — NO PROMOTION**
Date: 2026-09-09
Production base: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Research branch: `research-rb-r25-receptions-specialist-v1`

## Authoritative lineage

- Frozen plan: `docs/research/RB_R25_RECEPTIONS_SPECIALIST_V1_FROZEN_PLAN.md`
- Plan commit: `778c2ee531fbcfe4f1fa3db2f2ea9b322eeb8e58`
- Initial workflow head: `bc12575be6c538b762c889bd2d8d100e7e21dab7`
- Mechanical Run 1: `34347368040`, job `102452054367`
  - failed before R25 scientific scoring because 2019 weekly player stats contained Week-18-labelled rows while the actual 2019 REG schedule ended at Week 17;
  - frozen plan, implementation compile, and protected production-authority diff had already passed;
  - this was a historical scope/join failure, not a football-science result.
- Mechanical repair commit: `a74f8091edf12d3652345a945003509c58c9561c`
  - historical player logs now use exact REG schedule `(season, week, team)` keys as the authority for regular-season scope;
  - no R25 formula, window, pseudo-count, clip, cohort, threshold, gate, or production model changed.
- Authoritative Run 2: `34347752983`
- Authoritative job: `102453305462`
- Authoritative head: `a74f8091edf12d3652345a945003509c58c9561c`
- Artifact: `10102656324`
- Artifact digest: `sha256:2cc95d6dd718a40758ad3d706ae80dbeef67cb0469109ad9b228350f4e8851d3`
- Final disposition: **`MIXED_OR_FAIL_NO_PROMOTION`**

## Scientific design

R25 did not retune R23 after the known 2023-2025 R23/R24 results. It held the exact R23 target-entitlement/catch-conversion mechanism fixed and evaluated it on the earlier untouched 2020-2022 temporal confirmation block.

R25 was deliberately a **targets/receptions-only specialist test**. Receiving-yard point means were forced to exact baseline parity so the already-certified R22 receiving-yard tail could not be affected by this experiment.

## Integrity result

All integrity gates passed:

- sportsbook inputs upstream: **0**
- future/2026 outcomes: **0**
- strict-prior contract: **PASS**
- RB-room target mass conservation: **PASS**
- candidate targets/receptions finite and non-negative: **PASS**
- receiving-yard point-mean parity baseline vs candidate: **EXACT**
- protected production authority remained unchanged.

Therefore the no-promotion result is scientific rather than mechanical.

## Pooled 2020-2022 result

### Targets

| Metric | Baseline | R25 candidate | Direction |
|---|---:|---:|---|
| MAE | 1.361639 | 1.366606 | worse |
| RMSE | 1.893945 | 1.842168 | better |

### Receptions

| Metric | Baseline | R25 candidate | Direction |
|---|---:|---:|---|
| MAE | 1.129705 | 1.131417 | worse |
| RMSE | 1.564107 | 1.518092 | better |
| Bias | -0.302014 | -0.153073 | better |
| p90 abs error | 2.390131 | 2.503985 | worse |
| Spearman | 0.440387 | 0.463293 | better |

The candidate improved dispersion-sensitive RMSE, bias, and ranking, but worsened the primary individual-player MAE and the upper error tail. That is not sufficient for promotion under the frozen objective.

## Temporal replication

Receptions MAE by confirmation season:

- 2020: `1.181403 -> 1.185586` (**+0.354% worse**)
- 2021: `1.113002 -> 1.099354` (**-1.226% better**)
- 2022: `1.095304 -> 1.110699` (**+1.406% worse**)

Only one of three seasons improved, and 2022 exceeded the frozen maximum single-season worsening allowance.

## Role robustness

- RB1: `1.674877 -> 1.700855` (**+1.551% worse**)
- RB2+: `0.987452 -> 0.983368` (**-0.414% better**)

The candidate again transfers accuracy away from lead backs toward complementary backs. That is inconsistent with the required role robustness.

## Predeclared history-cohort diagnostic

This was a predeclared reporting cohort, **not a promotion rescue gate**:

- LOW_HISTORY: `0.876589 -> 0.814129` (**-7.125% improvement**)
- ESTABLISHED: `1.187596 -> 1.217766` (**+2.540% worse**)

This is scientifically interesting but cannot be used post hoc to redefine R25. It suggests the R23 mechanism behaves more like an uncertainty/role-transition aid for sparse-history players than a universal RB entitlement model. Any future test of that mechanism must be a newly frozen hypothesis and must first be reconciled against the prior R10-R12 identity/state research so old failed work is not repeated.

## Frozen-gate disposition

Failed scientific gates included:

- targets MAE improves
- receptions MAE improves by >=0.50%
- receptions p90 protected within 2%
- 2-of-3 directional temporal replication
- no season worse than 1%
- role robustness within 0.75%

Passed scientific protections included:

- target RMSE non-worse
- reception RMSE non-worse
- reception bias protection
- reception Spearman protection
- at least one role improved.

Final: **`MIXED_OR_FAIL_NO_PROMOTION`**.

## Production consequence

Nothing from R25 is authorized for `main`. The production Week-1 V4/R22 stack remains unchanged.

The experiment also confirmed an integration constraint: canonical MC uses shared target allocation for both receptions and receiving yards. Therefore even a future successful receptions specialist cannot simply overwrite RB target shares inside canonical MC without also disturbing R22 receiving-yard authority. Any eventual production integration must isolate the receptions/count path while preserving receiving-yard point mean, R22 tail shape, `rush_rec_yards` identity, and non-RB outputs exactly.

## Next authorized step

1. Audit R10-R12 identity/state research against R25's predeclared low-history vs established-history diagnostic.
2. Do **not** retune or cohort-rescue R25.
3. If the sparse-history/state mechanism was already tested and failed in R10-R12, bound this RB receiving mean/receptions lane for Week 1 and move to the next planned high-value lane (QB attempts/dropbacks/pass-rate/YPA decomposition).
4. Only if R10-R12 reveal a genuinely untested mechanism may a new RB receiving hypothesis be frozen and tested on an outcome-independent cohort.
