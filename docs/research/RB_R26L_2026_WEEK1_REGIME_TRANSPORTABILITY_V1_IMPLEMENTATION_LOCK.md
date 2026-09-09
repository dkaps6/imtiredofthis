# RB R26L 2026 Week-1 Source-Regime Transportability V1 — Implementation Lock

Status: LOCKED BEFORE 2026 AGGREGATE EXECUTION
Date: 2026-09-09

Frozen plan:
`docs/research/RB_R26L_2026_WEEK1_REGIME_TRANSPORTABILITY_V1_FROZEN_PLAN.md`

Evaluator:
`scripts/backtest/audit_rb_r26l_2026_week1_regime_transportability_v1.py`

## Immutable parents

Historical R26J:
- run `34374987828`
- artifact `10113466373`
- digest `sha256:7d886f228487e052ee1617cc5f42b974cd9da6a9d448f3a86b1de82cb4fc4f46`

Current production Full Slate:
- run `34317211395`
- artifact `10090547415`
- name `run_34317211395`
- digest `sha256:7eab77e41c5879d4f54d87497eee0d1186010784cfc4ef966178930e16fb7c3b`
- head `f8417f55b04ce0e19baf260e9d532765034c47f1`

Production comparison base:
`f8417f55b04ce0e19baf260e9d532765034c47f1`

## Locked source semantics

Primary classification state:
- 2026 Week 1 only;
- canonical weekly roster source already used by R26 historical research;
- canonical status after normalization: ACT/INA only;
- prior snapshot must resolve exactly to 2025 Week 18;
- same-week depth is not loaded into the classifier;
- departed-player receiving state uses strict-as-of R9 identity runtime.

Primary features are exactly the seven frozen in the plan. No production/Ourlads-only field may enter the primary classification.

The post-merge Full Slate artifact is used only for secondary production-current diagnostics. Only `roles_ourlads.csv` and `player_form_consensus.csv` are opened by the evaluator for this purpose. Those diagnostics are stamped non-decisive.

## Locked distance semantics

Per-feature preference, scale floors, normalized distances, 5-of-7 requirement, 0.75 aggregate-distance ratio, and <=2 beyond-2020-anomalous-direction rule are exactly those in the frozen plan.

No feature weight is fit.

## Authority ceiling

Even `2026_SOURCE_REGIME_MODERN_LIKE_FOR_PROSPECTIVE_QUALIFICATION` authorizes only design of a separately frozen R26M qualification/synthesis study.

R26L can never directly authorize:
- shadow;
- production;
- exclusion of 2020;
- R9 refit;
- prediction changes;
- R22/receiving-yard-mean changes.
