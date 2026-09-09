# RB R26E Week-1 Component Qualification V1 — Implementation Lock

Date: 2026-09-09
Status: LOCKED BEFORE EXECUTION

Frozen plan: `docs/research/RB_R26E_WEEK1_COMPONENT_QUALIFICATION_V1_FROZEN_PLAN.md`
Evaluator: `scripts/backtest/finalize_rb_r26e_week1_component_qualification_v1.py`
Immutable parent: R26 run `34356222339`, artifact `10106271075`, digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`.

Implementation contract:
- download and verify the immutable R26 parent artifact;
- read only the already-generated 2020-2025 R26 prediction/structural files;
- score only `week == 1` rows;
- primary cohort is vacancy-active same-team incumbents;
- do not fit or refit R9;
- do not regenerate predictions;
- do not use R26D's failed significance router;
- do not alter any coefficient, threshold, prediction, receiving-yard mean, or R22 behavior;
- apply exactly the 20 gates frozen in the plan;
- a mechanically green workflow must still emit either qualified-shadow or no-shadow scientific disposition;
- production promotion remains unauthorized by R26E regardless of result.
