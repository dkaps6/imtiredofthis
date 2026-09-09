# RB R26I Week-1 Selective Restoration V1 — Implementation Lock

Date: 2026-09-09
Status: LOCKED BEFORE EXECUTION

Frozen plan: `docs/research/RB_R26I_WEEK1_SELECTIVE_RESTORATION_V1_FROZEN_PLAN.md`
Evaluator: `scripts/backtest/finalize_rb_r26i_week1_selective_restoration_v1.py`

Immutable parents:
- R26 run `34356222339`, artifact `10106271075`, digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`
- R26H run `34369024680`, artifact `10111095834`, digest `sha256:c3fa00de4262bb98b1f73d6931005f963d0ded4cc9a9fe53734ae421e34ef18f`

Implementation contract:
- use R26H's persisted balanced-room source state directly; do not recompute or redefine significance/entry state;
- no model fit/refit and no regenerated football predictions;
- one complete endpoint per RB team-week: production baseline or original R26;
- non-vacancy -> baseline;
- unbalanced vacancy -> original R26;
- balanced + meaningful exit + (veteran entrant OR no-prior-NFL entrant) -> original R26;
- all other balanced rooms -> baseline;
- apply exactly the 27 frozen gates in the plan/evaluator;
- keep the <=2% single-season worsening gate unchanged;
- a mechanically green workflow may still carry a no-shadow scientific disposition;
- production promotion remains unauthorized regardless of R26I result.
