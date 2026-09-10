# RB R27C2 — Realized Target-Quality Forensic V1 Implementation Lock

Status: `IMPLEMENTATION LOCKED / RUN1 MECHANICAL FAILURE PRESERVED / IMPORT-PATH REPAIR V1 LOCKED`

This lock authorizes diagnostic-only execution. It does not authorize a predictive candidate or any production mutation.

## Parent authorities

- Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- R27C result commit: `b2907345c50dceffdfb5c074ce9d556ff6d764c6`
- R27C run: `34430754033`
- R27C artifact: `10134387002`
- R27C digest: `sha256:0c915a1c413d66c550990a1cc8bd45e0dc20a18e110fe1b96414a79eb7b598e9`
- R27B V2 run: `34428917229`
- R27B V2 artifact: `10134023092`
- R27B V2 digest: `sha256:1df1a14c9900b890fe86e04269849dc90d76fc03a9215ad8612c1820dee5341f`

## Frozen R27C2 blobs

- Plan: `docs/research/RB_R27C2_REALIZED_TARGET_QUALITY_FORENSIC_V1_FROZEN_PLAN.md`
- Plan blob: `57e26c75f8f551c28c86d49c55375ef4f590b6e9`
- Script: `scripts/backtest/forensic_rb_r27c2_realized_target_quality_v1.py`
- Script blob: `179cabe788afc9d995a827b597ba269fb7dcdded`
- Workflow: `.github/workflows/research-rb-r27c2-realized-target-quality-forensic-v1.yml`
- Run1 workflow blob: `cc1b20cd04a7134153e9ba9fe6be8e685cbe3fc0`
- Import-path repair V1 workflow blob: `26ca2ea9918f416d788567c190cf3587ea335911`

## Preserved Run1 mechanical failure

- Run: `34431105624`
- Job: `102726619356`
- Head: `aa9406750e23efa77f8186c3054512d55a7ff316`
- Failure: `ModuleNotFoundError: No module named 'scripts'`
- Failure occurred before PBP loading, cohort calculation, or forensic output.
- Exact repair record: `docs/research/RB_R27C2_RUN1_IMPORT_PATH_MECHANICAL_REPAIR.md`

The only repair is to invoke the identical pinned diagnostic script with repository root on the import path:

`PYTHONPATH=. python scripts/backtest/forensic_rb_r27c2_realized_target_quality_v1.py ...`

No diagnostic code or scientific/forensic design changed.

## Frozen execution rules

1. Verify exact R27B V2 and R27C artifacts by ID/digest.
2. Use R27B V2 preserved predictions as the row authority.
3. Reconstruct REG target-game PBP for 2020–2025 only.
4. Use target-game PBP solely as retrospective diagnostic labels.
5. Require at least 0.98 targeted-player-game PBP join coverage for each primary RB1/RB2+ cohort.
6. Materialize only the target-quality/decomposition tables frozen in the plan.
7. No model fit.
8. No hyperparameter tuning.
9. No new candidate projection.
10. No sportsbook input.
11. No production, R26 or R22 mutation.
12. Preserve exact first valid canonical output artifact and digest.

The machine disposition is intentionally conservative. Final forensic interpretation may use either non-production terminal label allowed in the frozen plan, based only on the pre-specified evidence tables.
