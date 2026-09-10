# RB R27C — Vacancy RB1 / 2023 Forensic V1 Implementation Lock

Status: `IMPLEMENTATION LOCKED BEFORE CANONICAL FORENSIC EXECUTION`

This lock authorizes one diagnostic-only execution of the frozen R27C plan. It does not authorize a new predictive candidate, production routing, threshold selection, or mutation of R26/R22/production.

## Frozen authorities

- Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- R27B V2 immutable result-record commit: `4a885207785255e7ffa2b1c33a35ddcaedfe6668`
- R27B V2 first valid run: `34428917229`
- R27B V2 job: `102720004328`
- R27B V2 artifact ID: `10134023092`
- R27B V2 artifact name: `rb-r27b-v2-novel-efficiency-context`
- R27B V2 artifact digest: `sha256:1df1a14c9900b890fe86e04269849dc90d76fc03a9215ad8612c1820dee5341f`
- R27B V2 implementation lock/head: `2c86520c84fbdd5a18aad3f4b88373e5f8c17051`

## Frozen R27C blobs

- Plan file: `docs/research/RB_R27C_RB1_2023_FORENSIC_V1_FROZEN_PLAN.md`
- Plan blob: `dc1883c40dd194c723453d76e1564de0c204a8e9`
- Diagnostic script: `scripts/backtest/forensic_rb_r27c_rb1_2023_v1.py`
- Diagnostic script blob: `b8a97f82e31725e7673957f0c999a2152d4ea993`
- Workflow: `.github/workflows/research-rb-r27c-rb1-2023-forensic-v1.yml`
- Workflow blob: `ad4b6a0cddc2d7004bd2fb67ee05379e525cc7a2`

## Execution constraints

1. Download exact parent artifact by ID and verify exact digest.
2. Consume preserved `r27b_v2_all_predictions.csv`; do not refit or rerun V2.
3. Materialize only the pre-specified R27C diagnostic tables.
4. No new model fit.
5. No new candidate projections.
6. No sportsbook input.
7. No threshold optimization or router selection.
8. No production change.
9. No R26 change.
10. No R22 change.
11. Preserve canonical output artifact and exact digest.
12. Any later predictive mechanism must be separately designed and frozen after R27C interpretation.

The diagnostic script's machine disposition is deliberately conservative; the canonical human-readable R27C result record may choose either allowed non-production forensic terminal label from the frozen plan after reviewing all pre-specified evidence. It may not authorize promotion.
