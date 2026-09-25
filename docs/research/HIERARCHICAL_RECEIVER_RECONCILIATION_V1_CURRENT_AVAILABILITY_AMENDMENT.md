# Hierarchical Receiver Reconciliation V1 — Current Availability Seam Amendment

Date: 2026-09-24

Status: **FROZEN BEFORE OUTCOME SCORING**

The first read-only audit run `36080048529` stopped before producing any
reconciliation output.

Exact stop:
`synthetic football-only comparison frame expected 32 teams, got 30`

This is a mechanical current-slate eligibility mismatch.

## Existing production authority

Current production already has a frozen availability-aware team-set contract:

`scripts/utils/eligible_team_set_v1.py`

When `ACTIVE_ROLES_CSV` is explicitly configured:
- the certified active-role artifact is the sole authority for the current
  production-eligible team set;
- the production football universe must match that set exactly;
- the production QB primary/starter set must match that set exactly;
- the separate QB state-context source remains a complete 32-team integrity
  source.

The production seam is applied by:
- `scripts/operations/apply_current_availability_eligible_team_seam_v1.py`
- `scripts/operations/apply_current_availability_qb_c2_eligible_team_seam_v1.py`
- `scripts/operations/apply_current_availability_qb_c2_primary_team_seam_v1.py`

## Frozen research amendment

The read-only reconciliation audit must inherit the same production contract.

Therefore:
1. replace the research helper's legacy hard-coded 32-team synthetic-frame check
   with `validate_current_team_set(...)`;
2. require the research football frame to equal the certified
   `ACTIVE_ROLES_CSV` team set exactly;
3. preserve the full 32-team QB state-context source-integrity check;
4. do not add, restore or impute any kicked-off/withheld team;
5. do not change any reconciliation equation, uncertainty weight, output metric
   or interpretation rule.

This amendment changes only current-slate mechanical scope.

No reconciliation output or target-game outcome was observed before this
amendment.
