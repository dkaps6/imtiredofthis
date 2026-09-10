#!/usr/bin/env python3
"""Apply the frozen availability-aware QB C2 primary-QB coverage validation.

Mechanical certification seam only. This transformer must be applied after
``apply_current_availability_qb_c2_eligible_team_seam_v1.py``. It replaces only
the downstream legacy 32-team primary-QB coverage assertion in
``apply_qb_c2_selector()``. Starter ranking/authority, complete state-context
source integrity, selector science, C2 distributions and sportsbook boundaries
remain unchanged.
"""
from pathlib import Path

PATH = Path("scripts/modeling/qb_c2_production_adapter_v1.py")
OLD = '''    primary = frame.loc[pd.to_numeric(frame["qb_projection_eligible"], errors="coerce").eq(1)].copy()\n    if len(primary) != 32 or primary["team"].nunique() != 32:\n        raise RuntimeError("QB C2 production adapter did not resolve exactly one primary QB per team")\n'''
NEW = '''    primary = frame.loc[pd.to_numeric(frame["qb_projection_eligible"], errors="coerce").eq(1)].copy()\n    from scripts.utils.eligible_team_set_v1 import validate_current_team_set\n    validate_current_team_set(primary["team"].astype(str), label="QB C2 primary QB coverage")\n'''


def main() -> int:
    text = PATH.read_text(encoding="utf-8")
    if text.count(OLD) != 1:
        raise RuntimeError(f"protected QB C2 primary coverage anchor count={text.count(OLD)}")
    # The first frozen QB seam must already have removed the starter-audit guard.
    starter_legacy = 'raise RuntimeError(f"QB C2 starter authority must cover 32 teams, got rows={len(audit)}")'
    if starter_legacy in text:
        raise RuntimeError("first QB C2 starter-audit availability seam was not applied before primary seam")
    starter_new = 'validate_current_team_set(audit["team"].astype(str), label="QB C2 starter authority")'
    if starter_new not in text:
        raise RuntimeError("first QB C2 starter-audit availability seam anchor missing")

    out = text.replace(OLD, NEW)

    # Complete source context remains a 32-team scientific/source-integrity contract.
    state_guard = 'raise RuntimeError(f"QB C2 state context must cover exactly 32 teams, got rows={len(ctx)}")'
    if state_guard not in out:
        raise RuntimeError("QB C2 state-context source integrity guard changed unexpectedly")
    if out.count('validate_current_team_set(primary["team"].astype(str), label="QB C2 primary QB coverage")') != 1:
        raise RuntimeError("QB C2 primary availability seam was not installed exactly once")

    PATH.write_text(out, encoding="utf-8")
    print("CURRENT_AVAILABILITY_QB_C2_PRIMARY_ELIGIBLE_TEAM_SEAM_TRANSFORM_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
