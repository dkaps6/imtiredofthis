#!/usr/bin/env python3
"""Apply frozen availability-aware team validation to QB C2 primary frame.

Mechanical certification seam only. It replaces exactly the second legacy 32-team
current-slate coverage guard in apply_qb_c2_selector(); all selection/model logic
and the 32-team state-context source-integrity guard remain untouched.
"""
from pathlib import Path
PATH=Path("scripts/modeling/qb_c2_production_adapter_v1.py")
OLD='''    primary = frame.loc[pd.to_numeric(frame["qb_projection_eligible"], errors="coerce").eq(1)].copy()\n    if len(primary) != 32 or primary["team"].nunique() != 32:\n        raise RuntimeError("QB C2 production adapter did not resolve exactly one primary QB per team")\n\n    anchors: dict[tuple[str, str], float] = {}\n'''
NEW='''    primary = frame.loc[pd.to_numeric(frame["qb_projection_eligible"], errors="coerce").eq(1)].copy()\n    from scripts.utils.eligible_team_set_v1 import validate_current_team_set\n    validate_current_team_set(primary["team"].astype(str), label="QB C2 primary projection")\n\n    anchors: dict[tuple[str, str], float] = {}\n'''
def main()->int:
    text=PATH.read_text(encoding="utf-8")
    if text.count(OLD)!=1: raise RuntimeError(f"QB C2 primary coverage anchor count={text.count(OLD)}")
    out=text.replace(OLD,NEW)
    if 'QB C2 state context must cover exactly 32 teams' not in out: raise RuntimeError("QB C2 state-context 32-team source guard changed unexpectedly")
    PATH.write_text(out,encoding="utf-8"); print("CURRENT_AVAILABILITY_QB_C2_PRIMARY_TEAM_SEAM_TRANSFORM_PASS"); return 0
if __name__=="__main__": raise SystemExit(main())
