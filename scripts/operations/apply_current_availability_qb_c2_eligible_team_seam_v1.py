#!/usr/bin/env python3
"""Apply the frozen availability-aware team-set validation to QB C2 starter audit.

Mechanical certification seam only. It replaces one legacy 32-team coverage guard
in annotate_primary_qbs() and leaves all starter selection, authority priority,
state context and C2 distribution logic byte-for-byte otherwise unchanged.
"""
from pathlib import Path
PATH=Path("scripts/modeling/qb_c2_production_adapter_v1.py")
OLD='''    audit = pd.DataFrame(rows).sort_values("team").reset_index(drop=True)\n    if len(audit) != 32 or audit["team"].nunique() != 32:\n        raise RuntimeError(f"QB C2 starter authority must cover 32 teams, got rows={len(audit)}")\n    if not audit["sportsbook_inputs_used"].eq(0).all():\n'''
NEW='''    audit = pd.DataFrame(rows).sort_values("team").reset_index(drop=True)\n    from scripts.utils.eligible_team_set_v1 import validate_current_team_set\n    validate_current_team_set(audit["team"].astype(str), label="QB C2 starter authority")\n    if not audit["sportsbook_inputs_used"].eq(0).all():\n'''
def main()->int:
    text=PATH.read_text(encoding="utf-8")
    if OLD not in text: raise RuntimeError("protected QB C2 legacy coverage anchor not found exactly once")
    if text.count(OLD)!=1: raise RuntimeError(f"protected QB C2 coverage anchor count={text.count(OLD)}")
    out=text.replace(OLD,NEW)
    anchor='raise RuntimeError(f"QB C2 state context must cover exactly 32 teams, got rows={len(ctx)}")'
    if anchor not in out: raise RuntimeError("QB C2 state-context source integrity guard changed unexpectedly")
    PATH.write_text(out,encoding="utf-8"); print("CURRENT_AVAILABILITY_QB_C2_ELIGIBLE_TEAM_SEAM_TRANSFORM_PASS"); return 0
if __name__=="__main__": raise SystemExit(main())
