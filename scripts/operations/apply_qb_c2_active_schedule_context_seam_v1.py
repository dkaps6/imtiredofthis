#!/usr/bin/env python3
"""Versioned QB C2 source-context availability seam; run AFTER both frozen starter seams.

The predecessor availability transforms deliberately assert the OLD 32-team
source anchor remains intact. This script only replaces that anchor afterward,
with a stricter schedule/roster/opponent parity validator for active bye weeks.
No C2 selector/means/distribution/football features are touched.
"""
from pathlib import Path

PATH=Path("scripts/modeling/qb_c2_production_adapter_v1.py")
OLD='''    if len(ctx) != 32 or ctx["team"].nunique() != 32 or ctx.duplicated("team").any():
        raise RuntimeError(f"QB C2 state context must cover exactly 32 teams, got rows={len(ctx)}")
'''
NEW='''    from scripts.utils.qb_c2_active_team_context_v1 import validate_qb_c2_state_context
    active_context_audit = validate_qb_c2_state_context(
        ctx, season=int(season), week=int(week),
    )
    if active_context_audit["sportsbook_inputs_used"] != 0:
        raise RuntimeError("QB C2 active state context sportsbook integrity failure")
'''

def main()->int:
    data=PATH.read_text(encoding="utf-8")
    if data.count(OLD)!=1:
        raise RuntimeError(f"QB C2 immutable source guard count={data.count(OLD)}; fail closed")
    # Both previously certified availability seams must have already run.
    if data.count('validate_current_team_set(primary["team"].astype(str), label="QB C2 primary projection")')!=1:
        raise RuntimeError("C2 primary availability seam not applied before source guard")
    if data.count('validate_current_team_set(audit["team"].astype(str), label="QB C2 starter authority")')!=1:
        raise RuntimeError("C2 starter availability seam not applied before source guard")
    output=data.replace(OLD,NEW)
    assert output.count(NEW)==1
    PATH.write_text(output,encoding="utf-8")
    print("QB_C2_ACTIVE_SCHEDULE_CONTEXT_V1_SEAM_TRANSFORM_PASS")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
