#!/usr/bin/env python3
"""Apply the frozen current-availability eligible-team coverage seam.

This transformer is intentionally surgical. It changes only legacy current-
coverage/control assumptions that require the full weekly universe even after
availability has certified a smaller current production slate. It refuses to
operate unless all exact source anchors are present once and refuses to touch
model parameters, assets, entitlement math, or sportsbook behavior.
"""
from __future__ import annotations

import argparse
from pathlib import Path

FULL = Path("scripts/run_pricing_with_full_roster_universe_v1.py")
R26 = Path("scripts/modeling/rb_r26_receptions_production_adapter_v1.py")

FULL_IMPORT_ANCHOR = "from scripts.simulation_v2 import MARKET_MAP, _player_key, lookup, simulate as canonical_simulate\n"
FULL_IMPORT = FULL_IMPORT_ANCHOR + "from scripts.utils.eligible_team_set_v1 import validate_current_team_set\n"
FULL_OLD = '''    if form["team"].nunique() != 32:\n        raise RuntimeError(f"football simulation universe must cover 32 teams, found {form['team'].nunique()}")\n'''
FULL_NEW = '''    team_coverage = validate_current_team_set(\n        form["team"].dropna().astype(str).unique(),\n        label="football simulation universe",\n    )\n'''
FULL_EVENT_ANCHOR = '''    form["event_id"] = [\n        _canonical_game(t, o, s, w)\n        for t, o, s, w in zip(form["team"], form["opponent"], form["season"], form["week"])\n    ]\n    form["market"] = "football_universe"\n'''
FULL_EVENT_NEW = '''    form["event_id"] = [\n        _canonical_game(t, o, s, w)\n        for t, o, s, w in zip(form["team"], form["opponent"], form["season"], form["week"])\n    ]\n    expected_games = int(team_coverage.get("canonical_games", 16))\n    observed_games = int(form["event_id"].nunique())\n    if observed_games != expected_games:\n        raise RuntimeError(\n            f"football simulation canonical-game coverage invalid expected={expected_games} observed={observed_games}"\n        )\n    form["market"] = "football_universe"\n'''

R26_IMPORT_ANCHOR = '''from scripts.simulation_explicit_entitlement_v1 import simulate as explicit_simulate\n'''
R26_IMPORT = R26_IMPORT_ANCHOR + "from scripts.utils.eligible_team_set_v1 import validate_current_team_set\n"
R26_OLD = '''    rb = frame.loc[frame.position_family.isin(RB_FAMILIES)].copy()\n    if rb.empty or rb["team"].nunique() != 32:\n        raise RuntimeError(f"R26 production RB/FB coverage invalid rows={len(rb)} teams={rb['team'].nunique()}")\n'''
R26_NEW = '''    rb = frame.loc[frame.position_family.isin(RB_FAMILIES)].copy()\n    if rb.empty:\n        raise RuntimeError("R26 production RB/FB coverage has zero rows")\n    validate_current_team_set(\n        rb["team"].dropna().astype(str).unique(),\n        label="R26 production RB/FB coverage",\n    )\n'''
R26_CONTROL_OLD = '''    cin = rb["team"].eq(CONTROL_TEAM)\n    max_cin_entitlement_delta = float(rb.loc[cin, "entitlement_delta"].abs().max()) if cin.any() else float("inf")\n'''
R26_CONTROL_NEW = '''    cin = rb["team"].eq(CONTROL_TEAM)\n    control_team_present = bool(cin.any())\n    # CIN is the frozen non-vacancy scientific control when it is part of the\n    # current certified slate. A kicked-off/withheld CIN game is legitimately\n    # absent from a partial production universe and must not make the adapter\n    # fail solely because an out-of-scope control row is unavailable.\n    max_cin_entitlement_delta = float(rb.loc[cin, "entitlement_delta"].abs().max()) if control_team_present else 0.0\n'''
R26_TRACE_CONTROL_OLD = '''    cin_trace = trace.loc[trace.team.eq(CONTROL_TEAM)]\n    if cin_trace.empty or cin_trace.rb_r26_receptions_applied.any() or cin_trace.final_minus_baseline_receptions_mean.abs().max() > 1e-12:\n        raise RuntimeError("R26 production CIN control did not remain exact baseline")\n'''
R26_TRACE_CONTROL_NEW = '''    cin_trace = trace.loc[trace.team.eq(CONTROL_TEAM)]\n    if not cin_trace.empty and (cin_trace.rb_r26_receptions_applied.any() or cin_trace.final_minus_baseline_receptions_mean.abs().max() > 1e-12):\n        raise RuntimeError("R26 production CIN control did not remain exact baseline")\n'''
R26_AUDIT_CONTROL_OLD = '''        "control_team": CONTROL_TEAM,\n        "applied_rb_fb_rows": int(trace.rb_r26_receptions_applied.sum()),\n'''
R26_AUDIT_CONTROL_NEW = '''        "control_team": CONTROL_TEAM,\n        "control_team_present_in_current_universe": bool(control_team_present),\n        "applied_rb_fb_rows": int(trace.rb_r26_receptions_applied.sum()),\n'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected exactly one frozen source anchor, found {n}")
    return text.replace(old, new, 1)


def transform(path: Path, operations: list[tuple[str, str, str]], *, write: bool) -> str:
    text = path.read_text(encoding="utf-8")
    for old, new, label in operations:
        text = replace_once(text, old, new, label)
    if write:
        path.write_text(text, encoding="utf-8")
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    full = transform(FULL, [
        (FULL_IMPORT_ANCHOR, FULL_IMPORT, "full-universe helper import"),
        (FULL_OLD, FULL_NEW, "full-universe team coverage guard"),
        (FULL_EVENT_ANCHOR, FULL_EVENT_NEW, "full-universe canonical game coverage guard"),
    ], write=not args.check_only)
    r26 = transform(R26, [
        (R26_IMPORT_ANCHOR, R26_IMPORT, "R26 helper import"),
        (R26_OLD, R26_NEW, "R26 team coverage guard"),
        (R26_CONTROL_OLD, R26_CONTROL_NEW, "R26 optional current-slate CIN entitlement control"),
        (R26_TRACE_CONTROL_OLD, R26_TRACE_CONTROL_NEW, "R26 optional current-slate CIN trace control"),
        (R26_AUDIT_CONTROL_OLD, R26_AUDIT_CONTROL_NEW, "R26 current-slate control audit"),
    ], write=not args.check_only)

    if "football simulation universe must cover 32 teams" in full:
        raise RuntimeError("legacy full-universe 32-team guard survived transformation")
    if 'rb["team"].nunique() != 32' in r26:
        raise RuntimeError("legacy R26 32-team guard survived transformation")
    if 'else float("inf")' in r26 or "if cin_trace.empty or" in r26:
        raise RuntimeError("legacy R26 mandatory-current-slate CIN control survived transformation")
    if '"control_team_present_in_current_universe": bool(control_team_present)' not in r26:
        raise RuntimeError("R26 current-slate control presence audit missing after transformation")
    for forbidden in ("EXPECTED_VACANCY_TEAMS =", "EXPECTED_MODEL_SHA256 =", "EXPECTED_ROOM_STATE_SHA256 ="):
        if forbidden not in r26:
            raise RuntimeError(f"R26 protected contract anchor unexpectedly absent: {forbidden}")
    print("CURRENT_AVAILABILITY_ELIGIBLE_TEAM_SEAM_TRANSFORM_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
