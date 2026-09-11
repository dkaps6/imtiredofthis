#!/usr/bin/env python3
"""Apply the frozen current-availability seam to downstream certification gates.

The football/model adapters already support the explicit current-eligible team
set. Several downstream governance scripts retained legacy Week-1 32-team /
full-roster cardinality assertions even when the certified current-availability
artifact intentionally withholds teams whose games have already been played.

This transformer changes coverage assertions only. It does not alter model
parameters, simulations, entitlements, sportsbook values, or scientific model
versions. Every replacement is anchored exactly once and fails closed if source
shape drifts.
"""
from __future__ import annotations

from pathlib import Path

LINEAGE_V1 = Path("scripts/audit_market_model_lineage_v1.py")
LINEAGE_V2 = Path("scripts/audit_market_model_lineage_v2_core.py")
LINEAGE_V3 = Path("scripts/audit_market_model_lineage_v3.py")
TARGET_POOL = Path("scripts/validate_team_target_pool_full_universe_v2.py")
STACK_V1 = Path("scripts/validate_certified_full_slate_stack_v1.py")
STACK_V2 = Path("scripts/validate_certified_full_slate_stack_v2_core.py")
STACK_V3 = Path("scripts/validate_certified_full_slate_stack_v3.py")


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one frozen source anchor, found {count}")
    return text.replace(old, new, 1)


def _transform(path: Path, operations: list[tuple[str, str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    for old, new, label in operations:
        text = _replace_once(text, old, new, label)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    import_anchor = "import pandas as pd\n"
    import_new = import_anchor + "\nfrom scripts.utils.eligible_team_set_v1 import expected_current_teams\n"

    lineage_scope_anchor = '''    stamp = _read_json(DATA / "qb_c2_pricing_lineage_stamp_audit.json")\n'''
    lineage_scope_new = lineage_scope_anchor + '''    current_teams = expected_current_teams()\n    expected_qbs = len(current_teams) if current_teams is not None else 32\n'''
    lineage_c2_old = '''    if int(c2.get("football_qb_rows", 0)) != 32 or int(c2.get("selected_qb_rows", 0)) <= 0:\n        raise RuntimeError(f"QB C2 production coverage invalid: {c2}")\n'''
    lineage_c2_new = '''    if int(c2.get("football_qb_rows", 0)) != expected_qbs or int(c2.get("selected_qb_rows", 0)) <= 0:\n        raise RuntimeError(f"QB C2 production coverage invalid expected={expected_qbs}: {c2}")\n'''
    lineage_stamp_old = '''    if int(stamp.get("pass_yard_qbs", 0)) != 32:\n        raise RuntimeError("QB C2 pricing stamp does not cover 32 QBs")\n    if int(stamp.get("c2_selected_qbs", -1)) != int(c2.get("selected_qb_rows", -2)):\n        raise RuntimeError("QB C2 selected-QB count differs between simulation and pricing stamp")\n'''
    lineage_stamp_new = '''    pass_yard_qbs = int(stamp.get("pass_yard_qbs", 0))\n    if current_teams is None:\n        if pass_yard_qbs != 32:\n            raise RuntimeError("QB C2 pricing stamp does not cover 32 QBs")\n        if int(stamp.get("c2_selected_qbs", -1)) != int(c2.get("selected_qb_rows", -2)):\n            raise RuntimeError("QB C2 selected-QB count differs between simulation and pricing stamp")\n    else:\n        if int(stamp.get("current_team_scope_expected", -1)) != expected_qbs:\n            raise RuntimeError("QB C2 pricing stamp current-team scope drift")\n        if int(stamp.get("football_qbs", -1)) != expected_qbs:\n            raise RuntimeError("QB C2 pricing stamp football-QB scope drift")\n        if int(stamp.get("c2_selected_football_qbs", -1)) != int(c2.get("selected_qb_rows", -2)):\n            raise RuntimeError("QB C2 selected football-QB count differs between simulation and pricing stamp")\n        if pass_yard_qbs <= 0 or pass_yard_qbs > expected_qbs:\n            raise RuntimeError("QB C2 priced pass-yard QB subset is invalid")\n'''
    lineage_priced_old = '''    if qb[["team", "player"]].drop_duplicates().shape[0] != 32:\n        raise RuntimeError("QB C2 priced pass-yard rows do not cover 32 unique QBs")\n'''
    lineage_priced_new = '''    priced_qbs = int(qb[["team", "player"]].drop_duplicates().shape[0])\n    expected_priced_qbs = 32 if current_teams is None else pass_yard_qbs\n    if priced_qbs != expected_priced_qbs:\n        raise RuntimeError(\n            f"QB C2 priced pass-yard QB coverage drift expected={expected_priced_qbs} actual={priced_qbs}"\n        )\n'''
    lineage_selected_old = '''    if selected_qbs != int(c2["selected_qb_rows"]):\n        raise RuntimeError(f"priced QB C2 selected count drift expected={c2['selected_qb_rows']} actual={selected_qbs}")\n'''
    lineage_selected_new = '''    expected_selected_priced = int(stamp.get("c2_selected_qbs", c2["selected_qb_rows"]))\n    if selected_qbs != expected_selected_priced:\n        raise RuntimeError(\n            f"priced QB C2 selected count drift expected={expected_selected_priced} actual={selected_qbs}"\n        )\n'''
    _transform(LINEAGE_V1, [
        (import_anchor, import_new, "lineage-v1 current-team import"),
        (lineage_scope_anchor, lineage_scope_new, "lineage-v1 current-team scope"),
        (lineage_c2_old, lineage_c2_new, "lineage-v1 C2 football-QB coverage"),
        (lineage_stamp_old, lineage_stamp_new, "lineage-v1 pricing-stamp coverage"),
        (lineage_priced_old, lineage_priced_new, "lineage-v1 priced-QB coverage"),
        (lineage_selected_old, lineage_selected_new, "lineage-v1 selected priced-QB coverage"),
    ])

    lineage_v2_old = '''    if int(wr.get("current_wr1_anchor_rows", 0)) != 32:\n        raise RuntimeError("WR-R15 does not have exactly one M38 WR1 anchor per team")\n'''
    lineage_v2_new = '''    expected_wr1_anchors = int(c2.get("football_qb_rows", 0))\n    if expected_wr1_anchors <= 0 or int(wr.get("current_wr1_anchor_rows", 0)) != expected_wr1_anchors:\n        raise RuntimeError(\n            f"WR-R15 current WR1 anchor coverage drift expected={expected_wr1_anchors} "\n            f"actual={wr.get('current_wr1_anchor_rows')}"\n        )\n'''
    _transform(LINEAGE_V2, [
        (lineage_v2_old, lineage_v2_new, "lineage-v2 WR1 current-team coverage"),
    ])

    r22_adapter_old = '''    _require(int(adapter.get("adapted_rb_rows", 0)) == 94, "R22 did not adapt all 94 Week-1 RBs")\n'''
    r22_adapter_new = '''    expected_r22_players = int(adapter.get("football_rb_rows", 0))\n    _require(expected_r22_players > 0, "R22 adapter reports zero football RB rows")\n    _require(\n        int(adapter.get("adapted_rb_rows", 0)) == expected_r22_players,\n        f"R22 adapted-RB coverage drift expected={expected_r22_players} actual={adapter.get('adapted_rb_rows')}",\n    )\n'''
    r22_pricing_old = '''    _require(int(pricing.get("adapted_player_keys", 0)) == 94, "R22 pricing lineage RB coverage drift")\n'''
    r22_pricing_new = '''    _require(\n        int(pricing.get("adapted_player_keys", 0)) == expected_r22_players,\n        f"R22 pricing RB coverage drift expected={expected_r22_players} actual={pricing.get('adapted_player_keys')}",\n    )\n'''
    r22_trace_old = '''    _require(int(trace["rb_receiving_tail_applied"].astype(bool).sum()) == 94, "R22 trace does not contain 94 adapted RBs")\n'''
    r22_trace_new = '''    _require(\n        int(trace["rb_receiving_tail_applied"].astype(bool).sum()) == expected_r22_players,\n        f"R22 trace adapted-RB coverage drift expected={expected_r22_players}",\n    )\n'''
    r22_payload_old = '''        "rb_r22_adapted_rb_rows": 94,\n'''
    r22_payload_new = '''        "rb_r22_adapted_rb_rows": expected_r22_players,\n'''
    _transform(LINEAGE_V3, [
        (r22_adapter_old, r22_adapter_new, "lineage-v3 R22 adapter coverage"),
        (r22_pricing_old, r22_pricing_new, "lineage-v3 R22 pricing coverage"),
        (r22_trace_old, r22_trace_new, "lineage-v3 R22 trace coverage"),
        (r22_payload_old, r22_payload_new, "lineage-v3 R22 payload coverage"),
    ])

    target_import_new = import_anchor + "\nfrom scripts.utils.eligible_team_set_v1 import expected_current_teams\n"
    target_scope_anchor = '''    source = json.loads(UNIVERSE_AUDIT.read_text(encoding="utf-8"))\n'''
    target_scope_new = target_scope_anchor + '''    current_teams = expected_current_teams()\n    expected_team_count = len(current_teams) if current_teams is not None else 32\n'''
    target_frame_old = '''    if frame["team"].nunique() != 32:\n        raise RuntimeError(f"target audit expected 32 teams, found {frame['team'].nunique()}")\n'''
    target_frame_new = '''    if frame["team"].nunique() != expected_team_count:\n        raise RuntimeError(\n            f"target audit expected {expected_team_count} teams, found {frame['team'].nunique()}"\n        )\n'''
    target_trace_old = '''        if trace["team"].nunique() != 32:\n            raise RuntimeError(f"explicit entitlement expected 32 teams, found {trace['team'].nunique()}")\n'''
    target_trace_new = '''        if trace["team"].nunique() != expected_team_count:\n            raise RuntimeError(\n                f"explicit entitlement expected {expected_team_count} teams, found {trace['team'].nunique()}"\n            )\n'''
    target_physical_old = '''        if len(physical) != 32:\n            raise RuntimeError(f"explicit entitlement physical audit expected 32 teams, found {len(physical)}")\n'''
    target_physical_new = '''        if len(physical) != expected_team_count:\n            raise RuntimeError(\n                f"explicit entitlement physical audit expected {expected_team_count} teams, found {len(physical)}"\n            )\n'''
    target_fail_explicit_old = '''            f"Explicit full-roster target entitlement is physically invalid for {len(explicit_bad)}/32 teams; see {OUT_CSV}."\n'''
    target_fail_explicit_new = '''            f"Explicit full-roster target entitlement is physically invalid for "\n            f"{len(explicit_bad)}/{expected_team_count} teams; see {OUT_CSV}."\n'''
    target_fail_raw_old = '''            f"Full-roster receiving entitlement pool is physically invalid for {len(raw_bad)}/32 teams; "\n'''
    target_fail_raw_new = '''            f"Full-roster receiving entitlement pool is physically invalid for "\n            f"{len(raw_bad)}/{expected_team_count} teams; "\n'''
    _transform(TARGET_POOL, [
        (import_anchor, target_import_new, "target-pool current-team import"),
        (target_scope_anchor, target_scope_new, "target-pool current-team scope"),
        (target_frame_old, target_frame_new, "target-pool football team coverage"),
        (target_trace_old, target_trace_new, "target-pool trace team coverage"),
        (target_physical_old, target_physical_new, "target-pool physical team coverage"),
        (target_fail_explicit_old, target_fail_explicit_new, "target-pool explicit failure denominator"),
        (target_fail_raw_old, target_fail_raw_new, "target-pool legacy failure denominator"),
    ])

    stack_import_anchor = "import pandas as pd\n"
    stack_import_new = stack_import_anchor + "\nfrom scripts.utils.eligible_team_set_v1 import expected_current_teams\n"
    stack_scope_anchor = '''    rb_final = _csv(DATA / "rb_rush_rec_conservation_final_audit.csv")\n'''
    stack_scope_new = stack_scope_anchor + '''\n    current_teams = expected_current_teams()\n    expected_team_count = len(current_teams) if current_teams is not None else 32\n    expected_game_count = expected_team_count // 2\n'''
    stack_football_old = '''    _require(int(football.get("football_players", 0)) == 469, f"football player universe drifted: {football.get('football_players')}")\n    _require(int(football.get("football_teams", 0)) == 32, "football universe does not cover 32 teams")\n    _require(int(football.get("canonical_games", 0)) == 16, "football universe does not cover 16 games")\n'''
    stack_football_new = '''    if current_teams is None:\n        _require(int(football.get("football_players", 0)) == 469, f"football player universe drifted: {football.get('football_players')}")\n    else:\n        football_players = int(football.get("football_players", 0))\n        _require(football_players > 0, "football universe contains zero players")\n        _require(football_players == int(football.get("football_player_rows", -1)), "football player row-count drift")\n        _require(football_players == int(football.get("model_context_players", -1)), "football/model-context player coverage drift")\n    _require(\n        int(football.get("football_teams", 0)) == expected_team_count,\n        f"football universe team coverage drift expected={expected_team_count} actual={football.get('football_teams')}",\n    )\n    _require(\n        int(football.get("canonical_games", 0)) == expected_game_count,\n        f"football universe game coverage drift expected={expected_game_count} actual={football.get('canonical_games')}",\n    )\n'''
    stack_c2_old = '''    _require(int(c2.get("football_qb_rows", 0)) == 32, "QB C2 does not cover 32 football starters")\n'''
    stack_c2_new = '''    _require(\n        int(c2.get("football_qb_rows", 0)) == expected_team_count,\n        f"QB C2 football starter coverage drift expected={expected_team_count} actual={c2.get('football_qb_rows')}",\n    )\n'''
    stack_stamp_old = '''    _require(int(stamp.get("pass_yard_qbs", 0)) == 32, "QB C2 pricing lineage does not cover 32 QBs")\n'''
    stack_stamp_new = '''    if current_teams is None:\n        _require(int(stamp.get("pass_yard_qbs", 0)) == 32, "QB C2 pricing lineage does not cover 32 QBs")\n    else:\n        _require(int(stamp.get("current_team_scope_expected", -1)) == expected_team_count, "QB C2 pricing current-team scope drift")\n        _require(int(stamp.get("football_qbs", -1)) == expected_team_count, "QB C2 pricing football-QB scope drift")\n        _require(\n            int(stamp.get("c2_selected_football_qbs", -1)) == int(c2.get("selected_qb_rows", -2)),\n            "QB C2 selected football-QB count differs between simulation and pricing stamp",\n        )\n        _require(\n            0 < int(stamp.get("pass_yard_qbs", 0)) <= expected_team_count,\n            "QB C2 priced pass-yard QB subset is invalid",\n        )\n'''
    _transform(STACK_V1, [
        (stack_import_anchor, stack_import_new, "stack-v1 current-team import"),
        (stack_scope_anchor, stack_scope_new, "stack-v1 current-team scope"),
        (stack_football_old, stack_football_new, "stack-v1 football universe coverage"),
        (stack_c2_old, stack_c2_new, "stack-v1 C2 coverage"),
        (stack_stamp_old, stack_stamp_new, "stack-v1 pricing-stamp coverage"),
    ])

    stack_v2_old = '''    _require(int(wr.get("current_wr1_anchor_rows", 0)) == 32, "WR-R15 does not have 32 WR1 anchors")\n'''
    stack_v2_new = '''    expected_wr1_anchors = int(c2.get("football_qb_rows", 0))\n    _require(expected_wr1_anchors > 0, "WR-R15 current-team authority has zero teams")\n    _require(\n        int(wr.get("current_wr1_anchor_rows", 0)) == expected_wr1_anchors,\n        f"WR-R15 current WR1 anchor coverage drift expected={expected_wr1_anchors} "\n        f"actual={wr.get('current_wr1_anchor_rows')}",\n    )\n'''
    _transform(STACK_V2, [
        (stack_v2_old, stack_v2_new, "stack-v2 WR1 current-team coverage"),
    ])

    stack_v3_adapter_old = '''    _require(int(adapter.get("adapted_rb_rows", 0)) == 94, "R22 adapted-RB coverage drift")\n'''
    stack_v3_adapter_new = '''    expected_r22_players = int(adapter.get("football_rb_rows", 0))\n    _require(expected_r22_players > 0, "R22 adapter reports zero football RB rows")\n    _require(\n        int(adapter.get("adapted_rb_rows", 0)) == expected_r22_players,\n        f"R22 adapted-RB coverage drift expected={expected_r22_players} actual={adapter.get('adapted_rb_rows')}",\n    )\n'''
    stack_v3_pricing_old = '''    _require(int(pricing.get("adapted_player_keys", 0)) == 94, "R22 pricing RB coverage drift")\n'''
    stack_v3_pricing_new = '''    _require(\n        int(pricing.get("adapted_player_keys", 0)) == expected_r22_players,\n        f"R22 pricing RB coverage drift expected={expected_r22_players} actual={pricing.get('adapted_player_keys')}",\n    )\n'''
    stack_v3_payload_old = '''        "rb_r22_adapted_rb_rows": 94,\n'''
    stack_v3_payload_new = '''        "rb_r22_adapted_rb_rows": expected_r22_players,\n'''
    _transform(STACK_V3, [
        (stack_v3_adapter_old, stack_v3_adapter_new, "stack-v3 R22 adapter coverage"),
        (stack_v3_pricing_old, stack_v3_pricing_new, "stack-v3 R22 pricing coverage"),
        (stack_v3_payload_old, stack_v3_payload_new, "stack-v3 R22 payload coverage"),
    ])

    print("CURRENT_AVAILABILITY_DOWNSTREAM_CERTIFICATION_SEAM_TRANSFORM_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
