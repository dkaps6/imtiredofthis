#!/usr/bin/env python3
"""Apply current-availability scope to legacy downstream certification counts.

This transformer changes governance/coverage assertions only. It does not alter
model parameters, simulations, entitlements, sportsbook values, or promoted
scientific authorities. `audit_market_model_lineage_v1.py` is intentionally NOT
patched here: it now owns a native football-vs-sportsbook QB coverage contract.
"""
from __future__ import annotations

from pathlib import Path

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
    # WR-R15: exactly one current WR1 anchor per current football team.
    _transform(LINEAGE_V2, [(
        '''    if int(wr.get("current_wr1_anchor_rows", 0)) != 32:\n        raise RuntimeError("WR-R15 does not have exactly one M38 WR1 anchor per team")\n''',
        '''    expected_wr1_anchors = int(c2.get("football_qb_rows", 0))\n    if expected_wr1_anchors <= 0 or int(wr.get("current_wr1_anchor_rows", 0)) != expected_wr1_anchors:\n        raise RuntimeError(\n            f"WR-R15 current WR1 anchor coverage drift expected={expected_wr1_anchors} "\n            f"actual={wr.get('current_wr1_anchor_rows')}"\n        )\n''',
        "lineage-v2 WR1 current-team coverage",
    )])

    # R22: all current football RBs must be adapted, not a frozen 94-player total.
    _transform(LINEAGE_V3, [
        (
            '''    _require(int(adapter.get("adapted_rb_rows", 0)) == 94, "R22 did not adapt all 94 Week-1 RBs")\n''',
            '''    expected_r22_players = int(adapter.get("football_rb_rows", 0))\n    _require(expected_r22_players > 0, "R22 adapter reports zero football RB rows")\n    _require(\n        int(adapter.get("adapted_rb_rows", 0)) == expected_r22_players,\n        f"R22 adapted-RB coverage drift expected={expected_r22_players} actual={adapter.get('adapted_rb_rows')}",\n    )\n''',
            "lineage-v3 R22 adapter coverage",
        ),
        (
            '''    _require(int(pricing.get("adapted_player_keys", 0)) == 94, "R22 pricing lineage RB coverage drift")\n''',
            '''    _require(\n        int(pricing.get("adapted_player_keys", 0)) == expected_r22_players,\n        f"R22 pricing RB coverage drift expected={expected_r22_players} actual={pricing.get('adapted_player_keys')}",\n    )\n''',
            "lineage-v3 R22 pricing coverage",
        ),
        (
            '''    _require(int(trace["rb_receiving_tail_applied"].astype(bool).sum()) == 94, "R22 trace does not contain 94 adapted RBs")\n''',
            '''    _require(\n        int(trace["rb_receiving_tail_applied"].astype(bool).sum()) == expected_r22_players,\n        f"R22 trace adapted-RB coverage drift expected={expected_r22_players}",\n    )\n''',
            "lineage-v3 R22 trace coverage",
        ),
        (
            '''        "rb_r22_adapted_rb_rows": 94,\n''',
            '''        "rb_r22_adapted_rb_rows": expected_r22_players,\n''',
            "lineage-v3 R22 payload coverage",
        ),
    ])

    # Target-pool physical conservation remains exact, but over the certified
    # current football team set (28 late in Week 1; legacy 32 otherwise).
    _transform(TARGET_POOL, [
        (
            "import pandas as pd\n",
            "import pandas as pd\n\nfrom scripts.utils.eligible_team_set_v1 import expected_current_teams\n",
            "target-pool current-team import",
        ),
        (
            '''    source = json.loads(UNIVERSE_AUDIT.read_text(encoding="utf-8"))\n''',
            '''    source = json.loads(UNIVERSE_AUDIT.read_text(encoding="utf-8"))\n    current_teams = expected_current_teams()\n    expected_team_count = len(current_teams) if current_teams is not None else 32\n''',
            "target-pool current-team scope",
        ),
        (
            '''    if frame["team"].nunique() != 32:\n        raise RuntimeError(f"target audit expected 32 teams, found {frame['team'].nunique()}")\n''',
            '''    if frame["team"].nunique() != expected_team_count:\n        raise RuntimeError(f"target audit expected {expected_team_count} teams, found {frame['team'].nunique()}")\n''',
            "target-pool football team coverage",
        ),
        (
            '''        if trace["team"].nunique() != 32:\n            raise RuntimeError(f"explicit entitlement expected 32 teams, found {trace['team'].nunique()}")\n''',
            '''        if trace["team"].nunique() != expected_team_count:\n            raise RuntimeError(f"explicit entitlement expected {expected_team_count} teams, found {trace['team'].nunique()}")\n''',
            "target-pool trace team coverage",
        ),
        (
            '''        if len(physical) != 32:\n            raise RuntimeError(f"explicit entitlement physical audit expected 32 teams, found {len(physical)}")\n''',
            '''        if len(physical) != expected_team_count:\n            raise RuntimeError(f"explicit entitlement physical audit expected {expected_team_count} teams, found {len(physical)}")\n''',
            "target-pool physical team coverage",
        ),
        (
            '''            f"Explicit full-roster target entitlement is physically invalid for {len(explicit_bad)}/32 teams; see {OUT_CSV}."\n''',
            '''            f"Explicit full-roster target entitlement is physically invalid for {len(explicit_bad)}/{expected_team_count} teams; see {OUT_CSV}."\n''',
            "target-pool explicit failure denominator",
        ),
        (
            '''            f"Full-roster receiving entitlement pool is physically invalid for {len(raw_bad)}/32 teams; "\n''',
            '''            f"Full-roster receiving entitlement pool is physically invalid for {len(raw_bad)}/{expected_team_count} teams; "\n''',
            "target-pool legacy failure denominator",
        ),
    ])

    # Final certified-stack V1: preserve every semantic gate, replacing only
    # frozen cardinalities with the explicit current-availability authority.
    _transform(STACK_V1, [
        (
            "import pandas as pd\n",
            "import pandas as pd\n\nfrom scripts.utils.eligible_team_set_v1 import expected_current_teams\n",
            "stack-v1 current-team import",
        ),
        (
            '''    rb_final = _csv(DATA / "rb_rush_rec_conservation_final_audit.csv")\n''',
            '''    rb_final = _csv(DATA / "rb_rush_rec_conservation_final_audit.csv")\n\n    current_teams = expected_current_teams()\n    expected_team_count = len(current_teams) if current_teams is not None else 32\n    expected_game_count = expected_team_count // 2\n''',
            "stack-v1 current-team scope",
        ),
        (
            '''    _require(int(football.get("football_players", 0)) == 469, f"football player universe drifted: {football.get('football_players')}")\n    _require(int(football.get("football_teams", 0)) == 32, "football universe does not cover 32 teams")\n    _require(int(football.get("canonical_games", 0)) == 16, "football universe does not cover 16 games")\n''',
            '''    if current_teams is None:\n        _require(int(football.get("football_players", 0)) == 469, f"football player universe drifted: {football.get('football_players')}")\n    else:\n        football_players = int(football.get("football_players", 0))\n        _require(football_players > 0, "football universe contains zero players")\n        _require(football_players == int(football.get("football_player_rows", -1)), "football player row-count drift")\n        _require(football_players == int(football.get("model_context_players", -1)), "football/model-context player coverage drift")\n    _require(int(football.get("football_teams", 0)) == expected_team_count, f"football universe team coverage drift expected={expected_team_count}")\n    _require(int(football.get("canonical_games", 0)) == expected_game_count, f"football universe game coverage drift expected={expected_game_count}")\n''',
            "stack-v1 football universe coverage",
        ),
        (
            '''    _require(int(c2.get("football_qb_rows", 0)) == 32, "QB C2 does not cover 32 football starters")\n''',
            '''    _require(int(c2.get("football_qb_rows", 0)) == expected_team_count, f"QB C2 football starter coverage drift expected={expected_team_count}")\n''',
            "stack-v1 C2 coverage",
        ),
        (
            '''    _require(int(stamp.get("pass_yard_qbs", 0)) == 32, "QB C2 pricing lineage does not cover 32 QBs")\n''',
            '''    if current_teams is None:\n        _require(int(stamp.get("pass_yard_qbs", 0)) == 32, "QB C2 pricing lineage does not cover 32 QBs")\n    else:\n        _require(int(stamp.get("current_team_scope_expected", -1)) == expected_team_count, "QB C2 pricing current-team scope drift")\n        _require(int(stamp.get("football_qbs", -1)) == expected_team_count, "QB C2 pricing football-QB scope drift")\n        _require(int(stamp.get("c2_selected_football_qbs", -1)) == int(c2.get("selected_qb_rows", -2)), "QB C2 selected football-QB count drift")\n        _require(0 < int(stamp.get("pass_yard_qbs", 0)) <= expected_team_count, "QB C2 priced pass-yard QB subset invalid")\n''',
            "stack-v1 pricing-stamp coverage",
        ),
    ])

    _transform(STACK_V2, [(
        '''    _require(int(wr.get("current_wr1_anchor_rows", 0)) == 32, "WR-R15 does not have 32 WR1 anchors")\n''',
        '''    expected_wr1_anchors = int(c2.get("football_qb_rows", 0))\n    _require(expected_wr1_anchors > 0, "WR-R15 current-team authority has zero teams")\n    _require(int(wr.get("current_wr1_anchor_rows", 0)) == expected_wr1_anchors, f"WR-R15 current WR1 anchor coverage drift expected={expected_wr1_anchors}")\n''',
        "stack-v2 WR1 current-team coverage",
    )])

    _transform(STACK_V3, [
        (
            '''    _require(int(adapter.get("adapted_rb_rows", 0)) == 94, "R22 adapted-RB coverage drift")\n''',
            '''    expected_r22_players = int(adapter.get("football_rb_rows", 0))\n    _require(expected_r22_players > 0, "R22 adapter reports zero football RB rows")\n    _require(int(adapter.get("adapted_rb_rows", 0)) == expected_r22_players, f"R22 adapted-RB coverage drift expected={expected_r22_players}")\n''',
            "stack-v3 R22 adapter coverage",
        ),
        (
            '''    _require(int(pricing.get("adapted_player_keys", 0)) == 94, "R22 pricing RB coverage drift")\n''',
            '''    _require(int(pricing.get("adapted_player_keys", 0)) == expected_r22_players, f"R22 pricing RB coverage drift expected={expected_r22_players}")\n''',
            "stack-v3 R22 pricing coverage",
        ),
        (
            '''        "rb_r22_adapted_rb_rows": 94,\n''',
            '''        "rb_r22_adapted_rb_rows": expected_r22_players,\n''',
            "stack-v3 R22 payload coverage",
        ),
    ])

    print("CURRENT_AVAILABILITY_DOWNSTREAM_CERTIFICATION_SEAM_TRANSFORM_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
