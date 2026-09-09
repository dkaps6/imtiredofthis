#!/usr/bin/env python3
"""Fail-closed Full Slate certification including promoted WR-R15.

Runs the established V1 stack certification first, then requires the frozen
M38+R15 WR contract and rewrites the certification payload so governance matches
the stack that actually ran.  This does not broaden any scientific claim beyond
WR entitlement; receiving efficiency/distribution and the full shared C2 stack
remain separate research lanes.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts import validate_certified_full_slate_stack_v1 as v1

DATA = Path("data")
OUT = DATA / "certified_full_slate_stack_audit.json"

WR_VERSION = "WR_R15_PRODUCTION_MODEL_V1"


def _load(path: Path) -> dict:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"certified stack V2 artifact missing/empty: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require(ok: bool, message: str) -> None:
    if not bool(ok):
        raise RuntimeError(message)


def main() -> int:
    rc = int(v1.main())
    if rc != 0:
        return rc

    wr = _load(DATA / "wr_r15_full_slate_entitlement_audit.json")
    football = _load(DATA / "football_simulation_universe_audit.json")
    entitlement = _load(DATA / "target_entitlement_v1_audit.json")
    lineage = _load(DATA / "market_model_lineage_current.json")
    c2 = _load(DATA / "qb_c2_production_integration_audit.json")
    payload = _load(OUT)

    _require(wr.get("disposition") == "WR_R15_FULL_SLATE_ENTITLEMENT_READY", "WR-R15 entitlement not production-ready")
    _require(wr.get("model_version") == WR_VERSION, "WR-R15 model version drift")
    _require(int(wr.get("authorized_by_run", 0)) == 34238301577, "WR-R15 OOS authorization run drift")
    _require(int(wr.get("authorized_by_artifact", 0)) == 10061328722, "WR-R15 OOS artifact drift")
    _require(int(wr.get("source_final_fit_run", 0)) == 34240496725, "WR-R15 final-fit run drift")
    _require(int(wr.get("source_final_fit_artifact", 0)) == 10062104621, "WR-R15 final-fit artifact drift")
    _require(wr.get("scientific_confirmation_seasons") == [2023, 2024], "WR-R15 confirmation seasons drift")
    _require(wr.get("scientific_confirmation_2025_used") is False, "WR-R15 incorrectly claims 2025 confirmation")
    _require(wr.get("m38_wr1_anchor_preserved") is True, "M38 WR1 anchor not preserved")
    _require(wr.get("wr2plus_pool_preserved") is True, "WR-R15 changed WR2+ pool")
    _require(wr.get("wr_room_mass_preserved") is True, "WR-R15 changed WR room mass")
    _require(wr.get("non_wr_entitlement_preserved") is True, "WR-R15 changed non-WR entitlement")
    _require(wr.get("team_total_player_entitlement_preserved") is True, "WR-R15 changed team player mass")
    _require(wr.get("target_residual_bucket_preserved") is True, "WR-R15 changed residual target bucket")
    _require(wr.get("sportsbook_inputs_used") is False, "sportsbook input leaked into WR-R15")
    _require(wr.get("current_or_future_outcomes_used") is False, "WR-R15 used current/future outcomes")
    _require(int(wr.get("current_wr1_anchor_rows", 0)) == 32, "WR-R15 does not have 32 WR1 anchors")
    _require(float(wr.get("max_wr1_anchor_entitlement_delta", 1.0)) == 0.0, "WR-R15 WR1 anchor changed")
    _require(float(wr.get("max_non_wr_entitlement_delta", 1.0)) == 0.0, "WR-R15 non-WR entitlement changed")
    for key in ("max_wr2plus_pool_gap", "max_wr_room_mass_gap", "max_team_player_entitlement_gap"):
        _require(float(wr.get(key, 1.0)) <= 1e-12, f"WR-R15 conservation drift {key}={wr.get(key)}")

    _require(football.get("wr_r15_full_slate_consumed") is True, "football universe does not consume WR-R15")
    _require(football.get("wr_r15_model_version") == WR_VERSION, "football universe WR-R15 version drift")
    _require(football.get("wr_r15_m38_wr1_anchor_preserved") is True, "football universe does not preserve M38 WR1")
    _require(entitlement.get("wr_r15_entitlement_scope_gate") == "PASS", "WR-R15 entitlement scope gate not passed")
    _require(c2.get("wr_r15_consumed_before_c2") is True, "C2 did not consume post-R15 football state")
    _require(int(c2.get("state_capture_changed_arrays", -1)) == 0, "C2 state-capture parity changed after WR-R15")
    _require(lineage.get("wr_r15_consumed") is True, "market lineage does not record WR-R15")
    _require(lineage.get("wr_m38_plus_r15_consumed") is True, "market lineage does not record M38+R15")
    _require(lineage.get("wr_r15_model_version") == WR_VERSION, "market lineage WR-R15 version drift")

    payload.update({
        "disposition": "FULL_SLATE_CERTIFIED_STACK_READY_M38_R15_TE_R5P_C2_WITH_DECLARED_SCIENCE_LIMITATIONS",
        "wr_m38_finite_pool_active": True,
        "wr_m38_wr1_anchor_active": True,
        "wr_r15_entitlement_active": True,
        "wr_r15_model_version": WR_VERSION,
        "wr_r15_authorized_oos_run": 34238301577,
        "wr_r15_final_fit_run": 34240496725,
        "wr_r15_current_wr1_anchors": int(wr.get("current_wr1_anchor_rows")),
        "wr_r15_current_wr2plus_rows": int(wr.get("current_wr2plus_rows")),
        "wr_r15_max_wr1_anchor_delta": float(wr.get("max_wr1_anchor_entitlement_delta")),
        "wr_r15_max_wr_room_gap": float(wr.get("max_wr_room_mass_gap")),
        "wr_r15_max_non_wr_delta": float(wr.get("max_non_wr_entitlement_delta")),
        "wr_r15_sportsbook_inputs_used": False,
        "remaining_science_lanes": [
            "RB receiving entitlement",
            "WR/TE/RB receiving efficiency and distribution calibration",
            "shared QB-receiver C2 conservation",
            "dedicated anytime-TD probability calibration",
        ],
    })
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[certified_full_slate_stack_v2] " + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
