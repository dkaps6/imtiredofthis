#!/usr/bin/env python3
"""Governed market lineage after WR-R15 production integration.

Runs the established V1 lineage audit first, then fail-closes the newly promoted
WR contract and rewrites only WR receiving lineage fields plus explicit R15
provenance.  This keeps all prior QB/RB/TE/ATD governance intact.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import audit_market_model_lineage_v1 as v1

DATA = Path("data")
CSV = DATA / "market_model_lineage_current.csv"
JSON = DATA / "market_model_lineage_current.json"
WR_AUDIT = DATA / "wr_r15_full_slate_entitlement_audit.json"
FOOTBALL = DATA / "football_simulation_universe_audit.json"
ENTITLEMENT = DATA / "target_entitlement_v1_audit.json"
C2 = DATA / "qb_c2_production_integration_audit.json"

WR_VERSION = "WR_R15_PRODUCTION_MODEL_V1"
WR_OOS_RUN = 34238301577
WR_OOS_ARTIFACT = 10061328722
WR_FINAL_FIT_RUN = 34240496725
WR_FINAL_FIT_ARTIFACT = 10062104621


def _load(path: Path) -> dict:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"WR lineage required artifact missing/empty: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    rc = int(v1.main())
    if rc != 0:
        return rc

    wr = _load(WR_AUDIT)
    football = _load(FOOTBALL)
    entitlement = _load(ENTITLEMENT)
    c2 = _load(C2)

    required = {
        "disposition": "WR_R15_FULL_SLATE_ENTITLEMENT_READY",
        "model_version": WR_VERSION,
        "authorized_by_run": WR_OOS_RUN,
        "authorized_by_artifact": WR_OOS_ARTIFACT,
        "source_final_fit_run": WR_FINAL_FIT_RUN,
        "source_final_fit_artifact": WR_FINAL_FIT_ARTIFACT,
        "scientific_confirmation_2025_used": False,
        "m38_wr1_anchor_preserved": True,
        "wr2plus_pool_preserved": True,
        "wr_room_mass_preserved": True,
        "non_wr_entitlement_preserved": True,
        "team_total_player_entitlement_preserved": True,
        "target_residual_bucket_preserved": True,
        "sportsbook_inputs_used": False,
        "current_or_future_outcomes_used": False,
    }
    for key, expected in required.items():
        if wr.get(key) != expected:
            raise RuntimeError(f"WR-R15 lineage contract drift {key}: expected={expected!r} actual={wr.get(key)!r}")
    if wr.get("scientific_confirmation_seasons") != [2023, 2024]:
        raise RuntimeError("WR-R15 scientific confirmation seasons drifted")
    if wr.get("training_seasons") != [2022, 2023, 2024, 2025]:
        raise RuntimeError("WR-R15 final-fit seasons drifted")
    if int(wr.get("current_wr1_anchor_rows", 0)) != 32:
        raise RuntimeError("WR-R15 does not have exactly one M38 WR1 anchor per team")
    if float(wr.get("max_wr1_anchor_entitlement_delta", 1.0)) != 0.0:
        raise RuntimeError("WR-R15 changed an M38 WR1 anchor")
    if float(wr.get("max_non_wr_entitlement_delta", 1.0)) != 0.0:
        raise RuntimeError("WR-R15 changed non-WR entitlement")
    for key in ("max_wr2plus_pool_gap", "max_wr_room_mass_gap", "max_team_player_entitlement_gap"):
        if float(wr.get(key, 1.0)) > 1e-12:
            raise RuntimeError(f"WR-R15 conservation drift {key}={wr.get(key)}")

    if football.get("wr_r15_full_slate_consumed") is not True:
        raise RuntimeError("football universe does not record WR-R15 consumption")
    if football.get("wr_r15_model_version") != WR_VERSION:
        raise RuntimeError("football universe WR-R15 version drift")
    if football.get("wr_r15_m38_wr1_anchor_preserved") is not True:
        raise RuntimeError("football universe does not certify M38 WR1 preservation")
    if entitlement.get("wr_r15_entitlement_scope_gate") != "PASS":
        raise RuntimeError("explicit target entitlement does not certify WR-R15 scope")
    if c2.get("wr_r15_consumed_before_c2") is not True:
        raise RuntimeError("QB C2 lineage does not record WR-R15 upstream")

    frame = pd.read_csv(CSV, low_memory=False)
    mask = (
        frame["position_family"].astype(str).eq("WR")
        & frame["market"].astype(str).isin(["player_reception_yds", "player_receptions"])
    )
    if int(mask.sum()) != 2:
        raise RuntimeError(f"expected exactly two WR receiving lineage rows, found {int(mask.sum())}")
    frame.loc[mask, "final_mean_owner"] = (
        "finite conserved team target pool + M38 WR1 anchor + "
        "WR_R15_PRODUCTION_MODEL_V1 WR2+ entitlement + canonical joint MC efficiency"
    )
    frame.loc[mask, "distribution_owner"] = (
        "finite-pool canonical receiving joint MC after M38 WR1 + WR-R15 WR2+ entitlement"
    )
    frame.loc[mask, "specialist_model_active"] = 1
    frame.loc[mask, "scientific_status"] = "PROMOTED_WR_M38_PLUS_R15_ENTITLEMENT_SPECIALISTS_ACTIVE"
    frame.loc[mask, "active_or_next_research"] = (
        "WR receiving efficiency/distribution calibration and matchup-quality refinement"
    )
    frame.loc[mask, "known_limitation"] = (
        "M38 remains the WR1 entitlement anchor; R15 redistributes only WR2+; direct WR-CB shadow is unavailable/gated"
    )
    frame.to_csv(CSV, index=False)

    payload = _load(JSON)
    payload.update({
        "wr_m38_plus_r15_consumed": True,
        "wr_m38_wr1_anchor_active": True,
        "wr_r15_consumed": True,
        "wr_r15_model_version": WR_VERSION,
        "wr_r15_authorized_oos_run": WR_OOS_RUN,
        "wr_r15_authorized_oos_artifact": WR_OOS_ARTIFACT,
        "wr_r15_final_fit_run": WR_FINAL_FIT_RUN,
        "wr_r15_final_fit_artifact": WR_FINAL_FIT_ARTIFACT,
        "wr_r15_scientific_confirmation_seasons": [2023, 2024],
        "wr_r15_scientific_confirmation_2025_used": False,
        "wr_r15_m38_wr1_anchor_preserved": True,
        "wr_r15_wr2plus_pool_preserved": True,
        "wr_r15_wr_room_mass_preserved": True,
        "wr_r15_non_wr_entitlement_preserved": True,
        "wr_r15_team_mass_preserved": True,
        "wr_r15_sportsbook_inputs_used": False,
        "wr_r15_entitlement_audit": str(WR_AUDIT),
        "lineage_version": "MARKET_MODEL_LINEAGE_V2_WR_R15_PROMOTED",
    })
    JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[market_model_lineage_v2] " + json.dumps(payload, sort_keys=True))
    print(frame.loc[mask].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
