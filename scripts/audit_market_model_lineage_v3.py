#!/usr/bin/env python3
"""Governed market lineage after RB-R22 Week-1 production integration.

Runs the certified V2 lineage audit first, then records the mean-preserving R22
RB receiving-yard distribution specialist. R22 changes distribution shape only:
RB targets/receptions and receiving-yard point means remain owned by the existing
finite-pool canonical receiving stack.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts import audit_market_model_lineage_v2 as v2

DATA = Path("data")
CSV = DATA / "market_model_lineage_current.csv"
JSON = DATA / "market_model_lineage_current.json"
ADAPTER = DATA / "rb_receiving_tail_production_audit.json"
PRICING = DATA / "rb_receiving_tail_pricing_lineage_audit.json"
TRACE = DATA / "rb_receiving_tail_production_trace.csv"

VERSION = "RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_V1"
R19_RUN = 34288244770
R22_RUN = 34298516960
R22_ARTIFACT = 10084118525


def _load(path: Path) -> dict:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"R22 lineage required artifact missing/empty: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require(ok: bool, message: str) -> None:
    if not bool(ok):
        raise RuntimeError(message)


def main() -> int:
    rc = int(v2.main())
    if rc != 0:
        return rc

    adapter = _load(ADAPTER)
    pricing = _load(PRICING)
    trace = pd.read_csv(TRACE, low_memory=False)

    _require(adapter.get("disposition") == "RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_ADAPTER_PASS", "R22 adapter not certified")
    _require(adapter.get("integration_valid") is True, "R22 adapter integration invalid")
    _require(int(adapter.get("adapted_rb_rows", 0)) == 94, "R22 did not adapt all 94 Week-1 RBs")
    _require(float(adapter.get("max_mean_delta", 1.0)) <= 1e-8, "R22 receiving mean drift")
    _require(float(adapter.get("min_spearman", -1.0)) >= 0.9999, "R22 rank preservation drift")
    _require(adapter.get("gates", {}).get("receptions_exact") is True, "R22 changed receptions")
    _require(adapter.get("gates", {}).get("rb_nonreceiving_markets_exact") is True, "R22 changed nonreceiving RB markets")
    _require(adapter.get("gates", {}).get("non_rb_exact") is True, "R22 changed non-RB distributions")
    _require(adapter.get("gates", {}).get("rush_rec_identity") is True, "R22 broke rush+rec identity")
    _require(int(adapter.get("sportsbook_inputs_added", 1)) == 0, "sportsbook leaked into R22 football distribution")
    _require(int(adapter.get("current_or_future_outcomes_used", 1)) == 0, "R22 used current/future outcomes")

    _require(pricing.get("disposition") == "RB_R22_WEEK1_RECEIVING_TAIL_PRICING_LINEAGE_PASS", "R22 pricing lineage not certified")
    _require(pricing.get("integration_valid") is True, "R22 pricing lineage invalid")
    _require(int(pricing.get("adapted_player_keys", 0)) == 94, "R22 pricing lineage RB coverage drift")
    _require(int(pricing.get("rec_yards_rows_stamped", 0)) > 0, "R22 does not reach RB receiving-yard pricing")
    _require(int(pricing.get("rush_rec_yards_rows_stamped", 0)) > 0, "R22 does not reach RB rush+rec pricing")
    _require(int(pricing.get("receptions_rows_present_unadapted", 0)) > 0, "R22 receptions protection not observed")
    _require(int(pricing.get("sportsbook_inputs_to_adapter", 1)) == 0, "sportsbook leaked into R22 adapter")
    _require(int(pricing.get("production_mean_parameters_changed", 1)) == 0, "R22 changed point-mean parameters")

    _require(not trace.empty, "R22 production trace empty")
    _require(int(trace["rb_receiving_tail_applied"].astype(bool).sum()) == 94, "R22 trace does not contain 94 adapted RBs")

    frame = pd.read_csv(CSV, low_memory=False)
    pos = frame["position_family"].astype(str)
    market = frame["market"].astype(str)
    rec = pos.eq("RB/FB") & market.eq("player_reception_yds")
    combo = pos.eq("RB/FB") & market.eq("player_rush_reception_yds")
    receptions = pos.eq("RB/FB") & market.eq("player_receptions")
    if int(rec.sum()) != 1 or int(combo.sum()) != 1 or int(receptions.sum()) != 1:
        raise RuntimeError(f"unexpected RB/FB receiving lineage rows rec={int(rec.sum())} combo={int(combo.sum())} receptions={int(receptions.sum())}")

    frame.loc[rec, "distribution_owner"] = (
        "RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_V1 mean-preserving receiving-yard tail distribution "
        "on finite-pool canonical RB receiving mean"
    )
    frame.loc[rec, "specialist_model_active"] = 1
    frame.loc[rec, "scientific_status"] = "PROMOTED_RB_R22_WEEK1_RECEIVING_DISTRIBUTION_ACTIVE"
    frame.loc[rec, "active_or_next_research"] = (
        "RB receiving entitlement/receptions and receiving-yard mean refinement remain separate lanes; prospectively grade R21/R22"
    )
    frame.loc[rec, "known_limitation"] = (
        "R22 is Week-1 distribution-shape only; receiving mean, targets and receptions remain canonical"
    )

    frame.loc[combo, "distribution_owner"] = (
        "RB P3 rushing distribution + RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_V1 receiving distribution with exact rush+rec identity"
    )
    frame.loc[combo, "specialist_model_active"] = 1
    frame.loc[combo, "scientific_status"] = "PROMOTED_RB_P3_PLUS_R22_WEEK1_DISTRIBUTION_ACTIVE"
    frame.loc[combo, "active_or_next_research"] = (
        "joint rush+rec calibration after independent rushing/receiving components accumulate prospective evidence"
    )
    frame.loc[combo, "known_limitation"] = (
        "P3 and R22 are Week-1 specialists; R22 leaves receiving mean/receptions unchanged"
    )

    # Explicitly preserve the receptions row as unpromoted by R22.
    frame.loc[receptions, "known_limitation"] = (
        "R22 does not change receptions; RB receiving-room entitlement/reception distribution remains a separate open lane"
    )
    frame.to_csv(CSV, index=False)

    payload = _load(JSON)
    payload.update({
        "rb_r22_receiving_tail_consumed": True,
        "rb_r22_version": VERSION,
        "rb_r22_week1_only": True,
        "rb_r22_integration_run": R22_RUN,
        "rb_r22_integration_artifact": R22_ARTIFACT,
        "rb_r22_r19_source_run": R19_RUN,
        "rb_r22_adapted_rb_rows": 94,
        "rb_r22_receiving_mean_preserved": True,
        "rb_r22_receptions_preserved": True,
        "rb_r22_non_rb_preserved": True,
        "rb_r22_sportsbook_inputs_used": False,
        "rb_r22_future_outcomes_used": False,
        "rb_r22_adapter_audit": str(ADAPTER),
        "rb_r22_pricing_lineage_audit": str(PRICING),
        "lineage_version": "MARKET_MODEL_LINEAGE_V3_R22_WEEK1_PROMOTED",
    })
    JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[market_model_lineage_v3] " + json.dumps(payload, sort_keys=True))
    print(frame.loc[rec | combo | receptions].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
