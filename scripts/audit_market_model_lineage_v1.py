#!/usr/bin/env python3
"""Materialize the actual model lineage consumed by the current priced Full Slate.

This is a governance/audit artifact, not a projection model. Its purpose is to
make the distinction between point-mean authority, distribution authority,
promoted specialists, generic canonical components, and research that is not yet
consumed impossible to hide behind a single 'production' label.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA = Path("data")
OUTPUTS = Path("outputs")
PRICED = OUTPUTS / "props_priced_clean.csv"
OUT_CSV = DATA / "market_model_lineage_current.csv"
OUT_JSON = DATA / "market_model_lineage_current.json"

QB_C2_VERSION = "C2_QB_MEAN_NEUTRAL_DISTRIBUTION_V1"
QB_C2_SELECTOR = "QB_DISTRIBUTION_STATE_SELECTOR_V1"


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"lineage required artifact missing/empty: {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if df.empty:
        raise RuntimeError(f"lineage required artifact has zero rows: {path}")
    return df


def _read_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"lineage required artifact missing/empty: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _pos(value) -> str:
    p = str(value or "").upper().strip()
    if p in {"RB", "HB", "TB", "FB"}: return "RB/FB"
    if p in {"WR", "LWR", "RWR", "SWR"}: return "WR"
    if p == "TE": return "TE"
    if p == "QB": return "QB"
    return p or "OTHER"


def _validate_qb_c2_coverage_contract(c2: dict, stamp: dict, *, priced_qbs: int | None = None, selected_priced_qbs: int | None = None) -> dict:
    """Validate current football-QB coverage separately from sportsbook coverage.

    The C2 production audit owns the certified current football-QB universe.
    The pricing stamp may legitimately cover fewer QBs because sportsbooks need
    not post a pass-yard line for every current starter. Sportsbook coverage may
    therefore be a subset, but it may never define or widen the football universe.
    """
    football_qbs = int(c2.get("football_qb_rows", 0))
    selected_football_qbs = int(c2.get("selected_qb_rows", 0))
    stamped_football_qbs = int(stamp.get("football_qbs", 0))
    current_scope = int(stamp.get("current_team_scope_expected", 0))
    stamped_priced_qbs = int(stamp.get("pass_yard_qbs", 0))
    stamped_selected_priced = int(stamp.get("c2_selected_qbs", -1))
    stamped_selected_football = int(stamp.get("c2_selected_football_qbs", -1))

    if football_qbs <= 0 or selected_football_qbs <= 0 or selected_football_qbs > football_qbs:
        raise RuntimeError(f"QB C2 production coverage invalid: {c2}")
    if stamped_football_qbs != football_qbs:
        raise RuntimeError(
            "QB C2 football universe differs between production audit and pricing lineage stamp; "
            f"production={football_qbs} stamp={stamped_football_qbs}"
        )
    if current_scope != football_qbs:
        raise RuntimeError(
            "QB C2 football universe differs from certified current team scope; "
            f"scope={current_scope} football={football_qbs}"
        )
    if stamped_selected_football != selected_football_qbs:
        raise RuntimeError(
            "QB C2 selected football-QB count differs between production audit and pricing lineage stamp; "
            f"production={selected_football_qbs} stamp={stamped_selected_football}"
        )
    if stamped_priced_qbs <= 0 or stamped_priced_qbs > football_qbs:
        raise RuntimeError(
            "QB C2 priced pass-yard coverage must be a nonempty subset of the football universe; "
            f"priced={stamped_priced_qbs} football={football_qbs}"
        )
    if stamped_selected_priced < 0 or stamped_selected_priced > selected_football_qbs:
        raise RuntimeError(
            "QB C2 selected priced-QB coverage invalid; "
            f"priced_selected={stamped_selected_priced} football_selected={selected_football_qbs}"
        )
    if bool(stamp.get("sportsbook_offer_coverage_defines_football_universe", True)):
        raise RuntimeError("QB C2 pricing stamp allows sportsbook coverage to define football universe")
    if priced_qbs is not None and int(priced_qbs) != stamped_priced_qbs:
        raise RuntimeError(
            "QB C2 priced pass-yard identity count differs from pricing lineage stamp; "
            f"priced={priced_qbs} stamp={stamped_priced_qbs}"
        )
    if selected_priced_qbs is not None and int(selected_priced_qbs) != stamped_selected_priced:
        raise RuntimeError(
            "QB C2 selected priced-QB count differs from pricing lineage stamp; "
            f"priced={selected_priced_qbs} stamp={stamped_selected_priced}"
        )
    return {
        "football_qbs": football_qbs,
        "selected_football_qbs": selected_football_qbs,
        "priced_qbs": stamped_priced_qbs,
        "selected_priced_qbs": stamped_selected_priced,
    }


def _certify_qb_c2(priced: pd.DataFrame) -> tuple[dict, dict]:
    c2 = _read_json(DATA / "qb_c2_production_integration_audit.json")
    stamp = _read_json(DATA / "qb_c2_pricing_lineage_stamp_audit.json")
    if c2.get("disposition") != "QB_C2_PRODUCTION_DISTRIBUTION_INTEGRATION_PASS":
        raise RuntimeError(f"QB C2 production integration not certified: {c2.get('disposition')}")
    if c2.get("production_distribution_specialist") != QB_C2_VERSION:
        raise RuntimeError("QB C2 production distribution version drift")
    if c2.get("selector_version") != QB_C2_SELECTOR:
        raise RuntimeError("QB C2 selector version drift")
    if float(c2.get("max_raw_qb_mean_gap", 1.0)) > 1e-10:
        raise RuntimeError("QB C2 raw mean-neutrality gate failed")
    if int(c2.get("state_capture_changed_arrays", -1)) != 0:
        raise RuntimeError("QB C2 state-capture parity changed arrays")
    if float(c2.get("state_capture_max_element_gap", 1.0)) > 1e-12:
        raise RuntimeError("QB C2 state-capture parity has element drift")
    if int(c2.get("receiver_outputs_replaced", 1)) != 0 or int(c2.get("rb_outputs_replaced", 1)) != 0:
        raise RuntimeError("QB C2 production adapter replaced non-QB outputs")
    for field in (
        "sportsbook_inputs_to_starter_selection",
        "sportsbook_inputs_to_selector",
        "sportsbook_inputs_to_c2_generation",
    ):
        if int(c2.get(field, 1)) != 0:
            raise RuntimeError(f"QB C2 sportsbook leakage flag {field}={c2.get(field)}")

    if stamp.get("disposition") != "QB_C2_PRICING_LINEAGE_STAMP_CERTIFIED":
        raise RuntimeError(f"QB C2 pricing stamp not certified: {stamp.get('disposition')}")
    if bool(stamp.get("pricing_values_modified", True)):
        raise RuntimeError("QB C2 pricing stamp reports pricing values modified")
    if int(stamp.get("protected_columns_changed", 1)) != 0:
        raise RuntimeError("QB C2 pricing stamp changed protected columns")
    _validate_qb_c2_coverage_contract(c2, stamp)
    if stamp.get("specialist_version") != QB_C2_VERSION or stamp.get("selector_version") != QB_C2_SELECTOR:
        raise RuntimeError("QB C2 pricing stamp version drift")

    required = {
        "qb_distribution_specialist_applied",
        "qb_distribution_specialist_version",
        "qb_distribution_candidate_version",
        "qb_distribution_selector_version",
        "qb_distribution_selector_delta_pass_attempts",
        "qb_distribution_starter_authority_source",
        "qb_distribution_route",
        "qb_distribution_raw_mean_gap",
    }
    missing = sorted(required - set(priced.columns))
    if missing:
        raise RuntimeError(f"priced output missing QB C2 lineage columns: {missing}")

    qb = priced.loc[priced["source_market"].astype(str).eq("player_pass_yds")].copy()
    if qb.empty:
        raise RuntimeError("QB C2 lineage found zero priced pass-yard rows")
    priced_qbs = int(qb[["team", "player"]].drop_duplicates().shape[0])
    if not qb["qb_distribution_candidate_version"].astype(str).eq(QB_C2_VERSION).all():
        raise RuntimeError("priced QB rows do not all record frozen C2 candidate version")
    if not qb["qb_distribution_selector_version"].astype(str).eq(QB_C2_SELECTOR).all():
        raise RuntimeError("priced QB rows do not all record frozen C2 selector version")
    applied = pd.to_numeric(qb["qb_distribution_specialist_applied"], errors="coerce")
    if applied.isna().any() or not applied.isin([0, 1]).all():
        raise RuntimeError("priced QB C2 applied flag invalid")
    selected = qb.loc[applied.eq(1)]
    selected_qbs = int(selected[["team", "player"]].drop_duplicates().shape[0])
    _validate_qb_c2_coverage_contract(c2, stamp, priced_qbs=priced_qbs, selected_priced_qbs=selected_qbs)

    if stamp.get("pass_yard_qbs") != priced_qbs:
        raise RuntimeError("QB C2 stamp pass-yard QB count does not match priced artifact")
    if stamp.get("c2_selected_qbs") != selected_qbs:
        raise RuntimeError("QB C2 stamp selected QB count does not match priced artifact")

    audit = {
        "production": c2,
        "pricing_stamp": stamp,
        "priced_pass_yard_rows": int(len(qb)),
        "priced_pass_yard_qbs": priced_qbs,
        "priced_selected_c2_qbs": selected_qbs,
    }
    return c2, audit


def main() -> int:
    priced = _read(PRICED)
    c2, c2_audit = _certify_qb_c2(priced)

    rows = []
    for market, group in priced.groupby("source_market", sort=True):
        families = sorted({_pos(x) for x in group.get("position", pd.Series([], dtype=str)).tolist()})
        if not families:
            families = ["UNKNOWN"]
        rows.append({
            "market": str(market),
            "position_families": ",".join(families),
            "rows": int(len(group)),
            "players": int(group[["team", "player"]].drop_duplicates().shape[0]) if {"team", "player"}.issubset(group.columns) else int(group["player"].nunique()),
            "qb_c2_specialist": QB_C2_VERSION if str(market) == "player_pass_yds" else "",
            "qb_c2_selector": QB_C2_SELECTOR if str(market) == "player_pass_yds" else "",
        })
    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    payload = {
        "disposition": "MARKET_MODEL_LINEAGE_CERTIFIED",
        "markets": int(out["market"].nunique()),
        "rows": int(len(priced)),
        "qb_c2": c2_audit,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
