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


def _certify_qb_c2(priced: pd.DataFrame) -> tuple[dict, dict]:
    c2 = _read_json(DATA / "qb_c2_production_integration_audit.json")
    stamp = _read_json(DATA / "qb_c2_pricing_lineage_stamp_audit.json")
    if c2.get("disposition") != "QB_C2_PRODUCTION_DISTRIBUTION_INTEGRATION_PASS":
        raise RuntimeError(f"QB C2 production integration not certified: {c2.get('disposition')}")
    if int(c2.get("football_qb_rows", 0)) != 32 or int(c2.get("selected_qb_rows", 0)) <= 0:
        raise RuntimeError(f"QB C2 production coverage invalid: {c2}")
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
    if int(stamp.get("pass_yard_qbs", 0)) != 32:
        raise RuntimeError("QB C2 pricing stamp does not cover 32 QBs")
    if int(stamp.get("c2_selected_qbs", -1)) != int(c2.get("selected_qb_rows", -2)):
        raise RuntimeError("QB C2 selected-QB count differs between simulation and pricing stamp")
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
    if qb[["team", "player"]].drop_duplicates().shape[0] != 32:
        raise RuntimeError("QB C2 priced pass-yard rows do not cover 32 unique QBs")
    if not qb["qb_distribution_candidate_version"].astype(str).eq(QB_C2_VERSION).all():
        raise RuntimeError("priced QB rows do not all record frozen C2 candidate version")
    if not qb["qb_distribution_selector_version"].astype(str).eq(QB_C2_SELECTOR).all():
        raise RuntimeError("priced QB rows do not all record frozen C2 selector version")
    applied = pd.to_numeric(qb["qb_distribution_specialist_applied"], errors="coerce")
    if applied.isna().any() or not applied.isin([0, 1]).all():
        raise RuntimeError("priced QB C2 applied flag invalid")
    selected = qb.loc[applied.eq(1)]
    selected_qbs = int(selected[["team", "player"]].drop_duplicates().shape[0])
    if selected_qbs != int(c2["selected_qb_rows"]):
        raise RuntimeError(f"priced QB C2 selected count drift expected={c2['selected_qb_rows']} actual={selected_qbs}")
    if not selected["qb_distribution_specialist_version"].astype(str).eq(QB_C2_VERSION).all():
        raise RuntimeError("selected priced QB rows missing C2 specialist version")
    unselected = qb.loc[applied.eq(0)]
    if unselected["qb_distribution_specialist_version"].fillna("").astype(str).str.strip().ne("").any():
        raise RuntimeError("unselected priced QB rows incorrectly claim C2 specialist consumption")
    if not selected["qb_distribution_route"].astype(str).eq("C2_SELECTED").all():
        raise RuntimeError("selected priced QB rows have wrong distribution route")
    if not unselected["qb_distribution_route"].astype(str).eq("CANONICAL_QB_DISTRIBUTION").all():
        raise RuntimeError("unselected priced QB rows have wrong canonical fallback route")
    gap = pd.to_numeric(qb["qb_distribution_raw_mean_gap"], errors="coerce")
    if gap.isna().any() or float(gap.abs().max()) > 1e-10:
        raise RuntimeError("priced QB C2 lineage raw mean gap failed")

    non_qb_market = priced.loc[~priced["source_market"].astype(str).eq("player_pass_yds")]
    if not pd.to_numeric(non_qb_market["qb_distribution_specialist_applied"], errors="coerce").fillna(0).eq(0).all():
        raise RuntimeError("non-pass-yards priced rows claim QB C2 specialist")
    return c2, stamp


def main() -> int:
    priced = _read(PRICED)
    metrics = _read(DATA / "metrics_ready.csv")
    football = _read_json(DATA / "football_simulation_universe_audit.json")
    te = _read_json(DATA / "te_r5p_full_slate_entitlement_audit.json")
    c2, c2_stamp = _certify_qb_c2(priced)

    te_consumed = bool(
        football.get("te_r5p_full_slate_consumed") is True
        and football.get("te_r5p_model_version") == "TE_R5P_PRODUCTION_MODEL_V1"
        and te.get("disposition") == "TE_R5P_FULL_SLATE_ENTITLEMENT_READY"
        and te.get("model_version") == "TE_R5P_PRODUCTION_MODEL_V1"
        and te.get("sportsbook_inputs_used") is False
        and te.get("current_or_future_outcomes_used") is False
        and te.get("team_te_pool_preserved") is True
        and te.get("non_te_entitlement_preserved") is True
        and te.get("team_total_player_entitlement_preserved") is True
    )
    if not te_consumed:
        raise RuntimeError(
            "Full Slate claims TE-R5P integration but lineage cannot certify its conservation/provenance contract"
        )

    pos_col = next((c for c in ("position_group", "position", "alignment_position") if c in metrics.columns), None)
    if pos_col is None:
        raise RuntimeError("metrics_ready has no position column for lineage audit")
    positions = metrics[["player", "team", pos_col]].drop_duplicates(["player", "team"]).copy()
    positions["position_family"] = positions[pos_col].map(_pos)
    p = priced.merge(positions[["player", "team", "position_family"]], on=["player", "team"], how="left", validate="many_to_one")
    if p["position_family"].isna().any():
        raise RuntimeError("lineage audit could not attach position to every priced player")

    rows: list[dict] = []
    def add(market, position, mean_owner, distribution_owner, specialist, science, active_research, limitation):
        part = p.loc[p["source_market"].astype(str).eq(market)]
        if position != "ALL":
            part = part.loc[part["position_family"].eq(position)]
        rows.append({
            "market": market,
            "position_family": position,
            "priced_side_rows": int(len(part)),
            "final_mean_owner": mean_owner,
            "distribution_owner": distribution_owner,
            "specialist_model_active": int(bool(specialist)),
            "scientific_status": science,
            "active_or_next_research": active_research,
            "known_limitation": limitation,
        })

    add(
        "player_pass_yds", "QB",
        "QB_PASS_SYNTHESIS_V1 / M89-M90",
        "QB_DISTRIBUTION_STATE_SELECTOR_V1 -> C2_QB_MEAN_NEUTRAL_DISTRIBUTION_V1 when selected; canonical QB distribution otherwise",
        True, "PROMOTED_QB_MEAN_PLUS_DISTRIBUTION_SPECIALISTS_ACTIVE",
        "shared QB-receiver C2 conservation remains a separate future integration; continue QB mean/distribution prospective scoring",
        "M89/M90 owns the point mean; C2 currently changes only selected QB pass-yard distribution shape, not receiver arrays",
    )
    add(
        "player_rush_yds", "RB/FB",
        "RB_P3_SYNTHESIS_V1 / WEEK1_STACK_OVERRIDE",
        "P3-scaled canonical rushing distribution",
        True, "PROMOTED_SPECIALIST_ACTIVE",
        "multiseason shared-room RB entitlement remains research",
        "Week-1 route only; room entitlement beyond P3 still pending",
    )
    for position in ("QB", "WR", "TE"):
        add(
            "player_rush_yds", position,
            "canonical calibrated rush-yards ensemble + joint MC",
            "canonical joint MC rushing distribution",
            False, "GENERIC_CANONICAL_ACTIVE",
            "position-specific rushing specialist not promoted for this family",
            "RB P3 is correctly gated off for non-RB/FB rushing",
        )
    for market in ("player_reception_yds", "player_receptions"):
        add(
            market, "WR",
            "finite conserved team target pool + WR M38 relative hierarchy + canonical joint MC efficiency",
            "finite-pool canonical receiving joint MC with M38 WR entitlement hierarchy",
            True, "PARTIAL_SPECIALIST_ACTIVE_FINITE_POOL_CERTIFIED",
            "dedicated WR-room entitlement around M38; receiving ensemble calibration",
            "team opportunity is finite, but WR room allocation beyond M38 remains the next specialist lane",
        )
        add(
            market, "TE",
            "finite conserved team target pool + TE_R5P_PRODUCTION_MODEL_V1 entitlement + canonical joint MC efficiency",
            "finite-pool canonical receiving joint MC after TE-R5P entitlement redistribution",
            True, "PROMOTED_TE_ENTITLEMENT_SPECIALIST_ACTIVE",
            "TE efficiency/distribution calibration beyond entitlement if it passes independent frozen gates",
            "TE-R5P reallocates only the existing TE room; team/TE-room mass and non-TE entitlement are conserved",
        )
        add(
            market, "RB/FB",
            "finite conserved team target pool + canonical RB receiving entitlement + joint MC efficiency",
            "finite-pool canonical RB receiving joint MC",
            False, "GENERIC_CANONICAL_ACTIVE_FINITE_POOL_CERTIFIED",
            "dedicated finite RB receiving-room entitlement while preserving P3 rushing",
            "team opportunity is finite, but RB receiving-room allocation has no promoted specialist yet",
        )
    add(
        "player_rush_reception_yds", "RB/FB",
        "RB P3-conserved rushing component + finite-pool canonical receiving joint MC",
        "P3-scaled rushing distribution + finite-pool canonical receiving distribution",
        True, "PARTIAL_SPECIALIST_ACTIVE_FINITE_POOL_CERTIFIED",
        "dedicated RB receiving entitlement; joint-market calibration later",
        "rushing is P3-consistent and receiving pool is finite; receiving allocation specialist remains pending",
    )
    add(
        "player_anytime_td", "ALL",
        "generic joint MC offensive_td_rate + red-zone modifiers",
        "generic joint MC Bernoulli TD distribution",
        False, "ATD_GENERIC_ACTIVE_NOT_DEDICATED_SCIENCE_CERTIFIED",
        "dedicated football-only anytime-TD opportunity/entitlement calibration",
        "current ATD execution is football-only but has no dedicated walk-forward probability certification",
    )

    out = pd.DataFrame(rows)
    observed = set(p["source_market"].astype(str).unique())
    represented = set(out.loc[out["priced_side_rows"].gt(0), "market"].astype(str))
    missing = sorted(observed - represented)
    if missing:
        raise RuntimeError(f"priced markets absent from lineage registry: {missing}")

    qb = p.loc[p["source_market"].astype(str).eq("player_pass_yds")]
    if not qb.empty and not pd.to_numeric(qb.get("qb_synthesis_applied", 0), errors="coerce").fillna(0).eq(1).all():
        raise RuntimeError("lineage claims QB M89/M90 specialist active but priced rows disagree")
    rb = p.loc[p["source_market"].astype(str).eq("player_rush_yds") & p["position_family"].eq("RB/FB")]
    if not rb.empty and not pd.to_numeric(rb.get("rb_synthesis_applied", 0), errors="coerce").fillna(0).eq(1).all():
        raise RuntimeError("lineage claims RB P3 active but priced rows disagree")
    atd = p.loc[p["source_market"].astype(str).eq("player_anytime_td")]
    if not atd.empty:
        if int(pd.to_numeric(atd.get("ml_applied", 0), errors="coerce").fillna(0).sum()) != 0:
            raise RuntimeError("ATD lineage expected no dedicated ML consumption but priced rows disagree")
        if int(pd.to_numeric(atd.get("state_applied", 0), errors="coerce").fillna(0).sum()) != 0:
            raise RuntimeError("ATD lineage expected no State consumption but priced rows disagree")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    payload = {
        "disposition": "MARKET_MODEL_LINEAGE_EXPLICIT",
        "priced_markets": sorted(observed),
        "lineage_rows": int(len(out)),
        "old_alternate_engine_active": False,
        "all_specialist_research_consumed": False,
        "te_r5p_consumed": True,
        "te_r5p_model_version": "TE_R5P_PRODUCTION_MODEL_V1",
        "te_r5p_final_fit_run": int(te.get("source_final_fit_run")),
        "te_r5p_team_pool_preserved": bool(te.get("team_te_pool_preserved")),
        "te_r5p_non_te_entitlement_preserved": bool(te.get("non_te_entitlement_preserved")),
        "qb_c2_distribution_consumed": True,
        "qb_c2_distribution_specialist_version": QB_C2_VERSION,
        "qb_c2_selector_version": QB_C2_SELECTOR,
        "qb_c2_selected_qbs_current_slate": int(c2.get("selected_qb_rows")),
        "qb_c2_state_capture_exact": int(c2.get("state_capture_changed_arrays", -1)) == 0,
        "qb_c2_max_raw_mean_gap": float(c2.get("max_raw_qb_mean_gap")),
        "qb_c2_pricing_lineage_stamp_certified": c2_stamp.get("disposition") == "QB_C2_PRICING_LINEAGE_STAMP_CERTIFIED",
        "qb_c2_distribution_audit": "data/qb_c2_production_integration_audit.json",
        "c2_full_stack_receiver_conservation_consumed": False,
        "c2_full_stack_consumed": False,
        "anytime_td_dedicated_science_certified": False,
        "sportsbook_inputs_used_to_define_lineage": False,
        "audit": str(OUT_CSV),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[market_model_lineage] " + json.dumps(payload, sort_keys=True))
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
