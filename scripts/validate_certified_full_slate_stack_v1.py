#!/usr/bin/env python3
"""Fail-closed certification for the currently approved Full Slate model stack.

This gate is source-agnostic: it can certify either a no-credit replay of a
previous sportsbook snapshot or a genuinely live sportsbook snapshot.  It does
NOT claim every market's science is finished.  It proves that the operational
pricing path consumed the football-first universe and the currently promoted
specialists exactly as intended:

- 469-player/32-team football universe independent of sportsbook offer coverage;
- explicit conserved team target entitlement;
- TE-R5P inside the conserved TE room;
- QB C2 mean-neutral distribution selector with M89/M90 retaining mean authority;
- RB P3 rushing plus rush+receiving conservation;
- exact bookmaker offer expansion and downstream-only sportsbook usage;
- explicit player-level model lineage;
- ATD execution allowed only with its not-yet-certified science status declared.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
OUTPUTS = Path("outputs")
OUT = DATA / "certified_full_slate_stack_audit.json"


def _json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"certified stack required JSON missing/empty: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"certified stack required CSV missing/empty: {path}")
    out = pd.read_csv(path, low_memory=False)
    if out.empty:
        raise RuntimeError(f"certified stack required CSV has zero rows: {path}")
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise RuntimeError(message)


def main() -> int:
    live = _json(DATA / "live_odds_status.json")
    pricing = _json(DATA / "live_pricing_offer_audit.json")
    football = _json(DATA / "football_simulation_universe_audit.json")
    entitlement = _json(DATA / "target_entitlement_v1_audit.json")
    te = _json(DATA / "te_r5p_full_slate_entitlement_audit.json")
    c2 = _json(DATA / "qb_c2_production_integration_audit.json")
    stamp = _json(DATA / "qb_c2_pricing_lineage_stamp_audit.json")
    lineage = _json(DATA / "market_model_lineage_current.json")
    atd = _json(DATA / "anytime_td_model_status_v1.json")
    target_pool = _json(DATA / "team_target_pool_audit.json")
    rb_input = _json(DATA / "rb_rush_rec_conservation_input_audit.json")

    offers = _csv(OUTPUTS / "props_pricing_offers.csv")
    priced = _csv(OUTPUTS / "props_priced_clean.csv")
    rb_final = _csv(DATA / "rb_rush_rec_conservation_final_audit.csv")

    _require(bool(live.get("available")), "certified stack requires an available sportsbook snapshot for pricing")
    _require(
        pricing.get("disposition") == "PRICING_OFFERS_MATERIALIZED_EXACTLY_ONCE",
        f"exact bookmaker pricing adapter not certified: {pricing.get('disposition')}",
    )
    _require(pricing.get("consensus_line_created") is False, "pricing adapter created a consensus line")
    _require(int(pricing.get("pricing_book_line_rows", -1)) == len(offers), "pricing audit/input row-count drift")
    _require(len(priced) == 2 * len(offers), "priced output is not exactly two model sides per bookmaker line")

    _require(football.get("disposition") == "FOOTBALL_SIMULATION_UNIVERSE_CERTIFIED", "football universe not certified")
    _require(int(football.get("football_players", 0)) == 469, f"football player universe drifted: {football.get('football_players')}")
    _require(int(football.get("football_teams", 0)) == 32, "football universe does not cover 32 teams")
    _require(int(football.get("canonical_games", 0)) == 16, "football universe does not cover 16 games")
    _require(int(football.get("priced_distribution_misses", -1)) == 0, "priced distributions missing from football universe")
    _require(int(football.get("priced_players_missing_from_football_universe", -1)) == 0, "priced player missing from football universe")
    _require(int(football.get("sportsbook_rows_used_to_define_player_universe", 1)) == 0, "sportsbook rows defined football universe")
    _require(football.get("sportsbook_inputs_used_to_generate_football_distributions") is False, "sportsbook input leaked into football distributions")
    _require(football.get("provider_event_ids_used_during_simulation") is False, "provider event IDs used during simulation")
    _require(football.get("team_wp_present_in_simulation_universe") is False, "market-derived team_wp leaked into simulation")
    _require(int(football.get("football_players_without_priced_offers", 0)) > 0, "football universe collapsed to priced-offer roster")

    _require(entitlement.get("projection_neutral_gate") == "PASS", "explicit target entitlement neutrality gate failed")
    _require(int(entitlement.get("projection_invariance_changed_arrays", -1)) == 0, "explicit entitlement changed baseline arrays")
    _require(float(entitlement.get("projection_invariance_max_element_gap", 1.0)) <= 1e-12, "explicit entitlement baseline element drift")
    _require(entitlement.get("sportsbook_inputs_used") is False, "sportsbook input leaked into target entitlement")
    _require(
        target_pool.get("disposition") == "EXPLICIT_TARGET_ENTITLEMENT_POOL_VALID",
        f"physical target pool not certified: {target_pool.get('disposition')}",
    )
    _require(float(target_pool.get("explicit_max_physical_gap", 1.0)) <= 1e-12, "team target probability does not conserve to one")
    _require(target_pool.get("sportsbook_inputs_used") is False, "sportsbook input leaked into target-pool certification")

    _require(football.get("te_r5p_full_slate_consumed") is True, "TE-R5P not consumed by football universe")
    _require(football.get("te_r5p_model_version") == "TE_R5P_PRODUCTION_MODEL_V1", "TE-R5P version drift")
    _require(te.get("disposition") == "TE_R5P_FULL_SLATE_ENTITLEMENT_READY", "TE-R5P entitlement audit not ready")
    _require(te.get("team_te_pool_preserved") is True, "TE-R5P changed TE-room pool")
    _require(te.get("non_te_entitlement_preserved") is True, "TE-R5P changed non-TE entitlement")
    _require(te.get("team_total_player_entitlement_preserved") is True, "TE-R5P changed team player entitlement mass")
    _require(te.get("sportsbook_inputs_used") is False, "sportsbook input leaked into TE-R5P")
    _require(te.get("current_or_future_outcomes_used") is False, "TE-R5P used current/future outcomes")

    _require(c2.get("disposition") == "QB_C2_PRODUCTION_DISTRIBUTION_INTEGRATION_PASS", "QB C2 integration not certified")
    _require(int(c2.get("football_qb_rows", 0)) == 32, "QB C2 does not cover 32 football starters")
    _require(int(c2.get("selected_qb_rows", 0)) > 0, "QB C2 selected zero QBs")
    _require(int(c2.get("state_capture_changed_arrays", -1)) == 0, "QB C2 state-capture seam changed canonical arrays")
    _require(float(c2.get("state_capture_max_element_gap", 1.0)) <= 1e-12, "QB C2 state-capture parity drift")
    _require(float(c2.get("max_raw_qb_mean_gap", 1.0)) <= 1e-10, "QB C2 changed raw QB mean")
    _require(c2.get("m89_m90_mean_authority_preserved_by_contract") is True, "M89/M90 QB mean authority not preserved")
    _require(int(c2.get("receiver_outputs_replaced", 1)) == 0, "QB C2 replaced receiver outputs")
    _require(int(c2.get("rb_outputs_replaced", 1)) == 0, "QB C2 replaced RB outputs")
    for field in (
        "sportsbook_inputs_to_starter_selection",
        "sportsbook_inputs_to_selector",
        "sportsbook_inputs_to_c2_generation",
    ):
        _require(int(c2.get(field, 1)) == 0, f"QB C2 sportsbook leakage: {field}")

    _require(stamp.get("disposition") == "QB_C2_PRICING_LINEAGE_STAMP_CERTIFIED", "QB C2 pricing lineage stamp not certified")
    _require(stamp.get("pricing_values_modified") is False, "QB C2 lineage stamp changed pricing values")
    _require(int(stamp.get("protected_columns_changed", 1)) == 0, "QB C2 lineage stamp changed protected columns")
    _require(int(stamp.get("pass_yard_qbs", 0)) == 32, "QB C2 pricing lineage does not cover 32 QBs")

    _require(rb_input.get("disposition") == "RB_RUSH_REC_DISTRIBUTION_CONSERVED_WITH_PROMOTED_P3", "RB P3 input conservation not certified")
    _require(rb_input.get("sportsbook_inputs_used") is False, "sportsbook input leaked into RB P3 conservation")
    rb_gap = pd.to_numeric(rb_final.get("conservation_gap"), errors="coerce")
    _require(rb_gap.notna().all() and np.isfinite(rb_gap.to_numpy(float)).all(), "RB final conservation audit has non-finite gaps")
    _require(float(rb_gap.abs().max()) <= 1e-6, "RB final rush+receiving conservation failed")

    _require(lineage.get("disposition") == "MARKET_MODEL_LINEAGE_EXPLICIT", "market model lineage not explicit")
    _require(lineage.get("qb_c2_distribution_consumed") is True, "lineage does not record QB C2 distribution consumption")
    _require(lineage.get("te_r5p_consumed") is True, "lineage does not record TE-R5P consumption")
    _require(lineage.get("sportsbook_inputs_used_to_define_lineage") is False, "sportsbook input used to define model lineage")
    _require(lineage.get("c2_full_stack_consumed") is False, "lineage incorrectly claims full shared C2 integration")

    _require(atd.get("execution_status") == "PASS", "ATD execution did not pass")
    _require(int(atd.get("current_sportsbook_inputs_used_for_football_probability", 1)) == 0, "sportsbook input leaked into ATD football probability")
    _require(int(atd.get("dedicated_probability_model_certified", 1)) == 0, "ATD science status unexpectedly changed; re-audit required")

    snapshot_mode = "NO_CREDIT_REPLAY" if live.get("replay_source_run") else "LIVE_SPORTSBOOK_SNAPSHOT"
    payload = {
        "disposition": "FULL_SLATE_CERTIFIED_STACK_READY_WITH_DECLARED_SCIENCE_LIMITATIONS",
        "snapshot_mode": snapshot_mode,
        "football_players": int(football.get("football_players")),
        "football_teams": int(football.get("football_teams")),
        "games": int(football.get("canonical_games")),
        "football_players_without_priced_offers": int(football.get("football_players_without_priced_offers")),
        "pricing_book_line_rows": int(len(offers)),
        "priced_side_rows": int(len(priced)),
        "priced_players": int(priced["player"].nunique()),
        "target_entitlement_version": str(football.get("explicit_target_entitlement_version", "")),
        "te_r5p_consumed": True,
        "qb_c2_distribution_consumed": True,
        "qb_c2_selected_qbs": int(c2.get("selected_qb_rows")),
        "qb_c2_max_raw_mean_gap": float(c2.get("max_raw_qb_mean_gap")),
        "rb_p3_rush_rec_conservation_max_gap": float(rb_gap.abs().max()),
        "sportsbook_defines_football_universe": False,
        "sportsbook_inputs_used_for_football_distributions": False,
        "sportsbook_role": "DOWNSTREAM_PRICING_COMPARISON_ONLY",
        "m89_m90_qb_mean_authority": True,
        "rb_p3_rushing_authority": True,
        "wr_m38_finite_pool_active": True,
        "te_r5p_entitlement_active": True,
        "atd_dedicated_science_certified": False,
        "c2_full_shared_qb_receiver_stack_consumed": False,
        "does_not_certify_all_market_science": True,
        "remaining_science_lanes": [
            "WR individual entitlement beyond M38",
            "RB receiving entitlement",
            "receiving efficiency/distribution calibration",
            "shared QB-receiver C2 conservation",
            "dedicated anytime-TD probability calibration",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[certified_full_slate_stack] " + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
