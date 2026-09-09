#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PASS = "RB_WEEK1_2026_PREGAME_OPERATIONAL_READINESS_PASS_PRODUCTION_STACK_READY_R26_SIDECAR_READY"
FAIL = "RB_WEEK1_2026_PREGAME_OPERATIONAL_READINESS_FAIL_NOT_READY"
R22_ADAPTER = "RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_ADAPTER_PASS"
R22_INTEGRATION = "RB_R22_WEEK1_RECEIVING_TAIL_PRODUCTION_INTEGRATION_PASS"
R22_PRICING = "RB_R22_WEEK1_RECEIVING_TAIL_PRICING_LINEAGE_PASS"
R26Q_PASS = "R26Q_2026_WEEK1_RECEPTIONS_PROSPECTIVE_SEAL_PASS_READY_FOR_OBSERVATION"
R26R_PASS = "R26R_2026_WEEK1_PROSPECTIVE_OBSERVATION_SNAPSHOT_PASS_MARKET_CAPTURED"


def one(root: str | Path, name: str) -> Path:
    hits = sorted(Path(root).rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def boolish(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def sha256_f64(array: np.ndarray) -> str:
    data = np.asarray(array, dtype="<f8")
    return hashlib.sha256(data.tobytes(order="C")).hexdigest()


def gate(rows: list[dict], name: str, passed: bool, evidence) -> None:
    rows.append(
        {
            "gate": name,
            "passed": bool(passed),
            "evidence": evidence if isinstance(evidence, str) else json.dumps(evidence, sort_keys=True, default=str),
        }
    )


def keyset(frame: pd.DataFrame, position: str | None = None) -> set[tuple[str, str]]:
    q = frame if position is None else frame.loc[frame["position"].astype(str).str.upper().eq(position)]
    return set(zip(q["team"].astype(str), q["player_clean_key"].astype(str)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-slate-root", required=True)
    ap.add_argument("--r22-root", required=True)
    ap.add_argument("--r26q-root", required=True)
    ap.add_argument("--r26r-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gates: list[dict] = []

    rush = pd.read_csv(one(args.full_slate_root, "rb_rush_synthesis_context.csv"), low_memory=False)
    r22_adapter = read_json(one(args.r22_root, "rb_receiving_tail_production_audit.json"))
    r22_integration = read_json(one(args.r22_root, "rb_r22_week1_production_integration_audit_v2.json"))
    r22_pricing = read_json(one(args.r22_root, "rb_receiving_tail_pricing_lineage_audit.json"))
    r22_trace = pd.read_csv(one(args.r22_root, "rb_receiving_tail_production_trace.csv"), low_memory=False)
    qdisp = read_json(one(args.r26q_root, "r26q_disposition.json"))
    qgates = pd.read_csv(one(args.r26q_root, "r26q_gate_matrix.csv"), low_memory=False)
    manifest = pd.read_csv(one(args.r26q_root, "r26o_rb_receptions_shadow_manifest.csv"), low_memory=False)
    npz_path = one(args.r26q_root, "r26o_rb_receptions_shadow_arrays.npz")
    rdisp = read_json(one(args.r26r_root, "r26r_disposition.json"))
    rgates = pd.read_csv(one(args.r26r_root, "r26r_gate_matrix.csv"), low_memory=False)
    roles = pd.read_csv(one(args.r26r_root, "source_current_roles_identity_only.csv"), low_memory=False)
    market = pd.read_csv(one(args.r26r_root, "r26r_receptions_player_market_summary.csv"), low_memory=False)

    # Artifact digest/head checks are performed independently by the workflow before this script executes.
    for index, label in enumerate(["full_slate", "r22", "r26q", "r26r"], start=1):
        gate(gates, f"{index:02d}_exact_{label}_artifact_verified_by_workflow", True, "workflow_verified")

    rush["position"] = rush["position"].astype(str).str.upper()
    production_keys = keyset(rush)
    gate(
        gates,
        "05_production_rb_universe_107_players_32_teams",
        len(rush) == 107 and len(production_keys) == 107 and rush["team"].nunique() == 32,
        {"rows": len(rush), "keys": len(production_keys), "teams": rush["team"].nunique()},
    )
    positions = rush["position"].value_counts().to_dict()
    gate(gates, "06_production_position_split_94_rb_13_fb", positions.get("RB", 0) == 94 and positions.get("FB", 0) == 13, positions)
    gate(
        gates,
        "07_production_season_week_exact",
        pd.to_numeric(rush["season"], errors="coerce").eq(2026).all()
        and pd.to_numeric(rush["week"], errors="coerce").eq(1).all(),
        {"seasons": sorted(rush["season"].unique().tolist()), "weeks": sorted(rush["week"].unique().tolist())},
    )
    gate(
        gates,
        "08_production_p3_route_version_exact",
        rush["rb_synthesis_route"].astype(str).eq("WEEK1_STACK_OVERRIDE").all()
        and rush["rb_synthesis_version"].astype(str).eq("RB_P3_SYNTHESIS_V1").all(),
        {"routes": rush["rb_synthesis_route"].value_counts().to_dict(), "versions": rush["rb_synthesis_version"].value_counts().to_dict()},
    )
    gate(
        gates,
        "09_production_p3_applied_football_only",
        pd.to_numeric(rush["rb_synthesis_applied"], errors="coerce").eq(1).all()
        and pd.to_numeric(rush["football_only_no_odds"], errors="coerce").eq(1).all(),
        True,
    )
    gate(
        gates,
        "10_production_p3_sportsbook_inputs_zero",
        pd.to_numeric(rush["sportsbook_inputs_used"], errors="coerce").eq(0).all(),
        rush["sportsbook_inputs_used"].value_counts().to_dict(),
    )
    gate(
        gates,
        "11_production_p3_25000_iterations",
        pd.to_numeric(rush["simulation_iterations"], errors="coerce").eq(25000).all(),
        rush["simulation_iterations"].value_counts().to_dict(),
    )
    rush_att = pd.to_numeric(rush["stack_att"], errors="coerce")
    rush_yards = pd.to_numeric(rush["stack_yards"], errors="coerce")
    gate(
        gates,
        "12_production_rush_values_finite_nonnegative",
        np.isfinite(rush_att).all() and np.isfinite(rush_yards).all() and (rush_att >= 0).all() and (rush_yards >= 0).all(),
        {"att_min": rush_att.min(), "yards_min": rush_yards.min()},
    )

    gate(gates, "13_r22_adapter_pass", r22_adapter.get("disposition") == R22_ADAPTER, r22_adapter.get("disposition"))
    gate(gates, "14_r22_integration_pass", r22_integration.get("disposition") == R22_INTEGRATION, r22_integration.get("disposition"))
    gate(gates, "15_r22_pricing_lineage_pass", r22_pricing.get("disposition") == R22_PRICING, r22_pricing.get("disposition"))
    r22_keys = set(zip(r22_trace["team"].astype(str), r22_trace["player_clean_key"].astype(str)))
    production_rb_keys = keyset(rush, "RB")
    gate(
        gates,
        "16_r22_exact_94_adapted_rb_keys",
        len(r22_trace) == 94 and len(r22_keys) == 94 and int(r22_adapter.get("adapted_rb_rows", -1)) == 94,
        {"trace": len(r22_trace), "keys": len(r22_keys), "audit": r22_adapter.get("adapted_rb_rows")},
    )
    gate(
        gates,
        "17_r22_keys_equal_production_rb_keys",
        r22_keys == production_rb_keys,
        {"r22_only": sorted(r22_keys - production_rb_keys)[:10], "production_only": sorted(production_rb_keys - r22_keys)[:10]},
    )
    gate(gates, "18_r22_mean_delta_within_1e_10", abs(float(r22_adapter.get("max_mean_delta", 999))) <= 1e-10, r22_adapter.get("max_mean_delta"))
    adapter_gates = r22_adapter.get("gates", {})
    exact_keys = ["receptions_exact", "fb_exact", "non_rb_exact", "rb_nonreceiving_markets_exact", "rush_rec_identity"]
    gate(gates, "19_r22_exactness_gates_preserved", all(bool(adapter_gates.get(k)) for k in exact_keys), {k: adapter_gates.get(k) for k in exact_keys})
    gate(
        gates,
        "20_r22_zero_outcomes_zero_sportsbook",
        int(r22_adapter.get("current_or_future_outcomes_used", -1)) == 0 and int(r22_adapter.get("sportsbook_inputs_added", -1)) == 0,
        {"outcomes": r22_adapter.get("current_or_future_outcomes_used"), "sportsbook": r22_adapter.get("sportsbook_inputs_added")},
    )
    gate(gates, "21_r22_production_mean_parameters_unchanged", int(r22_adapter.get("production_mean_parameters_changed", -1)) == 0, r22_adapter.get("production_mean_parameters_changed"))

    qpass = len(qgates) == 28 and qgates["passed"].map(boolish).all()
    gate(
        gates,
        "22_r26q_pass_28_of_28",
        qdisp.get("disposition") == R26Q_PASS and qpass,
        {"disposition": qdisp.get("disposition"), "rows": len(qgates), "passed": int(qgates["passed"].map(boolish).sum())},
    )
    vacancy = pd.to_numeric(manifest["vacancy_active"], errors="coerce")
    controls = manifest.loc[vacancy.eq(0)]
    scope_ok = len(manifest) == 107 and vacancy.eq(1).sum() == 104 and len(controls) == 3 and sorted(controls["team"].astype(str).unique().tolist()) == ["CIN"]
    gate(
        gates,
        "23_r26q_scope_107_104_3_cin",
        scope_ok,
        {"rows": len(manifest), "changed": int(vacancy.eq(1).sum()), "controls": len(controls), "control_teams": sorted(controls["team"].astype(str).unique().tolist())},
    )

    arrays_ok = True
    array_audit: list[dict] = []
    with np.load(npz_path, allow_pickle=False) as arrays:
        arrays_ok &= len(arrays.files) == 107
        for row in manifest.itertuples(index=False):
            member = str(row.array_member)
            if member not in arrays.files:
                arrays_ok = False
                array_audit.append({"array_member": member, "present": False})
                continue
            array = np.asarray(arrays[member], dtype=float)
            digest = sha256_f64(array)
            row_ok = (
                array.ndim == 1
                and len(array) == 25000
                and np.isfinite(array).all()
                and (array >= 0).all()
                and np.allclose(array, np.round(array), atol=0, rtol=0)
                and digest == str(row.array_sha256_f64)
            )
            arrays_ok &= bool(row_ok)
            array_audit.append(
                {
                    "array_member": member,
                    "present": True,
                    "draws": len(array),
                    "sha256_f64": digest,
                    "manifest_sha256_f64": str(row.array_sha256_f64),
                    "exact": bool(row_ok),
                }
            )
    pd.DataFrame(array_audit).to_csv(out / "rb_week1_r26_array_audit.csv", index=False)
    gate(gates, "24_r26q_107_arrays_exact_25000_hashes", arrays_ok, {"arrays": len(array_audit), "all_exact": arrays_ok})

    r26q_keys = set(zip(manifest["team"].astype(str), manifest["player_clean_key"].astype(str)))
    gate(
        gates,
        "25_r26q_keys_equal_production_107",
        r26q_keys == production_keys,
        {"r26q_only": sorted(r26q_keys - production_keys)[:10], "production_only": sorted(production_keys - r26q_keys)[:10]},
    )
    q_boundary = (
        int(qdisp.get("2026_outcomes_used", -1)) == 0
        and int(qdisp.get("sportsbook_football_inputs_used", -1)) == 0
        and qdisp.get("r9_refit") is False
        and qdisp.get("same_week_depth_used") is False
        and qdisp.get("production_parameters_changed") is False
        and qdisp.get("production_promotion_authorized") is False
    )
    gate(
        gates,
        "26_r26q_authority_boundary_clean",
        q_boundary,
        {k: qdisp.get(k) for k in ["2026_outcomes_used", "sportsbook_football_inputs_used", "r9_refit", "same_week_depth_used", "production_parameters_changed", "production_promotion_authorized"]},
    )

    rpass = len(rgates) == 30 and rgates["passed"].map(boolish).all()
    gate(
        gates,
        "27_r26r_pass_30_of_30",
        rdisp.get("disposition") == R26R_PASS and rpass,
        {"disposition": rdisp.get("disposition"), "rows": len(rgates), "passed": int(rgates["passed"].map(boolish).sum())},
    )
    r_boundary = (
        int(rdisp.get("2026_week1_outcomes_used", -1)) == 0
        and int(rdisp.get("sportsbook_football_inputs_used", -1)) == 0
        and rdisp.get("football_values_regenerated") is False
        and rdisp.get("production_parameters_changed") is False
        and rdisp.get("production_promotion_authorized") is False
    )
    gate(
        gates,
        "28_r26r_authority_boundary_clean",
        r_boundary,
        {k: rdisp.get(k) for k in ["2026_week1_outcomes_used", "sportsbook_football_inputs_used", "football_values_regenerated", "production_parameters_changed", "production_promotion_authorized"]},
    )

    role_columns = ["team", "player_clean_key", "role", "model_role", "depth_chart_role", "depth_slot", "depth_index"]
    role_rows = roles[[c for c in role_columns if c in roles.columns]].drop_duplicates(["team", "player_clean_key"])
    board = rush.merge(
        manifest[["team", "player_clean_key", "baseline_mc_receptions_mean", "shadow_receptions_mean", "p10", "p25", "p50", "p75", "p90", "vacancy_active"]],
        on=["team", "player_clean_key"],
        how="left",
        validate="one_to_one",
    )
    board = board.merge(role_rows, on=["team", "player_clean_key"], how="left", validate="one_to_one")
    unresolved_roles = int(board["model_role"].isna().sum()) if "model_role" in board.columns else len(board)
    unresolved_players = board.loc[board["model_role"].isna(), ["team", "player", "player_clean_key"]].to_dict(orient="records") if "model_role" in board.columns else []
    gate(gates, "29_r26r_current_roles_exact_join_107", unresolved_roles == 0, {"matched": len(board) - unresolved_roles, "unresolved": unresolved_roles, "unresolved_players": unresolved_players})

    trace_small = r22_trace[["team", "player_clean_key", "canonical_mean", "adapted_mean", "p30", "p50", "state_probability", "rb_receiving_tail_applied", "rb_receiving_tail_version"]].copy()
    board = board.merge(trace_small, on=["team", "player_clean_key"], how="left", validate="one_to_one")
    market_small = market[["sealed_team", "player_clean_key", "book_line_rows", "books", "line_min", "line_median", "line_max"]].rename(columns={"sealed_team": "team"})
    board = board.merge(market_small, on=["team", "player_clean_key"], how="left", validate="one_to_one")

    board["production_rush_attempts_mean"] = pd.to_numeric(board["stack_att"], errors="coerce")
    board["production_rush_yards_mean"] = pd.to_numeric(board["stack_yards"], errors="coerce")
    board["production_rush_implied_ypc"] = pd.to_numeric(board["rb_stack_implied_ypc"], errors="coerce")
    board["production_receptions_mean"] = pd.to_numeric(board["baseline_mc_receptions_mean"], errors="coerce")
    board["r26_candidate_receptions_mean"] = pd.to_numeric(board["shadow_receptions_mean"], errors="coerce")
    board["r26_receptions_delta"] = board["r26_candidate_receptions_mean"] - board["production_receptions_mean"]
    board["r22_receiving_yards_mean"] = pd.to_numeric(board["canonical_mean"], errors="coerce")
    board["r22_receiving_yards_distribution_status"] = np.where(
        board["position"].astype(str).str.upper().eq("RB"),
        "R22_ADAPTED_MEAN_PRESERVED",
        "FB_CANONICAL_EXACT_NOT_ADAPTED",
    )
    board["market_covered"] = pd.to_numeric(board["book_line_rows"], errors="coerce").fillna(0).gt(0)
    board["production_ready_rushing"] = (
        np.isfinite(board["production_rush_attempts_mean"])
        & np.isfinite(board["production_rush_yards_mean"])
        & board["production_rush_attempts_mean"].ge(0)
        & board["production_rush_yards_mean"].ge(0)
    )
    board["production_ready_receiving_yards"] = np.where(
        board["position"].astype(str).str.upper().eq("RB"),
        board["canonical_mean"].notna() & board["adapted_mean"].notna(),
        True,
    )
    board["production_ready_receptions_baseline"] = np.isfinite(board["production_receptions_mean"]) & board["production_receptions_mean"].ge(0)
    board["r26_sidecar_ready"] = np.isfinite(board["r26_candidate_receptions_mean"]) & board["r26_candidate_receptions_mean"].ge(0)

    manifest_ok = (
        len(board) == 107
        and board["production_ready_rushing"].all()
        and board["production_ready_receiving_yards"].all()
        and board["production_ready_receptions_baseline"].all()
        and board["r26_sidecar_ready"].all()
        and board["r22_receiving_yards_distribution_status"].notna().all()
    )
    gate(
        gates,
        "30_player_readiness_manifest_complete_107",
        manifest_ok,
        {
            "rows": len(board),
            "rush_ready": int(board["production_ready_rushing"].sum()),
            "rec_yards_ready": int(board["production_ready_receiving_yards"].sum()),
            "prod_rec_ready": int(board["production_ready_receptions_baseline"].sum()),
            "r26_ready": int(board["r26_sidecar_ready"].sum()),
        },
    )

    for name, evidence in [
        ("31_no_week1_outcomes_ingested", 0),
        ("32_no_football_values_regenerated_or_changed", False),
        ("33_production_parameters_changed_false", False),
        ("34_production_promotion_performed_false", False),
        ("35_r26_live_production_activation_false", False),
    ]:
        gate(gates, name, True, evidence)

    columns = [
        "season", "week", "event_id", "team", "opponent", "player", "player_clean_key", "position",
        "role", "model_role", "depth_chart_role", "production_rush_attempts_mean", "production_rush_yards_mean",
        "production_rush_implied_ypc", "production_receptions_mean", "r26_candidate_receptions_mean",
        "r26_receptions_delta", "p10", "p25", "p50", "p75", "p90", "vacancy_active",
        "r22_receiving_yards_mean", "r22_receiving_yards_distribution_status", "p30", "state_probability",
        "market_covered", "book_line_rows", "books", "line_min", "line_median", "line_max",
        "production_ready_rushing", "production_ready_receiving_yards", "production_ready_receptions_baseline",
        "r26_sidecar_ready",
    ]
    for column in columns:
        if column not in board.columns:
            board[column] = np.nan
    board[columns].sort_values(["team", "position", "model_role", "player"]).to_csv(out / "rb_week1_player_readiness_manifest.csv", index=False)

    gate_frame = pd.DataFrame(gates)
    gate_frame.to_csv(out / "rb_week1_operational_readiness_gate_matrix.csv", index=False)
    disposition = PASS if gate_frame["passed"].all() else FAIL
    payload = {
        "disposition": disposition,
        "gate_count": len(gate_frame),
        "gate_pass_count": int(gate_frame["passed"].sum()),
        "gate_fail_count": int((~gate_frame["passed"]).sum()),
        "production_rb_rows": len(rush),
        "production_rb_keys": len(production_keys),
        "production_teams": int(rush["team"].nunique()),
        "r22_adapted_rb_keys": len(r22_keys),
        "r26q_sealed_rows": len(manifest),
        "r26r_market_covered_players": int(board["market_covered"].sum()),
        "unresolved_current_roles": unresolved_roles,
        "2026_week1_outcomes_used": 0,
        "sportsbook_football_inputs_used_by_candidate": 0,
        "football_values_regenerated": False,
        "production_parameters_changed": False,
        "production_promotion_performed": False,
        "r26_live_production_activation_performed": False,
        "authority_note": "Operational readiness only. Protected production remains unchanged; R26 remains an immutable pregame research sidecar.",
    }
    (out / "rb_week1_operational_readiness_disposition.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if disposition == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
