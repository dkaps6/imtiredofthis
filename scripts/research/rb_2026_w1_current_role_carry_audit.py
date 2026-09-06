#!/usr/bin/env python3
"""No-fit audit of current role authority in the promoted 2026 W1 RB P3 path."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.bayesian_v2 import apply_bayesian_to_metrics
from scripts.modeling.ml_v2 import apply_ml_to_metrics
from scripts.modeling.simulation_rules import apply_rules_to_metrics
from scripts.modeling.state_v2 import apply_state_to_metrics
from scripts.run_rb_week1_no_odds import build_internal_rb_metrics

DATA = Path("data")
OUT_ROWS = DATA / "rb_2026_w1_current_role_carry_audit.csv"
OUT_TEAMS = DATA / "rb_2026_w1_current_role_team_summary.csv"
OUT_SUMMARY = DATA / "rb_2026_w1_current_role_audit_summary.json"
RB_GROUPS = {"RB", "HB", "TB", "FB", "RUNNING BACK", "BACKFIELD"}


def _key(value) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"{label} missing/empty: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    if x.empty:
        raise RuntimeError(f"{label} has zero rows: {path}")
    return x


def _rank_number(value) -> float:
    m = re.search(r"(\d+)", str(value or ""))
    return float(m.group(1)) if m else np.nan


def _static_path_audit() -> dict:
    player_form_path = Path("scripts/run_player_form_v2.py")
    sim_path = Path("scripts/simulation_v2.py")
    rules_path = Path("scripts/modeling/simulation_rules.py")
    texts = {}
    for name, path in {
        "player_form": player_form_path,
        "simulation": sim_path,
        "rules": rules_path,
    }.items():
        if not path.exists():
            raise RuntimeError(f"static audit source missing: {path}")
        texts[name] = path.read_text(encoding="utf-8")
    pf = texts["player_form"]
    return {
        "playerform_preserves_depth_role": bool('out["depth_role"]' in pf and 'out.get("role"' in pf),
        "playerform_effective_role_replaced_by_model_role": bool('out["role"] = out["model_role"]' in pf),
        "simulation_v2_depth_role_reference_count": int(texts["simulation"].count("depth_role")),
        "simulation_rules_depth_role_reference_count": int(texts["rules"].count("depth_role")),
    }


def _build_pre_sim(season: int, week: int) -> pd.DataFrame:
    metrics = build_internal_rb_metrics(season, week)
    metrics = apply_ml_to_metrics(metrics, _read(DATA / "model_ml_diagnostics.csv", "ML diagnostics"))
    metrics = apply_state_to_metrics(metrics, _read(DATA / "model_state_diagnostics.csv", "State diagnostics"))
    metrics = apply_bayesian_to_metrics(metrics)
    metrics = apply_rules_to_metrics(metrics)
    x = metrics.loc[metrics["market"].astype(str).str.lower().eq("rush_att")].copy()
    x["player_clean_key"] = x["player_clean_key"].fillna(x["player"].map(_key)).astype(str)
    if x.duplicated(["team", "player_clean_key"]).any():
        raise RuntimeError("pre-simulation rush_att metrics contain duplicate player-team rows")
    wanted = [
        "team", "opponent", "player", "player_clean_key", "position", "position_group",
        "depth_role", "model_role", "role", "rush_share", "bayes_rush_share",
        "rules_rush_share", "rules_role", "bayes_applied", "rules_applied",
        "ml_applied", "state_applied",
    ]
    wanted = [c for c in wanted if c in x.columns]
    return x[wanted].copy()


def run(season: int, week: int, out_rows: Path, out_teams: Path, out_summary: Path) -> dict:
    if int(week) != 1:
        raise RuntimeError(f"audit is frozen to Week 1, got week={week}")

    roles = _read(DATA / "roles_ourlads.csv", "Ourlads roles")
    roles["team"] = roles["team"].astype(str).str.upper().str.strip()
    roles["player_clean_key"] = roles["player"].map(_key)
    pos = roles.get("position", pd.Series("", index=roles.index)).fillna("").astype(str).str.upper().str.strip()
    grp = roles.get("position_group", pd.Series("", index=roles.index)).fillna("").astype(str).str.upper().str.strip()
    roles = roles.loc[pos.isin(RB_GROUPS) | grp.isin(RB_GROUPS)].copy()
    roles = roles.drop_duplicates(["team", "player_clean_key"], keep="first")
    role_keep = [c for c in [
        "team", "player_clean_key", "player", "position", "position_group", "depth_slot",
        "depth_index", "depth_chart_role", "role", "model_role", "status"
    ] if c in roles.columns]
    roles = roles[role_keep].rename(columns={
        "player": "ourlads_player",
        "position": "ourlads_position",
        "position_group": "ourlads_position_group",
        "role": "ourlads_role",
        "model_role": "ourlads_model_role",
        "status": "ourlads_status",
    })

    pre = _build_pre_sim(season, week)
    pre = pre.rename(columns={
        "player": "playerform_player",
        "position": "playerform_position",
        "position_group": "playerform_position_group",
        "role": "effective_role",
    })

    p3 = _read(DATA / "rb_rush_synthesis_context.csv", "promoted P3 context")
    p3["team"] = p3["team"].astype(str).str.upper().str.strip()
    p3["player_clean_key"] = p3["player_clean_key"].fillna(p3["player"].map(_key)).astype(str)
    if p3.duplicated(["team", "player_clean_key"]).any():
        raise RuntimeError("P3 context contains duplicate player-team rows")

    joined = p3.merge(roles, on=["team", "player_clean_key"], how="left", validate="one_to_one", indicator="_role_join")
    joined = joined.merge(pre, on=["team", "player_clean_key"], how="left", validate="one_to_one", suffixes=("", "_pre"))

    if "opponent_pre" in joined.columns:
        joined["opponent"] = joined["opponent"].combine_first(joined["opponent_pre"])

    joined["depth_rank"] = pd.to_numeric(joined.get("depth_index"), errors="coerce")
    missing_depth = joined["depth_rank"].isna()
    joined.loc[missing_depth, "depth_rank"] = joined.loc[missing_depth, "ourlads_role"].map(_rank_number)
    joined["depth_rank"] = pd.to_numeric(joined["depth_rank"], errors="coerce")
    joined["model_role_rank"] = pd.to_numeric(
        joined.get("model_role", pd.Series(np.nan, index=joined.index)).map(_rank_number),
        errors="coerce",
    )
    joined["effective_role_rank"] = pd.to_numeric(
        joined.get("effective_role", pd.Series(np.nan, index=joined.index)).map(_rank_number),
        errors="coerce",
    )
    joined["stack_att"] = pd.to_numeric(joined["stack_att"], errors="coerce")
    joined["stack_yards"] = pd.to_numeric(joined["stack_yards"], errors="coerce")
    joined["rb_synthesis_proj"] = pd.to_numeric(joined["rb_synthesis_proj"], errors="coerce")
    joined["p3_implied_ypc"] = np.where(joined["stack_att"] > 0, joined["rb_synthesis_proj"] / joined["stack_att"], np.nan)
    joined["team_rb_projected_carries"] = joined.groupby("team")["stack_att"].transform("sum")
    joined["projected_carry_share"] = np.where(
        joined["team_rb_projected_carries"] > 0,
        joined["stack_att"] / joined["team_rb_projected_carries"],
        np.nan,
    )
    joined["projected_carry_rank"] = pd.to_numeric(
        joined.groupby("team")["stack_att"].rank(method="first", ascending=False),
        errors="coerce",
    )
    joined["depth_vs_model_role_mismatch"] = (
        joined["depth_rank"].notna() & joined["model_role_rank"].notna() &
        ~np.isclose(joined["depth_rank"].to_numpy(dtype=float), joined["model_role_rank"].to_numpy(dtype=float), equal_nan=True)
    ).astype(int)
    joined["depth_vs_projected_carry_rank_mismatch"] = (
        joined["depth_rank"].notna() & joined["projected_carry_rank"].notna() &
        ~np.isclose(joined["depth_rank"].to_numpy(dtype=float), joined["projected_carry_rank"].to_numpy(dtype=float), equal_nan=True)
    ).astype(int)
    status = joined.get("ourlads_status", pd.Series("", index=joined.index)).fillna("").astype(str).str.lower()
    joined["inactive_positive_projection"] = (status.eq("inactive") & joined["stack_att"].gt(0)).astype(int)

    path_audit = _static_path_audit()
    share_fields_present = [c for c in ["rush_share", "bayes_rush_share", "rules_rush_share"] if c in joined.columns]

    role_match = joined["_role_join"].eq("both")
    role_coverage = float(role_match.mean()) if len(joined) else 0.0
    teams = int(joined["team"].nunique())
    no_odds = pd.to_numeric(joined.get("sportsbook_inputs_used", 1), errors="coerce").eq(0).all()
    version_ok = joined.get("rb_synthesis_version", pd.Series("", index=joined.index)).astype(str).eq("RB_P3_SYNTHESIS_V1").all()
    route_ok = joined.get("rb_synthesis_route", pd.Series("", index=joined.index)).astype(str).eq("WEEK1_STACK_OVERRIDE").all()
    parity_max = float(np.nanmax(np.abs(joined["rb_synthesis_proj"] - joined["stack_yards"])))

    gate = bool(teams == 32 and role_coverage >= 0.95 and no_odds and version_ok and route_ok and parity_max <= 1e-10)

    team_rows = []
    for team, g in joined.groupby("team", sort=True):
        active = ~g.get("ourlads_status", pd.Series("", index=g.index)).fillna("").astype(str).str.lower().eq("inactive")
        depth1 = g.loc[g["depth_rank"].eq(1)]
        depth2 = g.loc[g["depth_rank"].eq(2)]
        depth3 = g.loc[g["depth_rank"].ge(3)]
        total = float(g["stack_att"].sum())
        d1_att = float(depth1["stack_att"].sum())
        d2_att = float(depth2["stack_att"].sum())
        d3_att = float(depth3["stack_att"].sum())
        depth1_is_leader = bool((depth1["projected_carry_rank"] == 1).any()) if not depth1.empty else False
        shares = pd.to_numeric(g["projected_carry_share"], errors="coerce").fillna(0.0)
        team_rows.append({
            "team": team,
            "opponent": g["opponent"].dropna().astype(str).iloc[0] if g["opponent"].notna().any() else "",
            "active_rb_fb_count": int(active.sum()),
            "team_rb_projected_carries": total,
            "depth1_projected_carries": d1_att,
            "depth1_projected_share": d1_att / total if total > 0 else np.nan,
            "depth2_projected_carries": d2_att,
            "depth2_projected_share": d2_att / total if total > 0 else np.nan,
            "depth3plus_projected_carries": d3_att,
            "depth3plus_projected_share": d3_att / total if total > 0 else np.nan,
            "depth1_is_projected_carry_leader": int(depth1_is_leader),
            "depth_model_role_mismatches": int(g["depth_vs_model_role_mismatch"].sum()),
            "depth_projected_order_mismatches": int(g["depth_vs_projected_carry_rank_mismatch"].sum()),
            "inactive_positive_projection_count": int(g["inactive_positive_projection"].sum()),
            "projected_carry_hhi": float(np.square(shares).sum()),
        })
    team_df = pd.DataFrame(team_rows)

    if not gate:
        disposition = "MECHANICAL_AUDIT_FAILURE"
    elif int(joined["inactive_positive_projection"].sum()) > 0:
        disposition = "CURRENT_ROLE_INTEGRATION_INCOMPLETE"
    elif (
        path_audit["playerform_preserves_depth_role"]
        and path_audit["playerform_effective_role_replaced_by_model_role"]
        and path_audit["simulation_v2_depth_role_reference_count"] == 0
        and path_audit["simulation_rules_depth_role_reference_count"] == 0
    ):
        disposition = "CURRENT_ROLE_PRESERVED_BUT_NOT_DIRECT_ALLOCATION_INPUT"
    else:
        disposition = "CURRENT_ROLE_AUTHORITY_PRESENT_AND_COHERENT"

    summary = {
        "season": int(season),
        "week": int(week),
        "rows": int(len(joined)),
        "teams": teams,
        "ourlads_to_p3_identity_coverage": role_coverage,
        "sportsbook_inputs_used_zero": bool(no_odds),
        "rb_synthesis_version_ok": bool(version_ok),
        "rb_synthesis_route_ok": bool(route_ok),
        "p3_parent_parity_max_abs_diff": parity_max,
        "integrity_gate_pass": gate,
        "depth_model_role_mismatch_rows": int(joined["depth_vs_model_role_mismatch"].sum()),
        "depth_projected_order_mismatch_rows": int(joined["depth_vs_projected_carry_rank_mismatch"].sum()),
        "inactive_positive_projection_rows": int(joined["inactive_positive_projection"].sum()),
        "teams_where_depth1_not_projected_leader": int((team_df["depth1_is_projected_carry_leader"] == 0).sum()),
        "pre_sim_share_fields_present": share_fields_present,
        "static_path_audit": path_audit,
        "model_fit": 0,
        "sportsbook_used": 0,
        "production_change": 0,
        "disposition": disposition,
    }

    keep_front = [
        "player", "team", "opponent", "ourlads_player", "ourlads_position", "ourlads_position_group",
        "depth_slot", "depth_index", "depth_chart_role", "ourlads_role", "ourlads_status",
        "depth_role", "model_role", "effective_role", "rush_share", "bayes_rush_share",
        "rules_rush_share", "rules_role", "stack_att", "stack_yards", "rb_synthesis_proj",
        "p3_implied_ypc", "team_rb_projected_carries", "projected_carry_share",
        "projected_carry_rank", "depth_rank", "model_role_rank", "depth_vs_model_role_mismatch",
        "depth_vs_projected_carry_rank_mismatch", "inactive_positive_projection",
        "rb_synthesis_version", "rb_synthesis_route", "sportsbook_inputs_used",
    ]
    keep = [c for c in keep_front if c in joined.columns]
    audit_rows = joined[keep].sort_values(["team", "projected_carry_rank", "player"], na_position="last").reset_index(drop=True)

    out_rows.parent.mkdir(parents=True, exist_ok=True)
    audit_rows.to_csv(out_rows, index=False)
    team_df.to_csv(out_teams, index=False)
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\n--- CURRENT ROLE / CARRY AUDIT ---")
    print(audit_rows.to_string(index=False))
    print("\n--- TEAM SUMMARY ---")
    print(team_df.to_string(index=False))

    if not gate:
        raise RuntimeError("RB current-role carry-authority audit failed frozen mechanical integrity gate")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=1)
    ap.add_argument("--out-rows", type=Path, default=OUT_ROWS)
    ap.add_argument("--out-teams", type=Path, default=OUT_TEAMS)
    ap.add_argument("--out-summary", type=Path, default=OUT_SUMMARY)
    args = ap.parse_args()
    run(args.season, args.week, args.out_rows, args.out_teams, args.out_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
