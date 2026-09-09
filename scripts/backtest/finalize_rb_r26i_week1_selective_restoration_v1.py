#!/usr/bin/env python3
"""R26I Week-1 selective-restoration child qualification.

No model fit or prediction regeneration. Each Week-1 RB room selects one complete
immutable endpoint: production baseline or original frozen R26, using only the
room states frozen by R26H.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = tuple(range(2020, 2026))
EPS = 1e-12


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def read_many(root: Path, name: str) -> pd.DataFrame:
    paths = sorted(root.rglob(name))
    if len(paths) < 6:
        raise RuntimeError(f"expected >=6 {name} files under {root}, found {len(paths)}")
    return pd.concat([pd.read_csv(p, low_memory=False) for p in paths], ignore_index=True, sort=False)


def read_one(root: Path, name: str) -> pd.DataFrame:
    paths = sorted(root.rglob(name))
    if len(paths) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(paths)}")
    return pd.read_csv(paths[0], low_memory=False)


def json_one(root: Path, name: str) -> dict:
    paths = sorted(root.rglob(name))
    if len(paths) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(paths)}")
    return json.loads(paths[0].read_text())


def metric(g: pd.DataFrame, variant: str, market: str) -> dict:
    a = num(g[f"actual_{market}"]).to_numpy(float)
    p = num(g[f"{variant}_{market}"]).to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(p)
    a = a[ok]; p = p[ok]
    if len(a) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "p90_abs_error": np.nan}
    e = p - a
    ae = np.abs(e)
    return {
        "n": int(len(a)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(e.mean()),
        "p90_abs_error": float(np.quantile(ae, 0.90)),
    }


def rel_change(base: float, cand: float) -> float:
    if not np.isfinite(base) or not np.isfinite(cand):
        return np.inf
    if abs(base) < EPS:
        return 0.0 if abs(cand) < EPS else np.inf
    return float(cand / base - 1.0)


def improves(base: float, cand: float) -> bool:
    return bool(np.isfinite(base) and np.isfinite(cand) and cand < base)


def max_abs(a: pd.Series, b: pd.Series) -> float:
    x = num(a).to_numpy(float); y = num(b).to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return 0.0
    return float(np.max(np.abs(x[ok] - y[ok])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r26-root", type=Path, required=True)
    ap.add_argument("--r26h-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--protected-clean-marker", type=Path, required=True)
    a = ap.parse_args()

    if not a.protected_clean_marker.exists() or a.protected_clean_marker.read_text().strip() != "PASS":
        raise RuntimeError("R26I protected-production clean marker missing")

    pred = read_many(a.r26_root, "r26_predictions.csv")
    audit = read_many(a.r26_root, "r26_structural_audit.csv")
    hdisp = json_one(a.r26h_root, "r26h_disposition.json")
    hrooms = read_one(a.r26h_root, "r26h_balanced_room_state.csv")
    if hdisp.get("disposition") != "BALANCED_TURNOVER_ROLE_STATE_CHILD_DESIGN_SIGNAL":
        raise RuntimeError(f"R26H did not authorize child design: {hdisp.get('disposition')}")
    authorized = hdisp.get("child_design_authorized_states", [])
    authorized_names = {str(x.get("state")) for x in authorized}
    expected_auth = {
        "MEANINGFUL_EXIT+VETERAN_ENTRY_PRESENT",
        "MEANINGFUL_EXIT+NO_PRIOR_ENTRY_PRESENT",
    }
    if authorized_names != expected_auth:
        raise RuntimeError(f"R26H authorized-state mismatch: {sorted(authorized_names)}")

    for c in [
        "season", "week", "vacancy_active", "continuing_same_team", "rb_rank",
        "room_exits_n", "room_entrants_n", "actual_targets", "actual_receptions",
        "baseline_targets", "candidate_targets", "baseline_receptions", "candidate_receptions",
    ]:
        pred[c] = num(pred[c])
    pred = pred.loc[pred.season.isin(SEASONS)].copy()
    pred["season"] = pred.season.astype(int)
    pred["week"] = pred.week.astype(int)
    w1 = pred.loc[pred.week.eq(1)].copy()

    state_cols = ["vacancy_active", "room_exits_n", "room_entrants_n"]
    state_nunique = w1.groupby(["season", "week", "team"])[state_cols].nunique(dropna=False)
    if (state_nunique > 1).any().any():
        raise RuntimeError("R26I parent room-state inconsistency")

    # H artifact is authoritative for balanced-room frozen source state.
    hrooms["season"] = num(hrooms.season).astype(int)
    hrooms["week"] = num(hrooms.week).astype(int)
    for c in ["veteran_entry_present", "no_prior_entry_present"]:
        hrooms[c] = num(hrooms[c]).fillna(0).astype(int)
    hstate = hrooms[[
        "season", "week", "team", "exit_class", "veteran_entry_present", "no_prior_entry_present"
    ]].drop_duplicates(["season", "week", "team"])

    # Merge first, then derive boolean masks on the merged frame. This is a
    # mechanical index-alignment correction only; the frozen football states
    # and child-selection logic are unchanged.
    w1 = w1.merge(hstate, on=["season", "week", "team"], how="left", validate="many_to_one")
    balanced = w1.vacancy_active.eq(1) & w1.room_exits_n.eq(w1.room_entrants_n)
    unbalanced = w1.vacancy_active.eq(1) & ~balanced
    nonvac = ~w1.vacancy_active.eq(1)
    if w1.loc[balanced, "exit_class"].isna().any():
        raise RuntimeError("R26I missing R26H source state for balanced room")

    restore_balanced = (
        balanced
        & w1.exit_class.eq("MEANINGFUL_EXIT")
        & (w1.veteran_entry_present.eq(1) | w1.no_prior_entry_present.eq(1))
    )
    fallback_balanced = balanced & ~restore_balanced
    select_r26 = unbalanced | restore_balanced

    for market in ("targets", "receptions"):
        w1[f"child_{market}"] = np.where(
            select_r26,
            w1[f"candidate_{market}"],
            w1[f"baseline_{market}"],
        )

    # Exact endpoint inheritance checks.
    nonvac_t = max_abs(w1.loc[nonvac, "child_targets"], w1.loc[nonvac, "baseline_targets"])
    nonvac_r = max_abs(w1.loc[nonvac, "child_receptions"], w1.loc[nonvac, "baseline_receptions"])
    unbal_t = max_abs(w1.loc[unbalanced, "child_targets"], w1.loc[unbalanced, "candidate_targets"])
    unbal_r = max_abs(w1.loc[unbalanced, "child_receptions"], w1.loc[unbalanced, "candidate_receptions"])
    restore_t = max_abs(w1.loc[restore_balanced, "child_targets"], w1.loc[restore_balanced, "candidate_targets"])
    restore_r = max_abs(w1.loc[restore_balanced, "child_receptions"], w1.loc[restore_balanced, "candidate_receptions"])
    fallback_t = max_abs(w1.loc[fallback_balanced, "child_targets"], w1.loc[fallback_balanced, "baseline_targets"])
    fallback_r = max_abs(w1.loc[fallback_balanced, "child_receptions"], w1.loc[fallback_balanced, "baseline_receptions"])

    # Whole-room endpoint selection implies target-mass conservation if both parents conserve.
    room = w1.groupby(["season", "week", "team"], as_index=False).agg(
        baseline_room_targets=("baseline_targets", "sum"),
        r26_room_targets=("candidate_targets", "sum"),
        child_room_targets=("child_targets", "sum"),
        vacancy_active=("vacancy_active", "first"),
        room_exits_n=("room_exits_n", "first"),
        room_entrants_n=("room_entrants_n", "first"),
        exit_class=("exit_class", "first"),
        veteran_entry_present=("veteran_entry_present", "max"),
        no_prior_entry_present=("no_prior_entry_present", "max"),
    )
    room["child_minus_baseline_target_mass"] = room.child_room_targets - room.baseline_room_targets
    max_child_room_gap = float(room.child_minus_baseline_target_mass.abs().max())

    max_parent_room_gap = float(num(audit.room_mass_gap).abs().max())
    max_team_gap = float(num(audit.team_entitlement_gap).abs().max())
    max_non_rb = float(num(audit.max_non_rb_entitlement_delta).abs().max())
    max_ry = float(num(audit.max_receiving_yard_mean_delta).abs().max())
    max_r22 = float(num(audit.r22_authority_delta).abs().max())
    sportsbook = int(num(audit.sportsbook_inputs_used).fillna(0).sum())
    future = int(num(audit.future_outcomes_used).fillna(0).sum())

    inc = w1.loc[w1.vacancy_active.eq(1) & w1.continuing_same_team.eq(1)].copy()
    rb1 = inc.loc[inc.rb_rank.eq(1)].copy()
    rb2 = inc.loc[inc.rb_rank.ge(2)].copy()
    if inc.empty:
        raise RuntimeError("R26I found zero Week-1 vacancy incumbents")

    b_inc_rec = metric(inc, "baseline", "receptions")
    r_inc_rec = metric(inc, "candidate", "receptions")
    c_inc_rec = metric(inc, "child", "receptions")
    b_inc_tgt = metric(inc, "baseline", "targets")
    r_inc_tgt = metric(inc, "candidate", "targets")
    c_inc_tgt = metric(inc, "child", "targets")
    b_rb1 = metric(rb1, "baseline", "receptions"); c_rb1 = metric(rb1, "child", "receptions")
    b_rb2 = metric(rb2, "baseline", "receptions"); c_rb2 = metric(rb2, "child", "receptions")
    b_global = metric(w1, "baseline", "receptions")
    r_global = metric(w1, "candidate", "receptions")
    c_global = metric(w1, "child", "receptions")

    season_rows = []
    seasons_improved = 0
    seasons_n20 = 0
    max_season_worsen = -np.inf
    for s in SEASONS:
        g = inc.loc[inc.season.eq(s)]
        b = metric(g, "baseline", "receptions")
        r = metric(g, "candidate", "receptions")
        c = metric(g, "child", "receptions")
        rw = rel_change(b["mae"], c["mae"])
        if improves(b["mae"], c["mae"]):
            seasons_improved += 1
        if int(c["n"]) >= 20:
            seasons_n20 += 1
        max_season_worsen = max(max_season_worsen, rw)
        season_rows.append({
            "season": s,
            "n": int(c["n"]),
            "baseline_receptions_mae": b["mae"],
            "r26_receptions_mae": r["mae"],
            "r26i_receptions_mae": c["mae"],
            "r26i_vs_baseline_relative_change": rw,
            "r26i_vs_r26_relative_change": rel_change(r["mae"], c["mae"]),
            "baseline_rmse": b["rmse"],
            "r26_rmse": r["rmse"],
            "r26i_rmse": c["rmse"],
            "baseline_bias": b["bias"],
            "r26_bias": r["bias"],
            "r26i_bias": c["bias"],
            "baseline_p90": b["p90_abs_error"],
            "r26_p90": r["p90_abs_error"],
            "r26i_p90": c["p90_abs_error"],
        })

    post20 = inc.loc[inc.season.ge(2021)].copy()
    r_post20 = metric(post20, "candidate", "receptions")
    c_post20 = metric(post20, "child", "receptions")

    inheritance_integrity = bool(
        nonvac_t <= EPS and nonvac_r <= EPS
        and unbal_t <= EPS and unbal_r <= EPS
        and restore_t <= EPS and restore_r <= EPS
        and fallback_t <= EPS and fallback_r <= EPS
        and max_child_room_gap <= 1e-10
        and max_parent_room_gap <= 1e-10 and max_team_gap <= 1e-10 and max_non_rb <= EPS
        and max_ry == 0.0 and max_r22 == 0.0 and sportsbook == 0 and future == 0
    )

    gates = {
        "01_r26_immutable_parent_structural_integrity": bool(max_parent_room_gap <= 1e-10 and max_team_gap <= 1e-10),
        "02_r26h_authorized_child_design": True,
        "03_no_regeneration_no_r9_refit": True,
        "04_sportsbook_future_zero_recmean_r22_unchanged": bool(sportsbook == 0 and future == 0 and max_ry == 0.0 and max_r22 == 0.0),
        "05_nonvacancy_rooms_baseline_exact": bool(nonvac_t <= EPS and nonvac_r <= EPS),
        "06_unbalanced_vacancy_rooms_r26_exact": bool(unbal_t <= EPS and unbal_r <= EPS),
        "07_balanced_authorized_restoration_rooms_r26_exact": bool(restore_t <= EPS and restore_r <= EPS),
        "08_balanced_unsupported_rooms_baseline_exact": bool(fallback_t <= EPS and fallback_r <= EPS),
        "09_child_rb_room_target_mass_gap_le_1e10": max_child_room_gap <= 1e-10,
        "10_pooled_w1_vacancy_incumbent_rec_mae_improves": improves(b_inc_rec["mae"], c_inc_rec["mae"]),
        "11_pooled_w1_vacancy_incumbent_rec_rmse_nonworse": c_inc_rec["rmse"] <= b_inc_rec["rmse"],
        "12_pooled_w1_vacancy_incumbent_abs_bias_nonworse": abs(c_inc_rec["bias"]) <= abs(b_inc_rec["bias"]),
        "13_pooled_w1_vacancy_incumbent_p90_worsen_le_2pct": rel_change(b_inc_rec["p90_abs_error"], c_inc_rec["p90_abs_error"]) <= 0.02,
        "14_pooled_w1_vacancy_incumbent_target_mae_improves": improves(b_inc_tgt["mae"], c_inc_tgt["mae"]),
        "15_w1_vacancy_incumbent_rec_mae_improves_4_of_6": seasons_improved >= 4,
        "16_no_w1_season_worsens_more_than_2pct": max_season_worsen <= 0.02,
        "17_support_n150_n20_in_4_seasons": int(c_inc_rec["n"]) >= 150 and seasons_n20 >= 4,
        "18_rb1_mae_worsen_le_1pct": rel_change(b_rb1["mae"], c_rb1["mae"]) <= 0.01,
        "19_rb2plus_mae_worsen_le_1pct": rel_change(b_rb2["mae"], c_rb2["mae"]) <= 0.01,
        "20_at_least_one_role_improves": improves(b_rb1["mae"], c_rb1["mae"]) or improves(b_rb2["mae"], c_rb2["mae"]),
        "21_global_w1_rec_mae_worsen_le_0p5pct": rel_change(b_global["mae"], c_global["mae"]) <= 0.005,
        "22_global_w1_rec_rmse_worsen_le_0p5pct": rel_change(b_global["rmse"], c_global["rmse"]) <= 0.005,
        "23_global_w1_abs_bias_nonworse": abs(c_global["bias"]) <= abs(b_global["bias"]),
        "24_preserve_r26_pooled_inc_rec_mae_within_0p5pct": rel_change(r_inc_rec["mae"], c_inc_rec["mae"]) <= 0.005,
        "25_preserve_r26_pooled_inc_target_mae_within_0p5pct": rel_change(r_inc_tgt["mae"], c_inc_tgt["mae"]) <= 0.005,
        "26_preserve_r26_2021_2025_inc_rec_mae_within_0p5pct": rel_change(r_post20["mae"], c_post20["mae"]) <= 0.005,
        "27_preserve_r26_global_w1_rec_mae_within_0p5pct": rel_change(r_global["mae"], c_global["mae"]) <= 0.005,
    }

    all_pass = bool(all(gates.values()))
    if not inheritance_integrity:
        disposition = "WEEK1_SELECTIVE_RESTORATION_INTEGRITY_FAILURE"
    elif all_pass:
        disposition = "WEEK1_SELECTIVE_RESTORATION_RETROSPECTIVE_SUPPORT_FOR_2026_SHADOW"
    else:
        disposition = "WEEK1_SELECTIVE_RESTORATION_MIXED_NO_SHADOW"

    room["balanced"] = (room.vacancy_active.eq(1) & room.room_exits_n.eq(room.room_entrants_n)).astype(int)
    room["authorized_restore"] = (
        room.balanced.eq(1)
        & room.exit_class.eq("MEANINGFUL_EXIT")
        & (num(room.veteran_entry_present).fillna(0).eq(1) | num(room.no_prior_entry_present).fillna(0).eq(1))
    ).astype(int)

    result = {
        "candidate": "RB_R26I_WEEK1_SELECTIVE_RESTORATION_V1",
        "scientific_label": "RETROSPECTIVE_CHILD_CANDIDATE_ON_EXPOSED_HISTORY",
        "disposition": disposition,
        "all_frozen_gates_pass": all_pass,
        "gates_passed": int(sum(bool(v) for v in gates.values())),
        "gates_total": len(gates),
        "gates": gates,
        "prospective_2026_week1_shadow_authorized": all_pass,
        "production_promotion_authorized": False,
        "all_season_use_authorized": False,
        "parent_dispositions_preserved": {
            "R26": "RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW",
            "R26E": "WEEK1_COMPONENT_NOT_QUALIFIED_NO_SHADOW",
            "R26G": "WEEK1_BALANCED_TURNOVER_GUARD_MIXED_NO_SHADOW",
            "R26H": hdisp["disposition"],
        },
        "inheritance": {
            "r26_predictions_regenerated": False,
            "r9_refit": False,
            "sportsbook_inputs_added": 0,
            "production_parameters_changed": False,
            "receiving_yard_means_changed": False,
            "r22_changed": False,
            "balanced_authorized_room_count": int(room.authorized_restore.sum()),
            "balanced_fallback_room_count": int((room.balanced.eq(1) & room.authorized_restore.eq(0)).sum()),
            "unbalanced_vacancy_room_count": int((room.vacancy_active.eq(1) & room.balanced.eq(0)).sum()),
            "max_nonvacancy_target_child_vs_baseline_delta": nonvac_t,
            "max_nonvacancy_rec_child_vs_baseline_delta": nonvac_r,
            "max_unbalanced_target_child_vs_r26_delta": unbal_t,
            "max_unbalanced_rec_child_vs_r26_delta": unbal_r,
            "max_authorized_balanced_target_child_vs_r26_delta": restore_t,
            "max_authorized_balanced_rec_child_vs_r26_delta": restore_r,
            "max_fallback_balanced_target_child_vs_baseline_delta": fallback_t,
            "max_fallback_balanced_rec_child_vs_baseline_delta": fallback_r,
            "max_child_rb_room_target_mass_gap": max_child_room_gap,
        },
        "pooled_week1_vacancy_incumbent": {
            "baseline_receptions": b_inc_rec,
            "r26_receptions": r_inc_rec,
            "r26i_receptions": c_inc_rec,
            "baseline_targets": b_inc_tgt,
            "r26_targets": r_inc_tgt,
            "r26i_targets": c_inc_tgt,
        },
        "roles": {
            "RB1": {"baseline": b_rb1, "r26i": c_rb1},
            "RB2PLUS": {"baseline": b_rb2, "r26i": c_rb2},
        },
        "global_week1": {"baseline": b_global, "r26": r_global, "r26i": c_global},
        "preservation_2021_2025": {"r26": r_post20, "r26i": c_post20},
        "temporal": {
            "seasons_improved": seasons_improved,
            "seasons_n_ge_20": seasons_n20,
            "max_single_season_relative_worsening": float(max_season_worsen),
            "season_rows": season_rows,
        },
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    w1.to_csv(a.out_dir / "r26i_week1_player_predictions.csv", index=False)
    room.to_csv(a.out_dir / "r26i_room_selection_audit.csv", index=False)
    pd.DataFrame(season_rows).to_csv(a.out_dir / "r26i_week1_season_metrics.csv", index=False)
    (a.out_dir / "r26i_disposition.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("=== season metrics ===")
    print(pd.DataFrame(season_rows).to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
