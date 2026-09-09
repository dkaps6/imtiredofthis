#!/usr/bin/env python3
"""R26G retrospective child qualification.

Preserve frozen R26 Week-1 behavior except balanced-turnover vacancy rooms
(room_exits_n == room_entrants_n), where the entire RB/FB room falls back to
production baseline targets/receptions. No model is fit or regenerated.
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
    ap.add_argument("--r26f-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    pred = read_many(a.r26_root, "r26_predictions.csv")
    audit = read_many(a.r26_root, "r26_structural_audit.csv")
    fpaths = sorted(a.r26f_root.rglob("r26f_disposition.json"))
    if len(fpaths) != 1:
        raise RuntimeError(f"expected one r26f_disposition.json, found {len(fpaths)}")
    fdisp = json.loads(fpaths[0].read_text())
    if fdisp.get("disposition") != "WEEK1_FAILURE_MECHANISM_IDENTIFIED_REPLICATED" or not fdisp.get("child_candidate_design_authorized"):
        raise RuntimeError(f"R26F does not authorize child design: {fdisp.get('disposition')}")

    for c in ["season", "week", "vacancy_active", "continuing_same_team", "rb_rank", "room_exits_n", "room_entrants_n",
              "actual_targets", "actual_receptions", "baseline_targets", "candidate_targets", "baseline_receptions", "candidate_receptions"]:
        pred[c] = num(pred[c])
    pred = pred.loc[pred.season.isin(SEASONS)].copy()
    pred["season"] = pred.season.astype(int)
    pred["week"] = pred.week.astype(int)

    # Sanity: room state must be internally constant for each target team-week.
    state_cols = ["vacancy_active", "room_exits_n", "room_entrants_n"]
    w1_all = pred.loc[pred.week.eq(1)].copy()
    room_state_nunique = w1_all.groupby(["season", "week", "team"])[state_cols].nunique(dropna=False)
    if (room_state_nunique > 1).any().any():
        bad = room_state_nunique.loc[(room_state_nunique > 1).any(axis=1)].head(20)
        raise RuntimeError(f"room-state inconsistency: {bad.to_dict('index')}")

    balanced = w1_all.vacancy_active.eq(1) & w1_all.room_exits_n.eq(w1_all.room_entrants_n)
    unbalanced_vacancy = w1_all.vacancy_active.eq(1) & ~balanced
    nonvacancy = ~w1_all.vacancy_active.eq(1)

    # Room-complete child selection. These are the only child values R26G creates.
    for market in ("targets", "receptions"):
        w1_all[f"r26g_{market}"] = np.where(
            unbalanced_vacancy,
            w1_all[f"candidate_{market}"],
            w1_all[f"baseline_{market}"],
        )

    # Exact inheritance audits by state.
    balanced_target_exact = max_abs(w1_all.loc[balanced, "r26g_targets"], w1_all.loc[balanced, "baseline_targets"])
    balanced_rec_exact = max_abs(w1_all.loc[balanced, "r26g_receptions"], w1_all.loc[balanced, "baseline_receptions"])
    unbalanced_target_exact = max_abs(w1_all.loc[unbalanced_vacancy, "r26g_targets"], w1_all.loc[unbalanced_vacancy, "candidate_targets"])
    unbalanced_rec_exact = max_abs(w1_all.loc[unbalanced_vacancy, "r26g_receptions"], w1_all.loc[unbalanced_vacancy, "candidate_receptions"])
    nonvac_target_exact = max_abs(w1_all.loc[nonvacancy, "r26g_targets"], w1_all.loc[nonvacancy, "baseline_targets"])
    nonvac_rec_exact = max_abs(w1_all.loc[nonvacancy, "r26g_receptions"], w1_all.loc[nonvacancy, "baseline_receptions"])

    # Because each room selects one entire conserved endpoint, target mass must remain baseline-exact.
    room_mass = w1_all.groupby(["season", "week", "team"], as_index=False).agg(
        baseline_room_targets=("baseline_targets", "sum"),
        child_room_targets=("r26g_targets", "sum"),
        vacancy_active=("vacancy_active", "first"),
        room_exits_n=("room_exits_n", "first"),
        room_entrants_n=("room_entrants_n", "first"),
    )
    room_mass["child_minus_baseline_target_mass"] = room_mass.child_room_targets - room_mass.baseline_room_targets
    max_child_room_mass_gap = float(room_mass.child_minus_baseline_target_mass.abs().max())

    # The original R26 structural contract must still be exact.
    max_parent_room_gap = float(num(audit.room_mass_gap).abs().max())
    max_team_gap = float(num(audit.team_entitlement_gap).abs().max())
    max_non_rb = float(num(audit.max_non_rb_entitlement_delta).abs().max())
    max_ry = float(num(audit.max_receiving_yard_mean_delta).abs().max())
    max_r22 = float(num(audit.r22_authority_delta).abs().max())
    sportsbook = int(num(audit.sportsbook_inputs_used).fillna(0).sum())
    future = int(num(audit.future_outcomes_used).fillna(0).sum())

    # Metric helper expects variant columns; alias R26G child.
    w1_all["child_targets"] = w1_all.r26g_targets
    w1_all["child_receptions"] = w1_all.r26g_receptions

    inc = w1_all.loc[w1_all.vacancy_active.eq(1) & w1_all.continuing_same_team.eq(1)].copy()
    rb1 = inc.loc[inc.rb_rank.eq(1)].copy()
    rb2 = inc.loc[inc.rb_rank.ge(2)].copy()
    if inc.empty:
        raise RuntimeError("R26G found zero W1 vacancy incumbents")

    # Pooled metrics for baseline, original R26, and child.
    b_inc_rec = metric(inc, "baseline", "receptions")
    r_inc_rec = metric(inc, "candidate", "receptions")
    g_inc_rec = metric(inc, "child", "receptions")
    b_inc_tgt = metric(inc, "baseline", "targets")
    r_inc_tgt = metric(inc, "candidate", "targets")
    g_inc_tgt = metric(inc, "child", "targets")

    b_rb1 = metric(rb1, "baseline", "receptions"); g_rb1 = metric(rb1, "child", "receptions")
    b_rb2 = metric(rb2, "baseline", "receptions"); g_rb2 = metric(rb2, "child", "receptions")
    b_global = metric(w1_all, "baseline", "receptions")
    r_global = metric(w1_all, "candidate", "receptions")
    g_global = metric(w1_all, "child", "receptions")

    # Season replication.
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
        if improves(b["mae"], c["mae"]): seasons_improved += 1
        if int(c["n"]) >= 20: seasons_n20 += 1
        max_season_worsen = max(max_season_worsen, rw)
        season_rows.append({
            "season": s, "n": int(c["n"]),
            "baseline_receptions_mae": b["mae"],
            "r26_receptions_mae": r["mae"],
            "r26g_receptions_mae": c["mae"],
            "r26g_vs_baseline_relative_change": rw,
            "r26g_vs_r26_relative_change": rel_change(r["mae"], c["mae"]),
            "baseline_rmse": b["rmse"], "r26_rmse": r["rmse"], "r26g_rmse": c["rmse"],
            "baseline_bias": b["bias"], "r26_bias": r["bias"], "r26g_bias": c["bias"],
            "baseline_p90": b["p90_abs_error"], "r26_p90": r["p90_abs_error"], "r26g_p90": c["p90_abs_error"],
        })

    # 2021-25 pooled preservation check.
    post20 = inc.loc[inc.season.ge(2021)].copy()
    r_post20 = metric(post20, "candidate", "receptions")
    g_post20 = metric(post20, "child", "receptions")

    inheritance_integrity = bool(
        balanced_target_exact <= EPS and balanced_rec_exact <= EPS
        and unbalanced_target_exact <= EPS and unbalanced_rec_exact <= EPS
        and nonvac_target_exact <= EPS and nonvac_rec_exact <= EPS
        and max_child_room_mass_gap <= 1e-10
        and max_parent_room_gap <= 1e-10 and max_team_gap <= 1e-10 and max_non_rb <= EPS
        and max_ry == 0.0 and max_r22 == 0.0 and sportsbook == 0 and future == 0
    )

    gates = {
        "01_r26_immutable_and_parent_structural_integrity": bool(max_parent_room_gap <= 1e-10 and max_team_gap <= 1e-10),
        "02_r26f_authorized_forensic_disposition": True,
        "03_no_regeneration_no_r9_refit": True,
        "04_sportsbook_zero_production_recmean_r22_unchanged": bool(sportsbook == 0 and future == 0 and max_ry == 0.0 and max_r22 == 0.0),
        "05_balanced_turnover_rooms_baseline_exact": bool(balanced_target_exact <= EPS and balanced_rec_exact <= EPS),
        "06_unbalanced_vacancy_rooms_r26_exact": bool(unbalanced_target_exact <= EPS and unbalanced_rec_exact <= EPS),
        "07_nonvacancy_rooms_baseline_exact": bool(nonvac_target_exact <= EPS and nonvac_rec_exact <= EPS),
        "08_child_rb_room_target_mass_gap_le_1e10": max_child_room_mass_gap <= 1e-10,
        "09_pooled_w1_vacancy_incumbent_rec_mae_improves": improves(b_inc_rec["mae"], g_inc_rec["mae"]),
        "10_pooled_w1_vacancy_incumbent_rec_rmse_nonworse": g_inc_rec["rmse"] <= b_inc_rec["rmse"],
        "11_pooled_w1_vacancy_incumbent_abs_bias_nonworse": abs(g_inc_rec["bias"]) <= abs(b_inc_rec["bias"]),
        "12_pooled_w1_vacancy_incumbent_p90_worsen_le_2pct": rel_change(b_inc_rec["p90_abs_error"], g_inc_rec["p90_abs_error"]) <= 0.02,
        "13_pooled_w1_vacancy_incumbent_target_mae_improves": improves(b_inc_tgt["mae"], g_inc_tgt["mae"]),
        "14_w1_vacancy_incumbent_rec_mae_improves_4_of_6": seasons_improved >= 4,
        "15_no_w1_season_worsens_more_than_2pct": max_season_worsen <= 0.02,
        "16_support_n150_n20_in_4_seasons": int(g_inc_rec["n"]) >= 150 and seasons_n20 >= 4,
        "17_rb1_mae_worsen_le_1pct": rel_change(b_rb1["mae"], g_rb1["mae"]) <= 0.01,
        "18_rb2plus_mae_worsen_le_1pct": rel_change(b_rb2["mae"], g_rb2["mae"]) <= 0.01,
        "19_at_least_one_role_improves": improves(b_rb1["mae"], g_rb1["mae"]) or improves(b_rb2["mae"], g_rb2["mae"]),
        "20_global_w1_rec_mae_worsen_le_0p5pct": rel_change(b_global["mae"], g_global["mae"]) <= 0.005,
        "21_global_w1_rec_rmse_worsen_le_0p5pct": rel_change(b_global["rmse"], g_global["rmse"]) <= 0.005,
        "22_global_w1_abs_bias_nonworse": abs(g_global["bias"]) <= abs(b_global["bias"]),
        "23_preserve_r26_pooled_inc_rec_mae_within_0p5pct": rel_change(r_inc_rec["mae"], g_inc_rec["mae"]) <= 0.005,
        "24_preserve_r26_pooled_inc_target_mae_within_0p5pct": rel_change(r_inc_tgt["mae"], g_inc_tgt["mae"]) <= 0.005,
        "25_preserve_r26_2021_2025_inc_rec_mae_within_0p5pct": rel_change(r_post20["mae"], g_post20["mae"]) <= 0.005,
        "26_preserve_r26_global_w1_rec_mae_within_0p5pct": rel_change(r_global["mae"], g_global["mae"]) <= 0.005,
    }

    all_pass = bool(all(gates.values()))
    if not inheritance_integrity:
        disposition = "WEEK1_BALANCED_TURNOVER_GUARD_INTEGRITY_FAILURE"
    elif all_pass:
        disposition = "WEEK1_BALANCED_TURNOVER_GUARD_RETROSPECTIVE_SUPPORT_FOR_2026_SHADOW"
    else:
        disposition = "WEEK1_BALANCED_TURNOVER_GUARD_MIXED_NO_SHADOW"

    balanced_rooms = room_mass.loc[room_mass.vacancy_active.eq(1) & room_mass.room_exits_n.eq(room_mass.room_entrants_n)]
    unbalanced_rooms = room_mass.loc[room_mass.vacancy_active.eq(1) & ~room_mass.room_exits_n.eq(room_mass.room_entrants_n)]

    result = {
        "candidate": "RB_R26G_WEEK1_BALANCED_TURNOVER_GUARD_V1",
        "scientific_label": "RETROSPECTIVE_CHILD_CANDIDATE_ON_EXPOSED_HISTORY",
        "disposition": disposition,
        "all_frozen_gates_pass": all_pass,
        "gates_passed": int(sum(bool(v) for v in gates.values())),
        "gates_total": len(gates),
        "gates": gates,
        "prospective_2026_week1_shadow_design_authorized": all_pass,
        "production_promotion_authorized": False,
        "all_season_use_authorized": False,
        "parent_dispositions_unchanged": {
            "R26": "RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW",
            "R26E": "WEEK1_COMPONENT_MIXED_OR_FAIL_NO_SHADOW",
            "R26F": "WEEK1_FAILURE_MECHANISM_IDENTIFIED_REPLICATED",
        },
        "inheritance": {
            "r26_predictions_regenerated": False,
            "r9_refit": False,
            "sportsbook_inputs_added": 0,
            "production_parameters_changed": False,
            "receiving_yard_means_changed": False,
            "r22_changed": False,
            "balanced_turnover_room_count": int(len(balanced_rooms)),
            "unbalanced_vacancy_room_count": int(len(unbalanced_rooms)),
            "max_balanced_target_child_vs_baseline_delta": balanced_target_exact,
            "max_balanced_rec_child_vs_baseline_delta": balanced_rec_exact,
            "max_unbalanced_target_child_vs_r26_delta": unbalanced_target_exact,
            "max_unbalanced_rec_child_vs_r26_delta": unbalanced_rec_exact,
            "max_nonvacancy_target_child_vs_baseline_delta": nonvac_target_exact,
            "max_nonvacancy_rec_child_vs_baseline_delta": nonvac_rec_exact,
            "max_child_rb_room_target_mass_gap": max_child_room_mass_gap,
        },
        "pooled_week1_vacancy_incumbent": {
            "baseline_receptions": b_inc_rec,
            "r26_receptions": r_inc_rec,
            "r26g_receptions": g_inc_rec,
            "baseline_targets": b_inc_tgt,
            "r26_targets": r_inc_tgt,
            "r26g_targets": g_inc_tgt,
        },
        "roles": {
            "RB1": {"baseline": b_rb1, "r26g": g_rb1},
            "RB2PLUS": {"baseline": b_rb2, "r26g": g_rb2},
        },
        "global_week1": {"baseline": b_global, "r26": r_global, "r26g": g_global},
        "preservation_2021_2025": {"r26": r_post20, "r26g": g_post20},
        "temporal": {
            "seasons_improved": seasons_improved,
            "seasons_n_ge_20": seasons_n20,
            "max_single_season_relative_worsening": float(max_season_worsen),
            "season_rows": season_rows,
        },
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    w1_all.to_csv(a.out_dir / "r26g_week1_player_predictions.csv", index=False)
    room_mass.to_csv(a.out_dir / "r26g_room_mass_audit.csv", index=False)
    pd.DataFrame(season_rows).to_csv(a.out_dir / "r26g_week1_season_metrics.csv", index=False)
    (a.out_dir / "r26g_disposition.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("=== season metrics ===")
    print(pd.DataFrame(season_rows).to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
