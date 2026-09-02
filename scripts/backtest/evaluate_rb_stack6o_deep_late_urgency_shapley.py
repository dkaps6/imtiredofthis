#!/usr/bin/env python3
"""RB STACK6O: exact no-fit Shapley attribution inside deep-late urgency cells."""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

CELLS = (
    "two_score_early_q4",
    "three_plus_early_q4",
    "two_score_late_q4",
    "three_plus_late_q4",
)
START_WEEK = 6
ALPHA = 0.75
EXPECTED_N = 388
EXPECTED_EMPTY_MAE = 5.518381962346741
EXPECTED_ALL_MAE = 5.121110810459461
EXPECTED_RECOVERY = EXPECTED_EMPTY_MAE - EXPECTED_ALL_MAE


def num(v):
    return pd.to_numeric(v, errors="coerce")


def lower(df):
    z = df.copy()
    z.columns = [str(c).strip().lower() for c in z.columns]
    return z


def one(root: Path, name: str):
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name}; found {len(hits)}")
    return lower(pd.read_csv(hits[0], low_memory=False))


def metric(y, p):
    y, p = num(y), num(p)
    ok = y.notna() & p.notna()
    y, p = y[ok], p[ok]
    e = p - y
    return {
        "n": int(len(y)),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(e.mean()),
        "corr": float(p.corr(y)) if len(y) >= 3 and p.nunique() > 1 and y.nunique() > 1 else np.nan,
    }


def subset_name(sub):
    return "NONE" if not sub else "+".join(c.upper() for c in CELLS if c in sub)


def subsets():
    out = []
    for k in range(len(CELLS) + 1):
        out.extend(frozenset(x) for x in itertools.combinations(CELLS, k))
    return out


def pbp_cells():
    import nflreadpy as nfl

    p = lower(nfl.load_pbp(seasons=[2025]).to_pandas())
    if "season_type" in p.columns:
        reg = p.loc[p["season_type"].astype(str).str.upper().eq("REG")].copy()
        if len(reg):
            p = reg
    p["team"] = p["posteam"].map(canon_team)
    p["rush_attempt"] = num(p["rush_attempt"]).fillna(0)
    p["qb_dropback"] = num(p["qb_dropback"]).fillna(0)
    p = p.loc[(p["rush_attempt"].eq(1) | p["qb_dropback"].eq(1)) & p["team"].ne("")].copy()

    if "score_differential" in p.columns:
        diff = num(p["score_differential"])
    else:
        diff = num(p["posteam_score"]) - num(p["defteam_score"])
    p["score_diff"] = diff.fillna(0.0)
    p["qtr_num"] = num(p["qtr"]).fillna(0)
    p["gsr"] = num(p.get("game_seconds_remaining", pd.Series(np.nan, index=p.index)))
    if p.loc[p["qtr_num"].eq(4), "gsr"].isna().any():
        raise RuntimeError("STACK6O missing game_seconds_remaining on Q4 offensive plays")

    p["trail"] = p["score_diff"].lt(-3)
    p["deep_late"] = p["score_diff"].le(-9) & p["qtr_num"].ge(4)
    two_score = p["score_diff"].between(-16, -9, inclusive="both")
    three_plus = p["score_diff"].le(-17)
    early_q4 = p["qtr_num"].eq(4) & p["gsr"].gt(450)
    late_q4 = p["qtr_num"].gt(4) | (p["qtr_num"].eq(4) & p["gsr"].le(450))

    p["cell"] = np.select(
        [
            p["deep_late"] & two_score & early_q4,
            p["deep_late"] & three_plus & early_q4,
            p["deep_late"] & two_score & late_q4,
            p["deep_late"] & three_plus & late_q4,
        ],
        CELLS,
        default="other",
    )

    rows = []
    for (season, week, team), g in p.groupby(["season", "week", "team"], dropna=False):
        n = float(len(g))
        rec = {
            "season": int(season),
            "week": int(week),
            "team": canon_team(team),
            "pbp_actual_off_plays": n,
            "pbp_trail_share": float(g["trail"].mean()) if n else np.nan,
            "pbp_deep_late_plays": float(g["deep_late"].sum()),
            "pbp_deep_late_share": float(g["deep_late"].mean()) if n else np.nan,
        }
        for c in CELLS:
            q = g.loc[g["cell"].eq(c)]
            rec[f"{c}_plays"] = float(len(q))
            rec[f"{c}_rushes"] = float(q["rush_attempt"].sum())
            rec[f"{c}_share"] = float(len(q) / n) if n else np.nan
            rec[f"{c}_rush_rate"] = float(q["rush_attempt"].mean()) if len(q) else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values(["season", "week", "team"]).reset_index(drop=True)
    if out.empty or out.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("STACK6O PBP aggregation invalid")
    return out


def score_subsets(df, label):
    rows = []
    empty = None
    for sub in subsets():
        col = f"ORACLE_{subset_name(sub)}"
        m = metric(df["actual_team_rush_att"], df[col])
        rows.append({"population": label, "subset": subset_name(sub), "corrected_cells": ";".join(sorted(sub)), **m})
        if not sub:
            empty = m["mae"]
    out = pd.DataFrame(rows)
    out["recovery_vs_empty"] = float(empty) - out["mae"]
    return out


def shapley(table, population):
    q = table.loc[table["population"].eq(population)].copy()
    values = {}
    for _, r in q.iterrows():
        values[frozenset(x for x in str(r["corrected_cells"]).split(";") if x)] = float(r["recovery_vs_empty"])
    n = len(CELLS)
    total = values[frozenset(CELLS)]
    rows = []
    for c in CELLS:
        others = [x for x in CELLS if x != c]
        phi = 0.0
        for k in range(n):
            for co in itertools.combinations(others, k):
                S = frozenset(co)
                weight = math.factorial(len(S)) * math.factorial(n - len(S) - 1) / math.factorial(n)
                phi += weight * (values[S | {c}] - values[S])
        rows.append(
            {
                "population": population,
                "cell": c,
                "shapley_recovery": phi,
                "fraction_of_deep_late_recovery": phi / total if abs(total) > 1e-12 else np.nan,
                "total_deep_late_recovery": total,
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m94c-root", type=Path, required=True)
    ap.add_argument("--stack6h-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    m = one(a.m94c_root, "m94c_2025_team_trace.csv")
    h = one(a.stack6h_root, "stack6h_team_trace.csv")
    p = pbp_cells()
    for d in (m, h, p):
        d["season"] = num(d["season"]).astype(int)
        d["week"] = num(d["week"]).astype(int)
        d["team"] = d["team"].map(canon_team)

    req = [
        "season", "week", "team", "actual_team_rush_att", "baseline_team_rush_att", "pred_off_plays",
        "lead_play_share", "neutral_play_share", "trail_play_share",
        "gs_team_lead_rush_rate_shrunk", "gs_team_neutral_rush_rate_shrunk", "gs_team_trail_rush_rate_shrunk",
    ]
    bins = ["pool_over_5", "pool_under_5", "pool_abs_5", "non_extreme_abs_lt3"]
    t = (
        m[req]
        .merge(h[["season", "week", "team", *bins]], on=["season", "week", "team"], how="inner", validate="one_to_one")
        .merge(p, on=["season", "week", "team"], how="inner", validate="one_to_one")
    )
    if len(t) != 544:
        raise RuntimeError(f"expected 544 joined rows; got {len(t)}")
    for c in t.columns:
        if c != "team":
            t[c] = num(t[c])

    cell_sum = sum(t[f"{c}_share"] for c in CELLS)
    t["urgency_cell_share_sum"] = cell_sum
    non_dl_trail = (t["trail_play_share"] - t["pbp_deep_late_share"]).clip(lower=0.0)
    fixed = (
        t["lead_play_share"] * t["gs_team_lead_rush_rate_shrunk"]
        + t["neutral_play_share"] * t["gs_team_neutral_rush_rate_shrunk"]
        + non_dl_trail * t["gs_team_trail_rush_rate_shrunk"]
    )

    for sub in subsets():
        deep_con = pd.Series(0.0, index=t.index)
        for c in CELLS:
            if c in sub:
                deep_con += t[f"{c}_rushes"] / t["pbp_actual_off_plays"]
            else:
                deep_con += t[f"{c}_share"] * t["gs_team_trail_rush_rate_shrunk"]
        t[f"ORACLE_{subset_name(sub)}"] = (
            (1.0 - ALPHA) * t["baseline_team_rush_att"]
            + ALPHA * t["pred_off_plays"] * (fixed + deep_con)
        )

    w = t.loc[t["week"].ge(START_WEEK)].copy()
    if len(w) != EXPECTED_N:
        raise RuntimeError(f"expected {EXPECTED_N} W6-18 rows; got {len(w)}")

    scores = score_subsets(w, "ALL_W6_18")
    masks = {
        "POOL_OVER_5": w["pool_over_5"].eq(1),
        "POOL_UNDER_5": w["pool_under_5"].eq(1),
        "POOL_ABS_5": w["pool_abs_5"].eq(1),
        "NON_EXTREME_ABS_LT3": w["non_extreme_abs_lt3"].eq(1),
    }
    scores = pd.concat([scores] + [score_subsets(w.loc[mask], label) for label, mask in masks.items()], ignore_index=True)
    shp = pd.concat([shapley(scores, pop) for pop in ["ALL_W6_18", *masks.keys()]], ignore_index=True)

    summaries = []
    for c in CELLS:
        plays = float(w[f"{c}_plays"].sum())
        rushes = float(w[f"{c}_rushes"].sum())
        summaries.append(
            {
                "cell": c,
                "team_games_with_plays": int(w[f"{c}_plays"].gt(0).sum()),
                "mean_plays_per_team_game": float(w[f"{c}_plays"].mean()),
                "total_plays": plays,
                "total_rushes": rushes,
                "aggregate_rush_rate": float(rushes / plays) if plays > 0 else np.nan,
            }
        )
    cell_summary = pd.DataFrame(summaries)

    empty = float(scores.loc[(scores["population"].eq("ALL_W6_18")) & scores["subset"].eq("NONE"), "mae"].iloc[0])
    all_name = subset_name(frozenset(CELLS))
    all_mae = float(scores.loc[(scores["population"].eq("ALL_W6_18")) & scores["subset"].eq(all_name), "mae"].iloc[0])
    recovery = empty - all_mae
    overall_shp = shp.loc[shp["population"].eq("ALL_W6_18")].sort_values("shapley_recovery", ascending=False)
    shapley_sum = float(overall_shp["shapley_recovery"].sum())
    cell_share_err = float((w["urgency_cell_share_sum"] - w["pbp_deep_late_share"]).abs().max())
    trail_share_err = float((w["trail_play_share"] - w["pbp_trail_share"]).abs().max())

    integrity_pass = int(
        len(w) == EXPECTED_N
        and cell_share_err <= 1e-9
        and trail_share_err <= 1e-9
        and abs(empty - EXPECTED_EMPTY_MAE) <= 1e-9
        and abs(all_mae - EXPECTED_ALL_MAE) <= 1e-9
        and abs(recovery - EXPECTED_RECOVERY) <= 1e-9
        and abs(shapley_sum - EXPECTED_RECOVERY) <= 1e-9
    )

    top = overall_shp.iloc[0]
    top_cell = str(top["cell"])
    top_phi = float(top["shapley_recovery"])
    top_frac = float(top["fraction_of_deep_late_recovery"])
    over_row = shp.loc[(shp["population"].eq("POOL_OVER_5")) & shp["cell"].eq(top_cell)].iloc[0]
    under_row = shp.loc[(shp["population"].eq("POOL_UNDER_5")) & shp["cell"].eq(top_cell)].iloc[0]
    over_phi = float(over_row["shapley_recovery"])
    over_frac = float(over_row["fraction_of_deep_late_recovery"])
    under_phi = float(under_row["shapley_recovery"])

    if not integrity_pass:
        disposition = "STACK6O_INTEGRITY_FAILURE_DO_NOT_INTERPRET"
    elif top_phi >= 0.12 and top_frac >= 0.35 and over_phi > 0 and over_frac >= 0.40:
        disposition = "URGENCY_CELL_DOMINANT"
    else:
        disposition = "DEEP_LATE_URGENCY_DISTRIBUTED"

    integrity = pd.DataFrame([
        {
            "m94c_rows": len(m),
            "pbp_team_games": len(p),
            "joined_rows": len(t),
            "w6_18_n": len(w),
            "trail_share_max_abs_diff": trail_share_err,
            "urgency_cell_sum_vs_deep_late_max_abs_diff": cell_share_err,
            "expected_empty_mae": EXPECTED_EMPTY_MAE,
            "observed_empty_mae": empty,
            "expected_all_cell_mae": EXPECTED_ALL_MAE,
            "observed_all_cell_mae": all_mae,
            "expected_deep_late_recovery": EXPECTED_RECOVERY,
            "observed_deep_late_recovery": recovery,
            "shapley_sum": shapley_sum,
            "shapley_sum_abs_error": abs(shapley_sum - recovery),
            "integrity_pass": integrity_pass,
            "fitted_models": 0,
            "feature_search": 0,
            "hyperparameter_search": 0,
            "threshold_search": 0,
            "sportsbook_inputs": 0,
            "target_game_pbp_used_as_oracle_only": 1,
        }
    ])
    disposition_df = pd.DataFrame([
        {
            "top_cell": top_cell,
            "top_cell_shapley_recovery": top_phi,
            "top_cell_fraction": top_frac,
            "top_cell_pool_over5_shapley": over_phi,
            "top_cell_pool_over5_fraction": over_frac,
            "top_cell_pool_under5_shapley": under_phi,
            "disposition": disposition,
            "production_change": 0,
            "player_recomposition_authorized": 0,
            "predictive_model_authorized": 0,
        }
    ])

    t.to_csv(a.out_dir / "stack6o_team_trace.csv", index=False)
    scores.to_csv(a.out_dir / "stack6o_subset_scores.csv", index=False)
    shp.to_csv(a.out_dir / "stack6o_shapley.csv", index=False)
    cell_summary.to_csv(a.out_dir / "stack6o_cell_summary.csv", index=False)
    integrity.to_csv(a.out_dir / "stack6o_integrity.csv", index=False)
    disposition_df.to_csv(a.out_dir / "stack6o_disposition.csv", index=False)

    print("=== STACK6O integrity ===")
    print(integrity.to_string(index=False))
    print("=== STACK6O cell summary ===")
    print(cell_summary.to_string(index=False))
    print("=== STACK6O shapley ===")
    print(shp.to_string(index=False))
    print("=== STACK6O disposition ===")
    print(disposition_df.to_string(index=False))
    print(f"STACK6O_DISPOSITION={disposition}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
