#!/usr/bin/env python3
"""R26H diagnostic: explain balanced-turnover Week-1 R26 effects with frozen role state.

Reads immutable parent evidence only. No fit, no regenerated predictions, no production writes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]


def read_one(root: Path, name: str) -> pd.DataFrame:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return pd.read_csv(hits[0], low_memory=False)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def metric(g: pd.DataFrame, variant: str, market: str = "receptions") -> dict:
    a = num(g[f"actual_{market}"])
    p = num(g[f"{variant}_{market}"])
    mask = a.notna() & p.notna() & np.isfinite(a) & np.isfinite(p)
    a = a.loc[mask].to_numpy(float)
    p = p.loc[mask].to_numpy(float)
    if len(a) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "p90_abs_error": np.nan}
    e = p - a
    ae = np.abs(e)
    return {
        "n": int(len(a)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(e * e))),
        "bias": float(e.mean()),
        "p90_abs_error": float(np.quantile(ae, .90)),
    }


def collect_r26(root: Path) -> pd.DataFrame:
    hits = sorted(root.rglob("r26_predictions.csv"))
    if len(hits) != 6:
        raise RuntimeError(f"expected 6 R26 prediction files, found {len(hits)}")
    parts = [pd.read_csv(p, low_memory=False) for p in hits]
    x = pd.concat(parts, ignore_index=True)
    x["season"] = num(x.season).astype(int)
    x["week"] = num(x.week).astype(int)
    x["vacancy_active"] = num(x.vacancy_active).fillna(0).astype(int)
    x["room_exits_n"] = num(x.room_exits_n).fillna(0).astype(int)
    x["room_entrants_n"] = num(x.room_entrants_n).fillna(0).astype(int)
    x["continuing_same_team"] = num(x.continuing_same_team).fillna(0).astype(int)
    x["new_to_team_veteran"] = num(x.new_to_team_veteran).fillna(0).astype(int)
    x["no_prior_nfl_roster"] = num(x.no_prior_nfl_roster).fillna(0).astype(int)
    return x


def room_entry_state(x: pd.DataFrame) -> pd.DataFrame:
    keys = ["season", "week", "team"]
    rows = []
    for k, g in x.groupby(keys, sort=True):
        vet = bool(g.new_to_team_veteran.eq(1).any())
        nop = bool(g.no_prior_nfl_roster.eq(1).any())
        if vet and nop:
            state = "MIXED_ENTRY_STATE"
        elif vet:
            state = "VETERAN_ENTRY_PRESENT"
        elif nop:
            state = "NO_PRIOR_ENTRY_PRESENT"
        else:
            state = "UNRESOLVED_ENTRY_STATE"
        rows.append({
            **dict(zip(keys, k)),
            "veteran_entry_present": int(vet),
            "no_prior_entry_present": int(nop),
            "entry_state": state,
        })
    return pd.DataFrame(rows)


def classify_exit(r: pd.Series) -> str:
    hist = int(pd.to_numeric(pd.Series([r.get("any_exit_positive_history", 0)]), errors="coerce").fillna(0).iloc[0])
    tgt = pd.to_numeric(pd.Series([r.get("max_exit_prior_targets_pg", np.nan)]), errors="coerce").iloc[0]
    share = pd.to_numeric(pd.Series([r.get("max_exit_prior_rb_room_share", np.nan)]), errors="coerce").iloc[0]
    if (np.isfinite(tgt) and tgt > 1.0) or (np.isfinite(share) and share >= .25):
        return "MEANINGFUL_EXIT"
    if hist:
        return "LOW_EXIT"
    return "UNKNOWN_EXIT_HISTORY"


def state_masks(x: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "MEANINGFUL_EXIT+VETERAN_ENTRY_PRESENT": x.exit_class.eq("MEANINGFUL_EXIT") & x.veteran_entry_present.eq(1),
        "MEANINGFUL_EXIT+NO_PRIOR_ENTRY_PRESENT": x.exit_class.eq("MEANINGFUL_EXIT") & x.no_prior_entry_present.eq(1),
        "LOW_EXIT+VETERAN_ENTRY_PRESENT": x.exit_class.eq("LOW_EXIT") & x.veteran_entry_present.eq(1),
        "LOW_EXIT+NO_PRIOR_ENTRY_PRESENT": x.exit_class.eq("LOW_EXIT") & x.no_prior_entry_present.eq(1),
        "UNKNOWN_EXIT_HISTORY": x.exit_class.eq("UNKNOWN_EXIT_HISTORY"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r26-root", type=Path, required=True)
    ap.add_argument("--r26c-root", type=Path, required=True)
    ap.add_argument("--r26f-root", type=Path, required=True)
    ap.add_argument("--r26g-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    pred = collect_r26(a.r26_root)
    base = pred.loc[
        pred.season.isin(SEASONS)
        & pred.week.eq(1)
        & pred.vacancy_active.eq(1)
        & pred.room_exits_n.eq(pred.room_entrants_n)
    ].copy()
    if base.empty:
        raise RuntimeError("R26H found zero balanced-turnover Week-1 rows")

    # Verify forensic and child parent dispositions, without using their outcomes as features.
    fdisp = json.loads(next(a.r26f_root.rglob("r26f_disposition.json")).read_text())
    gdisp = json.loads(next(a.r26g_root.rglob("r26g_disposition.json")).read_text())
    if fdisp.get("disposition") != "WEEK1_FAILURE_MECHANISM_IDENTIFIED_REPLICATED":
        raise RuntimeError("R26H R26F parent disposition mismatch")
    if gdisp.get("disposition") != "WEEK1_BALANCED_TURNOVER_GUARD_MIXED_NO_SHADOW":
        raise RuntimeError("R26H R26G parent disposition mismatch")

    ent = room_entry_state(base)
    exit_tw = read_one(a.r26c_root, "r26c_vacancy_teamweek_aggregate.csv")
    exit_tw["season"] = num(exit_tw.season).astype(int)
    exit_tw["week"] = num(exit_tw.week).astype(int)
    exit_tw["exit_class"] = exit_tw.apply(classify_exit, axis=1)

    room = ent.merge(
        exit_tw,
        on=["season", "week", "team"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_exit"),
    )
    coverage = float(room.exit_class.notna().mean())
    if coverage < .99:
        raise RuntimeError(f"R26H exit-state coverage below 99%: {coverage}")

    inc = base.loc[base.continuing_same_team.eq(1)].copy()
    inc = inc.merge(
        room[["season", "week", "team", "exit_class", "entry_state", "veteran_entry_present", "no_prior_entry_present"]],
        on=["season", "week", "team"],
        how="left",
        validate="many_to_one",
    )
    if inc.exit_class.isna().any():
        raise RuntimeError("R26H missing room role state on incumbent rows")

    summary_rows = []
    season_rows = []
    authorization = []
    for state, mask in state_masks(inc).items():
        g = inc.loc[mask].copy()
        b = metric(g, "baseline", "receptions")
        c = metric(g, "candidate", "receptions")
        bt = metric(g, "baseline", "targets")
        ct = metric(g, "candidate", "targets")
        delta = float(c["mae"] - b["mae"]) if b["n"] else np.nan
        pooled_direction = "BENEFICIAL" if np.isfinite(delta) and delta < 0 else ("HARMFUL" if np.isfinite(delta) and delta > 0 else "NEUTRAL_OR_EMPTY")
        same_outside_2020 = 0
        supported_seasons = 0
        for season in SEASONS:
            sg = g.loc[g.season.eq(season)]
            sb = metric(sg, "baseline", "receptions")
            sc = metric(sg, "candidate", "receptions")
            sd = float(sc["mae"] - sb["mae"]) if sb["n"] else np.nan
            if sb["n"]:
                supported_seasons += 1
            same = False
            if season != 2020 and sb["n"]:
                same = (pooled_direction == "BENEFICIAL" and sd < 0) or (pooled_direction == "HARMFUL" and sd > 0)
                if same:
                    same_outside_2020 += 1
            season_rows.append({
                "state": state,
                "season": season,
                "n": int(sb["n"]),
                "baseline_rec_mae": sb["mae"],
                "r26_rec_mae": sc["mae"],
                "r26_minus_baseline_mae": sd,
                "same_direction_as_pooled_outside_2020": bool(same),
            })
        eligible = bool(b["n"] >= 20 and pooled_direction in {"BENEFICIAL", "HARMFUL"} and same_outside_2020 >= 2)
        if eligible:
            authorization.append({"state": state, "direction": pooled_direction, "n": int(b["n"]), "outside_2020_replications": same_outside_2020})
        summary_rows.append({
            "state": state,
            "n": int(b["n"]),
            "pooled_direction": pooled_direction,
            "baseline_rec_mae": b["mae"],
            "r26_rec_mae": c["mae"],
            "r26_minus_baseline_rec_mae": delta,
            "baseline_rec_rmse": b["rmse"],
            "r26_rec_rmse": c["rmse"],
            "baseline_rec_bias": b["bias"],
            "r26_rec_bias": c["bias"],
            "baseline_rec_p90": b["p90_abs_error"],
            "r26_rec_p90": c["p90_abs_error"],
            "baseline_target_mae": bt["mae"],
            "r26_target_mae": ct["mae"],
            "supported_seasons": supported_seasons,
            "outside_2020_same_direction_replications": same_outside_2020,
            "child_design_eligible": eligible,
        })

    structural = {
        "candidate": "RB_R26H_WEEK1_BALANCED_TURNOVER_ROLE_STATE_ATLAS_V1",
        "scientific_label": "RETROSPECTIVE_DIAGNOSTIC_ON_EXPOSED_HISTORY",
        "balanced_week1_rooms": int(room[["season", "team"]].drop_duplicates().shape[0]),
        "balanced_week1_incumbent_rows": int(len(inc)),
        "exit_state_room_coverage": coverage,
        "target_game_features_added": 0,
        "sportsbook_inputs_added": 0,
        "r26_predictions_regenerated": False,
        "r9_refit": False,
        "production_parameters_changed": False,
        "receiving_yard_means_changed": False,
        "r22_changed": False,
    }
    disposition = "BALANCED_TURNOVER_ROLE_STATE_CHILD_DESIGN_SIGNAL" if authorization else "BALANCED_TURNOVER_ROLE_STATE_MIXED_NO_CHILD_SIGNAL"
    result = {
        **structural,
        "disposition": disposition,
        "child_design_authorized_states": authorization,
        "prospective_shadow_authorized": False,
        "production_promotion_authorized": False,
        "parent_dispositions_preserved": {
            "R26": "RETROSPECTIVE_MIXED_OR_FAIL_NO_SHADOW",
            "R26F": fdisp["disposition"],
            "R26G": gdisp["disposition"],
        },
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(a.out_dir / "r26h_state_summary.csv", index=False)
    pd.DataFrame(season_rows).to_csv(a.out_dir / "r26h_state_season_metrics.csv", index=False)
    room.to_csv(a.out_dir / "r26h_balanced_room_state.csv", index=False)
    (a.out_dir / "r26h_disposition.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(pd.DataFrame(summary_rows).to_csv(index=False))
    print(pd.DataFrame(season_rows).to_csv(index=False))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
