"""Migration 92: decompose RB error into team volume, share, and efficiency.

This script consumes the frozen M91 temporal predictions. It does not rebuild
historical inputs or re-run MC/ML/State, so M92 is a fast diagnostic pass.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = {"mc": "mc_proj", "ml": "ml_proj", "state": "state_proj"}
KEYS = ["season", "week", "team", "player_clean_key"]
TEAM_KEYS = ["season", "week", "team"]


def _find_baseline(root: Path) -> Path:
    matches = list(root.rglob("rb_temporal_combined_predictions.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one frozen M91 combined prediction file under {root}; found={matches}")
    return matches[0]


def _read(root: Path) -> pd.DataFrame:
    path = _find_baseline(root)
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {"season", "week", "team", "player_clean_key", "player", "position", "market", "actual", *MODELS.values()}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"M91 predictions missing columns: {sorted(missing)}")
    x["season"] = pd.to_numeric(x["season"], errors="coerce").astype("Int64")
    x["week"] = pd.to_numeric(x["week"], errors="coerce").astype("Int64")
    x["actual"] = pd.to_numeric(x["actual"], errors="coerce")
    x["position"] = x["position"].astype(str).str.upper().str.strip()
    x["market"] = x["market"].astype(str).str.lower().str.strip()
    for c in MODELS.values():
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x


def _leader_flags(rb_att: pd.DataFrame, model_col: str) -> pd.DataFrame:
    x = rb_att[KEYS + ["actual", model_col]].copy()
    x["actual_rb1"] = x["actual"].eq(x.groupby(TEAM_KEYS)["actual"].transform("max"))
    x["projected_rb1"] = x[model_col].eq(x.groupby(TEAM_KEYS)[model_col].transform("max")) & x[model_col].notna()
    leader = []
    for key, g in x.groupby(TEAM_KEYS, dropna=False):
        actual_keys = set(g.loc[g["actual_rb1"], "player_clean_key"].astype(str))
        projected_keys = set(g.loc[g["projected_rb1"], "player_clean_key"].astype(str))
        leader.append(dict(zip(TEAM_KEYS, key)) | {"leader_correct": int(bool(actual_keys & projected_keys))})
    return x.merge(pd.DataFrame(leader), on=TEAM_KEYS, how="left", validate="many_to_one")


def build_trace(pred: pd.DataFrame) -> pd.DataFrame:
    all_att = pred.loc[pred["market"].eq("rush_att")].copy()
    rb_att = all_att.loc[all_att["position"].eq("RB")].copy()
    rb_yards = pred.loc[pred["market"].eq("rush_yards") & pred["position"].eq("RB")].copy()
    rb_combo = pred.loc[pred["market"].eq("rush_rec_yards") & pred["position"].eq("RB")].copy()

    actual_team = all_att.groupby(TEAM_KEYS, dropna=False)["actual"].sum().rename("actual_team_rush_att").reset_index()
    rows: list[pd.DataFrame] = []
    for model, col in MODELS.items():
        team_proj = all_att.groupby(TEAM_KEYS, dropna=False)[col].sum(min_count=1).rename("projected_team_rush_att").reset_index()
        flags = _leader_flags(rb_att, col)
        part = flags.rename(columns={"actual": "actual_rush_att", col: "projected_rush_att"})
        part = part.merge(actual_team, on=TEAM_KEYS, how="left", validate="many_to_one")
        part = part.merge(team_proj, on=TEAM_KEYS, how="left", validate="many_to_one")
        part["model"] = model
        part["actual_team_share"] = np.where(part["actual_team_rush_att"].gt(0), part["actual_rush_att"] / part["actual_team_rush_att"], np.nan)
        part["projected_team_share"] = np.where(part["projected_team_rush_att"].gt(0), part["projected_rush_att"] / part["projected_team_rush_att"], np.nan)
        part["carry_error"] = part["projected_rush_att"] - part["actual_rush_att"]
        # Exact two-part decomposition: projected_player - actual_player =
        # actual_share*(projected_team-actual_team) + projected_team*(projected_share-actual_share)
        part["team_volume_effect"] = part["actual_team_share"] * (part["projected_team_rush_att"] - part["actual_team_rush_att"])
        part["allocation_share_effect"] = part["projected_team_rush_att"] * (part["projected_team_share"] - part["actual_team_share"])
        part["decomp_residual"] = part["carry_error"] - part["team_volume_effect"] - part["allocation_share_effect"]
        part["oracle_team_volume_carries"] = part["actual_team_rush_att"] * part["projected_team_share"]
        part["oracle_allocation_share_carries"] = part["projected_team_rush_att"] * part["actual_team_share"]
        part["actual_bellcow_60"] = part["actual_rush_att"].ge(15) & part["actual_team_share"].ge(0.60)

        y = rb_yards[KEYS + ["actual", col]].rename(columns={"actual": "actual_rush_yards", col: "projected_rush_yards"})
        c = rb_combo[KEYS + ["actual", col]].rename(columns={"actual": "actual_rush_rec_yards", col: "projected_rush_rec_yards"})
        part = part.merge(y, on=KEYS, how="left", validate="one_to_one")
        part = part.merge(c, on=KEYS, how="left", validate="one_to_one")
        part["actual_ypc"] = np.where(part["actual_rush_att"].gt(0), part["actual_rush_yards"] / part["actual_rush_att"], np.nan)
        part["implied_projected_ypc"] = np.where(part["projected_rush_att"].gt(0.5), part["projected_rush_yards"] / part["projected_rush_att"], np.nan)
        part["implied_projected_ypc"] = part["implied_projected_ypc"].clip(lower=0.0, upper=12.0)
        part["oracle_carries_rush_yards"] = part["actual_rush_att"] * part["implied_projected_ypc"]
        part["oracle_efficiency_rush_yards"] = part["projected_rush_att"] * part["actual_ypc"]
        # Hold the direct receiving component fixed and replace only the rushing
        # piece with the carry-oracle rushing estimate.
        part["carry_oracle_rush_rec_yards"] = part["projected_rush_rec_yards"] + (part["oracle_carries_rush_yards"] - part["projected_rush_yards"])
        rows.append(part)
    return pd.concat(rows, ignore_index=True, sort=False)


def _mae(actual: pd.Series, projected: pd.Series) -> float:
    z = pd.DataFrame({"a": pd.to_numeric(actual, errors="coerce"), "p": pd.to_numeric(projected, errors="coerce")}).dropna()
    return float((z["p"] - z["a"]).abs().mean()) if len(z) else np.nan


def _bias(actual: pd.Series, projected: pd.Series) -> float:
    z = pd.DataFrame({"a": pd.to_numeric(actual, errors="coerce"), "p": pd.to_numeric(projected, errors="coerce")}).dropna()
    return float((z["p"] - z["a"]).mean()) if len(z) else np.nan


def summarize(trace: pd.DataFrame) -> pd.DataFrame:
    slices = {
        "all_rb": lambda g: pd.Series(True, index=g.index),
        "actual_15_plus": lambda g: g["actual_rush_att"].ge(15),
        "actual_20_plus": lambda g: g["actual_rush_att"].ge(20),
        "actual_25_plus": lambda g: g["actual_rush_att"].ge(25),
        "bellcow_60": lambda g: g["actual_bellcow_60"].fillna(False),
        "actual_rb1": lambda g: g["actual_rb1"].fillna(False),
        "projected_rb1": lambda g: g["projected_rb1"].fillna(False),
        "leader_correct": lambda g: g["leader_correct"].eq(1),
        "leader_wrong": lambda g: g["leader_correct"].eq(0),
    }
    rows: list[dict] = []
    seasons: list[int | str] = sorted(int(s) for s in trace["season"].dropna().unique()) + ["combined"]
    for season_scope in seasons:
        sx = trace if season_scope == "combined" else trace.loc[trace["season"].eq(int(season_scope))]
        for model in MODELS:
            mx = sx.loc[sx["model"].eq(model)]
            for name, fn in slices.items():
                g = mx.loc[fn(mx)].copy()
                rows.append({
                    "season_scope": season_scope,
                    "model": model,
                    "slice": name,
                    "n": int(len(g)),
                    "carry_mae": _mae(g["actual_rush_att"], g["projected_rush_att"]),
                    "carry_bias": _bias(g["actual_rush_att"], g["projected_rush_att"]),
                    "oracle_team_volume_carry_mae": _mae(g["actual_rush_att"], g["oracle_team_volume_carries"]),
                    "oracle_share_carry_mae": _mae(g["actual_rush_att"], g["oracle_allocation_share_carries"]),
                    "mean_team_volume_effect": float(g["team_volume_effect"].mean()) if len(g) else np.nan,
                    "mean_share_effect": float(g["allocation_share_effect"].mean()) if len(g) else np.nan,
                    "mean_abs_team_volume_effect": float(g["team_volume_effect"].abs().mean()) if len(g) else np.nan,
                    "mean_abs_share_effect": float(g["allocation_share_effect"].abs().mean()) if len(g) else np.nan,
                    "rush_yards_mae": _mae(g["actual_rush_yards"], g["projected_rush_yards"]),
                    "oracle_carries_rush_yards_mae": _mae(g["actual_rush_yards"], g["oracle_carries_rush_yards"]),
                    "oracle_efficiency_rush_yards_mae": _mae(g["actual_rush_yards"], g["oracle_efficiency_rush_yards"]),
                    "rush_rec_yards_mae": _mae(g["actual_rush_rec_yards"], g["projected_rush_rec_yards"]),
                    "carry_oracle_rush_rec_yards_mae": _mae(g["actual_rush_rec_yards"], g["carry_oracle_rush_rec_yards"]),
                })
    out = pd.DataFrame(rows)
    out["team_volume_oracle_carry_gain"] = out["carry_mae"] - out["oracle_team_volume_carry_mae"]
    out["share_oracle_carry_gain"] = out["carry_mae"] - out["oracle_share_carry_mae"]
    out["carry_oracle_rush_yards_gain"] = out["rush_yards_mae"] - out["oracle_carries_rush_yards_mae"]
    out["efficiency_oracle_rush_yards_gain"] = out["rush_yards_mae"] - out["oracle_efficiency_rush_yards_mae"]
    out["carry_oracle_rush_rec_gain"] = out["rush_rec_yards_mae"] - out["carry_oracle_rush_rec_yards_mae"]
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--m91-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_m92"))
    args = p.parse_args()
    pred = _read(args.m91_root)
    trace = build_trace(pred)
    summary = summarize(trace)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trace.to_csv(args.out_dir / "rb_opportunity_decomposition_trace.csv", index=False)
    summary.to_csv(args.out_dir / "rb_opportunity_decomposition_summary.csv", index=False)
    focus = summary.loc[(summary["season_scope"].astype(str).eq("combined")) & summary["slice"].isin(["all_rb", "actual_20_plus", "bellcow_60", "leader_correct", "leader_wrong"])]
    print("[rb_m92] combined opportunity decomposition")
    print(focus[[
        "model", "slice", "n", "carry_mae", "carry_bias",
        "team_volume_oracle_carry_gain", "share_oracle_carry_gain",
        "mean_team_volume_effect", "mean_share_effect",
        "rush_yards_mae", "carry_oracle_rush_yards_gain",
        "efficiency_oracle_rush_yards_gain", "rush_rec_yards_mae",
        "carry_oracle_rush_rec_gain",
    ]].to_string(index=False))
    max_resid = float(pd.to_numeric(trace["decomp_residual"], errors="coerce").abs().max())
    print(f"[rb_m92] exact carry decomposition max_abs_residual={max_resid:.12f}")
    if max_resid > 1e-7:
        raise RuntimeError(f"carry decomposition failed identity check: {max_resid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
