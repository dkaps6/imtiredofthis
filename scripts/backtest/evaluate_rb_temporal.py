"""Evaluate RB rushing markets across multiple leakage-safe target seasons.

Migration 91 is diagnostic only. It does not alter production projections,
model weights, simulation rules, or sportsbook behavior. The purpose is to
score the current canonical rushing architecture on multiple prospective
seasons and make the failure modes around bell-cow workloads explicit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.evaluate_components import MODEL_COLUMNS, prepare_errors

RB_MARKETS = ("rush_att", "rush_yards", "rush_rec_yards")
BIG_MISS = {
    "rush_att": 5.0,
    "rush_yards": 25.0,
    "rush_rec_yards": 30.0,
}
KEYS = ["season", "week", "team", "player_clean_key"]


def _read_many(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"missing prediction file: {path}")
        x = pd.read_csv(path)
        x["_source_file"] = str(path)
        frames.append(x)
    out = pd.concat(frames, ignore_index=True, sort=False)
    out.columns = [str(c).strip().lower() for c in out.columns]
    required = {
        "season", "week", "player", "team", "position", "market", "actual",
        *MODEL_COLUMNS.values(),
    }
    missing = required - set(out.columns)
    if missing:
        raise RuntimeError(f"prediction files missing columns: {sorted(missing)}")
    if "player_clean_key" not in out.columns:
        out["player_clean_key"] = (
            out["player"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
        )
    out["season"] = pd.to_numeric(out["season"], errors="coerce").astype("Int64")
    out["week"] = pd.to_numeric(out["week"], errors="coerce").astype("Int64")
    out["actual"] = pd.to_numeric(out["actual"], errors="coerce")
    out["position"] = out["position"].astype(str).str.upper().str.strip()
    out["market"] = out["market"].astype(str).str.lower().str.strip()
    # Each target season is generated independently. Duplicate player-market rows
    # indicate a harness error and must not be silently averaged away.
    dup = out.duplicated(KEYS + ["market"], keep=False)
    if dup.any():
        sample = out.loc[dup, KEYS + ["player", "market", "_source_file"]].head(20)
        raise RuntimeError(f"duplicate temporal prediction rows: {sample.to_dict(orient='records')}")
    return out


def _workload_context(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build actual/projected rushing-share and RB1 diagnostics at team-week grain."""
    rush = pred.loc[pred["market"].eq("rush_att")].copy()
    rush["actual_rush_att"] = pd.to_numeric(rush["actual"], errors="coerce").fillna(0.0)

    actual_team = (
        rush.groupby(["season", "week", "team"], dropna=False)["actual_rush_att"]
        .sum().rename("actual_team_rush_att").reset_index()
    )
    rb = rush.loc[rush["position"].eq("RB")].copy()
    rb = rb.merge(actual_team, on=["season", "week", "team"], how="left", validate="many_to_one")
    rb["actual_team_rush_share"] = np.where(
        rb["actual_team_rush_att"].gt(0),
        rb["actual_rush_att"] / rb["actual_team_rush_att"],
        np.nan,
    )
    rb_total = (
        rb.groupby(["season", "week", "team"], dropna=False)["actual_rush_att"]
        .sum().rename("actual_rb_pool_rush_att").reset_index()
    )
    rb = rb.merge(rb_total, on=["season", "week", "team"], how="left", validate="many_to_one")
    rb["actual_rb_pool_share"] = np.where(
        rb["actual_rb_pool_rush_att"].gt(0),
        rb["actual_rush_att"] / rb["actual_rb_pool_rush_att"],
        np.nan,
    )
    rb["actual_rb1"] = rb["actual_rush_att"].eq(
        rb.groupby(["season", "week", "team"], dropna=False)["actual_rush_att"].transform("max")
    )
    rb["actual_bellcow_55"] = rb["actual_rush_att"].ge(15) & rb["actual_team_rush_share"].ge(0.55)
    rb["actual_bellcow_60"] = rb["actual_rush_att"].ge(15) & rb["actual_team_rush_share"].ge(0.60)

    long_rows: list[pd.DataFrame] = []
    leader_rows: list[dict] = []
    for model, col in MODEL_COLUMNS.items():
        all_rush = rush.copy()
        all_rush["projection"] = pd.to_numeric(all_rush[col], errors="coerce").clip(lower=0.0)
        team_proj = (
            all_rush.groupby(["season", "week", "team"], dropna=False)["projection"]
            .sum(min_count=1).rename("projected_team_rush_att").reset_index()
        )
        part = rb.copy()
        part["model"] = model
        part["projection"] = pd.to_numeric(part[col], errors="coerce").clip(lower=0.0)
        part = part.merge(team_proj, on=["season", "week", "team"], how="left", validate="many_to_one")
        rb_proj_total = (
            part.groupby(["season", "week", "team"], dropna=False)["projection"]
            .transform(lambda s: s.sum(min_count=1))
        )
        part["projected_rb_pool_rush_att"] = rb_proj_total
        part["projected_team_rush_share"] = np.where(
            part["projected_team_rush_att"].gt(0),
            part["projection"] / part["projected_team_rush_att"],
            np.nan,
        )
        part["projected_rb_pool_share"] = np.where(
            part["projected_rb_pool_rush_att"].gt(0),
            part["projection"] / part["projected_rb_pool_rush_att"],
            np.nan,
        )
        part["projected_rb1"] = part["projection"].eq(
            part.groupby(["season", "week", "team"], dropna=False)["projection"].transform("max")
        ) & part["projection"].notna()
        part["rush_att_error"] = part["projection"] - part["actual_rush_att"]
        long_rows.append(part)

        for (season, week, team), g in part.groupby(["season", "week", "team"], dropna=False):
            if g.empty or g["projection"].notna().sum() == 0:
                continue
            actual_max = g["actual_rush_att"].max()
            proj_max = g["projection"].max()
            actual_keys = set(g.loc[g["actual_rush_att"].eq(actual_max), "player_clean_key"].astype(str))
            proj_keys = set(g.loc[g["projection"].eq(proj_max), "player_clean_key"].astype(str))
            leader_rows.append({
                "season": int(season),
                "week": int(week),
                "team": team,
                "model": model,
                "leader_match": int(bool(actual_keys & proj_keys)),
                "actual_leader_carries": float(actual_max),
                "projected_leader_carries": float(proj_max),
            })

    workload = pd.concat(long_rows, ignore_index=True, sort=False) if long_rows else pd.DataFrame()
    leaders = pd.DataFrame(leader_rows)
    return workload, leaders


def _metric(g: pd.DataFrame, market: str) -> dict[str, float | int]:
    z = g.loc[g["available"] & g["actual"].notna() & g["projection"].notna()].copy()
    n = int(len(z))
    row: dict[str, float | int] = {
        "n": n,
        "mae": np.nan,
        "median_ae": np.nan,
        "p90_ae": np.nan,
        "rmse": np.nan,
        "bias": np.nan,
        "correlation": np.nan,
        "actual_mean": np.nan,
        "projection_mean": np.nan,
        "underprojection_rate": np.nan,
        "big_under_rate": np.nan,
        "big_over_rate": np.nan,
    }
    if not n:
        return row
    err = z["projection"] - z["actual"]
    ae = err.abs()
    row.update({
        "mae": float(ae.mean()),
        "median_ae": float(ae.median()),
        "p90_ae": float(ae.quantile(0.90)),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "bias": float(err.mean()),
        "actual_mean": float(z["actual"].mean()),
        "projection_mean": float(z["projection"].mean()),
        "underprojection_rate": float(err.lt(0).mean()),
    })
    threshold = BIG_MISS[market]
    row["big_under_rate"] = float(err.le(-threshold).mean())
    row["big_over_rate"] = float(err.ge(threshold).mean())
    if n >= 2 and z["actual"].nunique() > 1 and z["projection"].nunique() > 1:
        row["correlation"] = float(z["actual"].corr(z["projection"]))
    return row


def _slice_mask(x: pd.DataFrame, name: str) -> pd.Series:
    if name == "all_rb":
        return pd.Series(True, index=x.index)
    if name == "actual_10_plus_carries":
        return x["actual_rush_att"].ge(10)
    if name == "actual_15_plus_carries":
        return x["actual_rush_att"].ge(15)
    if name == "actual_20_plus_carries":
        return x["actual_rush_att"].ge(20)
    if name == "actual_25_plus_carries":
        return x["actual_rush_att"].ge(25)
    if name == "actual_bellcow_55":
        return x["actual_bellcow_55"].fillna(False)
    if name == "actual_bellcow_60":
        return x["actual_bellcow_60"].fillna(False)
    if name == "actual_rb1":
        return x["actual_rb1"].fillna(False)
    if name == "projected_rb1":
        return x["projected_rb1"].fillna(False)
    raise KeyError(name)


def _summary(errors: pd.DataFrame, workload: pd.DataFrame) -> pd.DataFrame:
    rb = errors.loc[errors["position"].eq("RB") & errors["market"].isin(RB_MARKETS)].copy()
    attach_cols = KEYS + [
        "model", "actual_rush_att", "actual_team_rush_att", "actual_team_rush_share",
        "actual_rb_pool_share", "actual_rb1", "actual_bellcow_55", "actual_bellcow_60",
        "projected_rb1", "projected_team_rush_share", "projected_rb_pool_share",
    ]
    rb = rb.merge(
        workload[attach_cols].drop_duplicates(KEYS + ["model"]),
        on=KEYS + ["model"], how="left", validate="many_to_one",
    )
    slices = [
        "all_rb", "actual_10_plus_carries", "actual_15_plus_carries",
        "actual_20_plus_carries", "actual_25_plus_carries",
        "actual_bellcow_55", "actual_bellcow_60", "actual_rb1", "projected_rb1",
    ]
    rows: list[dict] = []
    season_values: list[int | str] = sorted(int(s) for s in rb["season"].dropna().unique()) + ["combined"]
    for season_scope in season_values:
        sx = rb if season_scope == "combined" else rb.loc[rb["season"].eq(int(season_scope))]
        for market in RB_MARKETS:
            mx = sx.loc[sx["market"].eq(market)]
            for model in MODEL_COLUMNS:
                mm = mx.loc[mx["model"].eq(model)]
                for slice_name in slices:
                    sm = mm.loc[_slice_mask(mm, slice_name)]
                    rec = {
                        "season_scope": season_scope,
                        "market": market,
                        "model": model,
                        "slice": slice_name,
                    }
                    rec.update(_metric(sm, market))
                    rows.append(rec)
    return pd.DataFrame(rows)


def _share_summary(workload: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    season_values: list[int | str] = sorted(int(s) for s in workload["season"].dropna().unique()) + ["combined"]
    for season_scope in season_values:
        sx = workload if season_scope == "combined" else workload.loc[workload["season"].eq(int(season_scope))]
        for model in MODEL_COLUMNS:
            g = sx.loc[sx["model"].eq(model)].copy()
            for kind, actual_col, proj_col in [
                ("team_rush_share", "actual_team_rush_share", "projected_team_rush_share"),
                ("rb_pool_share", "actual_rb_pool_share", "projected_rb_pool_share"),
            ]:
                z = g[[actual_col, proj_col]].apply(pd.to_numeric, errors="coerce").dropna()
                corr = np.nan
                if len(z) >= 2 and z[actual_col].nunique() > 1 and z[proj_col].nunique() > 1:
                    corr = float(z[actual_col].corr(z[proj_col]))
                rows.append({
                    "season_scope": season_scope,
                    "model": model,
                    "share_kind": kind,
                    "n": int(len(z)),
                    "share_mae": float((z[proj_col] - z[actual_col]).abs().mean()) if len(z) else np.nan,
                    "share_bias": float((z[proj_col] - z[actual_col]).mean()) if len(z) else np.nan,
                    "correlation": corr,
                })
    return pd.DataFrame(rows)


def _leader_summary(leaders: pd.DataFrame) -> pd.DataFrame:
    if leaders.empty:
        return leaders
    rows: list[dict] = []
    season_values: list[int | str] = sorted(int(s) for s in leaders["season"].dropna().unique()) + ["combined"]
    for season_scope in season_values:
        sx = leaders if season_scope == "combined" else leaders.loc[leaders["season"].eq(int(season_scope))]
        for model in MODEL_COLUMNS:
            g = sx.loc[sx["model"].eq(model)]
            rows.append({
                "season_scope": season_scope,
                "model": model,
                "team_weeks": int(len(g)),
                "rb1_identification_rate": float(g["leader_match"].mean()) if len(g) else np.nan,
                "actual_leader_carries_mean": float(g["actual_leader_carries"].mean()) if len(g) else np.nan,
                "projected_leader_carries_mean": float(g["projected_leader_carries"].mean()) if len(g) else np.nan,
            })
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_m91/evaluation"))
    args = p.parse_args()

    pred = _read_many(args.predictions)
    errors = prepare_errors(pred)
    workload, leaders = _workload_context(pred)
    summary = _summary(errors, workload)
    share_summary = _share_summary(workload)
    leader_summary = _leader_summary(leaders)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.out_dir / "rb_temporal_combined_predictions.csv", index=False)
    errors.loc[errors["position"].eq("RB") & errors["market"].isin(RB_MARKETS)].to_csv(
        args.out_dir / "rb_temporal_row_errors.csv", index=False
    )
    workload.to_csv(args.out_dir / "rb_workload_trace.csv", index=False)
    leaders.to_csv(args.out_dir / "rb1_leader_trace.csv", index=False)
    summary.to_csv(args.out_dir / "rb_temporal_summary.csv", index=False)
    share_summary.to_csv(args.out_dir / "rb_workload_share_summary.csv", index=False)
    leader_summary.to_csv(args.out_dir / "rb1_leader_summary.csv", index=False)

    print("[rb_m91] all-RB temporal baseline")
    view = summary.loc[summary["slice"].eq("all_rb")].copy()
    print(view[[
        "season_scope", "market", "model", "n", "mae", "rmse", "bias",
        "correlation", "underprojection_rate", "big_under_rate",
    ]].to_string(index=False))
    print("\n[rb_m91] workload concentration slices")
    focus = summary.loc[
        summary["market"].eq("rush_att")
        & summary["slice"].isin(["actual_15_plus_carries", "actual_20_plus_carries", "actual_bellcow_60"])
    ]
    print(focus[[
        "season_scope", "model", "slice", "n", "mae", "bias",
        "correlation", "underprojection_rate", "big_under_rate",
    ]].to_string(index=False))
    print("\n[rb_m91] projected RB1 identification")
    print(leader_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
