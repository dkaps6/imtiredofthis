"""Migration 93: test whether current RB workload shares are too compressed.

M93 is deliberately narrow. It changes only the distribution of the existing
projected RB rushing-attempt pool. Team/RB-pool volume, rushing efficiency,
receiving projection, defensive context, and every production coefficient stay
frozen.

The concentration exponent is selected on 2024 only, then frozen and evaluated
on 2025. This is exploratory research and does not alter production.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["season", "week", "team", "player_clean_key"]
TEAM_KEYS = ["season", "week", "team"]
GAMMAS = [1.00, 1.10, 1.20, 1.30, 1.40, 1.50]


def _read_predictions(root: Path, season: int) -> pd.DataFrame:
    path = root / str(season) / "component_predictions.csv"
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing M91 prediction file: {path}")
    x = pd.read_csv(path)
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {
        "season", "week", "team", "player", "player_clean_key", "position",
        "market", "actual", "ml_proj", "mc_proj", "state_proj",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"M91 predictions missing columns: {sorted(missing)}")
    x["position"] = x["position"].astype(str).str.upper().str.strip()
    x["market"] = x["market"].astype(str).str.lower().str.strip()
    for c in ["actual", "ml_proj", "mc_proj", "state_proj"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x


def _rb_frame(pred: pd.DataFrame, model: str = "ml") -> pd.DataFrame:
    col = f"{model}_proj"
    att = pred.loc[
        pred["position"].eq("RB") & pred["market"].eq("rush_att"),
        KEYS + ["player", "actual", col],
    ].rename(columns={"actual": "actual_rush_att", col: "base_rush_att"})
    ry = pred.loc[
        pred["position"].eq("RB") & pred["market"].eq("rush_yards"),
        KEYS + ["actual", col],
    ].rename(columns={"actual": "actual_rush_yards", col: "base_rush_yards"})
    rr = pred.loc[
        pred["position"].eq("RB") & pred["market"].eq("rush_rec_yards"),
        KEYS + ["actual", col],
    ].rename(columns={"actual": "actual_rush_rec_yards", col: "base_rush_rec_yards"})
    x = att.merge(ry, on=KEYS, how="inner", validate="one_to_one")
    x = x.merge(rr, on=KEYS, how="inner", validate="one_to_one")

    # Actual total team rushes include QB/WR/TE rushes and are used only to build
    # evaluation slices; they never enter the candidate pregame projection.
    total_actual = pred.loc[pred["market"].eq("rush_att")].groupby(TEAM_KEYS)["actual"].sum(min_count=1)
    x = x.merge(total_actual.rename("actual_team_rush_att").reset_index(), on=TEAM_KEYS, how="left")
    x["actual_team_rush_share"] = np.where(
        x["actual_team_rush_att"].gt(0), x["actual_rush_att"] / x["actual_team_rush_att"], np.nan
    )
    x["bellcow_60"] = x["actual_rush_att"].ge(15) & x["actual_team_rush_share"].ge(0.60)
    return x


def _apply_gamma(x: pd.DataFrame, gamma: float) -> pd.DataFrame:
    out = x.copy()
    pool = out.groupby(TEAM_KEYS)["base_rush_att"].transform(lambda s: s.sum(min_count=1))
    out["base_rb_pool_rush_att"] = pool
    base_share = np.where(pool.gt(0), out["base_rush_att"] / pool, 0.0)
    out["base_rb_pool_share"] = base_share

    raw = np.power(np.clip(base_share, 1e-12, None), gamma)
    raw = pd.Series(raw, index=out.index)
    denom = raw.groupby([out[c] for c in TEAM_KEYS]).transform("sum")
    out["candidate_rb_pool_share"] = np.where(denom.gt(0), raw / denom, base_share)
    out["candidate_rush_att"] = out["candidate_rb_pool_share"] * pool

    # Hold M91 rushing efficiency exactly fixed. This isolates the yardage change
    # attributable to workload redistribution rather than smuggling in YPC tuning.
    implied_ypc = np.where(
        out["base_rush_att"].abs().gt(1e-9),
        out["base_rush_yards"] / out["base_rush_att"],
        np.nan,
    )
    out["base_implied_ypc"] = implied_ypc
    out["candidate_rush_yards"] = np.where(
        np.isfinite(implied_ypc), out["candidate_rush_att"] * implied_ypc, out["base_rush_yards"]
    )
    # Hold the receiving component fixed exactly.
    out["candidate_rush_rec_yards"] = (
        out["base_rush_rec_yards"] + out["candidate_rush_yards"] - out["base_rush_yards"]
    )
    out["gamma"] = float(gamma)
    return out


def _mae(pred: pd.Series, actual: pd.Series) -> float:
    z = pd.DataFrame({"p": pd.to_numeric(pred, errors="coerce"), "a": pd.to_numeric(actual, errors="coerce")}).dropna()
    return float((z["p"] - z["a"]).abs().mean()) if len(z) else np.nan


def _metrics(x: pd.DataFrame, pred_col: str, actual_col: str) -> dict[str, float | int]:
    z = x[[pred_col, actual_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "correlation": np.nan}
    err = z[pred_col] - z[actual_col]
    corr = np.nan
    if len(z) >= 2 and z[pred_col].nunique() > 1 and z[actual_col].nunique() > 1:
        corr = float(z[pred_col].corr(z[actual_col]))
    return {
        "n": int(len(z)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "bias": float(err.mean()),
        "correlation": corr,
    }


def _slice(x: pd.DataFrame, name: str) -> pd.Series:
    a = pd.to_numeric(x["actual_rush_att"], errors="coerce")
    if name == "all_rb":
        return pd.Series(True, index=x.index)
    if name == "actual_0_5":
        return a.le(5)
    if name == "actual_6_10":
        return a.between(6, 10)
    if name == "actual_11_14":
        return a.between(11, 14)
    if name == "actual_15_plus":
        return a.ge(15)
    if name == "actual_20_plus":
        return a.ge(20)
    if name == "actual_25_plus":
        return a.ge(25)
    if name == "bellcow_60":
        return x["bellcow_60"].fillna(False)
    raise KeyError(name)


def _summary(base_by_season: dict[int, pd.DataFrame], gamma: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    traces: list[pd.DataFrame] = []
    rows: list[dict] = []
    slices = [
        "all_rb", "actual_0_5", "actual_6_10", "actual_11_14",
        "actual_15_plus", "actual_20_plus", "actual_25_plus", "bellcow_60",
    ]
    for season, base in sorted(base_by_season.items()):
        cand = _apply_gamma(base, gamma)
        traces.append(cand)
        for slice_name in slices:
            g = cand.loc[_slice(cand, slice_name)]
            for market, actual_col, base_col, cand_col in [
                ("rush_att", "actual_rush_att", "base_rush_att", "candidate_rush_att"),
                ("rush_yards", "actual_rush_yards", "base_rush_yards", "candidate_rush_yards"),
                ("rush_rec_yards", "actual_rush_rec_yards", "base_rush_rec_yards", "candidate_rush_rec_yards"),
            ]:
                b = _metrics(g, base_col, actual_col)
                c = _metrics(g, cand_col, actual_col)
                rows.append({
                    "season": season,
                    "slice": slice_name,
                    "market": market,
                    "gamma": gamma,
                    "n": b["n"],
                    "baseline_mae": b["mae"],
                    "candidate_mae": c["mae"],
                    "mae_gain": b["mae"] - c["mae"] if np.isfinite(b["mae"]) and np.isfinite(c["mae"]) else np.nan,
                    "baseline_rmse": b["rmse"],
                    "candidate_rmse": c["rmse"],
                    "baseline_bias": b["bias"],
                    "candidate_bias": c["bias"],
                    "baseline_correlation": b["correlation"],
                    "candidate_correlation": c["correlation"],
                })
    trace = pd.concat(traces, ignore_index=True, sort=False)
    per = pd.DataFrame(rows)

    # Add combined rows without letting 2025 influence gamma selection.
    combined_rows: list[dict] = []
    for slice_name in slices:
        g = trace.loc[_slice(trace, slice_name)]
        for market, actual_col, base_col, cand_col in [
            ("rush_att", "actual_rush_att", "base_rush_att", "candidate_rush_att"),
            ("rush_yards", "actual_rush_yards", "base_rush_yards", "candidate_rush_yards"),
            ("rush_rec_yards", "actual_rush_rec_yards", "base_rush_rec_yards", "candidate_rush_rec_yards"),
        ]:
            b = _metrics(g, base_col, actual_col)
            c = _metrics(g, cand_col, actual_col)
            combined_rows.append({
                "season": "combined", "slice": slice_name, "market": market, "gamma": gamma,
                "n": b["n"], "baseline_mae": b["mae"], "candidate_mae": c["mae"],
                "mae_gain": b["mae"] - c["mae"], "baseline_rmse": b["rmse"],
                "candidate_rmse": c["rmse"], "baseline_bias": b["bias"], "candidate_bias": c["bias"],
                "baseline_correlation": b["correlation"], "candidate_correlation": c["correlation"],
            })
    return pd.concat([per, pd.DataFrame(combined_rows)], ignore_index=True), trace


def _gamma_grid(base_2024: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for gamma in GAMMAS:
        c = _apply_gamma(base_2024, gamma)
        for slice_name in ["all_rb", "actual_0_5", "actual_6_10", "actual_11_14", "actual_15_plus", "actual_20_plus", "actual_25_plus", "bellcow_60"]:
            g = c.loc[_slice(c, slice_name)]
            rows.append({
                "gamma": gamma,
                "slice": slice_name,
                "n": int(len(g)),
                "rush_att_mae": _mae(g["candidate_rush_att"], g["actual_rush_att"]),
                "rush_yards_mae": _mae(g["candidate_rush_yards"], g["actual_rush_yards"]),
                "rush_rec_yards_mae": _mae(g["candidate_rush_rec_yards"], g["actual_rush_rec_yards"]),
            })
    return pd.DataFrame(rows)


def _legacy_guard(pred_by_season: dict[int, pd.DataFrame], trace: pd.DataFrame) -> pd.DataFrame:
    """Keep the old all-player rushing-yard scoreboard as a regression guard."""
    rows: list[dict] = []
    cand = trace[KEYS + ["candidate_rush_yards"]]
    for season, pred in sorted(pred_by_season.items()):
        all_ry = pred.loc[pred["market"].eq("rush_yards"), KEYS + ["actual", "ml_proj", "position"]].copy()
        all_ry = all_ry.merge(cand.loc[cand["season"].eq(season)], on=KEYS, how="left", validate="many_to_one")
        all_ry["candidate_all_player"] = np.where(
            all_ry["position"].eq("RB") & all_ry["candidate_rush_yards"].notna(),
            all_ry["candidate_rush_yards"], all_ry["ml_proj"],
        )
        b = _metrics(all_ry, "ml_proj", "actual")
        c = _metrics(all_ry, "candidate_all_player", "actual")
        rows.append({
            "season": season, "n": b["n"], "baseline_all_player_rush_yards_mae": b["mae"],
            "candidate_all_player_rush_yards_mae": c["mae"], "mae_gain": b["mae"] - c["mae"],
            "baseline_correlation": b["correlation"], "candidate_correlation": c["correlation"],
        })
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--m91-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_m93"))
    args = p.parse_args()

    preds = {s: _read_predictions(args.m91_root, s) for s in (2024, 2025)}
    bases = {s: _rb_frame(preds[s], "ml") for s in (2024, 2025)}

    grid = _gamma_grid(bases[2024])
    dev = grid.loc[grid["slice"].eq("all_rb")].sort_values(["rush_att_mae", "rush_yards_mae", "gamma"])
    selected_gamma = float(dev.iloc[0]["gamma"])

    summary, trace = _summary(bases, selected_gamma)
    guard = _legacy_guard(preds, trace)

    # This is a research disposition, not a production promotion gate.
    def gain(season: int, market: str, slice_name: str = "all_rb") -> float:
        q = summary.loc[
            summary["season"].astype(str).eq(str(season))
            & summary["market"].eq(market)
            & summary["slice"].eq(slice_name), "mae_gain"
        ]
        return float(q.iloc[0]) if len(q) else np.nan

    validation_ok = (
        gain(2025, "rush_att") > 0
        and gain(2025, "rush_yards") > 0
        and gain(2025, "rush_att", "actual_0_5") >= 0
        and gain(2025, "rush_att", "actual_20_plus") > 0
        and float(guard.loc[guard["season"].eq(2025), "mae_gain"].iloc[0]) >= 0
    )
    disposition = pd.DataFrame([{
        "selected_gamma_from_2024": selected_gamma,
        "development_season": 2024,
        "validation_season": 2025,
        "validation_pass": int(validation_ok),
        "disposition": "ADVANCE_CONCENTRATION_SIGNAL" if validation_ok else "DO_NOT_ADVANCE_FIXED_CONCENTRATION",
        "note": "No production change; M93 tests allocation compression only.",
    }])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.out_dir / "m93_gamma_grid_2024.csv", index=False)
    summary.to_csv(args.out_dir / "m93_summary.csv", index=False)
    trace.to_csv(args.out_dir / "m93_row_trace.csv", index=False)
    guard.to_csv(args.out_dir / "m93_legacy_rushing_guard.csv", index=False)
    disposition.to_csv(args.out_dir / "m93_disposition.csv", index=False)

    print("[rb_m93] 2024-only gamma development grid")
    print(dev.to_string(index=False))
    print("\n[rb_m93] selected gamma", selected_gamma)
    print("\n[rb_m93] headline RB results")
    print(summary.loc[
        summary["slice"].isin(["all_rb", "actual_0_5", "actual_15_plus", "actual_20_plus", "actual_25_plus", "bellcow_60"])
        & summary["market"].isin(["rush_att", "rush_yards", "rush_rec_yards"]),
        ["season", "slice", "market", "n", "baseline_mae", "candidate_mae", "mae_gain", "baseline_bias", "candidate_bias", "baseline_correlation", "candidate_correlation"],
    ].to_string(index=False))
    print("\n[rb_m93] legacy all-player rushing-yard guard")
    print(guard.to_string(index=False))
    print("\n[rb_m93] disposition")
    print(disposition.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
