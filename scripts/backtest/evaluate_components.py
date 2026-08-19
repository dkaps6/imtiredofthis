"""Evaluate leakage-safe walk-forward component predictions.

This module is diagnostic only: it never changes projections, model weights, or
production behavior. It converts the row-level OOS prediction table into stable
error/coverage reports so model changes can be measured before they are made.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_COLUMNS = {
    "mc": "mc_proj",
    "ml": "ml_proj",
    "state": "state_proj",
}


def _phase(week: pd.Series) -> pd.Series:
    w = pd.to_numeric(week, errors="coerce")
    return pd.Series(
        np.select(
            [w.le(6), w.le(12), w.ge(13)],
            ["early_1_6", "mid_7_12", "late_13_18"],
            default="unknown",
        ),
        index=week.index,
        dtype="string",
    )


def prepare_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    x = predictions.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {"season", "week", "player", "team", "opponent", "market", "actual", *MODEL_COLUMNS.values()}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"component predictions missing columns: {sorted(missing)}")

    x["actual"] = pd.to_numeric(x["actual"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce").astype("Int64")
    if "position" not in x.columns:
        x["position"] = "UNKNOWN"
    x["position"] = x["position"].astype("string").fillna("UNKNOWN").str.upper().str.strip().replace("", "UNKNOWN")
    x["season_phase"] = _phase(x["week"])

    rows: list[pd.DataFrame] = []
    id_cols = [c for c in [
        "season", "week", "season_phase", "player", "player_clean_key", "team", "opponent",
        "position", "role", "market", "actual", "prediction_cutoff",
    ] if c in x.columns]
    for model, col in MODEL_COLUMNS.items():
        part = x[id_cols].copy()
        part["model"] = model
        part["projection"] = pd.to_numeric(x[col], errors="coerce")
        part["available"] = part["projection"].notna()
        part["error"] = part["projection"] - part["actual"]
        part["abs_error"] = part["error"].abs()
        part["sq_error"] = part["error"].pow(2)
        rows.append(part)
    return pd.concat(rows, ignore_index=True, sort=False)


def _metric_row(g: pd.DataFrame) -> dict[str, float | int]:
    total = int(len(g))
    z = g.loc[g["available"] & g["actual"].notna()].copy()
    n = int(len(z))
    row: dict[str, float | int] = {
        "rows": total,
        "n": n,
        "coverage": float(n / total) if total else np.nan,
        "mae": np.nan,
        "median_ae": np.nan,
        "rmse": np.nan,
        "bias": np.nan,
        "correlation": np.nan,
        "r2": np.nan,
        "actual_mean": np.nan,
        "projection_mean": np.nan,
    }
    if not n:
        return row
    row["mae"] = float(z["abs_error"].mean())
    row["median_ae"] = float(z["abs_error"].median())
    row["rmse"] = float(np.sqrt(z["sq_error"].mean()))
    row["bias"] = float(z["error"].mean())
    row["actual_mean"] = float(z["actual"].mean())
    row["projection_mean"] = float(z["projection"].mean())
    if n >= 2 and z["actual"].nunique() > 1 and z["projection"].nunique() > 1:
        row["correlation"] = float(z["actual"].corr(z["projection"]))
    ss_tot = float(((z["actual"] - z["actual"].mean()) ** 2).sum())
    if ss_tot > 0:
        row["r2"] = float(1.0 - z["sq_error"].sum() / ss_tot)
    return row


def summarize(errors: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    for keys, g in errors.groupby(grouper, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        rec.update(_metric_row(g))
        rows.append(rec)
    return pd.DataFrame(rows)


def add_market_winner(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["mae_rank"] = out.groupby("market")["mae"].rank(method="min", na_option="bottom")
    out["market_winner"] = out["mae_rank"].eq(1) & out["mae"].notna()
    return out


def evaluate(predictions: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    errors = prepare_errors(predictions)

    summary = add_market_winner(summarize(errors, ["market", "model"]))
    by_week = summarize(errors, ["week", "market", "model"])
    by_position = summarize(errors, ["position", "market", "model"])
    by_phase = summarize(errors, ["season_phase", "market", "model"])
    by_player = summarize(errors, ["player", "position", "market", "model"])
    by_player = by_player.loc[by_player["n"].ge(3)].copy()

    observed = errors.loc[errors["available"] & errors["actual"].notna()].copy()
    worst = observed.sort_values("abs_error", ascending=False).head(500).copy()

    paths = {
        "errors": out_dir / "evaluation_row_errors.csv",
        "summary": out_dir / "evaluation_summary.csv",
        "by_week": out_dir / "evaluation_by_week.csv",
        "by_position": out_dir / "evaluation_by_position.csv",
        "by_phase": out_dir / "evaluation_by_phase.csv",
        "by_player": out_dir / "evaluation_by_player.csv",
        "worst": out_dir / "evaluation_worst_misses.csv",
    }
    errors.to_csv(paths["errors"], index=False)
    summary.to_csv(paths["summary"], index=False)
    by_week.to_csv(paths["by_week"], index=False)
    by_position.to_csv(paths["by_position"], index=False)
    by_phase.to_csv(paths["by_phase"], index=False)
    by_player.to_csv(paths["by_player"], index=False)
    worst.to_csv(paths["worst"], index=False)
    return paths


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, default=Path("data/backtests/component_predictions.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/evaluation"))
    args = p.parse_args()
    if not args.predictions.exists() or args.predictions.stat().st_size == 0:
        raise RuntimeError(f"missing component predictions: {args.predictions}")
    predictions = pd.read_csv(args.predictions)
    paths = evaluate(predictions, args.out_dir)
    summary = pd.read_csv(paths["summary"])
    print("[backtest_eval] market/model summary")
    cols = ["market", "model", "n", "coverage", "mae", "median_ae", "rmse", "bias", "correlation", "r2", "market_winner"]
    print(summary[cols].to_string(index=False))
    for name, path in paths.items():
        print(f"[backtest_eval] {name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
