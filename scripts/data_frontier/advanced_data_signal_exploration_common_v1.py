"""Common helpers for Advanced Data Signal Exploration V1.

Reconnaissance only: no production projections, no sportsbook inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def confidence_tier(n: pd.Series, low_min: int, medium_min: int, high_min: int) -> pd.Series:
    x = pd.to_numeric(n, errors="coerce")
    return pd.Series(
        np.select(
            [x >= high_min, x >= medium_min, x >= low_min],
            ["HIGH", "MEDIUM", "LOW"],
            default="ABSTAIN",
        ),
        index=n.index,
        dtype="object",
    )


def pair_metrics(frame: pd.DataFrame, pred: str, actual: str) -> dict:
    x = frame[[pred, actual]].copy()
    x[pred] = pd.to_numeric(x[pred], errors="coerce")
    x[actual] = pd.to_numeric(x[actual], errors="coerce")
    x = x.dropna()
    if x.empty:
        return {
            "n": 0,
            "mae": None,
            "median_abs_error": None,
            "bias": None,
            "pearson": None,
            "spearman": None,
        }
    err = x[pred] - x[actual]
    pearson = x[pred].corr(x[actual], method="pearson") if len(x) >= 3 else np.nan
    spearman = x[pred].corr(x[actual], method="spearman") if len(x) >= 3 else np.nan
    return {
        "n": int(len(x)),
        "mae": float(err.abs().mean()),
        "median_abs_error": float(err.abs().median()),
        "bias": float(err.mean()),
        "pearson": float(pearson) if pd.notna(pearson) else None,
        "spearman": float(spearman) if pd.notna(spearman) else None,
    }


def split_persistence(
    raw: pd.DataFrame,
    *,
    keys: list[str],
    metrics: list[str],
    early_mask: pd.Series,
    late_mask: pd.Series,
    min_obs_each: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    paired_frames = []
    for metric in metrics:
        q = raw[keys + [metric]].copy()
        q[metric] = pd.to_numeric(q[metric], errors="coerce")
        early = (
            q.loc[early_mask]
            .dropna(subset=[metric])
            .groupby(keys, dropna=False)[metric]
            .agg(["median", "count"])
            .reset_index()
            .rename(columns={"median": "early_value", "count": "early_n"})
        )
        late = (
            q.loc[late_mask]
            .dropna(subset=[metric])
            .groupby(keys, dropna=False)[metric]
            .agg(["median", "count"])
            .reset_index()
            .rename(columns={"median": "late_value", "count": "late_n"})
        )
        z = early.merge(late, on=keys, how="inner")
        z = z.loc[(z["early_n"] >= min_obs_each) & (z["late_n"] >= min_obs_each)].copy()
        z.insert(len(keys), "metric", metric)
        paired_frames.append(z)
        m = pair_metrics(z, "early_value", "late_value")
        rows.append(
            {
                "metric": metric,
                "min_obs_each": int(min_obs_each),
                "paired_profiles": int(len(z)),
                **m,
            }
        )
    pairs = pd.concat(paired_frames, ignore_index=True) if paired_frames else pd.DataFrame()
    return pd.DataFrame(rows), pairs


def future_geometry_validation(
    *,
    raw: pd.DataFrame,
    history: pd.DataFrame,
    target_keys: list[str],
    target_week_col: str,
    history_target_week_col: str,
    history_pred_actual_pairs: list[tuple[str, str]],
    history_count_col: str,
    tier_breaks: tuple[int, int, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual_aggs = []
    needed_actual = sorted(set(actual for _, actual in history_pred_actual_pairs))
    for target_week, g in raw.groupby(target_week_col):
        cols = target_keys + needed_actual
        q = g[cols].copy()
        agg_map = {c: "median" for c in needed_actual}
        a = q.groupby(target_keys, dropna=False).agg(agg_map).reset_index()
        a[target_week_col] = target_week
        actual_aggs.append(a)
    actual = pd.concat(actual_aggs, ignore_index=True) if actual_aggs else pd.DataFrame()

    hist = history.copy()
    if history_target_week_col != target_week_col:
        hist = hist.rename(columns={history_target_week_col: target_week_col})
    join_keys = [target_week_col] + target_keys
    z = hist.merge(actual, on=join_keys, how="inner", suffixes=("", "_actual"))
    z["confidence_tier"] = confidence_tier(
        z[history_count_col],
        low_min=tier_breaks[0],
        medium_min=tier_breaks[1],
        high_min=tier_breaks[2],
    )
    z = z.loc[z["confidence_tier"].ne("ABSTAIN")].copy()

    rows = []
    for pred, actual_name in history_pred_actual_pairs:
        actual_col = actual_name
        for tier in ["ALL", "LOW", "MEDIUM", "HIGH"]:
            q = z if tier == "ALL" else z.loc[z["confidence_tier"].eq(tier)]
            rows.append(
                {
                    "predictor": pred,
                    "actual_geometry": actual_col,
                    "confidence_tier": tier,
                    **pair_metrics(q, pred, actual_col),
                }
            )
    return pd.DataFrame(rows), z


def numeric_correlation_matrix(frame: pd.DataFrame, columns: list[str], method: str = "spearman") -> pd.DataFrame:
    q = frame[columns].apply(pd.to_numeric, errors="coerce")
    return q.corr(method=method)


def write_sanitized(out_dir: Path, name: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def dataframe_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    safe = df.replace({np.nan: None, np.inf: None, -np.inf: None})
    return safe.to_dict(orient="records")
