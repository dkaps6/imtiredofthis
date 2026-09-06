#!/usr/bin/env python3
"""WR-R1 frozen 2020-2025 M37 vs M38 multi-season replication evaluator.

This script does not fit or tune a model. It evaluates paired component prediction
files produced by exact M37 and M38 checkouts under the frozen WR-R1 plan.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

WR_POS = {"WR", "LWR", "RWR", "SWR"}
SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
EXPECTED_2025_ALL_REC_N = 4647
EXPECTED_2025_ALL_REC_MAE = 17.099904733366
PARITY_TOL = 1e-9
KEYS = ["season", "week", "team", "player_clean_key", "market"]


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path)
    x.columns = [str(c).strip().lower() for c in x.columns]
    for c in ("season", "week", "actual", "mc_proj"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    x["position"] = x.get("position", "").fillna("").astype(str).str.upper().str.strip()
    x["market"] = x.get("market", "").fillna("").astype(str)
    return x


def _metric_rows(frame: pd.DataFrame, model: str, slice_name: str) -> list[dict]:
    rows: list[dict] = []
    for market in ("rec_yards", "receptions"):
        g = frame.loc[frame["market"].eq(market)].copy()
        g = g.loc[g["actual"].notna() & g["mc_proj"].notna()].copy()
        err = g["mc_proj"] - g["actual"]
        rec = {
            "model": model,
            "slice": slice_name,
            "market": market,
            "n": int(len(g)),
            "mae": float(err.abs().mean()) if len(g) else np.nan,
            "rmse": float(np.sqrt(np.mean(np.square(err)))) if len(g) else np.nan,
            "bias": float(err.mean()) if len(g) else np.nan,
            "correlation": float(g["mc_proj"].corr(g["actual"])) if len(g) > 2 else np.nan,
        }
        if market == "rec_yards":
            resid = g["actual"] - g["mc_proj"]
            rec.update({
                "abs_error_ge25": int(err.abs().ge(25.0).sum()),
                "abs_error_ge50": int(err.abs().ge(50.0).sum()),
                "under_ge25": int(resid.ge(25.0).sum()),
                "under_ge50": int(resid.ge(50.0).sum()),
                "over_ge25": int(resid.le(-25.0).sum()),
                "over_ge50": int(resid.le(-50.0).sum()),
            })
        else:
            rec.update({k: np.nan for k in ["abs_error_ge25", "abs_error_ge50", "under_ge25", "under_ge50", "over_ge25", "over_ge50"]})
        rows.append(rec)
    return rows


def _paired(m37: pd.DataFrame, m38: pd.DataFrame, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = m37.loc[m37["season"].eq(season)].copy()
    b = m38.loc[m38["season"].eq(season)].copy()
    if a.duplicated(KEYS).any() or b.duplicated(KEYS).any():
        raise RuntimeError(f"duplicate prediction keys for {season}")
    common = a[KEYS + ["player", "position", "actual", "mc_proj"]].merge(
        b[KEYS + ["actual", "mc_proj"]], on=KEYS, how="inner", suffixes=("_m37", "_m38"), validate="one_to_one"
    )
    if len(common) != len(a) or len(common) != len(b):
        raise RuntimeError(f"M37/M38 row mismatch {season}: m37={len(a)} m38={len(b)} paired={len(common)}")
    actual_diff = (common["actual_m37"] - common["actual_m38"]).abs()
    if actual_diff.fillna(0.0).gt(1e-12).any():
        raise RuntimeError(f"actual outcome mismatch between M37/M38 for {season}")
    common["season"] = int(season)
    wr = common.loc[common["position"].isin(WR_POS)].copy()
    return common, wr


def _pooled_from_pair(pair_frames: list[pd.DataFrame], model: str) -> pd.DataFrame:
    x = pd.concat(pair_frames, ignore_index=True)
    out = x[KEYS + ["player", "position"]].copy()
    out["actual"] = x["actual_m37"]
    out["mc_proj"] = x[f"mc_proj_{model.lower()}"]
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="Directory containing YEAR/m37.csv and YEAR/m38.csv")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    metrics: list[dict] = []
    paired_wr_by_season: dict[int, pd.DataFrame] = {}
    paired_all_by_season: dict[int, pd.DataFrame] = {}
    parity = {}

    for season in SEASONS:
        m37 = _read(args.root / str(season) / "m37.csv", f"M37 {season}")
        m38 = _read(args.root / str(season) / "m38.csv", f"M38 {season}")
        pair_all, pair_wr = _paired(m37, m38, season)
        paired_all_by_season[season] = pair_all
        paired_wr_by_season[season] = pair_wr

        for model in ("M37", "M38"):
            f = _pooled_from_pair([pair_wr], model)
            metrics.extend(_metric_rows(f, model, str(season)))

        if season == 2025:
            all38 = _pooled_from_pair([pair_all], "M38")
            g = all38.loc[all38["market"].eq("rec_yards") & all38["actual"].notna() & all38["mc_proj"].notna()].copy()
            err = g["mc_proj"] - g["actual"]
            parity = {"all_receiver_rec_yards_n": int(len(g)), "all_receiver_rec_yards_mae": float(err.abs().mean())}
            if parity["all_receiver_rec_yards_n"] != EXPECTED_2025_ALL_REC_N or abs(parity["all_receiver_rec_yards_mae"] - EXPECTED_2025_ALL_REC_MAE) > PARITY_TOL:
                raise RuntimeError(f"2025 M38 parity failure: {parity}")

    aggregate_slices = {
        "2020_2025": SEASONS,
        "2021_2025": [2021, 2022, 2023, 2024, 2025],
        "2024_2025": [2024, 2025],
    }
    for slice_name, years in aggregate_slices.items():
        frames = [paired_wr_by_season[y] for y in years]
        for model in ("M37", "M38"):
            metrics.extend(_metric_rows(_pooled_from_pair(frames, model), model, slice_name))

    summary = pd.DataFrame(metrics)
    rec = summary.loc[summary["market"].eq("rec_yards"), ["model", "slice", "n", "mae", "rmse", "bias", "correlation"]].copy()
    m37 = rec.loc[rec["model"].eq("M37")].set_index("slice")
    m38 = rec.loc[rec["model"].eq("M38")].set_index("slice")
    shared = m37.index.intersection(m38.index)
    deltas = []
    for s in shared:
        deltas.append({
            "slice": s,
            "n": int(m38.loc[s, "n"]),
            "m37_mae": float(m37.loc[s, "mae"]),
            "m38_mae": float(m38.loc[s, "mae"]),
            "m38_minus_m37_mae": float(m38.loc[s, "mae"] - m37.loc[s, "mae"]),
            "m37_rmse": float(m37.loc[s, "rmse"]),
            "m38_rmse": float(m38.loc[s, "rmse"]),
            "m38_minus_m37_rmse": float(m38.loc[s, "rmse"] - m37.loc[s, "rmse"]),
            "m37_bias": float(m37.loc[s, "bias"]),
            "m38_bias": float(m38.loc[s, "bias"]),
            "m37_corr": float(m37.loc[s, "correlation"]),
            "m38_corr": float(m38.loc[s, "correlation"]),
        })
    delta_df = pd.DataFrame(deltas)

    def delta(slice_name: str) -> float:
        return float(delta_df.loc[delta_df["slice"].eq(slice_name), "m38_minus_m37_mae"].iloc[0])

    yearly = [delta(str(y)) for y in SEASONS]
    non_worse = int(sum(d <= 0.0 for d in yearly))
    worst_regression = float(max(yearly))
    pooled_improves = delta("2020_2025") < 0.0
    latest_improves = delta("2024_2025") < 0.0
    confirmed = pooled_improves and non_worse >= 4 and latest_improves and worst_regression <= 1.0
    if confirmed:
        disposition = "M38_MULTISEASON_CONFIRMED"
    elif pooled_improves:
        disposition = "M38_MULTISEASON_ERA_DEPENDENT"
    else:
        disposition = "M38_MULTISEASON_NOT_CONFIRMED"

    total_wr_rows = int(sum(len(paired_wr_by_season[y].loc[paired_wr_by_season[y]["market"].eq("rec_yards")]) for y in SEASONS))
    result = {
        "migration": "WR-R1",
        "target_seasons": SEASONS,
        "m37_ref": "ba83fd05412a36309822cac6aa9cc5003388b073",
        "m38_ref": "b98518d97b3038f471aee9ae3201009b2c70bb29",
        "iterations": 2000,
        "sportsbook_inputs_used": False,
        "retuning_used": False,
        "2025_parity": parity,
        "wr_rec_yards_rows_2020_2025": total_wr_rows,
        "yearly_non_worse_count": non_worse,
        "worst_single_season_mae_regression": worst_regression,
        "pooled_2020_2025_mae_delta": delta("2020_2025"),
        "pooled_2021_2025_mae_delta": delta("2021_2025"),
        "pooled_2024_2025_mae_delta": delta("2024_2025"),
        "disposition": disposition,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "wr_r1_metric_summary.csv", index=False)
    delta_df.to_csv(args.out_dir / "wr_r1_rec_yards_deltas.csv", index=False)
    pd.concat([paired_wr_by_season[y] for y in SEASONS], ignore_index=True).to_csv(args.out_dir / "wr_r1_paired_wr_casebook.csv", index=False)
    (args.out_dir / "wr_r1_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[wr-r1] receiving-yard deltas")
    print(delta_df.to_string(index=False))
    print("[wr-r1] result")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
