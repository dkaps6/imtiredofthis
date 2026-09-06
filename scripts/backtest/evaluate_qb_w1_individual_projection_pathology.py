#!/usr/bin/env python3
"""Frozen QB Week-1 individual-projection pathology audit.

No model is fit and no projection is changed. The Week-1 sportsbook line is
retained only as a downstream descriptive comparison. Historical individual-QB
error profiles come from the untouched M89 2024-2025 validation trace.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

W1_RUN = 34059395746
M89_RUN = 33331073376


def _one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} below {root}, found {len(hits)}")
    return hits[0]


def _read(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    if x.empty:
        raise RuntimeError(f"empty input: {path}")
    return x


def _key(v) -> str:
    return "".join(ch.lower() for ch in str(v or "") if ch.isalnum())


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _q(v: pd.Series, p: float) -> float:
    a = _num(v).dropna().to_numpy(dtype=float)
    return float(np.quantile(a, p)) if len(a) else np.nan


def build_history(trace: pd.DataFrame) -> pd.DataFrame:
    required = {"season", "player_clean_key", "actual_pass_yards", "base_proj", "football_synthesis"}
    missing = required - set(trace.columns)
    if missing:
        raise RuntimeError(f"M89 synthesis trace missing columns: {sorted(missing)}")
    q = trace.loc[_num(trace["season"]).isin([2024, 2025])].copy()
    if len(q) != 884:
        raise RuntimeError(f"M89 validation-row drift: expected 884 got {len(q)}")
    q["player_key"] = q["player_clean_key"].map(_key)
    q["actual"] = _num(q["actual_pass_yards"])
    q["base"] = _num(q["base_proj"])
    q["synth"] = _num(q["football_synthesis"])
    if q[["actual", "base", "synth"]].isna().any().any():
        raise RuntimeError("M89 validation trace contains non-finite core values")
    q["base_err"] = q["base"] - q["actual"]
    q["synth_err"] = q["synth"] - q["actual"]

    rows = []
    for player_key, g in q.groupby("player_key", sort=True):
        e = g["synth_err"].astype(float)
        be = g["base_err"].astype(float)
        rows.append({
            "player_key": player_key,
            "historical_games": int(len(g)),
            "historical_base_mae": float(be.abs().mean()),
            "historical_synthesis_mae": float(e.abs().mean()),
            "historical_synthesis_bias": float(e.mean()),
            "historical_median_abs_error": float(e.abs().median()),
            "historical_p90_abs_error": _q(e.abs(), 0.90),
            "historical_miss30_rate": float(e.abs().ge(30).mean()),
            "historical_miss50_rate": float(e.abs().ge(50).mean()),
            "historical_miss75_rate": float(e.abs().ge(75).mean()),
            "historical_synthesis_mae_delta_vs_base": float(e.abs().mean() - be.abs().mean()),
        })
    return pd.DataFrame(rows)


def build_week1(w1: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    required = {
        "player", "team", "opponent", "vegas_line", "model_proj", "projection_minus_line",
        "model_sd", "qb_pred_attempts", "qb_pred_ypa", "qb_synthesis_correction",
        "ensemble_proj", "mc_proj", "ml_proj", "state_proj",
        "football_projection_frozen_before_market", "sportsbook_inputs_used_in_football_projection",
    }
    missing = required - set(w1.columns)
    if missing:
        raise RuntimeError(f"Week-1 audit missing columns: {sorted(missing)}")
    if len(w1) != 31:
        raise RuntimeError(f"Week-1 posted-QB row drift: expected 31 got {len(w1)}")
    if not _num(w1["football_projection_frozen_before_market"]).eq(1).all():
        raise RuntimeError("Week-1 football projection was not frozen before market join")
    if not _num(w1["sportsbook_inputs_used_in_football_projection"]).eq(0).all():
        raise RuntimeError("sportsbook input entered football projection")

    x = w1.copy()
    x["player_key"] = x["player"].map(_key)
    for c in [
        "model_proj", "vegas_line", "projection_minus_line", "model_sd", "qb_pred_attempts",
        "qb_pred_ypa", "qb_synthesis_correction", "ensemble_proj", "mc_proj", "ml_proj", "state_proj",
    ]:
        x[c] = _num(x[c])
    x["component_range_recalc"] = x[["mc_proj", "ml_proj", "state_proj"]].max(axis=1) - x[["mc_proj", "ml_proj", "state_proj"]].min(axis=1)
    x["final_minus_ensemble"] = x["model_proj"] - x["ensemble_proj"]
    if float((x["final_minus_ensemble"] - x["qb_synthesis_correction"]).abs().max()) > 1e-8:
        raise RuntimeError("Week-1 final-minus-ensemble does not equal synthesis correction")

    x = x.merge(hist, on="player_key", how="left", validate="one_to_one")
    games = _num(x["historical_games"]).fillna(0)
    x["flag_synthesis_cap"] = x["qb_synthesis_correction"].abs().ge(44.999).astype(int)
    x["flag_large_synthesis_move"] = x["qb_synthesis_correction"].abs().ge(30.0).astype(int)
    x["flag_large_component_disagreement"] = x["component_range_recalc"].ge(40.0).astype(int)
    x["flag_player_history_high_mae"] = (games.ge(8) & _num(x["historical_synthesis_mae"]).ge(50.0)).astype(int)
    x["flag_player_history_directional_bias"] = (games.ge(8) & _num(x["historical_synthesis_bias"]).abs().ge(15.0)).astype(int)
    x["flag_player_history_synthesis_worsened"] = (
        games.ge(8)
        & (_num(x["historical_synthesis_mae"]) > _num(x["historical_base_mae"]))
    ).astype(int)
    x["internal_flag_count"] = x[[
        "flag_synthesis_cap", "flag_large_synthesis_move", "flag_large_component_disagreement",
        "flag_player_history_high_mae", "flag_player_history_directional_bias",
        "flag_player_history_synthesis_worsened",
    ]].sum(axis=1)
    x["market_gap_abs"] = x["projection_minus_line"].abs()
    x["market_gap_ge20_descriptive"] = x["market_gap_abs"].ge(20).astype(int)
    x["market_gap_ge30_descriptive"] = x["market_gap_abs"].ge(30).astype(int)
    x["market_gap_ge40_descriptive"] = x["market_gap_abs"].ge(40).astype(int)
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w1-root", type=Path, required=True)
    ap.add_argument("--m89-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/qb_w1_individual_projection_pathology"))
    a = ap.parse_args()

    w1 = _read(_one(a.w1_root, "qb_2026_w1_market_comparison.csv"))
    m89 = _read(_one(a.m89_root, "m89_2024_2025_synthesis_trace.csv"))
    hist = build_history(m89)
    audit = build_week1(w1, hist)

    flag_cols = [c for c in audit.columns if c.startswith("flag_")]
    summary = {
        "migration": "QB_W1_INDIVIDUAL_PROJECTION_PATHOLOGY",
        "w1_source_run": W1_RUN,
        "m89_source_run": M89_RUN,
        "week1_rows": int(len(audit)),
        "historical_validation_rows": int(len(m89)),
        "historical_players": int(len(hist)),
        "week1_with_historical_profile": int(_num(audit["historical_games"]).fillna(0).gt(0).sum()),
        "week1_with_8plus_historical_games": int(_num(audit["historical_games"]).fillna(0).ge(8).sum()),
        "mean_model_minus_market": float(audit["projection_minus_line"].mean()),
        "mean_abs_model_minus_market": float(audit["market_gap_abs"].mean()),
        "market_gap_ge20_rows_descriptive": int(audit["market_gap_ge20_descriptive"].sum()),
        "market_gap_ge30_rows_descriptive": int(audit["market_gap_ge30_descriptive"].sum()),
        "market_gap_ge40_rows_descriptive": int(audit["market_gap_ge40_descriptive"].sum()),
        "internal_flag_counts": {c: int(audit[c].sum()) for c in flag_cols},
        "sportsbook_used_as_model_feature": False,
        "model_fitting_used": False,
        "production_changed": False,
        "disposition": "DIAGNOSTIC_ONLY_NO_PRODUCTION_CHANGE",
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    hist.sort_values(["historical_synthesis_mae", "historical_games"], ascending=[False, False]).to_csv(
        a.out_dir / "qb_individual_historical_error_profiles.csv", index=False
    )
    audit.sort_values(["internal_flag_count", "market_gap_abs"], ascending=[False, False]).to_csv(
        a.out_dir / "qb_2026_w1_individual_pathology.csv", index=False
    )
    (a.out_dir / "qb_w1_individual_projection_pathology_result.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print("=== QB W1 INDIVIDUAL PATHOLOGY SUMMARY ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("=== HIGHEST INTERNAL-FLAG WEEK-1 ROWS ===")
    show = [
        "player", "team", "opponent", "model_proj", "vegas_line", "projection_minus_line",
        "ensemble_proj", "qb_synthesis_correction", "component_range_recalc", "qb_pred_attempts",
        "qb_pred_ypa", "historical_games", "historical_synthesis_mae", "historical_synthesis_bias",
        "historical_p90_abs_error", "internal_flag_count",
    ] + flag_cols
    print(audit.sort_values(["internal_flag_count", "market_gap_abs"], ascending=[False, False])[show].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
