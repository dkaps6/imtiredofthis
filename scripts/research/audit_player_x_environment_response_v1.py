#!/usr/bin/env python3
"""Player x Environment Response V1 frozen historical diagnostic.

Frozen in docs/research/PLAYER_X_ENVIRONMENT_RESPONSE_V1_FREEZE.md.

No projection change, coefficient fit, sportsbook input, 2026 outcome, or
production mutation. Tests whether a named player's own strictly-prior response
to the existing shared environment multiplier predicts his next residual.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

VERSION = "PLAYER_X_ENVIRONMENT_RESPONSE_V1"
PARENT_SHA = "812ac2cde79fe3846174613a2f2777e57b34b739"
TARGET_SEASONS = (2024, 2025)
TARGET_WEEKS = tuple(range(7, 19))
PRIOR_WINDOW = 8
MIN_PRIOR = 6
BOOT_REPS = 2000
GAME_SEED = 20261013
PLAYER_SEED = 20271013

CELL_SPECS = {
    "WR_REC_YARDS": {
        "positions": {"WR", "LWR", "RWR", "SWR"},
        "market": "rec_yards",
        "env_col": "rules_pass_eff_mult",
        "min_rows": 250,
        "min_players": 50,
        "min_games": 50,
    },
    "TE_REC_YARDS": {
        "positions": {"TE"},
        "market": "rec_yards",
        "env_col": "rules_pass_eff_mult",
        "min_rows": 125,
        "min_players": 25,
        "min_games": 50,
    },
    "RB_RUSH_YARDS": {
        "positions": {"RB", "HB", "FB"},
        "market": "rush_yards",
        "env_col": "rules_rush_eff_mult",
        "min_rows": 125,
        "min_players": 25,
        "min_games": 50,
    },
}

FORBIDDEN = (
    "sportsbook", "bookmaker", "over_odds", "under_odds", "decimal_odds",
    "american_odds", "prop_line", "market_line", "closing_line", "no_vig",
    "implied_prob", "oddsapi",
)


def read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def rho(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    return float(spearmanr(x, y).statistic)


def partial_rho(df: pd.DataFrame) -> float:
    cols = ["response_signal", "target_residual", "prior_mean_residual", "baseline_projection"]
    z = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(z) < 5:
        return np.nan
    ranks = np.column_stack([rankdata(z[c].to_numpy(float)) for c in cols])
    sx, target = ranks[:, 0], ranks[:, 1]
    controls = np.column_stack([np.ones(len(z)), ranks[:, 2], ranks[:, 3]])
    bs = np.linalg.lstsq(controls, sx, rcond=None)[0]
    bt = np.linalg.lstsq(controls, target, rcond=None)[0]
    rs = sx - controls @ bs
    rt = target - controls @ bt
    if np.std(rs) <= 0 or np.std(rt) <= 0:
        return np.nan
    return float(np.corrcoef(rs, rt)[0, 1])


def quartile_gap(df: pd.DataFrame) -> float:
    z = df[["response_signal", "target_residual"]].dropna().copy()
    if len(z) < 8 or z["response_signal"].nunique() < 4:
        return np.nan
    try:
        z["q"] = pd.qcut(z["response_signal"], 4, labels=False, duplicates="drop")
    except ValueError:
        return np.nan
    if z["q"].nunique() != 4:
        return np.nan
    means = z.groupby("q")["target_residual"].mean()
    return float(means.loc[3] - means.loc[0])


def _history_signal(h: pd.DataFrame, current_x: float) -> tuple[float, float, float]:
    hx = num(h["environment_x"]).to_numpy(float)
    he = num(h["target_residual"]).to_numpy(float)
    keep = np.isfinite(hx) & np.isfinite(he)
    hx, he = hx[keep], he[keep]
    if len(hx) < MIN_PRIOR:
        return np.nan, np.nan, np.nan
    if np.unique(np.round(hx, 8)).size < 3:
        return np.nan, np.nan, np.nan
    rr = rho(hx, he)
    if not np.isfinite(rr):
        return np.nan, np.nan, np.nan
    pct = (float(np.sum(hx <= current_x)) + 0.5) / (float(len(hx)) + 1.0)
    rel = 2.0 * pct - 1.0
    return float(rr), float(rel), float(rr * rel)


def build_panel(trace: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "season", "week", "game_id", "team", "opponent", "player_clean_key",
        "position", "market", "actual", "ensemble_proj",
        "rules_pass_eff_mult", "rules_rush_eff_mult",
    }
    missing = sorted(required - set(trace.columns))
    if missing:
        raise RuntimeError(f"projection trace missing columns: {missing}")
    bad = [c for c in trace.columns if any(t in str(c).lower() for t in FORBIDDEN)]
    if bad:
        raise RuntimeError(f"forbidden sportsbook/odds columns present: {bad}")

    x = trace.copy()
    x["season"] = num(x["season"]).astype(int)
    x["week"] = num(x["week"]).astype(int)
    x["position"] = x["position"].fillna("").astype(str).str.upper().str.strip()
    x["market"] = x["market"].fillna("").astype(str).str.lower().str.strip()
    x["player_clean_key"] = x["player_clean_key"].fillna("").astype(str).str.strip()
    x["actual"] = num(x["actual"])
    x["ensemble_proj"] = num(x["ensemble_proj"])

    rows: list[dict[str, Any]] = []
    chronology_violations = 0
    source_rows = 0
    valid_environment_rows = 0

    for cell, spec in CELL_SPECS.items():
        q = x.loc[
            x["season"].isin(TARGET_SEASONS)
            & x["position"].isin(spec["positions"])
            & x["market"].eq(spec["market"])
            & x["actual"].notna()
            & x["ensemble_proj"].notna()
            & x["player_clean_key"].ne("")
        ].copy()
        source_rows += int(len(q))
        q["environment_mult"] = num(q[spec["env_col"]])
        valid = q["environment_mult"].notna() & q["environment_mult"].gt(0)
        valid_environment_rows += int(valid.sum())
        q = q.loc[valid].copy()
        q["environment_x"] = np.log(q["environment_mult"].astype(float))
        q["target_residual"] = q["actual"] - q["ensemble_proj"]

        for (season, pid), g in q.groupby(["season", "player_clean_key"], sort=False):
            g = g.sort_values(["week", "game_id"]).reset_index(drop=True)
            for r in g.itertuples(index=False):
                week = int(r.week)
                if week not in TARGET_WEEKS:
                    continue
                h = g.loc[g["week"].lt(week)].tail(PRIOR_WINDOW).copy()
                if len(h) < MIN_PRIOR:
                    continue
                max_week = int(h["week"].max())
                chronology_violations += int(max_week >= week)
                rr, rel, signal = _history_signal(h, float(r.environment_x))
                if not np.isfinite(signal):
                    continue
                rows.append({
                    "cell": cell,
                    "season": int(season),
                    "week": week,
                    "game_id": str(r.game_id),
                    "team": str(r.team),
                    "opponent": str(r.opponent),
                    "player_clean_key": str(pid),
                    "position": str(r.position),
                    "market": str(r.market),
                    "baseline_projection": float(r.ensemble_proj),
                    "actual": float(r.actual),
                    "target_residual": float(r.target_residual),
                    "environment_mult": float(r.environment_mult),
                    "environment_x": float(r.environment_x),
                    "prior_games": int(len(h)),
                    "prior_feature_max_week": max_week,
                    "prior_mean_residual": float(num(h["target_residual"]).mean()),
                    "prior_response_rho": rr,
                    "relative_environment": rel,
                    "response_signal": signal,
                })

    panel = pd.DataFrame(rows)
    integrity = {
        "chronology_violations": int(chronology_violations),
        "source_rows": int(source_rows),
        "valid_environment_rows": int(valid_environment_rows),
    }
    return panel, integrity


def bootstrap(df: pd.DataFrame, cluster_col: str, seed: int) -> dict[str, float | int | None]:
    z = df.copy().reset_index(drop=True)
    groups = [g.copy() for _, g in z.groupby(cluster_col, sort=True)]
    if len(groups) < 2:
        return {
            "valid_reps": 0,
            "raw_ci_low": None, "raw_ci_high": None,
            "partial_ci_low": None, "partial_ci_high": None,
        }
    rng = np.random.default_rng(seed)
    raw = np.full(BOOT_REPS, np.nan)
    part = np.full(BOOT_REPS, np.nan)
    n = len(groups)
    for i in range(BOOT_REPS):
        picks = rng.integers(0, n, size=n)
        b = pd.concat([groups[j] for j in picks], ignore_index=True)
        raw[i] = rho(
            num(b["response_signal"]).to_numpy(float),
            num(b["target_residual"]).to_numpy(float),
        )
        part[i] = partial_rho(b)
    rv = raw[np.isfinite(raw)]
    pv = part[np.isfinite(part)]
    return {
        "valid_reps": int(min(len(rv), len(pv))),
        "raw_ci_low": float(np.quantile(rv, 0.025)) if len(rv) else None,
        "raw_ci_high": float(np.quantile(rv, 0.975)) if len(rv) else None,
        "partial_ci_low": float(np.quantile(pv, 0.025)) if len(pv) else None,
        "partial_ci_high": float(np.quantile(pv, 0.975)) if len(pv) else None,
    }


def score_cell(panel: pd.DataFrame, cell: str, season: int) -> dict[str, Any]:
    spec = CELL_SPECS[cell]
    z = panel.loc[panel["cell"].eq(cell) & panel["season"].eq(season)].copy()
    rows = int(len(z))
    players = int(z["player_clean_key"].nunique()) if rows else 0
    games = int(z["game_id"].nunique()) if rows else 0
    support = bool(
        rows >= spec["min_rows"]
        and players >= spec["min_players"]
        and games >= spec["min_games"]
    )
    out: dict[str, Any] = {
        "cell": cell, "season": int(season), "rows": rows,
        "players": players, "games": games, "support": support,
    }
    if not support:
        out["signal_supported"] = False
        return out

    raw = rho(
        num(z["response_signal"]).to_numpy(float),
        num(z["target_residual"]).to_numpy(float),
    )
    part = partial_rho(z)
    gap = quartile_gap(z)
    idx = list(CELL_SPECS).index(cell)
    gb = bootstrap(z, "game_id", GAME_SEED + season + idx * 100)
    pb = bootstrap(z, "player_clean_key", PLAYER_SEED + season + idx * 100)

    supported = bool(
        np.isfinite(raw) and raw > 0
        and gb["raw_ci_low"] is not None and gb["raw_ci_low"] > 0
        and pb["raw_ci_low"] is not None and pb["raw_ci_low"] > 0
        and np.isfinite(part) and part > 0
        and gb["partial_ci_low"] is not None and gb["partial_ci_low"] > 0
        and pb["partial_ci_low"] is not None and pb["partial_ci_low"] > 0
        and np.isfinite(gap) and gap > 0
    )
    out.update({
        "raw_spearman_rho": raw,
        "partial_spearman_rho": part,
        "q4_minus_q1_mean_residual": gap,
        "game_bootstrap": gb,
        "player_bootstrap": pb,
        "signal_supported": supported,
        "mean_response_signal": float(z["response_signal"].mean()),
        "mean_target_residual": float(z["target_residual"].mean()),
        "mean_abs_target_residual": float(z["target_residual"].abs().mean()),
    })
    return out


def player_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cell, season, pid), g in panel.groupby(
        ["cell", "season", "player_clean_key"], sort=True
    ):
        rows.append({
            "cell": cell,
            "season": int(season),
            "player_clean_key": str(pid),
            "rows": int(len(g)),
            "mean_signal": float(g["response_signal"].mean()),
            "mean_residual": float(g["target_residual"].mean()),
            "mean_abs_residual": float(g["target_residual"].abs().mean()),
            "signal_residual_rho": rho(
                num(g["response_signal"]).to_numpy(float),
                num(g["target_residual"]).to_numpy(float),
            ) if len(g) >= 3 else np.nan,
        })
    return pd.DataFrame(rows)


def evaluate(trace: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    panel, integrity = build_panel(trace)
    if panel.empty:
        raise RuntimeError("zero player x environment response rows")

    cells = []
    summary = []
    for cell in CELL_SPECS:
        per = {}
        for season in TARGET_SEASONS:
            rec = score_cell(panel, cell, season)
            cells.append(rec)
            per[season] = bool(rec.get("signal_supported", False))
        replicated = all(per.get(s, False) for s in TARGET_SEASONS)
        summary.append({
            "cell": cell,
            "season_2024_supported": bool(per.get(2024, False)),
            "season_2025_supported": bool(per.get(2025, False)),
            "disposition": (
                "PLAYER_X_ENVIRONMENT_RESPONSE_REPLICATED"
                if replicated
                else "PLAYER_X_ENVIRONMENT_RESPONSE_NOT_REPLICATED"
            ),
        })

    identity_coverage = float(panel["player_clean_key"].astype(str).str.strip().ne("").mean())
    gates = {
        "exact_target_seasons": sorted(panel["season"].unique().tolist()) == list(TARGET_SEASONS),
        "target_weeks_within_7_18": int(panel["week"].min()) >= 7 and int(panel["week"].max()) <= 18,
        "chronology_violations_zero": integrity["chronology_violations"] == 0,
        "stable_identity_coverage_100pct": identity_coverage == 1.0,
        "used_environment_values_finite_positive": bool(
            np.isfinite(panel["environment_mult"].to_numpy(float)).all()
            and (panel["environment_mult"].to_numpy(float) > 0).all()
        ),
        "sportsbook_inputs_zero": True,
        "oddsapi_calls_zero": True,
        "parameters_fit_zero": True,
        "projection_changes_zero": True,
        "production_changed_false": True,
    }
    any_replicated = any(
        r["disposition"] == "PLAYER_X_ENVIRONMENT_RESPONSE_REPLICATED"
        for r in summary
    )
    disposition = (
        "PLAYER_X_ENVIRONMENT_RESPONSE_HAS_REPLICATED_CELLS"
        if any_replicated else "PLAYER_X_ENVIRONMENT_RESPONSE_NULL"
    )
    if not all(gates.values()):
        disposition = "PLAYER_X_ENVIRONMENT_RESPONSE_INTEGRITY_FAILURE"

    result = {
        "version": VERSION,
        "disposition": disposition,
        "production_parent_sha": PARENT_SHA,
        "target_seasons": list(TARGET_SEASONS),
        "target_weeks": [7, 18],
        "prior_window_games": PRIOR_WINDOW,
        "minimum_prior_games": MIN_PRIOR,
        "bootstrap_reps": BOOT_REPS,
        "panel_rows": int(len(panel)),
        "panel_players": int(panel["player_clean_key"].nunique()),
        "identity_coverage": identity_coverage,
        "integrity": integrity,
        "summary": summary,
        "cells": cells,
        "gates": gates,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "outcomes_2026_used": 0,
        "parameters_fit": 0,
        "projection_changes": 0,
        "production_changed": False,
    }
    return panel, player_summary(panel), result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projection-file", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trace = read(args.projection_file, "full-stack projection trace")
    panel, players, result = evaluate(trace)
    panel.to_csv(args.out_dir / "player_x_environment_response_rows.csv", index=False, float_format="%.12f")
    players.to_csv(args.out_dir / "player_x_environment_response_players.csv", index=False, float_format="%.12f")
    pd.DataFrame(result["cells"]).to_csv(args.out_dir / "player_x_environment_response_cells.csv", index=False)
    (args.out_dir / "player_x_environment_response_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
