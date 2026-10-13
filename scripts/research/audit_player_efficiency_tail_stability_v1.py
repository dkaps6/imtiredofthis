#!/usr/bin/env python3
"""Player Efficiency Tail Stability V1.

Frozen in docs/research/PLAYER_EFFICIENCY_TAIL_STABILITY_V1_FREEZE.md.

Diagnostic only. Tests whether strictly-prior player-specific game-to-game
per-opportunity efficiency volatility predicts next-game absolute efficiency
surprise. No mean change, matchup transform, sportsbook data, model residuals,
2026 outcomes, or production changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _normalize_weekly

VERSION = "PLAYER_EFFICIENCY_TAIL_STABILITY_V1"
TARGET_SEASONS = (2023, 2024, 2025)
TARGET_WEEKS = tuple(range(5, 19))
PRIOR_WINDOW = 8
MIN_PRIOR_GAMES = 4
BOOT_REPS = 1000
GAME_SEED = 20261013
PLAYER_SEED = 20271013

RB_POS = {"RB", "HB", "FB"}
WR_POS = {"WR", "LWR", "RWR", "SWR"}
TE_POS = {"TE"}

CELL_SPECS = {
    "RB_RUSH_YPC": {"positions": RB_POS, "eff_col": "ypc_game", "opp_col": "rushes", "min_opp": 4},
    "WR_TARGET_YPT": {"positions": WR_POS, "eff_col": "ypt_game", "opp_col": "targets", "min_opp": 3},
    "TE_TARGET_YPT": {"positions": TE_POS, "eff_col": "ypt_game", "opp_col": "targets", "min_opp": 3},
    "RB_TARGET_YPT": {"positions": RB_POS, "eff_col": "ypt_game", "opp_col": "targets", "min_opp": 2},
}


def _to_pandas(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if hasattr(obj, "to_dicts"):
        return pd.DataFrame(obj.to_dicts())
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def _lower(x: pd.DataFrame) -> pd.DataFrame:
    y = x.copy()
    y.columns = [str(c).strip().lower() for c in y.columns]
    return y


def _load_schedule() -> pd.DataFrame:
    import nflreadpy as nfl

    rows = []
    for season in TARGET_SEASONS:
        try:
            raw = nfl.load_schedules(seasons=[season])
        except TypeError:
            raw = nfl.load_schedules(season)
        x = _lower(_to_pandas(raw))
        if "game_type" in x.columns:
            reg = x.loc[x["game_type"].astype(str).str.upper().eq("REG")].copy()
            if not reg.empty:
                x = reg
        elif "season_type" in x.columns:
            reg = x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
            if not reg.empty:
                x = reg
        hc = "home_team" if "home_team" in x.columns else "home"
        ac = "away_team" if "away_team" in x.columns else "away"
        gidc = "game_id" if "game_id" in x.columns else None
        for r in x.itertuples(index=False):
            week = int(getattr(r, "week"))
            if week > 18:
                continue
            home = canon_team(getattr(r, hc))
            away = canon_team(getattr(r, ac))
            if not home or not away:
                continue
            gid = str(getattr(r, gidc)) if gidc else f"{season}_{week:02d}_{away}_{home}"
            rows.append({"season": season, "week": week, "team": home, "opponent": away, "game_id": gid})
            rows.append({"season": season, "week": week, "team": away, "opponent": home, "game_id": gid})
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("zero schedule rows")
    if out.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate schedule team-week rows")
    return out


def _load_weekly() -> pd.DataFrame:
    import nflreadpy as nfl

    sched = _load_schedule()
    parts = []
    for season in TARGET_SEASONS:
        raw = nfl.load_player_stats(seasons=[season], summary_level="week")
        x = _normalize_weekly(_to_pandas(raw), season)
        ss = sched.loc[sched["season"].eq(season)].rename(
            columns={"game_id": "schedule_game_id", "opponent": "schedule_opponent"}
        )
        x = x.drop(columns=["game_id", "opponent"], errors="ignore").merge(
            ss,
            on=["season", "week", "team"],
            how="left",
            validate="many_to_one",
        )
        x["game_id"] = x["schedule_game_id"]
        x["opponent"] = x["schedule_opponent"]
        if x["game_id"].isna().any():
            raise RuntimeError(f"missing schedule game_id season={season}")
        parts.append(x)
    out = pd.concat(parts, ignore_index=True)
    if out.empty:
        raise RuntimeError("zero normalized weekly rows")
    return out.sort_values(["season", "week", "player_identity_key"]).reset_index(drop=True)


def _mad(values: np.ndarray) -> float:
    if not len(values):
        return np.nan
    med = float(np.median(values))
    return float(np.median(np.abs(values - med)))


def build_panel(weekly: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    rows = []
    chronology_violations = 0
    for cell, spec in CELL_SPECS.items():
        q = weekly.loc[
            weekly["position"].fillna("").astype(str).str.upper().isin(spec["positions"])
        ].copy()
        q["eff"] = pd.to_numeric(q[spec["eff_col"]], errors="coerce")
        q["opp"] = pd.to_numeric(q[spec["opp_col"]], errors="coerce")
        q = q.loc[q["eff"].notna() & q["opp"].ge(float(spec["min_opp"]))].copy()

        for (season, pid), g in q.groupby(["season", "player_identity_key"], sort=False):
            g = g.sort_values(["week", "game_id"]).copy()
            for r in g.itertuples(index=False):
                week = int(r.week)
                if week not in TARGET_WEEKS:
                    continue
                h = g.loc[g["week"].lt(week)].tail(PRIOR_WINDOW).copy()
                if len(h) < MIN_PRIOR_GAMES:
                    continue
                eff = pd.to_numeric(h["eff"], errors="coerce").dropna().to_numpy(float)
                opp = pd.to_numeric(h["opp"], errors="coerce").dropna().to_numpy(float)
                if len(eff) < MIN_PRIOR_GAMES or len(opp) < MIN_PRIOR_GAMES:
                    continue
                max_week = int(h["week"].max())
                chronology_violations += int(max_week >= week)
                prior_mean = float(np.mean(eff))
                prior_sd = float(np.std(eff, ddof=0))
                actual = float(r.eff)
                rows.append({
                    "season": int(season),
                    "week": week,
                    "game_id": str(r.game_id),
                    "team": str(r.team),
                    "opponent": str(r.opponent),
                    "player_identity_key": str(pid),
                    "player": str(r.player),
                    "position": str(r.position),
                    "cell": cell,
                    "actual_efficiency": actual,
                    "target_opportunity": float(r.opp),
                    "prior_games": int(len(eff)),
                    "prior_efficiency_mean": prior_mean,
                    "prior_efficiency_sd8": prior_sd,
                    "prior_efficiency_mad8": _mad(eff),
                    "prior_opportunity_mean": float(np.mean(opp)),
                    "prior_feature_max_week": max_week,
                    "efficiency_surprise": abs(actual - prior_mean),
                })
    return pd.DataFrame(rows), {"chronology_violations": int(chronology_violations)}


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan
    return float(spearmanr(x, y).statistic)


def _partial_spearman(df: pd.DataFrame) -> float:
    cols = [
        "prior_efficiency_sd8",
        "efficiency_surprise",
        "prior_efficiency_mean",
        "prior_opportunity_mean",
    ]
    z = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(z) < 5:
        return np.nan
    ranks = np.column_stack([rankdata(z[c].to_numpy(float)) for c in cols])
    x = ranks[:, 0]
    y = ranks[:, 1]
    controls = np.column_stack([np.ones(len(z)), ranks[:, 2], ranks[:, 3]])
    bx = np.linalg.lstsq(controls, x, rcond=None)[0]
    by = np.linalg.lstsq(controls, y, rcond=None)[0]
    rx = x - controls @ bx
    ry = y - controls @ by
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _bootstrap(df: pd.DataFrame, cluster_col: str, seed: int) -> dict[str, float | int]:
    z = df[
        [
            "prior_efficiency_sd8",
            "efficiency_surprise",
            "prior_efficiency_mean",
            "prior_opportunity_mean",
            cluster_col,
        ]
    ].dropna().copy()
    groups = [g.copy() for _, g in z.groupby(cluster_col, sort=True)]
    if len(groups) < 2:
        return {
            "valid_reps": 0,
            "raw_ci_low": np.nan,
            "raw_ci_high": np.nan,
            "partial_ci_low": np.nan,
            "partial_ci_high": np.nan,
        }
    rng = np.random.default_rng(seed)
    raw = np.full(BOOT_REPS, np.nan)
    partial = np.full(BOOT_REPS, np.nan)
    n = len(groups)
    for rep in range(BOOT_REPS):
        ids = rng.integers(0, n, size=n)
        b = pd.concat([groups[i] for i in ids], ignore_index=True)
        raw[rep] = _rho(
            pd.to_numeric(b["prior_efficiency_sd8"], errors="coerce").to_numpy(float),
            pd.to_numeric(b["efficiency_surprise"], errors="coerce").to_numpy(float),
        )
        partial[rep] = _partial_spearman(b)
    rv = raw[np.isfinite(raw)]
    pv = partial[np.isfinite(partial)]
    return {
        "valid_reps": int(min(len(rv), len(pv))),
        "raw_ci_low": float(np.quantile(rv, 0.025)) if len(rv) else np.nan,
        "raw_ci_high": float(np.quantile(rv, 0.975)) if len(rv) else np.nan,
        "partial_ci_low": float(np.quantile(pv, 0.025)) if len(pv) else np.nan,
        "partial_ci_high": float(np.quantile(pv, 0.975)) if len(pv) else np.nan,
    }


def _quartiles(df: pd.DataFrame) -> list[dict[str, float | int | str]]:
    z = df[["prior_efficiency_sd8", "efficiency_surprise"]].dropna().copy()
    if len(z) < 4 or z["prior_efficiency_sd8"].nunique() < 4:
        return []
    z["q"] = pd.qcut(z["prior_efficiency_sd8"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    out = []
    for label, g in z.groupby("q", observed=True):
        out.append({
            "quartile": str(label),
            "rows": int(len(g)),
            "mean_prior_efficiency_sd8": float(g["prior_efficiency_sd8"].mean()),
            "mean_efficiency_surprise": float(g["efficiency_surprise"].mean()),
        })
    return out


def score_cell(panel: pd.DataFrame, season: int, cell: str) -> dict[str, Any]:
    z = panel.loc[panel["season"].eq(season) & panel["cell"].eq(cell)].copy()
    rows = len(z)
    players = z["player_identity_key"].nunique()
    games = z["game_id"].nunique()
    support = rows >= 250 and players >= 60 and games >= 50
    out: dict[str, Any] = {
        "season": int(season),
        "cell": cell,
        "rows": int(rows),
        "players": int(players),
        "games": int(games),
        "support": bool(support),
        "quartiles": _quartiles(z),
    }
    if not support:
        out["signal_supported"] = False
        return out

    raw = _rho(
        pd.to_numeric(z["prior_efficiency_sd8"], errors="coerce").to_numpy(float),
        pd.to_numeric(z["efficiency_surprise"], errors="coerce").to_numpy(float),
    )
    partial = _partial_spearman(z)
    idx = list(CELL_SPECS).index(cell)
    gb = _bootstrap(z, "game_id", GAME_SEED + season + idx * 100)
    pb = _bootstrap(z, "player_identity_key", PLAYER_SEED + season + idx * 100)

    supported = bool(
        np.isfinite(raw)
        and raw > 0
        and np.isfinite(partial)
        and partial > 0
        and np.isfinite(gb["raw_ci_low"])
        and gb["raw_ci_low"] > 0
        and np.isfinite(pb["raw_ci_low"])
        and pb["raw_ci_low"] > 0
        and np.isfinite(gb["partial_ci_low"])
        and gb["partial_ci_low"] > 0
        and np.isfinite(pb["partial_ci_low"])
        and pb["partial_ci_low"] > 0
    )
    out.update({
        "raw_spearman_rho": raw,
        "partial_spearman_rho": partial,
        "game_bootstrap": gb,
        "player_bootstrap": pb,
        "signal_supported": supported,
        "mean_prior_efficiency_sd8": float(z["prior_efficiency_sd8"].mean()),
        "mean_prior_efficiency_mad8": float(z["prior_efficiency_mad8"].mean()),
        "mean_efficiency_surprise": float(z["efficiency_surprise"].mean()),
    })
    return out


def audit() -> tuple[pd.DataFrame, dict[str, Any]]:
    weekly = _load_weekly()
    panel, integrity = build_panel(weekly)
    if panel.empty:
        raise RuntimeError("zero efficiency-tail panel rows")

    cells = []
    summary = []
    for cell in CELL_SPECS:
        per_season = {}
        for season in TARGET_SEASONS:
            rec = score_cell(panel, season, cell)
            cells.append(rec)
            per_season[season] = bool(rec.get("signal_supported", False))
        replicated = all(per_season.get(s, False) for s in TARGET_SEASONS)
        summary.append({
            "cell": cell,
            **{f"season_{s}_supported": bool(per_season.get(s, False)) for s in TARGET_SEASONS},
            "disposition": (
                "PLAYER_EFFICIENCY_VOLATILITY_THREE_SEASON_CONFIRMED"
                if replicated
                else "PLAYER_EFFICIENCY_VOLATILITY_NOT_REPLICATED"
            ),
        })

    identity_cov = float(panel["player_identity_key"].astype(str).ne("").mean())
    gates = {
        "exact_target_seasons": sorted(panel["season"].unique().tolist()) == list(TARGET_SEASONS),
        "target_weeks_within_5_18": int(panel["week"].min()) >= 5 and int(panel["week"].max()) <= 18,
        "chronology_violations_zero": integrity["chronology_violations"] == 0,
        "stable_identity_coverage_ge_095": identity_cov >= 0.95,
        "sportsbook_inputs_zero": True,
        "oddsapi_calls_zero": True,
        "model_residuals_zero": True,
        "fitted_predictive_models_zero": True,
        "candidate_mean_changes_zero": True,
        "production_changed_false": True,
    }
    any_confirmed = any(
        x["disposition"] == "PLAYER_EFFICIENCY_VOLATILITY_THREE_SEASON_CONFIRMED"
        for x in summary
    )
    result = {
        "version": VERSION,
        "disposition": (
            "PLAYER_EFFICIENCY_TAIL_STABILITY_HAS_CONFIRMED_CELLS"
            if any_confirmed
            else "PLAYER_EFFICIENCY_TAIL_STABILITY_NULL"
        ),
        "target_seasons": list(TARGET_SEASONS),
        "target_weeks": [5, 18],
        "prior_window_games": PRIOR_WINDOW,
        "min_prior_games": MIN_PRIOR_GAMES,
        "bootstrap_reps": BOOT_REPS,
        "panel_rows": int(len(panel)),
        "panel_players": int(panel["player_identity_key"].nunique()),
        "stable_identity_coverage": identity_cov,
        "integrity": integrity,
        "summary": summary,
        "cells": cells,
        "gates": gates,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "model_residuals_used": 0,
        "fitted_predictive_models": 0,
        "candidate_mean_changes": 0,
        "production_changed": False,
    }
    if not all(bool(v) for v in gates.values()):
        result["disposition"] = "PLAYER_EFFICIENCY_TAIL_STABILITY_INTEGRITY_FAILURE"
    return panel, result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel, result = audit()
    panel.to_csv(
        args.out_dir / "player_efficiency_tail_stability_v1_rows.csv",
        index=False,
        float_format="%.12f",
    )
    (args.out_dir / "player_efficiency_tail_stability_v1_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
