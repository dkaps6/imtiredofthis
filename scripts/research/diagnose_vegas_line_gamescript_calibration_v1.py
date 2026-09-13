#!/usr/bin/env python3
"""docs/research/VEGAS_LINE_GAMESCRIPT_CALIBRATION_V1_PLAN.md.

Diagnostic only: how accurately does Vegas's own closing spread/total
(nflverse schedule, leakage-safe pregame data) describe what actually
happened in the game -- the final combined score (scoring environment) and
the final margin (competitiveness/blowout-vs-close) -- on the market's own
terms. Distinct from scripts/research/diagnose_market_implied_game_script_v1.py
(PR #558), which asked whether the market adds incremental value *on top of*
our own historical team-plays baseline. This script asks nothing about our
model at all: it is a direct calibration check of the market itself against
real outcomes.

No player-level or team-week data, no rolling history, no baseline fit
against our own numbers -- and therefore no cross-season leakage risk, since
every row (one game) is independent of every other row.

Research only. No production/model/threshold change.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

SPREAD_BINS = [0.0, 3.0, 7.0, 10.0, 14.0, np.inf]
SPREAD_BIN_LABELS = ["0-3", "3-7", "7-10", "10-14", "14+"]

TOTAL_BIN_EDGES_DEFAULT = [0.0, 38.0, 42.0, 46.0, 50.0, np.inf]
TOTAL_BIN_LABELS = ["<38", "38-42", "42-46", "46-50", "50+"]

MIN_ROWS_PER_FOLD = 30


def num(v):
    return pd.to_numeric(v, errors="coerce")


def load_game_outcomes(seasons: list[int]) -> pd.DataFrame:
    """One row per completed regular-season game: posted spread/total and
    actual final score, plus a signed home-perspective spread/margin.
    """
    import nflreadpy as nfl

    rows = []
    for season in seasons:
        s = nfl.load_schedules(int(season))
        if hasattr(s, "to_pandas"):
            s = s.to_pandas()
        s = pd.DataFrame(s)
        s.columns = [str(c).strip().lower() for c in s.columns]
        if "game_type" in s.columns:
            s = s.loc[s["game_type"].astype(str).str.upper().eq("REG")].copy()
        required = {"season", "week", "home_team", "away_team", "spread_line", "total_line", "home_score", "away_score"}
        missing = required - set(s.columns)
        if missing:
            raise RuntimeError(f"{season} schedule missing required columns: {sorted(missing)}")
        for _, r in s.iterrows():
            home_score = num(pd.Series([r.get("home_score")])).iloc[0]
            away_score = num(pd.Series([r.get("away_score")])).iloc[0]
            if pd.isna(home_score) or pd.isna(away_score):
                continue  # not yet played
            spread_line = num(pd.Series([r.get("spread_line")])).iloc[0]
            total_line = num(pd.Series([r.get("total_line")])).iloc[0]
            if pd.isna(spread_line) or pd.isna(total_line):
                continue
            rows.append({
                "season": int(r["season"]), "week": int(r["week"]),
                "game_id": r.get("game_id"),
                "home_team": canon_team(r["home_team"]), "away_team": canon_team(r["away_team"]),
                # nflverse convention (verified against real 2023 results,
                # e.g. DAL home spread_line=+17.5, won 49-17): POSITIVE means
                # the home team is favored by that many points.
                "predicted_margin_home": float(spread_line),
                "predicted_total": float(total_line),
                "actual_margin_home": float(home_score - away_score),
                "actual_total": float(home_score + away_score),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("no completed games found for requested seasons")
    if out.duplicated(["season", "week", "home_team", "away_team"]).any():
        raise RuntimeError("duplicate game rows")
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def direct_calibration(games: pd.DataFrame) -> pd.DataFrame:
    """Arm 1: pooled + per-season correlation/MAE/bias, no fitting at all."""
    rows = []
    groups = [("ALL_SEASONS", games)] + [
        (str(season), games.loc[games.season.eq(season)]) for season in sorted(games.season.unique())
    ]
    for label, g in groups:
        if g.empty:
            continue
        pred_t = g["predicted_total"].to_numpy(dtype=float)
        act_t = g["actual_total"].to_numpy(dtype=float)
        pred_m = g["predicted_margin_home"].to_numpy(dtype=float)
        act_m = g["actual_margin_home"].to_numpy(dtype=float)
        rows.append({
            "season": label, "n_games": int(len(g)),
            "total_corr": _corr(pred_t, act_t),
            "total_mae": float(np.mean(np.abs(pred_t - act_t))),
            "total_bias": float(np.mean(act_t - pred_t)),
            "margin_corr": _corr(pred_m, act_m),
            "margin_mae": float(np.mean(np.abs(pred_m - act_m))),
            "margin_bias": float(np.mean(act_m - pred_m)),
        })
    return pd.DataFrame(rows)


def _fit_and_apply(train: pd.DataFrame, test: pd.DataFrame, *, pred_col: str, actual_col: str) -> dict:
    if len(train) < MIN_ROWS_PER_FOLD or len(test) < MIN_ROWS_PER_FOLD:
        return {"status": "INSUFFICIENT_ROWS"}
    x_train = train[pred_col].to_numpy(dtype=float)
    y_train = train[actual_col].to_numpy(dtype=float)
    x_test = test[pred_col].to_numpy(dtype=float)
    y_test = test[actual_col].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x_train, y_train, 1)
    recalibrated_pred = intercept + slope * x_test

    raw_mae = float(np.mean(np.abs(x_test - y_test)))
    recalibrated_mae = float(np.mean(np.abs(recalibrated_pred - y_test)))
    return {
        "status": "OK",
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "fitted_slope": float(slope), "fitted_intercept": float(intercept),
        "raw_line_mae": raw_mae,
        "recalibrated_mae": recalibrated_mae,
        "recalibration_mae_improvement": raw_mae - recalibrated_mae,
    }


def out_of_sample_recalibration(games: pd.DataFrame) -> pd.DataFrame:
    """Arm 2: 3-fold rotation, one held-out season at a time. Safe to rotate
    in both directions since there is no cross-season history feature here
    (each game row is independent) -- unlike PR #558's rolling baseline.
    """
    seasons = sorted(games.season.unique())
    rows = []
    for test_season in seasons:
        train = games.loc[~games.season.eq(test_season)]
        test = games.loc[games.season.eq(test_season)]
        for target_label, pred_col, actual_col in [
            ("total", "predicted_total", "actual_total"),
            ("margin", "predicted_margin_home", "actual_margin_home"),
        ]:
            result = _fit_and_apply(train, test, pred_col=pred_col, actual_col=actual_col)
            result["test_season"] = int(test_season)
            result["target"] = target_label
            rows.append(result)
    return pd.DataFrame(rows)


def binned_game_script_accuracy(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Arm 3: pooled, no fitting. Spread-magnitude bins -> actual
    competitiveness + favorite win rate; total bins -> actual scoring level.
    """
    g = games.copy()
    g["abs_spread"] = g["predicted_margin_home"].abs()
    g["spread_bin"] = pd.cut(g["abs_spread"], bins=SPREAD_BINS, labels=SPREAD_BIN_LABELS, right=False)
    g["favorite_won"] = np.sign(g["predicted_margin_home"]) == np.sign(g["actual_margin_home"])
    # Games with predicted_margin_home == 0 (pick'em) have no defined favorite; exclude from favorite-win-rate only.
    pickem = g["predicted_margin_home"].eq(0.0)

    spread_rows = []
    for label in SPREAD_BIN_LABELS:
        bucket = g.loc[g["spread_bin"].eq(label)]
        if bucket.empty:
            continue
        decided = bucket.loc[~pickem.loc[bucket.index]]
        spread_rows.append({
            "spread_bucket": label,
            "n_games": int(len(bucket)),
            "mean_actual_abs_margin": float(bucket["actual_margin_home"].abs().mean()),
            "favorite_win_rate": float(decided["favorite_won"].mean()) if len(decided) else float("nan"),
        })
    spread_table = pd.DataFrame(spread_rows)

    total_edges = TOTAL_BIN_EDGES_DEFAULT
    g["total_bin"] = pd.cut(g["predicted_total"], bins=total_edges, labels=TOTAL_BIN_LABELS, right=False)
    total_rows = []
    for label in TOTAL_BIN_LABELS:
        bucket = g.loc[g["total_bin"].eq(label)]
        if bucket.empty:
            continue
        total_rows.append({
            "total_bucket": label,
            "n_games": int(len(bucket)),
            "mean_actual_total": float(bucket["actual_total"].mean()),
        })
    total_table = pd.DataFrame(total_rows)
    return spread_table, total_table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=str, default="2023,2024,2025")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    if len(seasons) < 2:
        raise RuntimeError("need at least 2 seasons for out-of-sample recalibration")

    games = load_game_outcomes(seasons)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    games.to_csv(a.out_dir / "vegas_line_gamescript_games.csv", index=False)

    direct = direct_calibration(games)
    direct.to_csv(a.out_dir / "vegas_line_calibration_direct.csv", index=False)

    recal = out_of_sample_recalibration(games)
    recal.to_csv(a.out_dir / "vegas_line_calibration_recalibration.csv", index=False)

    spread_table, total_table = binned_game_script_accuracy(games)
    spread_table.to_csv(a.out_dir / "vegas_line_spread_bucket_accuracy.csv", index=False)
    total_table.to_csv(a.out_dir / "vegas_line_total_bucket_accuracy.csv", index=False)

    print("=== VEGAS LINE -> ACTUAL GAME SCRIPT CALIBRATION ===")
    print("\n-- Direct calibration (no fit) --")
    print(direct.to_string(index=False))
    print("\n-- Out-of-sample linear recalibration (3-fold) --")
    print(recal.to_string(index=False))
    print("\n-- Spread-magnitude bucket -> actual competitiveness / favorite win rate --")
    print(spread_table.to_string(index=False))
    print("\n-- Total-line bucket -> actual combined score --")
    print(total_table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
