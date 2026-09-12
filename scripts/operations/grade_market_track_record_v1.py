#!/usr/bin/env python3
"""Grade the archived market track record against actual results.

Reads boards written by ``archive_priced_board_v1.py`` under
``data/market_track_record/boards/`` and, for weeks whose games have already
been played, joins the model's own pregame projection and the captured
sportsbook line/odds to the real outcome. The sportsbook line is graded
exactly the same way the model is; it is a benchmark only and was never an
input to the projection that produced these rows.

This is the forward, live-market half of the market-certification gap: no
historical player-prop odds purchase is required, because every row here was
already fetched (and paid for, when live odds were enabled) by a normal Full
Slate production run. This script only adds grading on top of data that
already exists.

Only markets with a direct actual-stat column are graded (anytime_td remains
research-only per project policy and is skipped here, matching production).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BOARDS_DIR = ROOT / "data" / "market_track_record" / "boards"
GRADED_DIR = ROOT / "data" / "market_track_record" / "graded"
ALL_TIME_SUMMARY = ROOT / "data" / "market_track_record" / "summary_all_time.csv"

MARKET_STAT_COLUMNS: dict[str, list[str]] = {
    "pass_yards": ["passing_yards"],
    "rush_yards": ["rushing_yards"],
    "rec_yards": ["receiving_yards"],
    "receptions": ["receptions"],
    "rush_rec_yards": ["rushing_yards", "receiving_yards"],
}


def num(v):
    return pd.to_numeric(v, errors="coerce")


def american_profit(odds) -> float:
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if not np.isfinite(o) or o == 0:
        return np.nan
    return o / 100.0 if o > 0 else 100.0 / abs(o)


def outcome_side(actual: float, line: float) -> str:
    if actual > line:
        return "OVER"
    if actual < line:
        return "UNDER"
    return "PUSH"


def model_side(proj: float, line: float) -> str:
    if proj > line:
        return "OVER"
    if proj < line:
        return "UNDER"
    return "NO_BET"


def edge_bucket(v: float) -> str:
    if not np.isfinite(v):
        return "missing"
    if v < 2:
        return "0-2"
    if v < 5:
        return "2-5"
    if v < 10:
        return "5-10"
    if v < 20:
        return "10-20"
    return "20+"


def _to_pandas(obj) -> pd.DataFrame:
    return obj.to_pandas() if hasattr(obj, "to_pandas") else pd.DataFrame(obj)


def load_boards(season: int, weeks: list[int] | None) -> pd.DataFrame:
    if not BOARDS_DIR.exists():
        return pd.DataFrame()
    paths = sorted(BOARDS_DIR.glob(f"{int(season)}_wk*.csv"))
    if weeks:
        wanted = {f"{int(season)}_wk{int(w):02d}.csv" for w in weeks}
        paths = [p for p in paths if p.name in wanted]
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True, sort=False)


def select_model_bet(board: pd.DataFrame) -> pd.DataFrame:
    """Collapse the archived OVER/UNDER row pair into one graded bet per
    (season, week, event_id, player, market): the side the model actually
    picked, keeping that row's own captured odds."""
    if board.empty:
        return board.copy()
    b = board.copy()
    b["model_pick_side"] = [
        model_side(p, l) for p, l in zip(num(b.model_proj), num(b.vegas_line))
    ]
    b = b.loc[b.side.astype(str).str.upper().eq(b.model_pick_side)].copy()
    key = [c for c in ["season", "week", "event_id", "player", "market"] if c in b.columns]
    return b.drop_duplicates(subset=key, keep="last")


def load_actual_stats(season: int, weeks: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    from scripts.player_form_v2 import _normalize_weekly

    raw = nfl.load_player_stats(seasons=[int(season)], summary_level="week")
    logs = _normalize_weekly(_to_pandas(raw), int(season))
    logs = logs.loc[logs.week.isin(weeks)].copy() if weeks else logs
    return logs


def match_bets_to_actuals(bets: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Join selected model bets to actual results for every recognized market.

    Pure/deterministic: no file or network I/O. `bets` are rows already
    reduced to the model's picked side (see `select_model_bet`); `actual` is
    a leakage-safe weekly player-stat frame with at least
    (season, week, team, player_clean_key, <stat columns>).
    """
    detail_parts = []
    for market, stat_cols in MARKET_STAT_COLUMNS.items():
        m = bets.loc[bets.market.eq(market)].copy()
        if m.empty:
            continue
        a = actual.copy()
        a["actual"] = sum(num(a[c]) for c in stat_cols if c in a.columns)
        a = a[["season", "week", "team", "player_clean_key", "actual"]].drop_duplicates(
            ["season", "week", "team", "player_clean_key"]
        )
        d = m.merge(a, on=["season", "week", "team", "player_clean_key"], how="left")
        d["has_actual"] = d.actual.notna()
        detail_parts.append(d)
    if not detail_parts:
        return pd.DataFrame()
    return pd.concat(detail_parts, ignore_index=True, sort=False)


def grade_matched_rows(detail: pd.DataFrame, *, season: int, present_weeks: list[int]) -> tuple[pd.DataFrame, dict]:
    """Compute per-row grading and the summary dict for already-matched rows.

    Pure/deterministic: no file or network I/O.
    """
    graded = detail.loc[detail.has_actual].copy()
    if graded.empty:
        return graded, {
            "status": "matched_zero_rows_to_actual_results",
            "season": season,
            "weeks": present_weeks,
            "archived_bet_rows": int(len(detail)),
        }

    graded["actual"] = num(graded.actual)
    graded["vegas_line"] = num(graded.vegas_line)
    graded["model_proj"] = num(graded.model_proj)
    graded["model_error"] = graded.model_proj - graded.actual
    graded["vegas_error"] = graded.vegas_line - graded.actual
    graded["model_closer_than_vegas"] = graded.model_error.abs() < graded.vegas_error.abs()
    graded["actual_side"] = [
        outcome_side(a, l) for a, l in zip(graded.actual, graded.vegas_line)
    ]
    graded["bet_result"] = np.select(
        [graded.actual_side.eq("PUSH"), graded.model_pick_side.eq(graded.actual_side)],
        ["PUSH", "WIN"],
        default="LOSS",
    )
    graded["unit_result"] = np.where(
        graded.bet_result.eq("WIN"),
        [american_profit(o) for o in graded.vegas_odds],
        np.where(graded.bet_result.eq("LOSS"), -1.0, 0.0),
    )
    graded["abs_edge"] = (graded.model_proj - graded.vegas_line).abs()
    graded["edge_bucket"] = graded.abs_edge.map(edge_bucket)

    decided = graded.loc[graded.bet_result.isin(["WIN", "LOSS"])]
    summary = {
        "status": "graded",
        "season": int(season),
        "weeks_graded": ",".join(str(w) for w in present_weeks),
        "archived_bet_rows": int(len(detail)),
        "matched_to_actual_rows": int(len(graded)),
        "decided_bets": int(len(decided)),
        "wins": int(decided.bet_result.eq("WIN").sum()),
        "losses": int(decided.bet_result.eq("LOSS").sum()),
        "win_rate": float(decided.bet_result.eq("WIN").mean()) if len(decided) else np.nan,
        "units": float(decided.unit_result.sum()) if len(decided) else np.nan,
        "roi_per_unit": float(decided.unit_result.mean()) if len(decided) else np.nan,
        "model_mae": float(graded.model_error.abs().mean()),
        "vegas_mae": float(graded.vegas_error.abs().mean()),
        "model_closer_than_vegas_rate": float(graded.model_closer_than_vegas.mean()),
    }
    return graded, summary


def grade(season: int, weeks: list[int] | None) -> dict:
    board = load_boards(season, weeks)
    if board.empty:
        return {"status": "no_archived_board_rows", "season": season, "weeks": weeks}

    present_weeks = sorted(set(num(board.week).dropna().astype(int)))
    actual = load_actual_stats(season, present_weeks)
    if actual.empty:
        return {"status": "no_actual_stats_available_yet", "season": season, "weeks": present_weeks}

    bets = select_model_bet(board)
    if bets.empty:
        return {"status": "no_graded_bet_rows_after_side_selection", "season": season}

    detail = match_bets_to_actuals(bets, actual)
    if detail.empty:
        return {"status": "no_recognized_gradeable_markets", "season": season}

    graded, summary = grade_matched_rows(detail, season=season, present_weeks=present_weeks)
    if summary["status"] != "graded":
        return summary

    GRADED_DIR.mkdir(parents=True, exist_ok=True)
    week_tag = "_".join(str(w) for w in present_weeks)
    detail_path = GRADED_DIR / f"{season}_wk{week_tag}_detail.csv"
    graded.to_csv(detail_path, index=False)

    summary_df = pd.DataFrame([summary])
    all_time = (
        pd.read_csv(ALL_TIME_SUMMARY)
        if ALL_TIME_SUMMARY.exists()
        else pd.DataFrame(columns=summary_df.columns)
    )
    all_time = pd.concat([all_time, summary_df], ignore_index=True, sort=False)
    all_time = all_time.drop_duplicates(subset=["season", "weeks_graded"], keep="last")
    all_time.to_csv(ALL_TIME_SUMMARY, index=False)

    return {"status": "graded", "detail_path": str(detail_path), **summary}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--weeks", default="", help="comma-separated weeks; omit to grade every archived week")
    args = ap.parse_args()
    weeks = [int(w) for w in args.weeks.split(",") if w.strip()] or None

    result = grade(args.season, weeks)
    print("=== MARKET TRACK RECORD GRADING ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
