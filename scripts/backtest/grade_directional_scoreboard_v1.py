#!/usr/bin/env python3
"""Simple pregame-projection-vs-Vegas-line directional scoreboard.

This answers a narrower, simpler question than grade_full_stack_vegas_benchmark_v1.py's
production-style EV/fair-probability/STRONG-LEAN side selection:

    projection > line  -> hypothetical OVER
    projection < line  -> hypothetical UNDER
    projection == line -> NO_BET
    then: actual > line -> OVER actually happened, actual < line -> UNDER,
    actual == line -> PUSH
    WIN if the hypothetical side matches what actually happened, LOSS if not,
    PUSH/NO_BET otherwise.

No probability, no EV, no odds-implied edge -- just "would our pregame number
have been on the right side of the closing line." Reuses the exact same
identity-checked join (assert_benchmark_identity, select_one_book_row) as the
production-style grader so both scoreboards are computed from the same rows
and are directly comparable, not two different cohorts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.benchmark_identity_v1 import assert_benchmark_identity
from scripts.backtest.grade_historical_market_vegas_benchmark_v1 import select_one_book_row
from scripts.operations.grade_market_track_record_v1 import american_profit, num, outcome_side


def _model_version(row: pd.Series) -> str:
    market = str(row.get("market", "")).lower()
    if market == "pass_yards" and int(row.get("qb_m89_synthesis_applied", 0) or 0) == 1:
        return "QB_M89_SYNTHESIS_V1"
    if market in {"rec_yards", "receptions", "rush_rec_yards"} and bool(row.get("wrte_authorized_treatment", False)):
        return str(row.get("wrte_route", "WR_R15_TE_R5P"))
    return "BASE_ENSEMBLE"


def score(proj: pd.DataFrame, props: pd.DataFrame, *, proj_col: str = "ensemble_proj") -> pd.DataFrame:
    assert_benchmark_identity(
        proj, label="directional scoreboard projection input",
        require_team=True, require_opponent=("opponent" in {str(c).strip().lower() for c in proj.columns}),
    )
    assert_benchmark_identity(props, label="directional scoreboard props input", require_team=False, require_opponent=False)

    selected = select_one_book_row(props)
    join_cols = ["game_id", "player_clean_key", "market"]
    keep = join_cols + ["book", "line", "over_odds", "under_odds", "player"]
    selected = selected[[c for c in keep if c in selected]].copy()

    proj_in = proj.drop(columns=[c for c in ("player",) if c in proj.columns])
    z = proj_in.merge(selected, on=join_cols, how="inner")
    if z.empty:
        return z

    z["pregame_projection"] = num(z[proj_col])
    z["vegas_line"] = num(z.line)
    z["projection_line_edge"] = z.pregame_projection - z.vegas_line
    z["actual_result"] = num(z.actual)

    z["directional_pick"] = np.select(
        [z.projection_line_edge.gt(0), z.projection_line_edge.lt(0)],
        ["OVER", "UNDER"], default="NO_BET",
    )
    z["actual_side"] = [outcome_side(a, l) for a, l in zip(z.actual_result, z.vegas_line)]
    z["win_loss_push"] = np.select(
        [z.directional_pick.eq("NO_BET"), z.actual_side.eq("PUSH"), z.directional_pick.eq(z.actual_side)],
        ["NO_BET", "PUSH", "WIN"], default="LOSS",
    )
    z["chosen_odds"] = np.select(
        [z.directional_pick.eq("OVER"), z.directional_pick.eq("UNDER")],
        [num(z.over_odds), num(z.under_odds)], default=np.nan,
    )
    z["unit_result"] = np.where(
        z.win_loss_push.eq("WIN"), [american_profit(o) for o in z.chosen_odds],
        np.where(z.win_loss_push.eq("LOSS"), -1.0, 0.0),
    )
    z["promoted_model_version"] = z.apply(_model_version, axis=1)

    cols = [
        "season", "week", "player", "team", "opponent", "market", "promoted_model_version",
        "pregame_projection", "vegas_line", "projection_line_edge", "directional_pick",
        "actual_result", "actual_side", "win_loss_push", "book", "chosen_odds", "unit_result",
    ]
    return z[[c for c in cols if c in z.columns]].copy()


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for (market, version), g in detail.groupby(["market", "promoted_model_version"], sort=True):
        decided = g.loc[g.win_loss_push.isin(["WIN", "LOSS"])]
        rows.append({
            "market": market, "promoted_model_version": version,
            "rows": int(len(g)),
            "decided_bets": int(len(decided)),
            "wins": int(decided.win_loss_push.eq("WIN").sum()),
            "losses": int(decided.win_loss_push.eq("LOSS").sum()),
            "pushes": int(g.win_loss_push.eq("PUSH").sum()),
            "no_bets": int(g.win_loss_push.eq("NO_BET").sum()),
            "win_rate": float(decided.win_loss_push.eq("WIN").mean()) if len(decided) else np.nan,
            "roi_per_unit": float(decided.unit_result.mean()) if len(decided) else np.nan,
        })
    all_rows = detail.loc[detail.win_loss_push.isin(["WIN", "LOSS"])]
    rows.append({
        "market": "ALL_MARKETS", "promoted_model_version": "ALL",
        "rows": int(len(detail)), "decided_bets": int(len(all_rows)),
        "wins": int(all_rows.win_loss_push.eq("WIN").sum()), "losses": int(all_rows.win_loss_push.eq("LOSS").sum()),
        "pushes": int(detail.win_loss_push.eq("PUSH").sum()), "no_bets": int(detail.win_loss_push.eq("NO_BET").sum()),
        "win_rate": float(all_rows.win_loss_push.eq("WIN").mean()) if len(all_rows) else np.nan,
        "roi_per_unit": float(all_rows.unit_result.mean()) if len(all_rows) else np.nan,
    })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projection-file", action="append", required=True)
    ap.add_argument("--proj-col", default="ensemble_proj")
    ap.add_argument("--props", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    proj = pd.concat([pd.read_csv(Path(p), low_memory=False) for p in a.projection_file], ignore_index=True)
    props = pd.read_csv(a.props, low_memory=False)

    detail = score(proj, props, proj_col=a.proj_col)
    summary = summarize(detail)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(a.out_dir / "directional_scoreboard_detail.csv", index=False)
    summary.to_csv(a.out_dir / "directional_scoreboard_summary.csv", index=False)
    print("=== DIRECTIONAL SCOREBOARD SUMMARY ===")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
