#!/usr/bin/env python3
"""Simple pregame-projection-vs-Vegas-line directional scoreboard.

This answers a narrower, simpler question than any production EV/fair-probability
selector:

    projection > line  -> hypothetical OVER
    projection < line  -> hypothetical UNDER
    projection == line -> NO_BET
    actual > line      -> OVER actually happened
    actual < line      -> UNDER actually happened
    actual == line     -> PUSH

WIN if the hypothetical side matches what actually happened, LOSS if not.
Sportsbook information is downstream only. The projection trace may carry an
explicit ``model_authority``/``authority_scope``; when present those fields are
preferred over heuristic lineage inference so an authority-exact benchmark
cannot be mislabeled.
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
    explicit = str(row.get("model_authority", "")).strip()
    if explicit and explicit.lower() not in {"nan", "none"}:
        return explicit
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
    keep = join_cols + [
        "book", "line", "over_odds", "under_odds", "player",
        "source_line_definition", "source_dataset",
    ]
    selected = selected[[c for c in keep if c in selected]].copy()

    proj_in = proj.drop(columns=[c for c in ("player",) if c in proj.columns])
    z = proj_in.merge(selected, on=join_cols, how="inner")
    if z.empty:
        return z

    z["pregame_projection"] = num(z[proj_col])
    z["vegas_line"] = num(z.line)
    z["projection_line_edge"] = z.pregame_projection - z.vegas_line
    z["actual_result"] = num(z.actual)
    if z[["pregame_projection", "vegas_line", "actual_result"]].isna().any().any():
        raise RuntimeError("directional scoreboard contains non-finite projection, line, or actual")

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
    z["abs_model_error"] = (z.pregame_projection - z.actual_result).abs()
    z["abs_vegas_error"] = (z.vegas_line - z.actual_result).abs()
    z["model_closer_than_vegas"] = z.abs_model_error < z.abs_vegas_error

    cols = [
        "benchmark_arm", "season", "week", "player", "player_clean_key", "team", "opponent", "position",
        "market", "promoted_model_version", "authority_scope", "authority_parity_pass",
        "pregame_projection", "vegas_line", "projection_line_edge", "directional_pick",
        "actual_result", "actual_side", "win_loss_push", "book", "chosen_odds", "unit_result",
        "abs_model_error", "abs_vegas_error", "model_closer_than_vegas",
        "source_line_definition", "source_dataset",
    ]
    return z[[c for c in cols if c in z.columns]].copy()


def _summary_row(g: pd.DataFrame, **labels) -> dict:
    decided = g.loc[g.win_loss_push.isin(["WIN", "LOSS"])]
    out = dict(labels)
    out.update({
        "rows": int(len(g)),
        "decided_bets": int(len(decided)),
        "wins": int(decided.win_loss_push.eq("WIN").sum()),
        "losses": int(decided.win_loss_push.eq("LOSS").sum()),
        "pushes": int(g.win_loss_push.eq("PUSH").sum()),
        "no_bets": int(g.win_loss_push.eq("NO_BET").sum()),
        "win_rate": float(decided.win_loss_push.eq("WIN").mean()) if len(decided) else np.nan,
        "roi_per_unit": float(decided.unit_result.mean()) if len(decided) else np.nan,
        "model_mae": float(g.abs_model_error.mean()) if "abs_model_error" in g.columns and len(g) else np.nan,
        "vegas_mae": float(g.abs_vegas_error.mean()) if "abs_vegas_error" in g.columns and len(g) else np.nan,
        "model_closer_than_vegas_rate": float(g.model_closer_than_vegas.mean()) if "model_closer_than_vegas" in g.columns and len(g) else np.nan,
    })
    return out


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    group_cols = [c for c in ["benchmark_arm", "position", "market", "promoted_model_version"] if c in detail.columns]
    for keys, g in detail.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        labels = dict(zip(group_cols, keys))
        rows.append(_summary_row(g, **labels))

    season_group_cols = [c for c in ["benchmark_arm", "season", "position", "market", "promoted_model_version"] if c in detail.columns]
    for keys, g in detail.groupby(season_group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        labels = dict(zip(season_group_cols, keys))
        labels["summary_scope"] = "SEASON"
        rows.append(_summary_row(g, **labels))

    all_labels = {
        "market": "ALL_MARKETS", "promoted_model_version": "ALL",
        "benchmark_arm": str(detail["benchmark_arm"].iloc[0]) if "benchmark_arm" in detail.columns and detail["benchmark_arm"].nunique() == 1 else "ALL",
        "position": "ALL",
        "summary_scope": "OVERALL",
    }
    rows.append(_summary_row(detail, **all_labels))
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
