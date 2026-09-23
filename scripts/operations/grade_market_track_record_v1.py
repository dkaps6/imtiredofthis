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
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.operations.quarantine_final_priced_props_v1 import (
    QUARANTINE as FINAL_BOARD_QUARANTINE,
    _load_quarantine_keys,
)
from scripts.repair_live_prop_identity_v1 import _name_keys

ROOT = Path(__file__).resolve().parents[2]
BOARDS_DIR = ROOT / "data" / "market_track_record" / "boards"
GRADED_DIR = ROOT / "data" / "market_track_record" / "graded"
ALL_TIME_SUMMARY = ROOT / "data" / "market_track_record" / "summary_all_time.csv"
PRODUCTION_GATE_EVIDENCE_COMMIT = "8133975f505365234dbdb75ff0fad0c715f68e31"
PRODUCTION_GATE_EVIDENCE_PATH = "data/market_track_record/PRODUCTION_DECISION_GATES_V1.json"

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
    """Bucket absolute projection-to-line gap in native stat units."""
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


def apply_final_board_quarantine(
    board: pd.DataFrame,
    quarantine_path: Path = FINAL_BOARD_QUARANTINE,
) -> pd.DataFrame:
    """Apply the verified final-board publication policy before grading."""
    if board.empty:
        return board.copy()
    required = {"season", "week", "team", "player"}
    missing = required - set(board.columns)
    if missing:
        raise RuntimeError(
            f"archived board missing final-quarantine columns: {sorted(missing)}"
        )
    out = board.copy()
    seasons = pd.to_numeric(out["season"], errors="coerce")
    weeks = pd.to_numeric(out["week"], errors="coerce")
    if seasons.isna().any() or weeks.isna().any():
        raise RuntimeError("archived board contains non-numeric season/week")

    remove = pd.Series(False, index=out.index)
    scope = pd.DataFrame(
        {"season": seasons.astype(int), "week": weeks.astype(int)}, index=out.index
    )
    for (season, week), idx in scope.groupby(["season", "week"]).groups.items():
        quarantine_keys = _load_quarantine_keys(
            quarantine_path, season=int(season), week=int(week)
        )
        if not quarantine_keys:
            continue
        teams = out.loc[idx, "team"].map(canon_team)
        player_keys = out.loc[idx, "player"].map(_name_keys)
        remove.loc[idx] = [
            bool({(team, key) for key in keys} & quarantine_keys)
            for team, keys in zip(teams, player_keys)
        ]
    return out.loc[~remove].copy()


def _load_immutable_production_gate_evidence() -> dict:
    """Load preserved pregame decision gates from their first immutable commit."""
    proc = subprocess.run(
        [
            "git",
            "show",
            f"{PRODUCTION_GATE_EVIDENCE_COMMIT}:{PRODUCTION_GATE_EVIDENCE_PATH}",
        ],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        detail = proc.stderr.strip()
        raise RuntimeError(
            "unable to load immutable production decision gate evidence from "
            f"{PRODUCTION_GATE_EVIDENCE_COMMIT}: {detail}"
        )
    payload = json.loads(proc.stdout)
    if str(payload.get("version", "")) != "PRODUCTION_DECISION_GATES_V1":
        raise RuntimeError("unexpected production decision gate evidence version")
    return payload


def apply_production_decision_gates(
    board: pd.DataFrame,
    *,
    evidence: dict | None = None,
) -> pd.DataFrame:
    """Replay preserved pregame workbook blockers before selecting a wager.

    These are downstream publication/eligibility gates only. They never feed a
    sportsbook line or outcome into a football projection. The evidence is
    pinned to the commit that first preserved it so a later board/edit cannot
    silently rewrite the historical decision contract.
    """
    if board.empty:
        return board.copy()

    required = {
        "season",
        "week",
        "team",
        "player_clean_key",
        "market",
        "source_run_id",
        "source_git_sha",
    }
    missing = required - set(board.columns)
    if missing:
        raise RuntimeError(
            f"archived board missing production-gate columns: {sorted(missing)}"
        )

    gates = (
        evidence
        if evidence is not None
        else _load_immutable_production_gate_evidence()
    )
    week_specs = gates.get("weeks", {})
    out = board.copy()
    seasons = pd.to_numeric(out["season"], errors="coerce")
    weeks = pd.to_numeric(out["week"], errors="coerce")
    if seasons.isna().any() or weeks.isna().any():
        raise RuntimeError("archived board contains non-numeric season/week")

    out["_gate_season"] = seasons.astype(int)
    out["_gate_week"] = weeks.astype(int)
    remove = pd.Series(False, index=out.index)

    for (season, week), idx in out.groupby(
        ["_gate_season", "_gate_week"], sort=False
    ).groups.items():
        spec = week_specs.get(str(int(week)))
        if int(season) != 2026 or spec is None:
            raise RuntimeError(
                f"no preserved production decision gate evidence for {season} week {week}"
            )

        run_ids = set(out.loc[idx, "source_run_id"].astype(str))
        source_shas = set(out.loc[idx, "source_git_sha"].astype(str))
        if run_ids != {str(spec.get("source_run_id", ""))}:
            raise RuntimeError(
                f"Week {week} production-gate source_run_id drift: {sorted(run_ids)}"
            )
        if source_shas != {str(spec.get("source_git_sha", ""))}:
            raise RuntimeError(
                f"Week {week} production-gate source_git_sha drift: {sorted(source_shas)}"
            )

        teams = out.loc[idx, "team"].map(canon_team)
        player_keys = (
            out.loc[idx, "player_clean_key"]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.lower()
        )
        markets = out.loc[idx, "market"].astype(str)

        for rule in spec.get("block_rules", []):
            scope = str(rule.get("scope", "")).strip().upper()
            rule_team = canon_team(rule.get("team", ""))
            if scope == "TEAM_ALL":
                mask = teams.eq(rule_team)
            elif scope == "PLAYER_MARKET":
                mask = (
                    teams.eq(rule_team)
                    & player_keys.eq(
                        str(rule.get("player_clean_key", "")).strip().lower()
                    )
                    & markets.eq(str(rule.get("market", "")))
                )
            else:
                raise RuntimeError(
                    f"Week {week} unknown production decision gate scope {scope!r}"
                )
            remove.loc[idx] = remove.loc[idx] | mask.to_numpy()

    return out.loc[~remove].drop(
        columns=["_gate_season", "_gate_week"], errors="ignore"
    ).copy()


BET_KEY = ["season", "week", "event_id", "player", "market"]


def _ev_roi(probability, odds) -> float:
    """Match the downstream production workbook's EV ROI arithmetic."""
    try:
        p = float(probability)
        o = float(odds)
    except Exception:
        return np.nan
    if not np.isfinite(p) or not np.isfinite(o) or o == 0:
        return np.nan
    profit = o / 100.0 if o > 0 else 100.0 / abs(o)
    return p * profit - (1.0 - p)


def _normalized_book_key(frame: pd.DataFrame) -> pd.Series:
    if "book" in frame.columns:
        book = frame["book"].astype("string").fillna("").str.strip().str.lower()
    else:
        book = pd.Series("", index=frame.index, dtype="string")
    if "book_title" in frame.columns:
        title = (
            frame["book_title"].astype("string").fillna("").str.strip().str.lower()
        )
        book = book.mask(book.eq(""), title)
    return book.mask(book.eq(""), "~missing-book")


def select_model_bet(board: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the deployed Best Snapshot EV decision from an archived board.

    Production does *not* choose a side from mean-vs-line geometry. For every
    real book+line offer it compares OVER and UNDER expected ROI using the
    model's side-specific fair probability and that side's captured American
    price, chooses the higher-EV side, then the Best Snapshot sheet keeps the
    player-market offer with the highest EV. If that best EV is nonpositive,
    production says PASS and there is no bet to grade.

    This function reproduces that deployed decision using only already-captured
    downstream sportsbook data. Sportsbook information never feeds the football
    projection itself.

    Exact-EV cross-book ties are made deterministic for replay. If tied offers
    imply different wagers (side/line/odds), the player-market fails closed
    rather than letting archive row order decide. Ties that represent the same
    wager use the lexicographically smallest normalized provider book key.
    """
    if board.empty:
        return board.copy()

    b = board.copy()
    key = [c for c in BET_KEY if c in b.columns]
    if not key:
        return b.iloc[0:0].copy()

    b["_line"] = num(b.get("vegas_line"))
    b["_fair_prob"] = num(b.get("fair_prob"))
    b["_odds"] = num(b.get("vegas_odds"))
    b["_row_ev"] = [
        _ev_roi(p, o) for p, o in zip(b["_fair_prob"], b["_odds"])
    ]
    b["_side"] = b.get("side", "").astype(str).str.upper().str.strip()
    b["_book_key"] = _normalized_book_key(b)

    # Keep the consensus number as a diagnostic only. It has no role in the
    # deployed betting decision.
    consensus = (
        b.loc[b["_line"].notna()]
        .groupby(key, dropna=False)["_line"]
        .median()
        .rename("consensus_line")
    )
    b = b.merge(consensus, left_on=key, right_index=True, how="left")

    # Use the normalized row-level book identity in the quote key so legacy
    # blank `book` rows from different `book_title` providers never collapse
    # into one synthetic offer.
    quote_key = [
        c for c in (
            "season", "week", "event_id", "player_clean_key", "team",
            "opponent", "source_market",
        ) if c in b.columns
    ] + ["_book_key", "_line"]
    if not quote_key:
        return b.iloc[0:0].copy()

    # Production chooses the higher-EV side at each concrete book+line quote.
    # Its >= tie goes to OVER; preserve that semantic deterministically.
    b["_side_rank"] = b["_side"].map({"OVER": 0, "UNDER": 1}).fillna(9)
    q = b.loc[
        b["_row_ev"].notna() & b["_side"].isin(["OVER", "UNDER"])
    ].copy()
    if q.empty:
        return q.drop(
            columns=["_line", "_fair_prob", "_odds", "_row_ev", "_side",
                     "_book_key", "_side_rank"],
            errors="ignore",
        )
    q = q.sort_values(
        quote_key + ["_row_ev", "_side_rank"],
        ascending=[True] * len(quote_key) + [False, True],
        kind="mergesort",
    )
    offers = q.drop_duplicates(subset=quote_key, keep="first").copy()
    offers["production_best_ev"] = offers["_row_ev"]

    # Best Snapshot keeps the highest-EV concrete offer per player-market.
    max_ev = offers.groupby(key, dropna=False)["production_best_ev"].transform("max")
    candidates = offers.loc[
        np.isclose(
            offers["production_best_ev"],
            max_ev,
            rtol=0.0,
            atol=1e-12,
        )
    ].copy()

    # Exact EV ties must not silently choose different wagers by archive order.
    sig_cols = [c for c in ("side", "vegas_line", "vegas_odds") if c in candidates.columns]
    if sig_cols:
        sig_count = candidates.groupby(key, dropna=False)[sig_cols].transform("nunique").max(axis=1)
        ambiguous_keys = candidates.loc[sig_count.gt(1), key].drop_duplicates()
        if len(ambiguous_keys):
            marker = ambiguous_keys.assign(_ambiguous_offer_tie=True)
            candidates = candidates.merge(marker, on=key, how="left")
            candidates = candidates.loc[candidates["_ambiguous_offer_tie"].isna()].copy()
            candidates = candidates.drop(columns=["_ambiguous_offer_tie"], errors="ignore")

    if candidates.empty:
        return candidates.drop(
            columns=["_line", "_fair_prob", "_odds", "_row_ev", "_side",
                     "_book_key", "_side_rank"],
            errors="ignore",
        )

    candidates = candidates.sort_values(
        key + ["_book_key", "_line", "_odds"],
        kind="mergesort",
        na_position="last",
    )
    out = candidates.drop_duplicates(subset=key, keep="first").copy()

    # Production signals HAS EDGE only for strictly positive best EV.
    out = out.loc[out["production_best_ev"].gt(0)].copy()
    out["model_pick_side"] = out["side"].astype(str).str.upper()
    out["production_decision"] = "BET"
    return out.drop(
        columns=["_line", "_fair_prob", "_odds", "_row_ev", "_side",
                 "_book_key", "_side_rank"],
        errors="ignore",
    )

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
    board = apply_final_board_quarantine(board)
    board = apply_production_decision_gates(board)
    if board.empty:
        return {
            "status": "no_publishable_board_rows_after_final_quarantine",
            "season": season,
            "weeks": weeks,
        }

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
