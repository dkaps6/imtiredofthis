#!/usr/bin/env python3
"""Corrected STACK6D source audit wrapper.

The frozen gates live in audit_rb_stack6d_availability_source.py. This wrapper fixes
only cross-season source filtering: regular-season eligibility is inherited from the
canonical schedule team-game keys rather than each source table's drifting
season_type field.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from scripts.backtest import audit_rb_stack6d_availability_source as base
from scripts._opponent_map import canon_team


def _schedule_keys(team_games: pd.DataFrame) -> pd.DataFrame:
    return team_games[
        ["season", "week", "team", "game_key", "kickoff_utc", "decision_deadline_utc"]
    ].drop_duplicates(["season", "week", "team"])


def prepare_rosters_corrected(x: pd.DataFrame, team_games: pd.DataFrame):
    z = x.copy()
    required = ["season", "week", "team", "position", "status"]
    missing = [c for c in required if c not in z.columns]
    if missing:
        raise RuntimeError(f"STACK6D weekly rosters missing {missing}")
    id_col = base.choose(z, ["gsis_id", "player_id"])
    if not id_col:
        raise RuntimeError("STACK6D weekly rosters missing GSIS/player identity")

    z["season"] = base.num(z.season)
    z["week"] = base.num(z.week)
    z = z.loc[z.season.notna() & z.week.notna()].copy()
    z["season"] = z.season.astype(int)
    z["week"] = z.week.astype(int)
    z["team"] = z.team.map(canon_team)
    z["position"] = z.position.astype(str).str.upper().str.strip()
    z["status"] = z.status.map(base.clean_status)
    z["player_id"] = z[id_col].map(base.clean_id)
    z = z.loc[z.position.isin(["RB", "FB"]) & z.player_id.ne("")].copy()
    z = z.merge(_schedule_keys(team_games), on=["season", "week", "team"], how="inner", validate="many_to_one")

    timing_cols = []
    for c in x.columns:
        lc = str(c).lower()
        if any(tok in lc for tok in ["modified", "updated", "timestamp", "transaction_date", "status_date"]):
            timing_cols.append(c)
    return z, timing_cols


def prepare_injuries_corrected(x: pd.DataFrame, team_games: pd.DataFrame):
    z = x.copy()
    required = ["season", "week", "team", "position", "report_status", "date_modified"]
    missing = [c for c in required if c not in z.columns]
    if missing:
        raise RuntimeError(f"STACK6D injuries missing {missing}")
    id_col = base.choose(z, ["gsis_id", "player_id"])
    if not id_col:
        raise RuntimeError("STACK6D injuries missing GSIS/player identity")

    z["season"] = base.num(z.season)
    z["week"] = base.num(z.week)
    z = z.loc[z.season.notna() & z.week.notna()].copy()
    z["season"] = z.season.astype(int)
    z["week"] = z.week.astype(int)
    z["team"] = z.team.map(canon_team)
    z["position"] = z.position.astype(str).str.upper().str.strip()
    z["player_id"] = z[id_col].map(base.clean_id)
    z["report_status_norm"] = z.report_status.map(base.clean_status)
    z = z.loc[z.position.isin(["RB", "FB"]) & z.player_id.ne("")].copy()
    z = z.merge(_schedule_keys(team_games), on=["season", "week", "team"], how="inner", validate="many_to_one")

    raw = z.date_modified.astype(str)
    z["date_modified_missing"] = z.date_modified.isna() | raw.str.strip().str.lower().isin(["", "nan", "none", "<na>"])
    z["date_modified_has_clock"] = raw.str.contains(r"\d{1,2}:\d{2}", regex=True, na=False)
    z["date_modified_explicit_tz"] = base.explicit_timezone_mask(raw)
    z["date_modified_utc"] = pd.to_datetime(z.date_modified, utc=True, errors="coerce")
    z["date_modified_parseable"] = z.date_modified_utc.notna()
    z["date_modified_unparseable_nonmissing"] = (~z.date_modified_missing) & (~z.date_modified_parseable)
    z["timestamp_provenance_usable"] = (
        z.date_modified_parseable & z.date_modified_has_clock & z.date_modified_explicit_tz
    )
    z["modified_by_deadline"] = (
        z.timestamp_provenance_usable
        & z.decision_deadline_utc.notna()
        & z.date_modified_utc.le(z.decision_deadline_utc)
    )
    return z


def participation_benchmark_corrected(x: pd.DataFrame, team_games: pd.DataFrame):
    z = x.copy()
    if "season" not in z.columns or "week" not in z.columns:
        gid = base.choose(z, ["nflverse_game_id", "game_id", "old_game_id"])
        if gid:
            parts = z[gid].astype(str).str.split("_", expand=True)
            if "season" not in z.columns:
                z["season"] = base.num(parts[0])
            if "week" not in z.columns and parts.shape[1] > 1:
                z["week"] = base.num(parts[1])

    z["season"] = base.num(z.season)
    z["week"] = base.num(z.week)
    z = z.loc[z.season.notna() & z.week.notna()].copy()
    z["season"] = z.season.astype(int)
    z["week"] = z.week.astype(int)
    team_col = base.choose(z, ["posteam", "possession_team", "team"])
    if not team_col:
        raise RuntimeError("STACK6D participation missing possession team")
    z["team"] = z[team_col].map(canon_team)

    valid = _schedule_keys(team_games)[["season", "week", "team"]]
    z = z.merge(valid, on=["season", "week", "team"], how="inner", validate="many_to_one")

    player_col = base.choose(z, ["offense_players"])
    pos_col = base.choose(z, ["offense_positions"])
    if not player_col or not pos_col:
        raise RuntimeError("STACK6D participation missing offense arrays")

    present = set()
    game_rows = set()
    aligned = 0
    eligible = 0
    for _, r in z.iterrows():
        if not r.team:
            continue
        game_rows.add((int(r.season), int(r.week), str(r.team)))
        ids = base.split_list(r[player_col])
        pos = base.split_list(r[pos_col])
        if ids or pos:
            eligible += 1
        if not ids or len(ids) != len(pos):
            continue
        aligned += 1
        for pid, pp in zip(ids, pos):
            if str(pp).upper() in {"RB", "FB"} and base.clean_id(pid):
                present.add((int(r.season), int(r.week), str(r.team), base.clean_id(pid)))
    return present, game_rows, aligned / max(eligible, 1)


def _arg_value(flag: str) -> str | None:
    try:
        i = sys.argv.index(flag)
    except ValueError:
        return None
    return sys.argv[i + 1] if i + 1 < len(sys.argv) else None


def emit_season_timing(out_dir: Path) -> None:
    p = out_dir / "stack6d_rbfb_injury_casebook.csv"
    if not p.exists():
        return
    z = pd.read_csv(p, low_memory=False)
    if z.empty or "season" not in z.columns:
        return
    rows = []
    for season, g in z.groupby("season", dropna=False):
        status = g.get("report_status_norm", pd.Series("", index=g.index)).fillna("").astype(str).str.upper()
        out = status.eq("OUT")
        rows.append(
            {
                "season": season,
                "rbfb_injury_rows": int(len(g)),
                "date_modified_nonmissing_rate": float((~g.get("date_modified_missing", pd.Series(True, index=g.index)).astype(bool)).mean()),
                "date_modified_parse_rate": float(g.get("date_modified_parseable", pd.Series(False, index=g.index)).astype(bool).mean()),
                "date_modified_unparseable_nonmissing_rate": float(g.get("date_modified_unparseable_nonmissing", pd.Series(False, index=g.index)).astype(bool).mean()),
                "usable_explicit_timing_rate": float(g.get("timestamp_provenance_usable", pd.Series(False, index=g.index)).astype(bool).mean()),
                "out_rows": int(out.sum()),
                "out_rows_usable_timestamp": int((out & g.get("timestamp_provenance_usable", pd.Series(False, index=g.index)).astype(bool)).sum()),
                "predeadline_out_rows": int((out & g.get("modified_by_deadline", pd.Series(False, index=g.index)).astype(bool)).sum()),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "stack6d_injury_timing_by_season.csv", index=False)


def main() -> int:
    base.prepare_rosters = prepare_rosters_corrected
    base.prepare_injuries = prepare_injuries_corrected
    base.participation_benchmark = participation_benchmark_corrected
    rc = base.main()
    out = _arg_value("--out-dir")
    if out:
        emit_season_timing(Path(out))
    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
