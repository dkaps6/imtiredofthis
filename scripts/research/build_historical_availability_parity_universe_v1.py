#!/usr/bin/env python3
"""Build an ACT-only diagnostic universe from the exact historical baseline universe.

No result/stat/snap/sportsbook field is used. The only selector is the same
nflreadpy weekly-roster status field already consumed by historical_inputs.py.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team

SEASON = 2026
WEEKS = (1, 2, 3, 4)
ALLOWED = {"ACT", "INA"}


def _to_pandas(obj) -> pd.DataFrame:
    return obj.to_pandas() if hasattr(obj, "to_pandas") else pd.DataFrame(obj)


def _join_key(value) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def load_weekly_roster_status() -> pd.DataFrame:
    import nflreadpy as nfl
    raw = _to_pandas(nfl.load_rosters_weekly(SEASON))
    x = raw.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {"season", "week", "team", "status", "position"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"weekly roster missing columns: {sorted(missing)}")
    name_col = "full_name" if "full_name" in x.columns else "football_name" if "football_name" in x.columns else None
    if name_col is None:
        raise RuntimeError("weekly roster missing full_name/football_name")
    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x = x.loc[x["season"].eq(SEASON) & x["week"].isin(WEEKS)].copy()
    x["team"] = x["team"].map(canon_team)
    x["status"] = x["status"].astype(str).str.upper().str.strip()
    x["player"] = x[name_col].astype(str).str.strip()
    x["player_join"] = x["player"].map(_join_key)
    x = x.loc[x["status"].isin(ALLOWED)].copy()

    key = ["week", "team", "player_join"]
    conflicts = (
        x.groupby(key)["status"].nunique().reset_index(name="n").loc[lambda d: d["n"] > 1]
    )
    if not conflicts.empty:
        raise RuntimeError(f"conflicting roster statuses: {conflicts.head(20).to_dict('records')}")
    x = x.sort_values(key).drop_duplicates(key, keep="last")
    return x[["season", "week", "team", "player", "player_join", "status", "position"]]


def build(*, baseline_dir: Path, out_dir: Path, status_out: Path, summary_out: Path) -> dict:
    roster = load_weekly_roster_status()
    out_dir.mkdir(parents=True, exist_ok=True)
    status_rows = []
    week_meta = {}

    for week in WEEKS:
        src = baseline_dir / f"{SEASON}_week_{week:02d}.csv"
        if not src.exists() or src.stat().st_size <= 0:
            raise RuntimeError(f"missing baseline universe: {src}")
        u = pd.read_csv(src, low_memory=False)
        u.columns = [str(c).strip().lower() for c in u.columns]
        if not {"player", "team", "season", "week"}.issubset(u.columns):
            raise RuntimeError(f"baseline universe W{week} missing identity columns")
        u["team"] = u["team"].map(canon_team)
        u["player_join"] = u["player"].map(_join_key)
        r = roster.loc[roster["week"].eq(week), ["week", "team", "player_join", "status", "position"]].copy()
        merged = u.merge(r, on=["week", "team", "player_join"], how="left", validate="one_to_one", suffixes=("", "_roster"))
        if merged["status"].isna().any():
            bad = merged.loc[merged["status"].isna(), ["week", "team", "player"]].head(30)
            raise RuntimeError(f"baseline universe rows missing exact roster status W{week}: {bad.to_dict('records')}")
        if not merged["status"].isin(ALLOWED).all():
            raise RuntimeError(f"unexpected baseline roster status W{week}: {merged['status'].value_counts().to_dict()}")

        status_rows.append(
            merged[["season", "week", "team", "player", "player_join", "status"]].copy()
        )
        act = merged.loc[merged["status"].eq("ACT")].copy()
        removed = merged.loc[merged["status"].eq("INA")].copy()
        preserve_cols = [c for c in u.columns if c != "player_join"]
        act[preserve_cols].to_csv(out_dir / f"{SEASON}_week_{week:02d}.csv", index=False)

        week_meta[str(week)] = {
            "baseline_rows": int(len(merged)),
            "act_rows": int(len(act)),
            "ina_rows_removed": int(len(removed)),
            "baseline_teams": int(merged["team"].nunique()),
            "act_teams": int(act["team"].nunique()),
        }
        if act["team"].nunique() != merged["team"].nunique():
            missing_teams = sorted(set(merged["team"]) - set(act["team"]))
            raise RuntimeError(f"ACT-only universe lost all players for teams W{week}: {missing_teams}")

    status_map = pd.concat(status_rows, ignore_index=True, sort=False)
    key = ["season", "week", "team", "player_join"]
    if status_map.duplicated(key).any():
        raise RuntimeError("status map duplicate identity")
    status_out.parent.mkdir(parents=True, exist_ok=True)
    status_map.to_csv(status_out, index=False)

    summary = {
        "version": "HISTORICAL_AVAILABILITY_PARITY_UNIVERSE_V1",
        "season": SEASON,
        "weeks": list(WEEKS),
        "status_source": "nflreadpy.load_rosters_weekly",
        "selection_field": "status",
        "baseline_allowed_statuses": ["ACT", "INA"],
        "act_only_status": "ACT",
        "sportsbook_inputs_used": False,
        "outcome_inputs_used_for_selection": False,
        "weeks_meta": week_meta,
        "baseline_rows": int(len(status_map)),
        "act_rows": int(status_map["status"].eq("ACT").sum()),
        "ina_rows_removed": int(status_map["status"].eq("INA").sum()),
    }
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--status-out", type=Path, required=True)
    p.add_argument("--summary-out", type=Path, required=True)
    a = p.parse_args()
    build(
        baseline_dir=a.baseline_dir,
        out_dir=a.out_dir,
        status_out=a.status_out,
        summary_out=a.summary_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
