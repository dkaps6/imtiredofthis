"""Build historical weekly injury snapshots for leakage-safe backtests.

Uses nflverse/nflreadpy weekly injury reports. Rows are preserved at their
season/week grain so the walk-forward runner can select only the target week's
pregame report; prior-week reports are never silently substituted.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team


def _to_pandas(value) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    if hasattr(value, "to_dicts"):
        return pd.DataFrame(value.to_dicts())
    return pd.DataFrame(value)


def _first(df: pd.DataFrame, names: list[str], default=pd.NA) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)


def normalize_historical_injuries(raw: pd.DataFrame) -> pd.DataFrame:
    cols = ["player", "team", "season", "week", "status", "practice_status", "body_part", "designation", "source", "report_available"]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=cols)
    x = raw.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    season_col = next((c for c in ("season",) if c in x.columns), None)
    week_col = next((c for c in ("week", "report_week") if c in x.columns), None)
    if season_col is None or week_col is None:
        raise RuntimeError("historical nflverse injuries require season/week")
    out = pd.DataFrame(index=x.index)
    out["player"] = _first(x, ["full_name", "player_name", "player", "name"]).astype("string").str.strip()
    out["team"] = _first(x, ["team", "team_abbr", "team_abbreviation", "club"]).map(canon_team)
    out["season"] = pd.to_numeric(x[season_col], errors="coerce").astype("Int64")
    out["week"] = pd.to_numeric(x[week_col], errors="coerce").astype("Int64")
    out["status"] = _first(x, ["report_status", "game_status", "status"]).astype("string").str.strip()
    out["practice_status"] = _first(x, ["practice_status", "practice_participation"]).astype("string").str.strip()
    out["body_part"] = _first(x, ["report_primary_injury", "primary_injury", "injury", "injury_type"]).astype("string").str.strip()
    out["designation"] = out["status"]
    out["source"] = "nflverse_historical"
    out["report_available"] = 1
    out = out.loc[out["season"].notna() & out["week"].notna() & out["player"].notna() & out["player"].ne("") & out["team"].notna() & out["team"].ne("")].copy()
    return out[cols].drop_duplicates(["season", "week", "player", "team"], keep="last").sort_values(["season", "week", "team", "player"]).reset_index(drop=True)


def load_historical_injuries(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl
    raw = nfl.load_injuries(seasons=sorted(set(int(s) for s in seasons)))
    return normalize_historical_injuries(_to_pandas(raw))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seasons", default="2024,2025")
    p.add_argument("--out", type=Path, default=Path("data/backtests/injuries_history.csv"))
    args = p.parse_args()
    seasons = [int(v.strip()) for v in args.seasons.split(",") if v.strip()]
    out = load_historical_injuries(seasons)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[historical_injuries] rows={len(out)} seasons={seasons} -> {args.out}")
    if len(out):
        print(out.groupby(["season", "week"]).size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
