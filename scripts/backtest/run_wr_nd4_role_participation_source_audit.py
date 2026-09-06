#!/usr/bin/env python3
"""Mechanical runner for the frozen WR-ND4 source audit.

The original ND4 evaluator expects target-game dates on the reconstructed
schedule frame. The canonical historical schedule artifact intentionally stores
only season/week/team/opponent, so this runner restores game dates from the
same public nflverse schedule source already used elsewhere in the repository.

No ND4 source thresholds, source qualification rules, casebook rows, or model
logic are changed here.
"""
from __future__ import annotations

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest import audit_wr_nd4_role_participation_sources as nd4


def _resolved_schedule_team_dates(schedule: pd.DataFrame) -> pd.DataFrame:
    import nflreadpy as nfl

    s = schedule.copy()
    s.columns = [str(c).strip().lower() for c in s.columns]
    seasons = sorted(
        int(v)
        for v in pd.to_numeric(s.get("season"), errors="coerce").dropna().unique().tolist()
    )
    if not seasons:
        raise RuntimeError("WR-ND4 mechanical schedule repair found no seasons")

    rows: list[dict] = []
    for season in seasons:
        raw = nfl.load_schedules(int(season))
        q = raw.to_pandas() if hasattr(raw, "to_pandas") else pd.DataFrame(raw)
        q.columns = [str(c).strip().lower() for c in q.columns]
        if "game_type" in q.columns:
            reg = q["game_type"].fillna("").astype(str).str.upper().eq("REG")
            if reg.any():
                q = q.loc[reg].copy()
        date_col = next((c for c in ["gameday", "game_date", "date"] if c in q.columns), None)
        away_col = "away_team" if "away_team" in q.columns else "away"
        home_col = "home_team" if "home_team" in q.columns else "home"
        if date_col is None or away_col not in q.columns or home_col not in q.columns or "week" not in q.columns:
            raise RuntimeError(
                f"nflverse schedule missing required date/team fields for {season}: {list(q.columns)}"
            )
        q["game_date"] = pd.to_datetime(q[date_col], errors="coerce").dt.date
        q["week"] = pd.to_numeric(q["week"], errors="coerce")
        for _, r in q.loc[q["game_date"].notna() & q["week"].notna()].iterrows():
            for team in (r[away_col], r[home_col]):
                rows.append(
                    {
                        "season": int(season),
                        "week": int(r["week"]),
                        "team": canon_team(team),
                        "game_date": r["game_date"],
                    }
                )

    out = pd.DataFrame(rows).drop_duplicates(["season", "week", "team"], keep="last")
    required = s[["season", "week", "team"]].copy()
    required["season"] = pd.to_numeric(required["season"], errors="coerce")
    required["week"] = pd.to_numeric(required["week"], errors="coerce")
    required["team"] = required["team"].fillna("").map(canon_team)
    required = required.dropna(subset=["season", "week"]).drop_duplicates()
    check = required.merge(out, on=["season", "week", "team"], how="left")
    if check["game_date"].isna().any():
        bad = check.loc[check["game_date"].isna(), ["season", "week", "team"]].head(20)
        raise RuntimeError(f"WR-ND4 schedule-date repair incomplete:\n{bad.to_string(index=False)}")
    return out


def main() -> int:
    nd4._schedule_team_dates = _resolved_schedule_team_dates
    return nd4.main()


if __name__ == "__main__":
    raise SystemExit(main())
