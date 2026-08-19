#!/usr/bin/env python3
"""Build leakage-safe historical game weather context for walk-forward backtests.

Historical *observed* kickoff weather is not used here because it is only known
at/after kickoff and would overstate what a pregame projection knew.  This first
weather recovery layer therefore uses only deterministic venue architecture:
controlled/non-outdoor venues are weather-neutral, while outdoor games remain
explicitly unavailable until a trustworthy archived pregame forecast source is
added.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.stadium_locations import STADIUM_LOCATION

SCHEMA = [
    "season", "week", "game_id", "home", "away", "stadium", "outdoor",
    "controlled_environment", "temp_f", "wind_mph", "precip_flag",
    "forecast_ok", "weather_source",
]


def build_historical_weather(schedule_history: pd.DataFrame) -> pd.DataFrame:
    x = schedule_history.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {"season", "week", "team", "opponent", "home_away"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"schedule history missing weather identity columns: {sorted(missing)}")
    if "game_id" not in x.columns:
        x["game_id"] = ""

    home_rows = x.loc[x["home_away"].astype(str).str.lower().eq("home")].copy()
    if home_rows.empty:
        raise RuntimeError("schedule history contains no home rows for weather reconstruction")

    rows: list[dict] = []
    for _, g in home_rows.iterrows():
        home = canon_team(g.get("team"))
        away = canon_team(g.get("opponent"))
        if not home or not away:
            raise RuntimeError("historical weather encountered unresolved team identity")
        meta = STADIUM_LOCATION.get(home)
        if meta is None:
            raise RuntimeError(f"historical weather has no stadium metadata for home team {home}")

        outdoor = bool(meta.get("outdoor", True))
        controlled = not outdoor
        # Do not invent an outdoor forecast. For controlled venues the external
        # weather impact is deterministically neutral from venue architecture.
        rows.append({
            "season": int(g["season"]),
            "week": int(g["week"]),
            "game_id": str(g.get("game_id", "") or ""),
            "home": home,
            "away": away,
            "stadium": str(meta.get("stadium", "") or ""),
            "outdoor": int(outdoor),
            "controlled_environment": int(controlled),
            "temp_f": np.nan,
            "wind_mph": 0.0 if controlled else np.nan,
            "precip_flag": 0 if controlled else np.nan,
            "forecast_ok": 1 if controlled else 0,
            "weather_source": "venue_architecture_neutral" if controlled else "archived_pregame_forecast_unavailable",
        })

    out = pd.DataFrame(rows, columns=SCHEMA)
    if out.empty:
        raise RuntimeError("historical weather builder produced zero rows")
    keys = ["season", "week", "game_id"] if out["game_id"].astype(str).str.len().gt(0).all() else ["season", "week", "home", "away"]
    if out.duplicated(keys).any():
        raise RuntimeError("historical weather contains duplicate game rows")
    return out.sort_values(["season", "week", "home", "away"]).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--schedule", type=Path, default=Path("data/backtests/schedule_history.csv"))
    p.add_argument("--out", type=Path, default=Path("data/backtests/weather_history.csv"))
    args = p.parse_args()
    if not args.schedule.exists() or args.schedule.stat().st_size == 0:
        raise RuntimeError(f"missing historical schedule: {args.schedule}")
    out = build_historical_weather(pd.read_csv(args.schedule))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    controlled = int(out["controlled_environment"].sum())
    print(f"[historical_weather] wrote {len(out)} games -> {args.out}; controlled={controlled} outdoor_unavailable={len(out)-controlled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
