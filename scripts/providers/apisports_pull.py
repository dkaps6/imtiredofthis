#!/usr/bin/env python3
"""
API-Sports NFL pull (american-football API).
ENV: APISPORTS_KEY (or API_SPORTS_KEY)
Outputs:
  data/apisports_team.csv    (basic team shell)
  data/apisports_player.csv  with: player, team, ypc, ypa  (targets/route_rate may not be available)
Docs: https://api-sports.io/documentation/nfl/v1
"""
import os, sys, time
from pathlib import Path
import requests
import pandas as pd
import numpy as np

BASE = "https://v1.american-football.api-sports.io"

def _headers():
    key = os.getenv("APISPORTS_KEY") or os.getenv("API_SPORTS_KEY")
    if not key:
        return None
    return {"x-apisports-key": key}

def _get(path, params=None):
    h = _headers()
    if not h: 
        raise RuntimeError("APISPORTS_KEY missing")
    r = requests.get(BASE + path, headers=h, params=params or {}, timeout=20)
    r.raise_for_status()
    return r.json()

def main(season: int):
    if not _headers():
        print("[apisports_pull] APISPORTS_KEY missing; skipping.")
        return 0

    # League id: 1 (NFL) per docs
    league = 1

    # Teams
    try:
        tjs = _get("/teams", {"league": league, "season": season})
        teams = []
        for it in tjs.get("response", []):
            abbr = it.get("team", {}).get("code") or it.get("team", {}).get("name")
            teams.append({"team": str(abbr).upper()})
        teams_df = pd.DataFrame(teams).drop_duplicates()
        Path("data").mkdir(exist_ok=True)
        teams_df.to_csv("data/apisports_team.csv", index=False)
        print(f"[apisports_pull] wrote data/apisports_team.csv rows={len(teams_df)}")
    except Exception as e:
        print("[apisports_pull] teams fetch failed:", e)

    # Players + basic stats (paged)
    rows = []
    page = 1
    try:
        while True:
            pjs = _get("/players", {"league": league, "season": season, "page": page})
            resp = pjs.get("response", [])
            if not resp:
                break
            for it in resp:
                teamc = it.get("team", {}).get("code") or it.get("team", {}).get("name")
                player_name = it.get("player", {}).get("name")
                stats = it.get("statistics", [])
                # statistics may come per game or per season; try to aggregate rough ypc/ypa
                rush_att = 0; rush_yds = 0; pass_att = 0; pass_yds = 0
                for st in stats:
                    r = st.get("rushing", {})
                    p = st.get("passing", {})
                    rush_att += int(r.get("attempts", 0) or 0)
                    rush_yds += int(r.get("yards", 0) or 0)
                    pass_att += int(p.get("attempts", 0) or 0)
                    pass_yds += int(p.get("yards", 0) or 0)
                rows.append({
                    "player": player_name,
                    "team": str(teamc).upper(),
                    "route_rate": np.nan,
                    "target_share": np.nan,
                    "rush_share": np.nan,
                    "yprr": np.nan,
                    "ypc": (rush_yds / rush_att) if rush_att else np.nan,
                    "ypa": (pass_yds / pass_att) if pass_att else np.nan,
                })
            page += 1
            if page > int(pjs.get("paging", {}).get("total", page)):
                break
    except Exception as e:
        print("[apisports_pull] players fetch failed:", e)

    if rows:
        out = pd.DataFrame(rows)
        out.to_csv("data/apisports_player.csv", index=False)
        print(f"[apisports_pull] wrote data/apisports_player.csv rows={len(out)}")
    else:
        Path("data").mkdir(exist_ok=True)
        pd.DataFrame(columns=["player","team"]).to_csv("data/apisports_player.csv", index=False)
        print("[apisports_pull] no rows; wrote empty data/apisports_player.csv")
    return 0

if __name__ == "__main__":
    season = int(os.getenv("SEASON", "2025"))
    sys.exit(main(season))
