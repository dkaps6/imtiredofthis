#!/usr/bin/env python3
"""
ESPN (unofficial) pull — conservative.
No key required; endpoints can change. We fetch rosters + a few season stats
to compute only ypc/ypa where available. Route/targets are typically not provided here.

Outputs:
  data/espn_team.csv    (team codes)
  data/espn_player.csv  with: player, team, ypc?, ypa? (others NaN)
"""
import sys, requests
from pathlib import Path
import pandas as pd
import numpy as np

TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"

def _get(url, params=None):
    r = requests.get(url, params=params or {}, timeout=20)
    r.raise_for_status()
    return r.json()

def main():
    teams = []
    try:
        js = _get(TEAMS_URL)
        for it in js.get("sports", [])[0].get("leagues", [])[0].get("teams", []):
            t = it.get("team", {})
            code = t.get("abbreviation") or t.get("shortDisplayName") or t.get("name")
            teams.append({"id": t.get("id"), "team": str(code).upper()})
    except Exception as e:
        print("[espn_pull] team list failed:", e)

    Path("data").mkdir(exist_ok=True)
    if teams:
        pd.DataFrame(teams)[["team"]].drop_duplicates().to_csv("data/espn_team.csv", index=False)
    else:
        pd.DataFrame(columns=["team"]).to_csv("data/espn_team.csv", index=False)

    # Try rosters and per-player simple rushing/passing rates (ypc/ypa)
    rows = []
    for t in teams:
        tid = t["id"]
        try:
            roster = _get(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{tid}", {"enable":"roster"})
            rr = roster.get("team", {}).get("athletes", [])
            for grp in rr:
                for a in grp.get("items", []):
                    name = a.get("displayName")
                    pos  = a.get("position", {}).get("abbreviation", "")
                    # Season stats may or may not be embedded; keep conservative
                    ypc = np.nan; ypa = np.nan
                    try:
                        stats = a.get("statistics", {}).get("splits", {}).get("categories", [])
                        # heuristic parse
                        for cat in stats:
                            if cat.get("name","").lower().startswith("rushing"):
                                for stat in cat.get("stats", []):
                                    if stat.get("name") == "yardsPerRushAttempt":
                                        ypc = float(stat.get("value"))
                            if cat.get("name","").lower().startswith("passing"):
                                for stat in cat.get("stats", []):
                                    if stat.get("name") == "yardsPerPassAttempt":
                                        ypa = float(stat.get("value"))
                    except Exception:
                        pass
                    rows.append({
                        "player": name, "team": t["team"],
                        "route_rate": np.nan, "target_share": np.nan, "rush_share": np.nan,
                        "yprr": np.nan, "ypc": ypc, "ypa": ypa
                    })
        except Exception as e:
            print(f"[espn_pull] roster for team {t['team']} failed:", e)

    if rows:
        pd.DataFrame(rows).to_csv("data/espn_player.csv", index=False)
        print(f"[espn_pull] wrote data/espn_player.csv rows={len(rows)}")
    else:
        pd.DataFrame(columns=["player","team"]).to_csv("data/espn_player.csv", index=False)
        print("[espn_pull] no players found; wrote empty data/espn_player.csv")
    return 0

if __name__ == "__main__":
    sys.exit(main())
