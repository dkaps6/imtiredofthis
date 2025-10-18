#!/usr/bin/env python3
"""
ESPN pull (conservative, unauth endpoints).
Writes:
  data/espn_team_form.csv
  data/espn_player_form.csv
Schema matches team/player enrichers. Unknown fields remain NaN.
"""

import sys, requests
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = "data"
TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"

TEAM_COLS = [
    "team","def_pass_epa","def_rush_epa","def_sack_rate",
    "pace","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
    "light_box_rate","heavy_box_rate"
]
PLAYER_COLS = [
    "player","team",
    "tgt_share","route_rate","rush_share",
    "yprr","ypt","ypc","ypa",
    "receptions_per_target",
    "rz_share","rz_tgt_share","rz_rush_share",
]

def _write_csv(path: str, df: pd.DataFrame, cols):
    out = (df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame())
    out.columns = [c.lower() for c in out.columns]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    out = out[cols]
    out.to_csv(path, index=False)
    return out

def _get(url, params=None):
    r = requests.get(url, params=params or {}, timeout=20)
    r.raise_for_status()
    return r.json()

def main():
    Path(DATA_DIR).mkdir(exist_ok=True)

    # ---- team list
    teams = []
    try:
        js = _get(TEAMS_URL)
        for it in js.get("sports", [])[0].get("leagues", [])[0].get("teams", []):
            t = it.get("team", {})
            code = t.get("abbreviation") or t.get("shortDisplayName") or t.get("name")
            teams.append({"id": t.get("id"), "team": str(code).upper()})
    except Exception as e:
        print("[espn_pull] team list failed:", e)

    # write team_form with schema (ESPN doesn't provide these metrics directly)
    team_df = pd.DataFrame(teams)[["team"]] if teams else pd.DataFrame(columns=["team"])
    _write_csv(f"{DATA_DIR}/espn_team_form.csv", team_df, TEAM_COLS)

    # ---- per-player ypc/ypa if present in roster JSON
    rows = []
    for t in teams:
        tid = t["id"]
        try:
            roster = _get(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{tid}", {"enable":"roster"})
            rr = roster.get("team", {}).get("athletes", [])
            for grp in rr:
                for a in grp.get("items", []):
                    name = a.get("displayName")
                    ypc = np.nan; ypa = np.nan
                    try:
                        stats = a.get("statistics", {}).get("splits", {}).get("categories", [])
                        for cat in stats:
                            nm = (cat.get("name") or "").lower()
                            if nm.startswith("rushing"):
                                for stat in cat.get("stats", []):
                                    if stat.get("name") == "yardsPerRushAttempt":
                                        ypc = float(stat.get("value"))
                            if nm.startswith("passing"):
                                for stat in cat.get("stats", []):
                                    if stat.get("name") == "yardsPerPassAttempt":
                                        ypa = float(stat.get("value"))
                    except Exception:
                        pass
                    rows.append({"player": name, "team": t["team"], "ypc": ypc, "ypa": ypa})
        except Exception as e:
            print(f"[espn_pull] roster for team {t['team']} failed:", e)

    ply = pd.DataFrame(rows)
    _write_csv(f"{DATA_DIR}/espn_player_form.csv", ply, PLAYER_COLS)

    print(f"[espn_pull] wrote team rows={len(team_df)} → data/espn_team_form.csv")
    print(f"[espn_pull] wrote player rows={len(ply)} → data/espn_player_form.csv")
    return 0

if __name__ == "__main__":
    sys.exit(main())
