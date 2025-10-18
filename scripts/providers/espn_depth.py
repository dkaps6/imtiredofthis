#!/usr/bin/env python3
# scripts/providers/espn_depth.py
from __future__ import annotations
import time, os, sys, requests, pandas as pd

DATA_DIR = "data"
HEADERS = {"User-Agent":"Mozilla/5.0","Accept":"application/json, text/plain, */*"}

ESPN_TEAMS = {
    "ARI":22,"ATL":1,"BAL":33,"BUF":2,"CAR":29,"CHI":3,"CIN":4,"CLE":5,"DAL":6,"DEN":7,
    "DET":8,"GB":9,"HOU":34,"IND":11,"JAX":30,"KC":12,"LAC":24,"LAR":14,"LV":13,"MIA":15,
    "MIN":16,"NE":17,"NO":18,"NYG":19,"NYJ":20,"PHI":21,"PIT":23,"SEA":26,"SF":25,"TB":27,"TEN":10,"WAS":28
}

def fetch_depth(team_id: int) -> dict:
    url = f"https://site.api.espn.com/apis/v2/sports/football/nfl/teams/{team_id}/depthchart"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()

def rows_from_json(team_abbr: str, js: dict) -> list[dict]:
    out = []
    for group in js.get("items", []):
        pos = group.get("position", {}).get("abbreviation") or group.get("position", {}).get("name")
        for ent in group.get("entries", []):
            pl = ent.get("player") or {}
            out.append({
                "team": team_abbr,
                "position": str(pos).upper(),
                "order": ent.get("order"),
                "player_id": pl.get("id"),
                "player": pl.get("fullName") or pl.get("name"),
            })
    return out

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    rows = []
    for abbr, tid in ESPN_TEAMS.items():
        try:
            js = fetch_depth(tid)
            rows.extend(rows_from_json(abbr, js))
            time.sleep(0.6)
        except Exception as e:
            print(f"[espn_depth] {abbr} failed: {e}", file=sys.stderr)
    df = pd.DataFrame(rows, columns=["team","position","order","player_id","player"]).sort_values(["team","position","order"])
    df["player"] = df["player"].astype(str).str.replace(".","", regex=False).str.strip()
    df.to_csv(os.path.join(DATA_DIR, "espn_depth.csv"), index=False)
    # roles outputs for downstream make_* merging
    df[["player","team"]].assign(role=df["position"].str.upper() + df["order"].fillna(1).astype(int).astype(str)).to_csv(os.path.join(DATA_DIR,"roles_espn.csv"), index=False)
    # merge roles preference: ESPN first, then ourlads if present
    try:
        espn = pd.read_csv(os.path.join(DATA_DIR,"roles_espn.csv"))
        try:
            ol = pd.read_csv(os.path.join(DATA_DIR,"roles_ourlads.csv"))
        except Exception:
            ol = pd.DataFrame(columns=espn.columns)
        merged = pd.concat([espn, ol], ignore_index=True).drop_duplicates(subset=["player","team"], keep="first")
        merged.to_csv(os.path.join(DATA_DIR,"roles.csv"), index=False)
    except Exception:
        pass
    print(f"[espn_depth] wrote rows={len(df)} → data/espn_depth.csv (+roles_espn.csv, roles.csv)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
