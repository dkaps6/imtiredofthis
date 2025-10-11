#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# --------- Fallback team maps if data/id_map.csv isn't present ----------
TEAM_ABBR_TO_NAME = {
    "ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens","BUF":"Buffalo Bills",
    "CAR":"Carolina Panthers","CHI":"Chicago Bears","CIN":"Cincinnati Bengals","CLE":"Cleveland Browns",
    "DAL":"Dallas Cowboys","DEN":"Denver Broncos","DET":"Detroit Lions","GB":"Green Bay Packers","GNB":"Green Bay Packers",
    "HOU":"Houston Texans","IND":"Indianapolis Colts","JAX":"Jacksonville Jaguars","JAC":"Jacksonville Jaguars",
    "KC":"Kansas City Chiefs","KAN":"Kansas City Chiefs","LAC":"Los Angeles Chargers","LAR":"Los Angeles Rams",
    "LV":"Las Vegas Raiders","LVR":"Las Vegas Raiders","MIA":"Miami Dolphins","MIN":"Minnesota Vikings",
    "NE":"New England Patriots","NWE":"New England Patriots","NO":"New Orleans Saints","NOR":"New Orleans Saints",
    "NYG":"New York Giants","NYJ":"New York Jets","PHI":"Philadelphia Eagles","PIT":"Pittsburgh Steelers",
    "SEA":"Seattle Seahawks","SF":"San Francisco 49ers","SFO":"San Francisco 49ers","TB":"Tampa Bay Buccaneers","TAM":"Tampa Bay Buccaneers",
    "TEN":"Tennessee Titans","WAS":"Washington Commanders","WSH":"Washington Commanders",
}
# reverse map
TEAM_NAME_TO_ABBR = {}
for a,n in TEAM_ABBR_TO_NAME.items():
    TEAM_NAME_TO_ABBR.setdefault(n, a)

def load_id_map() -> Optional[pd.DataFrame]:
    """Optional: data/id_map.csv with columns like: team, team_name (or full_name)."""
    p = Path("data/id_map.csv")
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(p)
        # normalize expected columns
        cols = {c.lower(): c for c in df.columns}
        # Look for 'team' abbrev and some variant of name
        name_col = None
        for cand in ("team_name","full_name","name"):
            if cand in cols:
                name_col = cols[cand]
                break
        if "team" in cols and name_col:
            out = df[[cols["team"], name_col]].dropna().copy()
            out.columns = ["team","team_name"]
            out["team"] = out["team"].astype(str).str.upper()
            out["team_name"] = out["team_name"].astype(str)
            return out
    except Exception:
        return None
    return None

def abbr_to_name(abbr: str, idmap: Optional[pd.DataFrame]) -> Optional[str]:
    if not isinstance(abbr, str):
        return None
    a = abbr.strip().upper()
    if idmap is not None:
        row = idmap[idmap["team"] == a]
        if not row.empty:
            return row.iloc[0]["team_name"]
    return TEAM_ABBR_TO_NAME.get(a)

def name_to_abbr(name: str, idmap: Optional[pd.DataFrame]) -> Optional[str]:
    if not isinstance(name, str):
        return None
    n = name.strip()
    if idmap is not None:
        row = idmap[idmap["team_name"].str.lower() == n.lower()]
        if not row.empty:
            return row.iloc[0]["team"]
    # fallback dictionary
    return TEAM_NAME_TO_ABBR.get(n)

def build_schedule_map(odds_path: Path, idmap: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    From outputs/odds_game.csv (built by fetcher) create a 2-row-per-game table:
       team (abbr), opp_team (abbr), event_id, home_away, team_wp
    """
    cols = ["event_id","home_team","away_team","home_wp","away_wp"]
    if not odds_path.exists() or odds_path.stat().st_size == 0:
        print("[metrics] WARNING: outputs/odds_game.csv missing/empty; opp_team will be NaN")
        return pd.DataFrame(columns=["team","opp_team","event_id","home_away","team_wp"])

    og = pd.read_csv(odds_path)
    missing = [c for c in cols if c not in og.columns]
    if missing:
        print(f"[metrics] WARNING: outputs/odds_game.csv missing columns {missing}; opp_team will be NaN")
        return pd.DataFrame(columns=["team","opp_team","event_id","home_away","team_wp"])

    # Map team names -> abbrev
    og["home_abbr"] = og["home_team"].apply(lambda n: name_to_abbr(n, idmap))
    og["away_abbr"] = og["away_team"].apply(lambda n: name_to_abbr(n, idmap))

    # Warn if any could not be mapped
    if og["home_abbr"].isna().any() or og["away_abbr"].isna().any():
        bad = og[(og["home_abbr"].isna()) | (og["away_abbr"].isna())][["event_id","home_team","away_team"]]
        print("[metrics] WARNING: could not map some team names to abbreviations:\n", bad.to_string(index=False))

    # explode to team-level rows
    home_rows = og.assign(
        team=og["home_abbr"],
        opp_team=og["away_abbr"],
        home_away="home",
        team_wp=og["home_wp"],
    )[["event_id","team","opp_team","home_away","team_wp"]]

    away_rows = og.assign(
        team=og["away_abbr"],
        opp_team=og["home_abbr"],
        home_away="away",
        team_wp=og["away_wp"],
    )[["event_id","team","opp_team","home_away","team_wp"]]

    sched = pd.concat([home_rows, away_rows], ignore_index=True)
    # enforce uppercase strings for team/opp_team
    for c in ("team","opp_team"):
        if c in sched.columns:
            sched[c] = sched[c].astype(str).str.upper()
    return sched

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    a = ap.parse_args()

    Path("data").mkdir(exist_ok=True)

    tfp = Path("data/team_form.csv")
    pfp = Path("data/player_form.csv")
    ogp = Path("outputs/odds_game.csv")

    # Required inputs guard
    if not tfp.exists() or tfp.stat().st_size == 0:
        print("[metrics] team_form missing/empty; wrote empty metrics_ready.csv")
        (Path("data")/"metrics_ready.csv").write_text("")
        return
    if not pfp.exists() or pfp.stat().st_size == 0:
        print("[metrics] player_form missing/empty; wrote empty metrics_ready.csv")
        (Path("data")/"metrics_ready.csv").write_text("")
        return

    tf = pd.read_csv(tfp)
    pf = pd.read_csv(pfp)

    # normalize team codes in sources
    if "team" in tf.columns:
        tf["team"] = tf["team"].astype(str).str.upper()
    if "team" in pf.columns:
        pf["team"] = pf["team"].astype(str).str.upper()

    # Optional team↔name map
    idmap = load_id_map()

    # Build schedule map to determine opp_team for this slate
    sched = build_schedule_map(ogp, idmap)

    # Attach opp_team + slate context to player_form
    if not sched.empty:
        pf = pf.merge(sched, on="team", how="left")
    else:
        # still add the columns so downstream never KeyErrors
        pf["opp_team"] = pd.NA
        pf["event_id"] = pd.NA
        pf["home_away"] = pd.NA
        pf["team_wp"] = pd.NA

    # Merge team features (our own team)
    mf = pf.merge(tf, on="team", how="left", suffixes=("","_team"))

    # Optional: you may also pre-merge opponent defensive features here,
    # but pricing.py currently performs that merge itself using `opp_team`.
    # We just ensure opp_team exists and matches tf["team"] abbreviations.

    outp = Path("data/metrics_ready.csv")
    mf.to_csv(outp, index=False)
    print(f"[metrics] rows={len(mf)} → {outp}")

if __name__ == "__main__":
    main()
