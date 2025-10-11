#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Dict

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
TEAM_NAME_TO_ABBR: Dict[str,str] = {}
for a, n in TEAM_ABBR_TO_NAME.items():
    TEAM_NAME_TO_ABBR.setdefault(n, a)

def load_id_map() -> Optional[pd.DataFrame]:
    p = Path("data/id_map.csv")
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(p)
        cols = {c.lower(): c for c in df.columns}
        name_col = None
        for cand in ("team_name","full_name","name"):
            if cand in cols:
                name_col = cols[cand]; break
        if "team" in cols and name_col:
            out = df[[cols["team"], name_col]].dropna().copy()
            out.columns = ["team","team_name"]
            out["team"] = out["team"].astype(str).str.upper()
            out["team_name"] = out["team_name"].astype(str)
            return out
    except Exception:
        return None
    return None

def name_to_abbr(name: str, idmap: Optional[pd.DataFrame]) -> Optional[str]:
    if not isinstance(name, str): return None
    n = name.strip()
    if idmap is not None:
        row = idmap[idmap["team_name"].str.lower() == n.lower()]
        if not row.empty:
            return row.iloc[0]["team"]
    return TEAM_NAME_TO_ABBR.get(n)

def build_schedule_map(odds_path: Path, idmap: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    From outputs/odds_game.csv create team-level slate rows:
      team (abbr), opp_team (abbr), event_id, home_away, team_wp
    """
    required = ["event_id","home_team","away_team","home_wp","away_wp"]
    if not odds_path.exists() or odds_path.stat().st_size == 0:
        print("[metrics] INFO: odds_game.csv missing or empty; continuing without opp_team")
        return pd.DataFrame(columns=["team","opp_team","event_id","home_away","team_wp"])

    try:
        og = pd.read_csv(odds_path)
    except pd.errors.EmptyDataError:
        print("[metrics] INFO: odds_game.csv has no columns; continuing without opp_team")
        return pd.DataFrame(columns=["team","opp_team","event_id","home_away","team_wp"])

    miss = [c for c in required if c not in og.columns]
    if miss:
        print(f"[metrics] INFO: odds_game.csv missing columns {miss}; continuing without opp_team")
        return pd.DataFrame(columns=["team","opp_team","event_id","home_away","team_wp"])

    og["home_abbr"] = og["home_team"].apply(lambda n: name_to_abbr(n, idmap))
    og["away_abbr"] = og["away_team"].apply(lambda n: name_to_abbr(n, idmap))

    if og["home_abbr"].isna().any() or og["away_abbr"].isna().any():
        bad = og[(og["home_abbr"].isna()) | (og["away_abbr"].isna())][["event_id","home_team","away_team"]]
        print("[metrics] WARNING: could not map some team names:\n", bad.to_string(index=False))

    home_rows = og.assign(team=og["home_abbr"], opp_team=og["away_abbr"],
                          home_away="home", team_wp=og["home_wp"])[["event_id","team","opp_team","home_away","team_wp"]]
    away_rows = og.assign(team=og["away_abbr"], opp_team=og["home_abbr"],
                          home_away="away", team_wp=og["away_wp"])[["event_id","team","opp_team","home_away","team_wp"]]

    sched = pd.concat([home_rows, away_rows], ignore_index=True)
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

    if "team" in tf.columns: tf["team"] = tf["team"].astype(str).str.upper()
    if "team" in pf.columns: pf["team"] = pf["team"].astype(str).str.upper()

    idmap = load_id_map()
    sched = build_schedule_map(ogp, idmap)

    # attach slate context
    if not sched.empty:
        pf = pf.merge(sched, on="team", how="left")
    else:
        for c in ("opp_team","event_id","home_away","team_wp"):
            pf[c] = pd.NA

    # our own team features
    mf = pf.merge(tf, on="team", how="left", suffixes=("","_team"))

    # --- attach opponent defensive context here (pre-merge for pricing) ---
    opp_cols = [
        "team",
        "def_sack_rate","def_pass_epa","def_rush_epa",
        "light_box_rate","heavy_box_rate",
    ]
    present = [c for c in opp_cols if c in tf.columns]
    tf_opp = tf[present].copy()
    tf_opp = tf_opp.rename(columns={"team":"opp_team"})

    if "opp_team" in mf.columns:
        mf["opp_team"] = mf["opp_team"].astype(str).str.upper()
    if "opp_team" in tf_opp.columns:
        tf_opp["opp_team"] = tf_opp["opp_team"].astype(str).str.upper()

    mf = mf.merge(tf_opp, on="opp_team", how="left", suffixes=("", "_opp"))

    # clarify opponent columns with explicit suffix when needed
    rename_map = {}
    for c in ("def_sack_rate","def_pass_epa","def_rush_epa","light_box_rate","heavy_box_rate"):
        if c in mf.columns and f"{c}_opp" not in mf.columns:
            rename_map[c] = f"{c}_opp"
    if rename_map:
        mf = mf.rename(columns=rename_map)

    outp = Path("data/metrics_ready.csv")
    mf.to_csv(outp, index=False)
    print(f"[metrics] rows={len(mf)} → {outp}")

if __name__ == "__main__":
    main()
