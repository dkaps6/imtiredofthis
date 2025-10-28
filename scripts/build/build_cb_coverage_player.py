#!/usr/bin/env python3
# build_cb_coverage_player.py (upgraded)
# Goal: Build player-level WR/CB file for top-3 WRs per team, Weeks 1–8 (2025), with weekly opponent join.
# Free sources cited in README/comments:
# - Coverage scheme (team man%/zone%): Sharp Football Analysis
#   https://www.sharpfootballanalysis.com/stats-nfl/nfl-coverage-schemes/
# - Receiver Alignment (slot/outside): RotoWire Receiver Alignment Breakdown (public page)
#   https://www.rotowire.com/football/player-alignment.php
#   (JS-rendered – use requests_html headless render; otherwise fall back to static HTML or leave blanks w/ warning)
# - Weekly WR/CB matchups (primary CB + occasional shadow): RotoBaller
#   Wk6: https://www.rotoballer.com/wr-cb-matchups-for-fantasy-football-sleepers-targets-for-week-6-2025/1719201
#   Wk7: https://www.rotoballer.com/wr-cb-matchups-for-fantasy-football-sleepers-targets-for-week-7-2025/1724870
#   Wk8: https://www.rotoballer.com/wr-cb-matchups-for-fantasy-football-sleepers-targets-for-week-8-2025/1732461
# - Weekly opponent mapping: Pro-Football-Reference 2025 Games page (all weeks on one page)
#   https://www.pro-football-reference.com/years/2025/games.htm
#
# Output schema:
# player,team,primary_cb,exp_vs_man,exp_vs_zone,slot_pct,wide_pct,shadow_flag,week
#
# Notes:
# - Alignment page may not expose week filters server-side; if rendering doesn't include weekly splits,
#   we use season alignment as a proxy and keep top-3 WRs constant per team across weeks.
# - exp_vs_man/zone = opponent team man%/zone% from Sharp (proxy; transparent and reproducible).
# - The script is defensive: if a source fails, it logs and keeps writing a CSV with available fields.
#
import sys, io, re, time, math, itertools
from typing import Dict, List, Tuple
import pandas as pd
import requests

# Optional headless renderer (free). If not installed, we degrade gracefully.
try:
    from requests_html import HTMLSession
    HAVE_RENDER = True
except Exception:
    HAVE_RENDER = False

SHARP_URL = "https://www.sharpfootballanalysis.com/stats-nfl/nfl-coverage-schemes/"
ROTOWIRE_URL = "https://www.rotowire.com/football/player-alignment.php"
ROTOBALLER_WEEKS = {
    6: "https://www.rotoballer.com/wr-cb-matchups-for-fantasy-football-sleepers-targets-for-week-6-2025/1719201",
    7: "https://www.rotoballer.com/wr-cb-matchups-for-fantasy-football-sleepers-targets-for-week-7-2025/1724870",
    8: "https://www.rotoballer.com/wr-cb-matchups-for-fantasy-football-sleepers-targets-for-week-8-2025/1732461",
}
PFR_GAMES_URL = "https://www.pro-football-reference.com/years/2025/games.htm"

TEAM_CODE = {
    "Cardinals":"ARI","Falcons":"ATL","Ravens":"BAL","Bills":"BUF","Panthers":"CAR","Bears":"CHI",
    "Bengals":"CIN","Browns":"CLE","Cowboys":"DAL","Broncos":"DEN","Lions":"DET","Packers":"GB",
    "Texans":"HOU","Colts":"IND","Jaguars":"JAX","Chiefs":"KC","Raiders":"LV","Chargers":"LAC",
    "Rams":"LAR","Dolphins":"MIA","Vikings":"MIN","Patriots":"NE","Saints":"NO","Giants":"NYG",
    "Jets":"NYJ","Eagles":"PHI","Steelers":"PIT","Seahawks":"SEA","49ers":"SF","Buccaneers":"TB",
    "Titans":"TEN","Commanders":"WAS"
}

TEAM_NAME_TO_ABBR = TEAM_CODE.copy()  # PFR uses full names; map via name→abbr
# Add common alt names seen on PFR (capitalized)
TEAM_NAME_TO_ABBR.update({
    "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
    "Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
    "Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
    "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX","Kansas City Chiefs":"KC",
    "Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LAR","Miami Dolphins":"MIA",
    "Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
    "New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","Seattle Seahawks":"SEA",
    "San Francisco 49ers":"SF","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS"
})

HDRS = {"User-Agent": "Mozilla/5.0"}

def read_html_tables(url: str, render: bool=False) -> List[pd.DataFrame]:
    if render and HAVE_RENDER:
        try:
            sess = HTMLSession()
            r = sess.get(url, headers=HDRS, timeout=45)
            r.html.render(timeout=60, sleep=2)  # headless Chromium via pyppeteer
            return pd.read_html(io.StringIO(r.html.html))
        except Exception as e:
            print(f"WARN: headless render failed for {url}: {e}", file=sys.stderr)
    # Fallback: plain requests
    resp = requests.get(url, headers=HDRS, timeout=45)
    resp.raise_for_status()
    try:
        return pd.read_html(io.StringIO(resp.text))
    except ValueError:
        return []

def fetch_sharp_coverages() -> Dict[str, Tuple[float, float]]:
    """Return {team_abbr: (man_frac, zone_frac)} from Sharp (strip % and convert to 0-1)."""
    tables = read_html_tables(SHARP_URL, render=False)
    df = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("team" in c for c in cols) and any("man" in c for c in cols) and any("zone" in c for c in cols):
            df = t
            break
    if df is None:
        if tables:
            df = tables[0]
        else:
            print("ERROR: Could not read Sharp coverage table.", file=sys.stderr)
            return {}
    cmap = {}
    for c in df.columns:
        lc = str(c).lower()
        if "team" in lc: cmap[c] = "team_name"
        elif "man" in lc: cmap[c] = "man_rate"
        elif "zone" in lc: cmap[c] = "zone_rate"
    df = df.rename(columns=cmap)
    df = df[[c for c in ["team_name","man_rate","zone_rate"] if c in df.columns]].copy()
    for c in ["man_rate","zone_rate"]:
        df[c] = df[c].astype(str).str.replace("%","",regex=False).str.extract(r"([0-9]+\.?[0-9]*)")[0]
        df[c] = pd.to_numeric(df[c], errors="coerce")/100.0
    df["team"] = df["team_name"].map(TEAM_NAME_TO_ABBR)
    df = df.dropna(subset=["team"])
    return {r["team"]:(float(r["man_rate"]), float(r["zone_rate"])) for _, r in df.iterrows()}

def fetch_rotowire_alignment(week: int) -> pd.DataFrame:
    """Read alignment table; attempt headless render first. Returns player, team, slot_pct, wide_pct, week."""
    tables = read_html_tables(ROTOWIRE_URL, render=True)
    if not tables:
        print(f"WARN: RotoWire alignment table unavailable (week {week}).", file=sys.stderr)
        return pd.DataFrame(columns=["player","team","slot_pct","wide_pct","week"])
    # Heuristic select
    df = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("player" in c for c in cols) and any("slot" in c for c in cols) and any("outside" in c for c in cols):
            df = t
            break
    if df is None:
        df = tables[0]
    colmap = {}
    for c in df.columns:
        lc = str(c).lower()
        if "player" in lc: colmap[c]="player"
        elif "team" in lc: colmap[c]="team_name"
        elif "slot" in lc: colmap[c]="slot_pct"
        elif "outside" in lc: colmap[c]="outside_pct"
    d = df.rename(columns=colmap)
    keep = [c for c in ["player","team_name","slot_pct","outside_pct"] if c in d.columns]
    d = d[keep].copy()
    # Parse percents
    for c in ["slot_pct","outside_pct"]:
        if c in d.columns:
            d[c] = d[c].astype(str).str.replace("%","",regex=False).str.extract(r"([0-9]+\.?[0-9]*)")[0]
            d[c] = pd.to_numeric(d[c], errors="coerce")/100.0
    d["team"] = d["team_name"].map(TEAM_NAME_TO_ABBR)
    d = d.dropna(subset=["team"]).copy()
    d["wide_pct"] = d.get("outside_pct", pd.NA)
    d = d.drop(columns=[c for c in ["team_name","outside_pct"] if c in d.columns])
    d["week"] = week
    return d[["player","team","slot_pct","wide_pct","week"]]

def fetch_rotoballer_matchups(week: int) -> pd.DataFrame:
    """Parse RotoBaller article for WR vs CB pairings; returns player, primary_cb, shadow_flag, week."""
    url = ROTO BALLER_WEEKS.get(week)
    if not url:
        return pd.DataFrame(columns=["player","primary_cb","shadow_flag","week"])
    resp = requests.get(url, headers=HDRS, timeout=45)
    resp.raise_for_status()
    text = resp.text
    pairs = []
    pat = re.compile(r'([A-Z][a-zA-Z\'\.]+(?:\s[A-Z][a-zA-Z\'\.]+){0,2})\s+(?:vs\.?|will see|draws)\s+([A-Z][a-zA-Z\'\-\.]+(?:\s[A-Z][a-zA-Z\'\-\.]+){0,2})')
    for m in pat.finditer(text):
        wr = m.group(1).strip()
        cb = m.group(2).strip()
        if len(wr.split())<=3 and len(cb.split())<=4:
            pairs.append((wr, cb))
    df = pd.DataFrame(pairs, columns=["player","primary_cb"]).drop_duplicates()
    df["shadow_flag"] = "projected_shadow" if re.search(r'[Ss]hadow', text) else ""
    df["week"] = week
    return df

def fetch_schedule_week_map() -> Dict[Tuple[str,int], str]:
    """Return {(team_abbr, week)-> opponent_abbr} for 2025 weeks 1..18 from PFR games table."""
    tables = read_html_tables(PFR_GAMES_URL, render=False)
    if not tables:
        print("ERROR: Could not read PFR games page.", file=sys.stderr)
        return {}
    # PFR page has multiple 'Schedule & Game Results' tables per week; concatenate all rows with Week, Visitor, Home
    frames = []
    for t in tables:
        cols = [c for c in t.columns]
        lower = [str(c).lower() for c in cols]
        if any("week" in s for s in lower) and any("visitor" in s for s in lower) and any("home" in s for s in lower):
            frames.append(t)
    if not frames:
        frames = [tables[0]]
    df = pd.concat(frames, ignore_index=True)
    cmap = {}
    for c in df.columns:
        lc = str(c).lower()
        if "week" in lc: cmap[c]="week"
        elif "visitor" in lc: cmap[c]="visitor"
        elif "home" in lc: cmap[c]="home"
    df = df.rename(columns=cmap)
    df = df[[c for c in ["week","visitor","home"] if c in df.columns]].copy()
    # Clean rows (drop NaNs and non-integer weeks)
    df = df.dropna(subset=["week","visitor","home"])
    # Map team full names to abbr
    df["visitor_abbr"] = df["visitor"].map(TEAM_NAME_TO_ABBR)
    df["home_abbr"] = df["home"].map(TEAM_NAME_TO_ABBR)
    df = df.dropna(subset=["visitor_abbr","home_abbr"])
    # Build mapping
    wk_map = {}
    for _, r in df.iterrows():
        try:
            wk = int(str(r["week"]).strip().split()[0])
        except Exception:
            continue
        va, ha = r["visitor_abbr"], r["home_abbr"]
        wk_map[(va, wk)] = ha
        wk_map[(ha, wk)] = va
    return wk_map

def select_top3_per_team(alignment_df: pd.DataFrame) -> pd.DataFrame:
    """Select top-3 WRs by (slot+wide) usage within each team for the given week."""
    d = alignment_df.copy()
    d["usage"] = d[["slot_pct","wide_pct"]].fillna(0).sum(axis=1)
    d = d.sort_values(["team","usage"], ascending=[True,False])
    d = d.groupby("team").head(3).copy()
    return d.drop(columns=["usage"])

def main(out_csv: str = "cb_coverage_player.csv"):
    # Coverage rates by defense
    mz = fetch_sharp_coverages()  # {team: (man_frac, zone_frac)}
    # 2025 schedule team-week->opponent
    opp_map = fetch_schedule_week_map()

    rows = []
    prev_top3_by_team: Dict[str, List[str]] = {}
    for week in range(1, 9):
        align = fetch_rotowire_alignment(week)  # may be season-level proxy
        if align.empty:
            # Fall back to previous week’s top-3
            recs = []
            for team, players in prev_top3_by_team.items():
                for p in players:
                    recs.append({"player":p, "team":team, "slot_pct": pd.NA, "wide_pct": pd.NA, "week":week})
            top3 = pd.DataFrame(recs) if recs else pd.DataFrame(columns=["player","team","slot_pct","wide_pct","week"])
        else:
            top3 = select_top3_per_team(align)
            prev_top3_by_team = top3.groupby("team")["player"].apply(list).to_dict()

        # Primary CB / shadow from RotoBaller for targeted weeks
        rb = fetch_rotoballer_matchups(week)
        merged = top3.merge(rb, on="player", how="left")

        # Exposure vs man/zone via opponent mapping
        exp_m, exp_z = [], []
        for _, r in merged.iterrows():
            team = r["team"]
            opp = opp_map.get((team, week))
            if opp and opp in mz:
                man, zone = mz[opp]
                exp_m.append(round(man,4))
                exp_z.append(round(zone,4))
            else:
                exp_m.append(pd.NA)
                exp_z.append(pd.NA)
        merged["exp_vs_man"] = exp_m
        merged["exp_vs_zone"] = exp_z

        # Ensure columns exist
        if "primary_cb" not in merged.columns: merged["primary_cb"] = ""
        if "shadow_flag" not in merged.columns: merged["shadow_flag"] = ""

        rows.append(merged[["player","team","primary_cb","exp_vs_man","exp_vs_zone","slot_pct","wide_pct","shadow_flag","week"]])

    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["player","team","primary_cb","exp_vs_man","exp_vs_zone","slot_pct","wide_pct","shadow_flag","week"])
    # Deduplicate & sort
    df = df.drop_duplicates(subset=["player","team","week"]).sort_values(["team","player","week"])
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows.")

if __name__ == "__main__":
    out_csv = "cb_coverage_player.csv" if len(sys.argv) < 2 else sys.argv[1]
    main(out_csv)
