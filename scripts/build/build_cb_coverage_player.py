#!/usr/bin/env python3
# build_cb_coverage_player.py (upgraded)
# Player-level WR/CB exposure for top-3 WRs per team (weeks 1–8), joined to opponent and man/zone exposure.
import sys, io, re
from typing import Dict, List, Tuple
from pathlib import Path
import pandas as pd
import requests

# Optional renderer (degrades gracefully)
try:
    from requests_html import HTMLSession
    HAVE_RENDER = True
except Exception:
    HAVE_RENDER = False

HDRS = {"User-Agent": "Mozilla/5.0"}

SHARP_URL = "https://www.sharpfootballanalysis.com/stats-nfl/nfl-coverage-schemes/"
ROTOWIRE_URL = "https://www.rotowire.com/football/player-alignment.php"
ROTOBALLER_WEEKS = {
    6: "https://www.rotoballer.com/wr-cb-matchups-for-fantasy-football-sleepers-targets-for-week-6-2025/1719201",
    7: "https://www.rotoballer.com/wr-cb-matchups-for-fantasy-football-sleepers-targets-for-week-7-2025/1724870",
    8: "https://www.rotoballer.com/wr-cb-matchups-for-fantasy-football-sleepers-targets-for-week-8-2025/1732461",
}
PFR_GAMES_URL = "https://www.pro-football-reference.com/years/2025/games.htm"

TEAM_NAME_TO_ABBR = {
    "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL","Buffalo Bills":"BUF",
    "Carolina Panthers":"CAR","Chicago Bears":"CHI","Cincinnati Bengals":"CIN","Cleveland Browns":"CLE",
    "Dallas Cowboys":"DAL","Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
    "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX","Kansas City Chiefs":"KC",
    "Las Vegas Raiders":"LV","Los Angeles Chargers":"LAC","Los Angeles Rams":"LAR","Miami Dolphins":"MIA",
    "Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO","New York Giants":"NYG",
    "New York Jets":"NYJ","Philadelphia Eagles":"PHI","Pittsburgh Steelers":"PIT","Seattle Seahawks":"SEA",
    "San Francisco 49ers":"SF","Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS"
}

def _read_html_tables(url: str, render: bool=False) -> List[pd.DataFrame]:
    if render and HAVE_RENDER:
        try:
            sess = HTMLSession()
            r = sess.get(url, headers=HDRS, timeout=45)
            r.html.render(timeout=60, sleep=2)
            return pd.read_html(io.StringIO(r.html.html))
        except Exception:
            pass
    try:
        resp = requests.get(url, headers=HDRS, timeout=45)
        resp.raise_for_status()
        return pd.read_html(io.StringIO(resp.text))
    except Exception:
        return []

def _fetch_sharp_coverages() -> Dict[str, Tuple[float, float]]:
    """Return {team_abbr: (man_frac, zone_frac)}."""
    tables = _read_html_tables(SHARP_URL, render=False)
    if not tables:
        return {}
    # best-effort pick
    df = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if "team" in " ".join(cols) and "man" in " ".join(cols) and "zone" in " ".join(cols):
            df = t; break
    if df is None:
        df = tables[0]
    cmap = {}
    for c in df.columns:
        lc = str(c).lower()
        if "team" in lc: cmap[c] = "team_name"
        elif "man" in lc: cmap[c] = "man_rate"
        elif "zone" in lc: cmap[c] = "zone_rate"
    df = df.rename(columns=cmap)[[c for c in ["team_name","man_rate","zone_rate"] if c in cmap.values()]]
    for c in ["man_rate","zone_rate"]:
        if c in df:
            df[c] = (
                df[c].astype(str)
                     .str.replace("%","",regex=False)
                     .str.extract(r"([0-9]+\.?[0-9]*)")[0]
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")/100.0
    df["team"] = df["team_name"].map(TEAM_NAME_TO_ABBR)
    df = df.dropna(subset=["team"])
    return {r["team"]:(float(r.get("man_rate", 0) or 0), float(r.get("zone_rate", 0) or 0)) for _, r in df.iterrows()}

def _fetch_rotowire_alignment(week: int) -> pd.DataFrame:
    tables = _read_html_tables(ROTOWIRE_URL, render=True)
    if not tables:
        return pd.DataFrame(columns=["player","team","slot_pct","wide_pct","week"])
    df = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("player" in c for c in cols) and any("slot" in c for c in cols) and any("outside" in c for c in cols):
            df = t; break
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
    for c in ["slot_pct","outside_pct"]:
        if c in d.columns:
            d[c] = d[c].astype(str).str.replace("%","",regex=False).str.extract(r"([0-9]+\.?[0-9]*)")[0]
            d[c] = pd.to_numeric(d[c], errors="coerce")/100.0
    d["wide_pct"] = d.get("outside_pct", pd.NA)
    d["team"] = d.get("team_name", "").map(TEAM_NAME_TO_ABBR)
    d = d.dropna(subset=["team"]).copy()
    d = d.drop(columns=[c for c in ["team_name","outside_pct"] if c in d.columns])
    d["week"] = week
    return d[["player","team","slot_pct","wide_pct","week"]]

def _fetch_rotoballer_matchups(week: int) -> pd.DataFrame:
    url = ROTOBALLER_WEEKS.get(week)
    if not url:
        return pd.DataFrame(columns=["player","primary_cb","shadow_flag","week"])
    try:
        resp = requests.get(url, headers=HDRS, timeout=45)
        resp.raise_for_status()
        text = resp.text
    except Exception:
        return pd.DataFrame(columns=["player","primary_cb","shadow_flag","week"])
    pairs = []
    pat = re.compile(r'([A-Z][a-zA-Z\'\.]+(?:\s[A-Z][a-zA-Z\'\.]+){0,2})\s+(?:vs\.?|will see|draws)\s+([A-Z][a-zA-Z\'\-\.]+(?:\s[A-Z][a-zA-Z\'\-\.]+){0,2})')
    for m in pat.finditer(text):
        wr = m.group(1).strip(); cb = m.group(2).strip()
        if len(wr.split())<=3 and len(cb.split())<=4:
            pairs.append((wr, cb))
    df = pd.DataFrame(pairs, columns=["player","primary_cb"]).drop_duplicates()
    df["shadow_flag"] = "projected_shadow" if re.search(r'[Ss]hadow', text) else ""
    df["week"] = week
    return df

def _fetch_schedule_week_map() -> Dict[Tuple[str,int], str]:
    tables = _read_html_tables(PFR_GAMES_URL, render=False)
    if not tables:
        return {}
    frames = []
    for t in tables:
        cols_lower = [str(c).lower() for c in t.columns]
        if any("week" in c for c in cols_lower) and any("visitor" in c for c in cols_lower) and any("home" in c for c in cols_lower):
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
    df = df[[c for c in ["week","visitor","home"] if c in df.columns]].dropna()
    name_to_abbr = TEAM_NAME_TO_ABBR
    df["visitor_abbr"] = df["visitor"].map(name_to_abbr)
    df["home_abbr"] = df["home"].map(name_to_abbr)
    df = df.dropna(subset=["visitor_abbr","home_abbr"])
    wk_map = {}
    for _, r in df.iterrows():
        try:
            wk = int(str(r["week"]).split()[0])
        except Exception:
            continue
        va, ha = r["visitor_abbr"], r["home_abbr"]
        wk_map[(va, wk)] = ha
        wk_map[(ha, wk)] = va
    return wk_map

def _select_top3_per_team(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    x["usage"] = x[["slot_pct","wide_pct"]].fillna(0).sum(axis=1)
    x = x.sort_values(["team","usage"], ascending=[True,False])
    x = x.groupby("team").head(3).copy()
    return x.drop(columns=["usage"])

def build_cb_coverage_player(season: int = 2025, max_week: int = 8) -> pd.DataFrame:
    mz = _fetch_sharp_coverages()
    opp_map = _fetch_schedule_week_map()

    rows = []
    prev_top3 = {}
    for week in range(1, max_week+1):
        align = _fetch_rotowire_alignment(week)
        if align.empty and prev_top3:
            # carry forward last known top-3
            recs = []
            for team, players in prev_top3.items():
                for p in players:
                    recs.append({"player":p, "team":team, "slot_pct": pd.NA, "wide_pct": pd.NA, "week":week})
            top3 = pd.DataFrame(recs)
        else:
            top3 = _select_top3_per_team(align) if not align.empty else pd.DataFrame(columns=["player","team","slot_pct","wide_pct","week"])
            if not top3.empty:
                prev_top3 = top3.groupby("team")["player"].apply(list).to_dict()

        rb = _fetch_rotoballer_matchups(week)
        merged = top3.merge(rb, on="player", how="left")

        exp_m, exp_z = [], []
        for _, r in merged.iterrows():
            team = r.get("team")
            opp = opp_map.get((team, week))
            if opp and opp in mz:
                man, zone = mz[opp]
                exp_m.append(round(man,4)); exp_z.append(round(zone,4))
            else:
                exp_m.append(pd.NA); exp_z.append(pd.NA)
        merged["exp_vs_man"] = exp_m
        merged["exp_vs_zone"] = exp_z

        if "primary_cb" not in merged.columns: merged["primary_cb"] = ""
        if "shadow_flag" not in merged.columns: merged["shadow_flag"] = ""

        rows.append(merged[["player","team","primary_cb","exp_vs_man","exp_vs_zone","slot_pct","wide_pct","shadow_flag","week"]])

    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["player","team","primary_cb","exp_vs_man","exp_vs_zone","slot_pct","wide_pct","shadow_flag","week"])
    df = df.drop_duplicates(subset=["player","team","week"]).sort_values(["team","player","week"])
    return df

if __name__ == "__main__":
    from datetime import datetime
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=datetime.utcnow().year)
    ap.add_argument("--max-week", type=int, default=8)
    args = ap.parse_args()

    out = Path("data") / "cb_coverage_player.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = build_cb_coverage_player(season=args.season, max_week=args.max_week)
        df.to_csv(out, index=False)
        print(f"Wrote {out.resolve()} with {len(df)} rows.")
    except Exception as e:
        # graceful header-only fallback
        print(f"[build_cb_coverage_player] ERROR: {e}")
        hdr = ["player","team","primary_cb","exp_vs_man","exp_vs_zone","slot_pct","wide_pct","shadow_flag","week"]
        pd.DataFrame(columns=hdr).to_csv(out, index=False)
        print(f"[build_cb_coverage_player] wrote header-only CSV fallback at {out.resolve()}")
