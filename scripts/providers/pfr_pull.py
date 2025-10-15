#!/usr/bin/env python3
"""
Pro-Football-Reference (PFR) fallback to estimate routes-per-dropback.

What we scrape:
- League receiving page: targets (Tgt), receiving yards (Yds), team code (Tm)
- Each team page: team pass attempts (Att), times sacked (Sk)

We compute:
- team_dropbacks = pass_att + sacks
- routes_est     = targets / TPRR   (TPRR ~ targets per route run; default 0.22)
- routes_per_dropback (route_rate_est) = routes_est / team_dropbacks
- yprr_proxy_est = rec_yards / routes_est

Outputs:
- data/pfr_player_enrich.csv  with columns:
    player, team, routes_per_dropback, yprr_proxy_est
- data/pfr_team_enrich.csv    with columns:
    team_abbr, team_pass_att, team_times_sacked, team_dropbacks

Notes:
- This is an ESTIMATE. When participation data is present, your pipeline will prefer the real route_rate.
- Team code mapping normalizes PFR's 3-letter codes (e.g., GNB) to standard (GB), etc.
"""

from __future__ import annotations
import sys, time
from pathlib import Path
import re
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

LEAGUE_RECEIVING_URL = "https://www.pro-football-reference.com/years/{season}/receiving.htm"
TEAM_PAGE_URL        = "https://www.pro-football-reference.com/teams/{pfr}/{season}.htm"

# Map PFR team codes -> common 2/3-letter you use in your data (adjust if your repo uses different)
PFR_TO_STD = {
    "ARI":"ARI","ATL":"ATL","BAL":"BAL","BUF":"BUF","CAR":"CAR","CHI":"CHI","CIN":"CIN","CLE":"CLE","DAL":"DAL",
    "DEN":"DEN","DET":"DET","GNB":"GB","GNB ":"GB","GBP":"GB","GB":"GB",
    "HOU":"HOU","IND":"IND","JAX":"JAX","KAN":"KC","KCC":"KC","KC":"KC",
    "LVR":"LV","RAI":"LV","OAK":"LV","LV":"LV",
    "LAC":"LAC","SDG":"LAC","SD":"LAC",
    "LAR":"LAR","STL":"LAR","RAM":"LAR","LA":"LAR",
    "MIA":"MIA","MIN":"MIN","NWE":"NE","NE":"NE","NOS":"NO","NO":"NO",
    "NYG":"NYG","NYJ":"NYJ","PHI":"PHI","PIT":"PIT","SEA":"SEA","SFO":"SF","SF":"SF",
    "TAM":"TB","TBB":"TB","TB":"TB","TEN":"TEN","OTI":"TEN","WAS":"WAS","WFT":"WAS","WSH":"WAS",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Model/1.0; +https://example.com/bot)"}

def _read_html(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")

def _clean_num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return np.nan

def fetch_league_receiving(season: int) -> pd.DataFrame:
    """Return columns: player, team (STD), targets, rec_yards"""
    url = LEAGUE_RECEIVING_URL.format(season=season)
    soup = _read_html(url)
    table = soup.find("table", {"id": "receiving"})
    if table is None:
        return pd.DataFrame()

    # Use pandas read_html on the table HTML to get rows
    df_list = pd.read_html(str(table))
    if not df_list:
        return pd.DataFrame()
    df = df_list[0]
    # Normalize columns (PFR uses multi-row headers sometimes)
    cols = [re.sub(r"[\W]+", "_", str(c)).strip("_").lower() for c in df.columns]
    df.columns = cols

    # Expect columns including 'player', 'tm', 'tgt', 'yds'
    rename_map = {}
    if "player" in df.columns: rename_map["player"] = "player"
    if "tm" in df.columns:     rename_map["tm"]     = "team_pfr"
    if "tgt" in df.columns:    rename_map["tgt"]    = "targets"
    if "yds" in df.columns:    rename_map["yds"]    = "rec_yards"

    df = df.rename(columns=rename_map)
    keep = [c for c in ["player","team_pfr","targets","rec_yards"] if c in df.columns]
    df = df[keep].copy()

    # Clean
    df["targets"]   = pd.to_numeric(df["targets"], errors="coerce")
    df["rec_yards"] = pd.to_numeric(df["rec_yards"], errors="coerce")

    # Drop summary rows
    df = df[df["player"].astype(str).str.lower() != "team total"]

    # Map team codes
    df["team"] = df["team_pfr"].map(PFR_TO_STD).fillna(df["team_pfr"].astype(str).str.upper())
    df = df.dropna(subset=["team"])
    return df[["player","team","targets","rec_yards"]]

def fetch_team_dropbacks(season: int, teams_std: list[str]) -> pd.DataFrame:
    """Return columns: team (STD), team_pass_att, team_times_sacked, team_dropbacks"""
    rows = []
    # Build inverse map to PFR code (best-effort)
    std_to_pfr = {}
    for pfr, std in PFR_TO_STD.items():
        std_to_pfr.setdefault(std, pfr)
    for team in sorted(set(teams_std)):
        pfr_code = std_to_pfr.get(team, team)
        url = TEAM_PAGE_URL.format(pfr=pfr_code, season=season)
        try:
            soup = _read_html(url)
        except Exception as e:
            print(f"[pfr_pull] team page failed {team} ({pfr_code}): {e}")
            continue

        # Team Offense table often has IDs like 'team_stats' or a caption 'Team Offense'
        # Easiest robust approach: use read_html and select the table containing 'Pass Att' and 'Times Sacked'
        dfs = pd.read_html(str(soup))
        pass_att = np.nan
        sacks    = np.nan
        for tdf in dfs:
            lower_cols = [str(c).lower() for c in tdf.columns]
            if any("pass att" in c or "att" == c.strip() for c in lower_cols) and any("times sacked" in c or "sk" == c.strip() for c in lower_cols):
                # Try to find a single row totals
                t = tdf.copy()
                # Try to find totals row by label or take the last row
                last = t.tail(1)
                # Search for likely columns
                # Prefer exact matches if present
                cand_pass_cols = [c for c in t.columns if str(c).lower() in ("pass att","att","pass_att")]
                cand_sack_cols = [c for c in t.columns if str(c).lower() in ("times sacked","sk","sacked","times_sacked")]
                if not cand_pass_cols:
                    cand_pass_cols = [c for c in t.columns if "Att" in str(c)]
                if not cand_sack_cols:
                    cand_sack_cols = [c for c in t.columns if "Sk" in str(c) or "Sack" in str(c)]
                try:
                    pass_att = _clean_num(last[cand_pass_cols[0]].values[0])
                except Exception:
                    pass
                try:
                    sacks = _clean_num(last[cand_sack_cols[0]].values[0])
                except Exception:
                    pass
                if np.isfinite(pass_att) or np.isfinite(sacks):
                    break

        # If we missed via the heuristic above, try a second heuristic: look for any table with 'Passing' and sum 'Att' and 'Sk'
        if not np.isfinite(pass_att) or not np.isfinite(sacks):
            # Fallback pass: try more specific searches
            try:
                passing_tables = [tdf for tdf in dfs if any("passing" in str(h).lower() for h in tdf.columns)]
                for tdf in passing_tables:
                    # Try to sum columns named 'Att' and 'Sk'
                    cols = [str(c).lower() for c in tdf.columns]
                    att_col = None; sk_col = None
                    for c in tdf.columns:
                        lc = str(c).lower()
                        if lc in ("att","pass att","pass_att"): att_col = c
                        if lc in ("sk","times sacked","times_sacked","sacked"): sk_col = c
                    if att_col is not None and sk_col is not None:
                        pass_att = _clean_num(tdf[att_col].sum())
                        sacks    = _clean_num(tdf[sk_col].sum())
                        break
            except Exception:
                pass

        if not np.isfinite(pass_att) and not np.isfinite(sacks):
            print(f"[pfr_pull] WARN: could not find pass_att/sacks for {team}")
            continue

        dropbacks = (0 if not np.isfinite(pass_att) else pass_att) + (0 if not np.isfinite(sacks) else sacks)
        rows.append({
            "team_abbr": team,
            "team_pass_att": pass_att,
            "team_times_sacked": sacks,
            "team_dropbacks": dropbacks
        })
        time.sleep(0.6)  # be polite

    return pd.DataFrame(rows)

def main(season: int, tprr_default: float = 0.22):
    Path("data").mkdir(exist_ok=True)

    rec = fetch_league_receiving(season)
    if rec.empty:
        print("[pfr_pull] receiving page empty; writing placeholders")
        pd.DataFrame(columns=["player","team","routes_per_dropback","yprr_proxy_est"]).to_csv("data/pfr_player_enrich.csv", index=False)
        pd.DataFrame(columns=["team_abbr","team_pass_att","team_times_sacked","team_dropbacks"]).to_csv("data/pfr_team_enrich.csv", index=False)
        return 0

    teams = sorted(rec["team"].dropna().unique().tolist())
    team_db = fetch_team_dropbacks(season, teams)
    if team_db.empty:
        print("[pfr_pull] team dropbacks empty; writing players without route rates")
        rec.assign(routes_per_dropback=np.nan, yprr_proxy_est=np.nan)[["player","team","routes_per_dropback","yprr_proxy_est"]].to_csv("data/pfr_player_enrich.csv", index=False)
        pd.DataFrame(columns=["team_abbr","team_pass_att","team_times_sacked","team_dropbacks"]).to_csv("data/pfr_team_enrich.csv", index=False)
        return 0

    # Join team dropbacks onto player rows
    rec = rec.merge(team_db.rename(columns={"team_abbr":"team"}), on="team", how="left")

    # Estimate routes and route rate
    # routes_est = targets / TPRR; route_rate_est = routes_est / team_dropbacks
    # yprr_proxy_est = rec_yards / routes_est
    rec["routes_est"] = rec["targets"] / max(1e-9, tprr_default)
    rec["routes_per_dropback"] = rec["routes_est"] / rec["team_dropbacks"].replace(0, np.nan)
    rec["yprr_proxy_est"] = rec["rec_yards"] / rec["routes_est"].replace(0, np.nan)

    # Clamp to reasonable ranges
    rec["routes_per_dropback"] = rec["routes_per_dropback"].clip(lower=0, upper=1)
    # yprr often 1.2–3.2, but leave unclamped; downstream will sanity-check

    # Write outputs
    out_players = rec[["player","team","routes_per_dropback","yprr_proxy_est"]].copy()
    out_players.to_csv("data/pfr_player_enrich.csv", index=False)
    team_db.to_csv("data/pfr_team_enrich.csv", index=False)

    print(f"[pfr_pull] wrote data/pfr_player_enrich.csv rows={len(out_players)}")
    print(f"[pfr_pull] wrote data/pfr_team_enrich.csv rows={len(team_db)}")
    return 0

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=int(Path(".").read_text() if False else 2025))
    ap.add_argument("--tprr", type=float, default=0.22)
    args = ap.parse_args()
    sys.exit(main(args.season, args.tprr))
