#!/usr/bin/env python3
"""
Pro-Football-Reference (PFR) fallback to estimate routes-per-dropback.

Outputs:
- data/pfr_player_enrich.csv  with columns:
    player, team, routes_per_dropback, yprr_proxy_est
- data/pfr_team_enrich.csv    with columns:
    team_abbr, team_pass_att, team_times_sacked, team_dropbacks
"""
from __future__ import annotations
import sys, time, re
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment  # ← NEW: Comment

LEAGUE_RECEIVING_URL = "https://www.pro-football-reference.com/years/{season}/receiving.htm"
TEAM_PAGE_URL        = "https://www.pro-football-reference.com/teams/{pfr}/{season}.htm"

# Map PFR team codes -> your standard codes
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

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DepthBot/1.0)"}

# ── NEW: small retry wrapper with 403-safe behavior ───────────────────────────
def _get_html(url: str, tries: int = 3, sleep: float = 0.8) -> str:
    for k in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            # 403 → return empty html so caller can gracefully proceed to placeholders
            if r.status_code == 403:
                return ""
            r.raise_for_status()
            return r.text
        except requests.HTTPError as e:
            # If not a 403 or on last try, propagate
            if getattr(e.response, "status_code", None) == 403:
                return ""
            if k == tries - 1:
                raise
            time.sleep(sleep)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(sleep)
    # Should not reach here
    return ""

def _read_html(url: str) -> BeautifulSoup:
    return BeautifulSoup(_get_html(url), "lxml")

# ── include commented tables in the HTML we give to read_html ────────────────
def _html_with_uncomment(soup: BeautifulSoup) -> str:
    parts = [str(soup)]
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        text = str(c)
        if "<table" in text:
            parts.append(text)
    return "\n".join(parts)

def _clean_num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return np.nan

def fetch_league_receiving(season: int) -> pd.DataFrame:
    """Return columns: player, team (STD), targets, rec_yards"""
    url = LEAGUE_RECEIVING_URL.format(season=season)
    soup = _read_html(url)

    # If we were 403-blocked, _get_html returned ""; soup will be empty → no tables → empty DF
    html = _html_with_uncomment(soup)
    dfs = pd.read_html(html, match="Receiving", flavor="lxml") if "table" in html else []
    if not dfs:
        table = soup.find("table", {"id": "receiving"})
        if table is None:
            return pd.DataFrame()
        dfs = pd.read_html(str(table))

    if not dfs:
        return pd.DataFrame()
    df = dfs[0]

    cols = [re.sub(r"[\W]+", "_", str(c)).strip("_").lower() for c in df.columns]
    df.columns = cols

    rename_map = {}
    if "player" in df.columns: rename_map["player"] = "player"
    if "tm" in df.columns:     rename_map["tm"]     = "team_pfr"
    if "tgt" in df.columns:    rename_map["tgt"]    = "targets"
    if "yds" in df.columns:    rename_map["yds"]    = "rec_yards"
    df = df.rename(columns=rename_map)

    keep = [c for c in ["player","team_pfr","targets","rec_yards"] if c in df.columns]
    df = df[keep].copy()

    df["targets"]   = pd.to_numeric(df["targets"], errors="coerce")
    df["rec_yards"] = pd.to_numeric(df["rec_yards"], errors="coerce")
    df = df[df["player"].astype(str).str.lower() != "team total"]

    df["team"] = df["team_pfr"].map(PFR_TO_STD).fillna(df["team_pfr"].astype(str).str.upper())
    df = df.dropna(subset=["team"])
    return df[["player","team","targets","rec_yards"]]

def fetch_team_dropbacks(season: int, teams_std: list[str]) -> pd.DataFrame:
    """Return columns: team_abbr, team_pass_att, team_times_sacked, team_dropbacks"""
    rows = []
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

        html = _html_with_uncomment(soup)
        dfs = pd.read_html(html) if "table" in html else []

        pass_att = np.nan
        sacks    = np.nan

        # Heuristic #1: totals row in a table containing Att & Sk
        for tdf in dfs:
            low = [str(c).lower() for c in tdf.columns]
            if any("att" in c for c in low) and any(("sk" in c) or ("sack" in c) for c in low):
                t = tdf.copy()
                last = t.tail(1)
                cand_pass = [c for c in t.columns if str(c).lower() in ("att","pass att","pass_att")]
                cand_sack = [c for c in t.columns if str(c).lower() in ("sk","times sacked","times_sacked","sacked")]
                try: pass_att = _clean_num(last[cand_pass[0]].values[0])
                except Exception: pass
                try: sacks = _clean_num(last[cand_sack[0]].values[0])
                except Exception: pass
                if np.isfinite(pass_att) or np.isfinite(sacks):
                    break

        # Heuristic #2: sum columns if no clear totals row
        if not np.isfinite(pass_att) or not np.isfinite(sacks):
            for tdf in dfs:
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
        time.sleep(0.5)  # be polite

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

    rec = rec.merge(team_db.rename(columns={"team_abbr":"team"}), on="team", how="left")
    rec["routes_est"] = rec["targets"] / max(1e-9, tprr_default)
    rec["routes_per_dropback"] = rec["routes_est"] / rec["team_dropbacks"].replace(0, np.nan)
    rec["yprr_proxy_est"] = rec["rec_yards"] / rec["routes_est"].replace(0, np.nan)
    rec["routes_per_dropback"] = rec["routes_per_dropback"].clip(lower=0, upper=1)

    out_players = rec[["player","team","routes_per_dropback","yprr_proxy_est"]].copy()
    out_players.to_csv("data/pfr_player_enrich.csv", index=False)
    team_db.to_csv("data/pfr_team_enrich.csv", index=False)

    print(f"[pfr_pull] wrote data/pfr_player_enrich.csv rows={len(out_players)}")
    print(f"[pfr_pull] wrote data/pfr_team_enrich.csv rows={len(team_db)}")
    return 0

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--tprr", type=float, default=0.22)
    args = ap.parse_args()
    sys.exit(main(args.season, args.tprr))
