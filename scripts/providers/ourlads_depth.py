#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/providers/ourlads_depth.py (corrected save path + postprocess)
import os, re, time, warnings
from typing import Dict
import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_DIR = "data"
OUT_ROLES = os.path.join(DATA_DIR, "roles_ourlads.csv")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DepthBot/1.0)"}

TEAM_URLS: Dict[str, str] = {
    "ARI":"https://www.ourlads.com/nfldepthcharts/depthchart/ARI",
    "ATL":"https://www.ourlads.com/nfldepthcharts/depthchart/ATL",
    "BAL":"https://www.ourlads.com/nfldepthcharts/depthchart/BAL",
    "BUF":"https://www.ourlads.com/nfldepthcharts/depthchart/BUF",
    "CAR":"https://www.ourlads.com/nfldepthcharts/depthchart/CAR",
    "CHI":"https://www.ourlads.com/nfldepthcharts/depthchart/CHI",
    "CIN":"https://www.ourlads.com/nfldepthcharts/depthchart/CIN",
    "CLE":"https://www.ourlads.com/nfldepthcharts/depthchart/CLE",
    "DAL":"https://www.ourlads.com/nfldepthcharts/depthchart/DAL",
    "DEN":"https://www.ourlads.com/nfldepthcharts/depthchart/DEN",
    "DET":"https://www.ourlads.com/nfldepthcharts/depthchart/DET",
    "GB":"https://www.ourlads.com/nfldepthcharts/depthchart/GB",
    "HOU":"https://www.ourlads.com/nfldepthcharts/depthchart/HOU",
    "IND":"https://www.ourlads.com/nfldepthcharts/depthchart/IND",
    "JAX":"https://www.ourlads.com/nfldepthcharts/depthchart/JAX",
    "KC":"https://www.ourlads.com/nfldepthcharts/depthchart/KC",
    "LAC":"https://www.ourlads.com/nfldepthcharts/depthchart/LAC",
    "LAR":"https://www.ourlads.com/nfldepthcharts/depthchart/LAR",
    "LV":"https://www.ourlads.com/nfldepthcharts/depthchart/LV",
    "MIA":"https://www.ourlads.com/nfldepthcharts/depthchart/MIA",
    "MIN":"https://www.ourlads.com/nfldepthcharts/depthchart/MIN",
    "NE":"https://www.ourlads.com/nfldepthcharts/depthchart/NE",
    "NO":"https://www.ourlads.com/nfldepthcharts/depthchart/NO",
    "NYG":"https://www.ourlads.com/nfldepthcharts/depthchart/NYG",
    "NYJ":"https://www.ourlads.com/nfldepthcharts/depthchart/NYJ",
    "PHI":"https://www.ourlads.com/nfldepthcharts/depthchart/PHI",
    "PIT":"https://www.ourlads.com/nfldepthcharts/depthchart/PIT",
    "SEA":"https://www.ourlads.com/nfldepthcharts/depthchart/SEA",
    "SF":"https://www.ourlads.com/nfldepthcharts/depthchart/SF",
    "TB":"https://www.ourlads.com/nfldepthcharts/depthchart/TB",
    "TEN":"https://www.ourlads.com/nfldepthcharts/depthchart/TEN",
    "WAS":"https://www.ourlads.com/nfldepthcharts/depthchart/WAS",
}

SUFFIX_RE = re.compile(r"\s+(JR|SR|II|III|IV|V)\.?$", re.IGNORECASE)
LEADING_NUM_RE = re.compile(r"^\s*(?:#\s*)?\d+\s*[-–—:]?\s*", re.UNICODE)

def _norm_player(name: str) -> str:
    if not isinstance(name, str): return ""
    s = name.replace(".", "").strip()
    s = LEADING_NUM_RE.sub("", s)
    s = SUFFIX_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _role_rank(role: str) -> int:
    m = re.search(r"(\d+)$", str(role))
    return int(m.group(1)) if m else 999

def fetch_team_roles(team: str) -> pd.DataFrame:
    url = TEAM_URLS[team]
    r = requests.get(url, headers=HEADERS, timeout=25); r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    rows = []
    for tr in soup.select("table tbody tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        pos = tds[0].get_text(" ", strip=True).upper()
        if pos not in {"QB","RB","WR","TE"}:
            continue
        depth_idx = 1
        for td in tds[1:]:
            a = td.find("a")
            raw = a.get_text(" ", strip=True) if a else td.get_text(" ", strip=True)
            raw = re.sub(r"\s*\(.*?\)\s*$", "", raw).strip()
            player = _norm_player(raw)
            if not player or player.isdigit():
                continue
            role = f"{pos}{depth_idx}"
            rows.append({"team": team, "player": player, "role": role})
            depth_idx += 1

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["rank"] = df["role"].map(_role_rank)
    df = (df.sort_values(["team","player","rank"])
            .drop_duplicates(["team","player"], keep="first")
            .drop(columns=["rank"]))
    return df

def _postprocess_roles_df_ourlads(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return df
    df = df.copy()
    df["player"] = df["player"].astype(str)
    df["player"] = df["player"].str.replace(r"^\s*(?:#\s*)?\d+\s*[-–—:]?\s*", "", regex=True)
    df = df[~df["player"].str.fullmatch(r"\d+")]
    if "role" in df.columns:
        rk = df["role"].astype(str).str.extract(r"(\d+)$", expand=False).astype(float).fillna(999).astype(int)
        df = df.assign(_rk=rk).sort_values(["team","player","_rk"]).drop_duplicates(["team","player"], keep="first").drop(columns=["_rk"])
    return df

def main():
    warnings.simplefilter("ignore")
    os.makedirs(DATA_DIR, exist_ok=True)
    all_roles = []
    for tm in sorted(TEAM_URLS.keys()):
        try:
            df = fetch_team_roles(tm)
            if not df.empty:
                all_roles.append(df)
        except Exception as e:
            print(f"[ourlads_depth] WARN: failed {tm}: {e}", flush=True)
        time.sleep(0.4)

    roles = (pd.concat(all_roles, ignore_index=True).drop_duplicates()
             if all_roles else pd.DataFrame(columns=["team","player","role"]))
    roles = _postprocess_roles_df_ourlads(roles)
    roles.to_csv(OUT_ROLES, index=False)
    print(f"[ourlads_depth] wrote rows={len(roles)} → {OUT_ROLES}")

if __name__ == "__main__":
    main()
