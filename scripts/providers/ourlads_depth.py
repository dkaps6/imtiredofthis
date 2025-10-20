#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/providers/ourlads_depth.py (corrected save path + postprocess)

import os, re, time, warnings
from typing import Dict, List
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

# ----------------------------
# Normalization helpers (improved)
# ----------------------------

SUFFIX_RE = re.compile(r"\s+(JR|SR|II|III|IV|V)\.?$", re.IGNORECASE)
LEADING_NUM_RE = re.compile(r"^\s*(?:#\s*)?\d+\s*[-–—:]?\s*", re.UNICODE)

# Additional debris often seen in OurLads cells (draft/school/UDFA tags):
TAG_CODE_RE = re.compile(r"\b[A-Z]{1,3}\d{2}\b")       # SF23, CF25, RS22...
UDFA_SCHOOL_RE = re.compile(r"\b[Uu]/[A-Za-z]{2,4}\b") # U/Min, U/Mia...
DATE_FRACTION_RE = re.compile(r"\b\d{1,2}/\d{1,2}\b")  # 22/5

def _norm_player(name: str) -> str:
    """
    Normalize OurLads player strings:
    - Swap 'Last, First' -> 'First Last'
    - Remove jersey numbers, dots, suffixes, parentheses, and debris tags (e.g., 'SF23', 'U/Min', '22/5')
    - Collapse whitespace; Title Case for readability
    """
    if not isinstance(name, str):
        return ""
    s = name.strip()
    # Strip leading jersey numbers like "#17 -", "12—"
    s = LEADING_NUM_RE.sub("", s)

    # Convert "Last, First" → "First Last"
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            s = f"{parts[1]} {parts[0]}"

    # Kill parentheses blocks and common debris tags
    s = re.sub(r"\(.*?\)", "", s)
    s = TAG_CODE_RE.sub("", s)
    s = UDFA_SCHOOL_RE.sub("", s)
    s = DATE_FRACTION_RE.sub("", s)

    # Remove suffixes (Jr, Sr, II, III, IV, V)
    s = SUFFIX_RE.sub("", s)

    # Normalize punctuation/spaces
    s = s.replace(".", " ")
    s = re.sub(r"[^\w\s'\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Title-case for consistent human-readable text
    return s.title()

# For splitting multi-name cells (co-starters)
SPLIT_RE = re.compile(r"(?:<br\s*/?>|/| & | and )", flags=re.I)

def _split_candidates(cell_text: str) -> List[str]:
    if not cell_text:
        return []
    parts = re.split(SPLIT_RE, cell_text)
    out: List[str] = []
    for p in parts:
        txt = BeautifulSoup(p, "lxml").get_text(" ", strip=True)
        nm = _norm_player(txt)
        if nm and not nm.isdigit():
            out.append(nm)
    return out

# Optional: standardize some position variants to stable buckets
# POS_MAP = {
#     "WR-X":"WR","WR-Z":"WR","WR-Y":"TE","SL":"WR","FB":"RB",
#     "LEO":"EDGE","JACK":"EDGE","STAR":"NB"
# }

def _role_rank(role: str) -> int:
    m = re.search(r"(\d+)$", str(role))
    return int(m.group(1)) if m else 999

# ----------------------------
# Scrape & parse
# ----------------------------

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
        # pos = POS_MAP.get(pos, pos)  # enable if you decide to standardize
        if pos not in {"QB","RB","WR","TE"}:
            continue

        depth_idx = 1
        for td in tds[1:]:
            a = td.find("a")
            raw = a.get_text(" ", strip=True) if a else td.get_text(" ", strip=True)
            raw = re.sub(r"\s*\(.*?\)\s*$", "", raw).strip()

            # NEW: split multi-name cells → each candidate becomes its own row at the same depth
            candidates = _split_candidates(raw)
            if not candidates:
                continue

            for player in candidates:
                role = f"{pos}{depth_idx}"
                rows.append({"team": team, "player": player, "role": role})

            depth_idx += 1

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["rank"] = df["role"].map(_role_rank)
    df = (
        df.sort_values(["team","player","rank"])
          .drop_duplicates(["team","player"], keep="first")
          .drop(columns=["rank"])
    )
    return df

# ----------------------------
# Post-processing & join keys
# ----------------------------

def _fi_last(s: str) -> str:
    """
    'First Last' -> 'f last' (lowercase) — helpful for space-separated joins.
    """
    toks = s.strip().split()
    if not toks:
        return ""
    first, last = toks[0], toks[-1]
    return f"{first[0].lower()} {last.lower()}" if last else s.lower()

def _fi_last_concat(s: str) -> str:
    """
    'First Last' -> 'FLast' — matches your player_form keys like 'DJohnson', 'EDemercado'.
    """
    toks = s.strip().split()
    if not toks:
        return ""
    first, last = toks[0], toks[-1]
    return f"{first[0].upper()}{last.title().replace(' ', '')}"

def _postprocess_roles_df_ourlads(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return df
    df = df.copy()

    # Keep your existing cleanup
    df["player"] = df["player"].astype(str)
    df["player"] = df["player"].str.replace(r"^\s*(?:#\s*)?\d+\s*[-–—:]?\s*", "", regex=True)
    df = df[~df["player"].str.fullmatch(r"\d+")]

    # Ensure best (lowest) depth per team/player
    if "role" in df.columns:
        rk = (
            df["role"]
            .astype(str)
            .str.extract(r"(\d+)$", expand=False)
            .astype(float)
            .fillna(999)
            .astype(int)
        )
        df = (
            df.assign(_rk=rk)
              .sort_values(["team","player","_rk"])
              .drop_duplicates(["team","player"], keep="first")
              .drop(columns=["_rk"])
        )

    # NEW: deterministic join keys for your merges
    df["player_key_filast"] = df["player"].map(_fi_last)        # e.g., 'd johnson'
    df["player_key_concat"] = df["player"].map(_fi_last_concat) # e.g., 'DJohnson'

    return df

# ----------------------------
# Main
# ----------------------------

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
        time.sleep(0.4)  # be polite

    roles = (
        pd.concat(all_roles, ignore_index=True).drop_duplicates()
        if all_roles else pd.DataFrame(columns=["team","player","role"])
    )
    roles = _postprocess_roles_df_ourlads(roles)
    roles.to_csv(OUT_ROLES, index=False)
    print(f"[ourlads_depth] wrote rows={len(roles)} → {OUT_ROLES}")

if __name__ == "__main__":
    main()
