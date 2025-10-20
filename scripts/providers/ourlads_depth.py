#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/providers/ourlads_depth.py  (surgically patched)

import os, re, time, warnings
from typing import Dict, List
import pandas as pd
import requests
from bs4 import BeautifulSoup

VALID = {"ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU",
         "IND","JAX","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG","NYJ",
         "PHI","PIT","SEA","SF","TB","TEN","WAS"}

# ---- Team aliases (canonicalization) ----
TEAM_ALIASES = {
    "ARZ": "ARI",
    "JAC": "JAX",
    "WSH": "WAS",
    "LA":  "LAR",
    "STL": "LAR",
    "SD":  "LAC",
    "OAK": "LV",
}
def _canon_team(code: str) -> str:
    code = (code or "").upper().strip()
    return TEAM_ALIASES.get(code, code)

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
TAG_CODE_RE = re.compile(r"\b[A-Z]{1,3}\d{2}\b")       # SF23, CF25, RS22...
UDFA_SCHOOL_RE = re.compile(r"\b[Uu]/[A-Za-z]{2,4}\b") # U/Min, U/Mia...
DATE_FRACTION_RE = re.compile(r"\b\d{1,2}/\d{1,2}\b")  # 22/5

def _norm_player(name: str) -> str:
    """Normalize OurLads player strings to 'First Last' (title case) free of debris."""
    if not isinstance(name, str):
        return ""
    s = name.strip()
    s = LEADING_NUM_RE.sub("", s)  # strip jersey numbers like "#17 -", "12—"
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            s = f"{parts[1]} {parts[0]}"
    s = re.sub(r"\(.*?\)", "", s)
    s = TAG_CODE_RE.sub("", s)
    s = UDFA_SCHOOL_RE.sub("", s)
    s = DATE_FRACTION_RE.sub("", s)
    s = SUFFIX_RE.sub("", s)
    s = s.replace(".", " ")
    s = re.sub(r"[^\w\s'\-]", " ", s)
    s = re.sub(r"\b\d+\b", "", s)
    s = re.sub(r"\s+", " ", s).strip().title()
    return "" if s == "U" else s

SPLIT_RE = re.compile(r"(?:<br\s*/?>|/| & | and )", flags=re.I)

def _split_candidates(cell_text: str) -> List[str]:
    if not cell_text:
        return []
    parts = re.split(SPLIT_RE, cell_text)
    out: List[str] = []
    for p in parts:
        txt = BeautifulSoup(p, "lxml").get_text(" ", strip=True)
        nm = _norm_player(txt)
        if nm and len(nm.split()) >= 2 and nm.upper() not in VALID:
            out.append(nm)
    return out

def _role_rank(role: str) -> int:
    m = re.search(r"(\d+)$", str(role))
    return int(m.group(1)) if m else 999

def _normalize_pos(p: str) -> str:
    p = (p or "").upper().strip()
    # WR aliases → WR (this is what your screenshot shows: LWR/RWR/SWR)
    if ("WR" in p) or p in {"SL","SLOT","FL","SE","SWR","LWR","RWR"}:
        return "WR"
    # RB aliases → RB
    if p in {"RB","HB","TB","FB"}:
        return "RB"
    # TE aliases → TE (Y/H frequently used)
    if p in {"TE","Y","H"}:
        return "TE"
    return p

# ----------------------------
# Scrape & parse (depth chart)
# ----------------------------

def _get_depth_soup(team: str) -> BeautifulSoup:
    url = TEAM_URLS[team]
    r = requests.get(url, headers=HEADERS, timeout=25); r.raise_for_status()
    return BeautifulSoup(r.text, "lxml")

def fetch_team_roles(team: str, soup: BeautifulSoup) -> pd.DataFrame:
    rows: List[dict] = []

    # Only parse the OFFENSE depth-chart table
    off = soup.find(lambda tag: tag.name in ["h2","h3"] and "OFFENSE" in tag.get_text().upper())
    if not off:
        return pd.DataFrame(columns=["team","player","role"])
    table = off.find_next("table")
    if not table:
        return pd.DataFrame(columns=["team","player","role"])

    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue

        pos_raw = tds[0].get_text(" ", strip=True).upper()
        pos = _normalize_pos(pos_raw)
        if pos not in {"QB","RB","WR","TE"}:
            continue

        # Columns = 1,2,3,... → WR1/WR2/WR3 etc; keep in-cell order for co-starters
        for depth_idx, td in enumerate(tds[1:], start=1):
            a = td.find("a")
            raw = a.get_text(" ", strip=True) if a else td.get_text(" ", strip=True)
            raw = re.sub(r"\s*\(.*?\)\s*$", "", raw).strip()
            candidates = _split_candidates(raw)
            if not candidates:
                continue
            for slot_order, player in enumerate(candidates, start=1):
                rows.append({
                    "team": team,
                    "player": player,
                    "role": f"{pos}{depth_idx}",
                    "orig_depth": depth_idx,
                    "slot_order": slot_order,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Prefer lowest numeric depth per (team,player) if duplicated
    df["rank"] = df["role"].map(_role_rank)
    df = (
        df.sort_values(["team","player","rank"])
          .drop_duplicates(["team","player"], keep="first")
          .drop(columns=["rank"])
    )
    return df

# ----------------------------
# Scrape roster (fallback coverage)
# ----------------------------

def fetch_team_roster(team: str, soup_depth: BeautifulSoup) -> pd.DataFrame:
    """
    Follow the 'Roster' link from the team's depth-chart page and parse the roster table.
    We keep only offense positions QB/RB/WR/TE and normalize player names.
    """
    try:
        roster_link = None
        for a in soup_depth.find_all("a", href=True):
            h = a.get("href","")
            t = a.get_text(" ", strip=True).upper()
            if "ROSTER" in t and "/nfldepthcharts/roster/" in h:
                roster_link = h if h.startswith("http") else f"https://www.ourlads.com{h}"
                break
        if not roster_link:
            return pd.DataFrame(columns=["team","player","pos_roster"])

        r = requests.get(roster_link, headers=HEADERS, timeout=25); r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        table = soup.find("table")
        if not table:
            return pd.DataFrame(columns=["team","player","pos_roster"])

        rows = []
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue
            # Guess first two cols as position and player (site is consistent here)
            pos_txt = BeautifulSoup(tds[0].decode_contents(), "lxml").get_text(" ", strip=True).upper()
            name_txt = BeautifulSoup(tds[1].decode_contents(), "lxml").get_text(" ", strip=True)
            pos = _normalize_pos(pos_txt)
            if pos not in {"QB","RB","WR","TE"}:
                continue
            player = _norm_player(name_txt)
            if not player:
                continue
            rows.append({"team": team, "player": player, "pos_roster": pos})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=["team","player","pos_roster"])

# ----------------------------
# Depth post-processing
# ----------------------------

def _reassign_sequential_depth(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure unique depth numbers per (team, position) using (orig_depth, slot_order)."""
    if df is None or getattr(df, "empty", True):
        return df
    df = df.copy()
    df["pos"] = df["role"].str.extract(r"([A-Z]+)")
    if "orig_depth" not in df.columns:
        df["orig_depth"] = df["role"].str.extract(r"(\d+)").astype(int)
    if "slot_order" not in df.columns:
        df["slot_order"] = 1

    def assign(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(["orig_depth", "slot_order"], na_position="last").copy()
        g["depth"] = range(1, len(g) + 1)
        g["role"] = g["pos"].iloc[0] + g["depth"].astype(str)
        return g.drop(columns=["depth"])
    return df.groupby(["team", "pos"], group_keys=False).apply(assign)

def _fi_last(s: str) -> str:
    toks = s.strip().split()
    if not toks:
        return ""
    first, last = toks[0], toks[-1]
    return f"{first[0].lower()} {last.lower()}" if last else s.lower()

def _fi_last_concat(s: str) -> str:
    toks = s.strip().split()
    if not toks:
        return ""
    first, last = toks[0], toks[-1]
    return f"{first[0].upper()}{last.title().replace(' ', '')}"

def _postprocess_roles_df_ourlads(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or getattr(df, "empty", True):
        return df.copy()

    df = df.copy()
    df["team"] = df["team"].map(_canon_team)
    df = df[df["team"].isin(VALID)]

    df["player"] = df["player"].astype(str)
    df["player"] = df["player"].str.replace(r"^\s*(?:#\s*)?\d+\s*[-–—:]?\s*", "", regex=True)
    df = df[~df["player"].str.fullmatch(r"\d+")]

    if "role" in df.columns:
        rk = (
            df["role"].astype(str).str.extract(r"(\d+)$", expand=False)
              .astype(float).fillna(999).astype(int)
        )
        df = (
            df.assign(_rk=rk)
              .sort_values(["team","player","_rk"])
              .drop_duplicates(["team","player"], keep="first")
              .drop(columns=["_rk"])
        )

    df = _reassign_sequential_depth(df)

    df["player_key_filast"]  = df["player"].map(_fi_last)
    df["player_key_concat"]  = df["player"].map(_fi_last_concat)
    return df

# ----------------------------
# Main
# ----------------------------

def main():
    warnings.simplefilter("ignore")
    os.makedirs(DATA_DIR, exist_ok=True)

    all_dfs = []

    for tm in sorted(TEAM_URLS.keys()):
        try:
            soup = _get_depth_soup(tm)

            # 1) depth chart roles
            roles = fetch_team_roles(tm, soup)

            # 2) roster fallback (offense only)
            roster = fetch_team_roster(tm, soup)

            # 3) union: keep all roster players; overlay roles where present
            if not roster.empty:
                # left-merge on normalized player name
                r = roster.copy()
                r["player_join"] = r["player"].str.replace(r"[^A-Za-z]", "", regex=True)
                if not roles.empty:
                    d = roles.copy()
                    d["player_join"] = d["player"].str.replace(r"[^A-Za-z]", "", regex=True)
                    merged = r.merge(
                        d[["team","player_join","role","orig_depth","slot_order"]],
                        on=["player_join"], how="left", suffixes=("","_d")
                    )
                    # ensure correct team if any discrepancy
                    merged["team"] = tm
                    merged.drop(columns=["player_join"], inplace=True, errors="ignore")
                    df_team = merged[["team","player","role","orig_depth","slot_order"]].copy()
                else:
                    # only roster available → no roles
                    df_team = r[["team","player"]].copy()
                    df_team["role"] = pd.NA
                    df_team["orig_depth"] = pd.NA
                    df_team["slot_order"] = pd.NA
            else:
                df_team = roles

            if df_team is not None and not df_team.empty:
                all_dfs.append(df_team)

        except Exception as e:
            print(f"[ourlads_depth] WARN: failed {tm}: {e}", flush=True)

        time.sleep(0.4)  # be polite

    roles_all = (
        pd.concat(all_dfs, ignore_index=True).drop_duplicates()
        if all_dfs else pd.DataFrame(columns=["team","player","role"])
    )
    roles_all = _postprocess_roles_df_ourlads(roles_all)
    roles_all.to_csv(OUT_ROLES, index=False)
    print(f"[ourlads_depth] wrote rows={len(roles_all)} → {OUT_ROLES}")

if __name__ == "__main__":
    main()
