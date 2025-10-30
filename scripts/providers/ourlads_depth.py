#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/providers/ourlads_depth.py  (hardened: retries + robust selectors + roster URL fallback)

import os, re, time, warnings, sys

from typing import Dict, List, Optional
import pandas as pd
import requests
from bs4 import BeautifulSoup

VALID = {"ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU",
         "IND","JAX","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG","NYJ",
         "PHI","PIT","SEA","SF","TB","TEN","WAS"}

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
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DepthBot/1.0; +https://example.com)"}

TEAM_URLS: Dict[str, str] = {
    "ARI":"https://www.ourlads.com/nfldepthcharts/depthchart/ARZ",
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
TAG_CODE_RE = re.compile(r"\b[A-Z]{1,3}\d{2}\b")
UDFA_SCHOOL_RE = re.compile(r"\b[Uu]/[A-Za-z]{2,4}\b")
DATE_FRACTION_RE = re.compile(r"\b\d{1,2}/\d{1,2}\b")

def clean_ourlads_name(raw: str) -> str:
    """
    Convert 'Allen, Josh 18/1' -> 'Josh Allen'.
    Strip draft/UDFA markers like '18/1', 'CF23', 'U/LAC', etc.
    Keep only letters, spaces, periods, hyphens in the final name.
    """
    if not isinstance(raw, str):
        return ""

    parts = [p.strip() for p in raw.split(",", 1)]
    if len(parts) == 2:
        last, first_and_junk = parts[0], parts[1]
        base = f"{first_and_junk} {last}"
    else:
        base = raw

    tokens = []
    for tok in re.split(r"\s+", base):
        if re.search(r"[\d/]", tok):
            continue
        tokens.append(tok)

    clean = " ".join(tokens).strip()
    clean = re.sub(r"\s+", " ", clean)
    clean = clean.replace(",", "").strip()
    clean = re.sub(r"[^A-Za-z.\-\s]", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()

    return clean

def _norm_player(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = clean_ourlads_name(name)
    s = LEADING_NUM_RE.sub("", s)
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
        nm = clean_ourlads_name(txt)
        nm = _norm_player(nm)
        if nm and len(nm.split()) >= 2 and nm.upper() not in VALID:
            out.append(nm)
    return out


OL_POSITIONS = {"LT", "LG", "C", "RG", "RT"}

ROLE_SLOT_MAPPING = {
    ("LWR", "player 1"): "WR1",
    ("RWR", "player 1"): "WR2",
    ("SWR", "player 1"): "WR3",
    ("QB", "player 1"): "QB1",
    ("RB", "player 1"): "RB1",
    ("TB", "player 1"): "RB1",
    ("HB", "player 1"): "RB1",
    ("TE", "player 1"): "TE1",
}


def map_role(base_pos: str, depth_slot: str) -> Optional[str]:
    base = (base_pos or "").upper().strip()
    slot = re.sub(r"\s+", " ", (depth_slot or "").strip().lower())
    return ROLE_SLOT_MAPPING.get((base, slot))

# ----------------------------
# Robust fetch with retries
# ----------------------------
def _get_html(url: str, max_tries: int = 3, backoff: float = 0.8) -> str:
    last = None
    for i in range(1, max_tries+1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
            if r.status_code == 200 and r.text:
                return r.text
            last = f"{r.status_code}"
        except Exception as e:
            last = str(e)
        time.sleep(backoff * i)
    raise RuntimeError(f"GET failed for {url} → {last}")

def _get_depth_soup(team: str) -> BeautifulSoup:
    url = TEAM_URLS[team]
    html = _get_html(url)
    return BeautifulSoup(html, "lxml")

# ----------------------------
# OFFENSE table detection
# ----------------------------
def _find_offense_table(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    # 1) Preferred: header "OFFENSE" then first table
    off = soup.find(lambda tag: tag.name in ["h1","h2","h3","h4"] and "OFFENSE" in tag.get_text().upper())
    if off:
        tbl = off.find_next("table")
        if tbl:
            return tbl
    # 2) Fall back: any table whose first column looks like positions
    pos_like = {"QB","RB","WR","TE","HB","TB","FB","FL","SE","SLOT","SL","Y","H"}
    for tbl in soup.find_all("table"):
        rows = tbl.find_all("tr")
        if not rows:
            continue
        # examine up to first 6 body rows
        score, seen = 0, 0
        for tr in rows[:6]:
            tds = tr.find_all(["td","th"])
            if not tds:
                continue
            col0 = BeautifulSoup(tds[0].decode_contents(), "lxml").get_text(" ", strip=True).upper()
            if col0 in pos_like or col0.startswith("WR"):
                score += 1
            seen += 1
        if seen and score >= max(2, seen//2):  # at least half of sampled rows look like positions
            return tbl
    return None

def _extract_table_structure(table: BeautifulSoup) -> (List[str], List[BeautifulSoup]):
    rows = table.find_all("tr")
    if not rows:
        return [], []

    def _cell_text(cell: BeautifulSoup) -> str:
        return BeautifulSoup(cell.decode_contents(), "lxml").get_text(" ", strip=True)

    header_cells = rows[0].find_all(["th", "td"])
    header_has_th = any(cell.name == "th" for cell in header_cells)

    if header_has_th:
        header = []
        for idx, cell in enumerate(header_cells):
            text = _cell_text(cell)
            if not text:
                text = "Position" if idx == 0 else f"Player {idx}"
            else:
                match = re.match(r"player\s*(\d+)", text, flags=re.I)
                if match:
                    text = f"Player {match.group(1)}"
            header.append(text.strip())
        data_rows = rows[1:]
    else:
        header = ["Position"]
        header.extend(f"Player {idx}" for idx in range(1, len(header_cells)))
        data_rows = rows

    return header, data_rows


def fetch_team_roles(team: str, soup: BeautifulSoup) -> List[dict]:
    table = _find_offense_table(soup)
    if not table:
        return []

    header, data_rows = _extract_table_structure(table)
    if not data_rows:
        return []

    try:
        player1_idx = next(
            idx
            for idx, label in enumerate(header)
            if idx > 0 and re.sub(r"\s+", " ", label.strip()).lower() == "player 1"
        )
    except StopIteration:
        player1_idx = 1 if len(header) > 1 else None

    if player1_idx is None:
        return []

    records: List[dict] = []
    team_code = _canon_team(team)

    for tr in data_rows:
        tds = tr.find_all("td")
        if len(tds) <= player1_idx:
            continue

        pos_raw = BeautifulSoup(tds[0].decode_contents(), "lxml").get_text(" ", strip=True).upper()
        base_pos = pos_raw.strip().upper()
        if not base_pos or base_pos in OL_POSITIONS:
            continue

        depth_slot_label = header[player1_idx] if player1_idx < len(header) else "Player 1"
        depth_slot_label = re.sub(r"\s+", " ", depth_slot_label).strip() or "Player 1"

        cell_text = BeautifulSoup(tds[player1_idx].decode_contents(), "lxml").get_text(" ", strip=True)
        cell_text = re.sub(r"\s*\(.*?\)\s*$", "", cell_text).strip()
        candidates = _split_candidates(cell_text)
        if not candidates:
            continue

        player_name = clean_ourlads_name(candidates[0])
        if not player_name:
            continue
        player_name = _norm_player(player_name)
        role = map_role(base_pos, depth_slot_label)
        if not role:
            continue

        records.append(
            {
                "team": team_code,
                "player": player_name,
                "position": base_pos,
                "depth_slot": "Player 1",
                "role": role,
            }
        )

    return records


# ----------------------------
# Main
# ----------------------------
def main():
    warnings.simplefilter("ignore")
    os.makedirs(DATA_DIR, exist_ok=True)

    all_rows: List[dict] = []
    for tm in sorted(TEAM_URLS.keys()):
        try:
            soup = _get_depth_soup(tm)
            team_rows = fetch_team_roles(tm, soup)
            if not team_rows:
                print(f"[ourlads_depth] NOTE: 0 Player 1 rows for {tm}", file=sys.stderr)
            else:
                all_rows.extend(team_rows)
        except Exception as e:
            print(f"[ourlads_depth] WARN: failed {tm}: {e}", flush=True)

        time.sleep(0.5)  # polite

    if all_rows:
        roles_all = pd.DataFrame(all_rows)
    else:
        roles_all = pd.DataFrame(columns=["team", "player", "position", "depth_slot", "role"])

    roles_all = roles_all.dropna(subset=["player", "role"])
    if not roles_all.empty:
        roles_all["player"] = roles_all["player"].map(clean_ourlads_name)
        roles_all = roles_all[roles_all["player"] != ""]
        roles_all["player"] = roles_all["player"].map(_norm_player)
        roles_all = roles_all[roles_all["player"] != ""]
        roles_all["team"] = roles_all["team"].astype(str).str.upper().str.strip().map(_canon_team)
        roles_all = roles_all[roles_all["team"].isin(VALID)]
        roles_all = roles_all.drop_duplicates(subset=["team", "player", "role"])
        roles_all = roles_all.sort_values(["team", "role", "player"])

    output_df = roles_all[["team", "player", "role", "position"]].copy()
    output_df.to_csv(OUT_ROLES, index=False)
    print(f"[ourlads_depth] wrote rows={len(output_df)} → {OUT_ROLES}")

if __name__ == "__main__":
    main()
