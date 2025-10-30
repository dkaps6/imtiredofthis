#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/providers/ourlads_depth.py
#
# Scrapes Ourlads depth charts for every NFL team and writes a clean,
# canonical roles file:
#   data/roles_ourlads.csv  with cols: team,player,role,position
#
# Key details:
# - Only "Player 1" at fantasy positions (QB/RB/TE/LWR/RWR/SWR)
# - OL (LT/LG/C/RG/RT) is ignored
# - Maps:
#     LWR -> WR1
#     RWR -> WR2
#     SWR -> WR3
#     QB  -> QB1
#     RB -> RB1
#     TE  -> TE1
# - Cleans names to sportsbook style:
#     "Allen, Josh 18/1 cc" -> "Josh Allen"
#     removes "cc", "ps", "ir", "pup", "22/5", "U/LAC", "(R)", etc.
# - Canonicalizes team abbreviations (BUF, KC, WAS, etc.)

import os, re, time, warnings, sys
from typing import Dict, List, Optional
import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_DIR = "data"
OUT_ROLES = os.path.join(DATA_DIR, "roles_ourlads.csv")

# canonical team set
VALID = {
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU",
    "IND","JAX","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG","NYJ",
    "PHI","PIT","SEA","SF","TB","TEN","WAS"
}

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
    code = TEAM_ALIASES.get(code, code)
    return code if code in VALID else code

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; DepthBot/1.0; +https://example.com)"
    )
}

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

BAD_TOKENS = {
    "cc", "ps", "ir", "pup", "rs", "na", "out", "doubtful", "questionable",
    "inactive", "probable", "act", "act.", "actv", "res", "res.", "p-sq",
    "(r)", "(ir)", "(ps)", "(pup)"
}


def clean_ourlads_name(raw: str) -> str:
    """
    Convert Ourlads depth text like:
      'Allen, Josh 18/1 cc'
      'Kelce, Travis CF23'
      'Pacheco, Isiah 22/5'
      'Brown, Amon-Ra (R)'
    into canonical sportsbook-style:
      'Josh Allen'
      'Travis Kelce'
      'Isiah Pacheco'
      'Amon-Ra Brown'

    Rules:
    - Flip 'Last, First ...' -> 'First Last'
    - Drop tokens that contain digits or '/' (draft/UDFA markers like '22/5', 'CF23', 'U/LAC')
    - Drop common junk/status tokens like 'cc', 'ps', 'ir', '(R)', 'pup', etc.
    - Strip commas/parentheses and collapse whitespace.
    """

    if not isinstance(raw, str):
        return ""

    # 1. Flip "Last, First ..." -> "First Last ..."
    parts = [p.strip() for p in raw.split(",", 1)]
    if len(parts) == 2:
        last, first_rest = parts[0], parts[1]
        candidate = f"{first_rest} {last}"
    else:
        candidate = raw

    # 2. Token clean
    cleaned_tokens = []
    for tok in re.split(r"\s+", candidate):
        tnorm = tok.lower().strip(",()")

        # Drop obvious garbage:
        # - numeric or slash tokens (draft round, UDFA markers)
        # - bad status/junk tokens (cc, ps, ir, pup, etc.)
        if re.search(r"[\d/]", tok):
            continue
        if tnorm in BAD_TOKENS:
            continue

        # Drop lone punctuation leftovers
        if tnorm in {"", "-", "--"}:
            continue

        cleaned_tokens.append(tok.strip(",()"))

    # 3. Rebuild "Firstname Lastname" string
    name = " ".join(cleaned_tokens)
    name = re.sub(r"\s+", " ", name).strip()

    return name


# regex to split multi-player cells
SPLIT_RE = re.compile(r"(?:<br\s*/?>|/| & | and )", flags=re.I)


def _split_candidates(cell_text: str) -> List[str]:
    if not cell_text:
        return []
    parts = re.split(SPLIT_RE, cell_text)
    out: List[str] = []
    for p in parts:
        txt = BeautifulSoup(p, "lxml").get_text(" ", strip=True)
        if txt:
            out.append(txt)
    return out


# offensive line positions we IGNORE
OL_POSITIONS = {"LT","LG","C","RG","RT"}

# how we convert Ourlads position column + "Player 1" slot into fantasy role
POSITION_ALIASES = {
    "HB": "RB",
    "TB": "RB",
}

ROLE_SLOT_MAPPING = {
    "LWR": "WR1",
    "RWR": "WR2",
    "SWR": "WR3",
    "QB": "QB1",
    "RB": "RB1",
    "TE": "TE1",
}


def map_role(base_pos: str, depth_slot: str) -> Optional[str]:
    base = (base_pos or "").upper().strip()
    slot = re.sub(r"\s+", " ", (depth_slot or "").strip().lower())
    base = POSITION_ALIASES.get(base, base)
    if slot != "player 1":
        return None
    return ROLE_SLOT_MAPPING.get(base)


# --- HTTP fetch with retries -------------------------------------------------

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


# --- locate the OFFENSE depth table -----------------------------------------

def _find_offense_table(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    # Priority: find header 'OFFENSE', then first following table
    off = soup.find(
        lambda tag: tag.name in ["h1","h2","h3","h4"]
        and "OFFENSE" in tag.get_text().upper()
    )
    if off:
        tbl = off.find_next("table")
        if tbl:
            return tbl

    # Fallback: guess which table is the offensive chart by scanning first col
    pos_like = {"QB","RB","WR","TE","HB","TB","FB","FL","SE","SLOT","SL","Y","H"}
    for tbl in soup.find_all("table"):
        rows = tbl.find_all("tr")
        if not rows:
            continue
        score, seen = 0, 0
        for tr in rows[:6]:
            tds = tr.find_all(["td","th"])
            if not tds:
                continue
            col0 = BeautifulSoup(tds[0].decode_contents(), "lxml").get_text(
                " ", strip=True
            ).upper()
            if col0 in pos_like or col0.startswith("WR"):
                score += 1
            seen += 1
        if seen and score >= max(2, seen//2):  # looks like offensive skill positions
            return tbl
    return None


def _extract_table_structure(table: BeautifulSoup) -> (List[str], List[BeautifulSoup]):
    rows = table.find_all("tr")
    if not rows:
        return [], []

    def _cell_text(cell: BeautifulSoup) -> str:
        return BeautifulSoup(cell.decode_contents(), "lxml").get_text(
            " ", strip=True
        )

    header_cells = rows[0].find_all(["th","td"])
    header_has_th = any(cell.name == "th" for cell in header_cells)

    if header_has_th:
        header = []
        for idx, cell in enumerate(header_cells):
            text = _cell_text(cell)
            if not text:
                text = "Position" if idx == 0 else f"Player {idx}"
            else:
                m = re.match(r"player\s*(\d+)", text, flags=re.I)
                if m:
                    text = f"Player {m.group(1)}"
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

    # find column index for "Player 1"
    try:
        player1_idx = next(
            idx
            for idx, label in enumerate(header)
            if idx > 0 and re.sub(r"\s+"," ",label.strip()).lower() == "player 1"
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

        pos_raw = BeautifulSoup(tds[0].decode_contents(), "lxml").get_text(
            " ", strip=True
        ).upper()
        base_pos = pos_raw.strip().upper()
        if not base_pos or base_pos in OL_POSITIONS:
            # ignore OL entirely
            continue

        # The column header for this slot ("Player 1", etc.)
        depth_slot_label = header[player1_idx] if player1_idx < len(header) else "Player 1"
        depth_slot_label = re.sub(r"\s+", " ", depth_slot_label).strip() or "Player 1"

        # Grab the text from the Player 1 cell
        cell_text = BeautifulSoup(
            tds[player1_idx].decode_contents(), "lxml"
        ).get_text(" ", strip=True)

        # Some cells contain multiple slash/and separated names -> split and take first
        candidates = _split_candidates(cell_text)
        if not candidates:
            continue

        raw_player_name = candidates[0]
        player_clean = clean_ourlads_name(raw_player_name)
        if not player_clean or len(player_clean.split()) < 2:
            continue

        role = map_role(base_pos, depth_slot_label)
        if not role:
            continue

        position_out = POSITION_ALIASES.get(base_pos, base_pos)
        records.append(
            {
                "team": team_code,
                "player": player_clean,
                "position": position_out,
                "depth_slot": "Player 1",
                "role": role,
            }
        )

    return records


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
        time.sleep(0.5)  # don't hammer

    if all_rows:
        roles_all = pd.DataFrame(all_rows)
    else:
        roles_all = pd.DataFrame(columns=["team","player","position","depth_slot","role"])

    # Final cleanup & canonicalization
    roles_all = roles_all.dropna(subset=["player","role"])
    if not roles_all.empty:
        # re-clean in case anything weird leaked
        roles_all["player"] = roles_all["player"].map(clean_ourlads_name)
        roles_all["team"] = (
            roles_all["team"].astype(str).str.upper().str.strip().map(_canon_team)
        )
        roles_all = roles_all[roles_all["team"].isin(VALID)]
        # remove dupes like same player under multiple slots
        roles_all = roles_all.drop_duplicates(subset=["team","player","role"])
        roles_all = roles_all.sort_values(["team","role","player"])

    output_df = roles_all[["team","player","role","position"]].copy()
    output_df.to_csv(OUT_ROLES, index=False)
    print(f"[ourlads_depth] wrote rows={len(output_df)} → {OUT_ROLES}")

if __name__ == "__main__":
    main()
