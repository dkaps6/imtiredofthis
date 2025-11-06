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
from typing import Dict, List, Optional, Tuple
import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

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

DROP_TOKENS = {
    "U", "CC", "T",
    "II", "III", "IV", "V", "VI", "VII",
    "Sr", "Sr.", "Jr", "Jr.", "III",
}

NON_NAME_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v", "vi", "vii"}


def canonical_player_key(full_name: str) -> str:
    """
    "ZAY CC JONES"      -> "Zjones"
    "Darnell U Mooney"  -> "Dmooney"
    "JOE T FLACCO"      -> "Jflacco"
    "Kyle Pitts Sr."    -> "Kpitts"
    Behavior:
    - strip '.' and "'"
    - split on whitespace
    - drop anything in DROP_TOKENS
    - first remaining token = first name
    - last remaining token = last name
    - build key = first_initial + last_name
    - lowercase then capitalize first char
    """

    if not isinstance(full_name, str):
        return ""
    cleaned = (
        full_name.replace(".", "")
        .replace("'", "")
        .strip()
    )
    if not cleaned:
        return ""
    parts = [p for p in cleaned.split() if p and p not in DROP_TOKENS]
    if not parts:
        return ""
    first = parts[0]
    last = parts[-1]
    if not first or not last:
        return ""
    key = (first[0] + last).lower()
    return key[0].upper() + key[1:]


def clean_ourlads_name(raw: str) -> str:
    """Return strictly "First Last" with suffixes/status tokens removed."""

    if not isinstance(raw, str):
        return ""

    working = raw.strip()
    if "," in working:
        last, first_rest = [p.strip() for p in working.split(",", 1)]
        working = f"{first_rest} {last}".strip()

    # Remove parenthetical/suffix clutter before tokenization
    working = re.sub(r"\(.*?\)", " ", working)
    working = working.replace("/", " ")

    tokens: List[str] = []
    for tok in re.split(r"\s+", working):
        piece = tok.strip(",()")
        if not piece:
            continue
        lower = piece.lower().strip(".")
        if re.search(r"\d", piece):
            continue
        if lower in BAD_TOKENS or lower in NON_NAME_TOKENS:
            continue
        if lower in {"u", "cc", "t"}:
            continue
        if len(lower) == 1:  # middle initial
            continue
        clean_piece = re.sub(r"[^A-Za-z\-']", "", piece)
        clean_piece = clean_piece.strip("'")
        if not clean_piece:
            continue
        tokens.append(clean_piece)

    if not tokens:
        return ""

    first = tokens[0]
    last = tokens[-1]
    if not last:
        return first
    return f"{first} {last}".strip()


def _cell_status(cell: Tag) -> str:
    """Detect inactive players (rendered in red)."""

    if cell is None:
        return "active"

    def _has_red(tag: Tag) -> bool:
        style = (tag.get("style") or "").lower()
        if "color" in style and "red" in style:
            return True
        color = (tag.get("color") or "").lower()
        return color == "red"

    if _has_red(cell):
        return "inactive"
    for desc in cell.descendants:
        if isinstance(desc, Tag) and _has_red(desc):
            return "inactive"
    return "active"


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
    "LWR": {1: "WR1"},
    "RWR": {1: "WR2"},
    "SWR": {1: "WR3"},
    "QB":  {1: "QB1"},
    "RB":  {1: "RB1"},
    "TE":  {1: "TE1"},
    # NEW: when the table uses a single WR row with Player 1..3
    "WR":  {1: "WR1", 2: "WR2", 3: "WR3"},
}

def map_role(base_pos: str, depth_slot: str):
    """
    Maps an Ourlads base position + slot label ('Player N') to a depth role (WR1, WR2, WR3, etc).
    Supports pages that list one 'WR' row with Player 1–3 columns.
    """
    import re
    base = (base_pos or "").upper().strip()
    m = re.search(r"player\s*(\d+)", (depth_slot or ""), flags=re.I)
    if not m:
        return None
    n = int(m.group(1))
    if base in ROLE_SLOT_MAPPING:
        return ROLE_SLOT_MAPPING[base].get(n)
    if base.startswith("WR"):
        return {1: "WR1", 2: "WR2", 3: "WR3"}.get(n)
    return None


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

    # --- BEGIN: detect slot columns ---
    slot_idxs = []
    for want in ("player 1", "player 2", "player 3"):
        idx = None
        for i, label in enumerate(header):
            if re.sub(r"\s+", " ", str(label).strip()).lower() == want:
                idx = i
                break
        if idx is not None:
            slot_idxs.append((idx, want.title()))
    # fallback if missing headers but wide table
    if not slot_idxs and len(header) > 3:
        slot_idxs = [(1, "Player 1"), (2, "Player 2"), (3, "Player 3")]
    # --- END: detect slot columns ---

    records: List[dict] = []
    team_code = _canon_team(team)

    for tr in data_rows:
        tds = tr.find_all("td")
        pos_raw = BeautifulSoup(tds[0].decode_contents(), "lxml").get_text(
            " ", strip=True
        ).upper()
        base_pos = pos_raw.strip().upper()
        if not base_pos or base_pos in OL_POSITIONS:
            # ignore OL entirely
            continue
        # --- BEGIN: player extraction across slots ---
        for idx, slot_label in slot_idxs:
            if idx >= len(tds):
                continue
            cell_text = BeautifulSoup(tds[idx].decode_contents(), "lxml").get_text(" ", strip=True)
            candidates = _split_candidates(cell_text)
            if not candidates:
                continue
            raw_player_name = candidates[0]
            player_clean = clean_ourlads_name(raw_player_name)
            if not player_clean or len(player_clean.split()) < 2:
                continue
            role = map_role(base_pos, slot_label)
            if not role:
                continue
            position_out = POSITION_ALIASES.get(base_pos, base_pos)
            status = _cell_status(tds[idx])
            records.append({
                "team": team_code,
                "player": player_clean,
                "position": position_out,
                "depth_slot": slot_label,
                "role": role,
                "status": status,
            })
        # --- END: player extraction across slots ---

    return records


def main(*, season: Optional[int] = None, include_inactive: bool = False):
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
        roles_all = pd.DataFrame(columns=["team","player","position","depth_slot","role","status"])

    # Final cleanup & canonicalization
    roles_all = roles_all.dropna(subset=["player","role"])
    if not roles_all.empty:
        # re-clean in case anything weird leaked
        roles_all["player"] = roles_all["player"].map(clean_ourlads_name)
        roles_all["team"] = (
            roles_all["team"].astype(str).str.upper().str.strip().map(_canon_team)
        )
        roles_all = roles_all[roles_all["team"].isin(VALID)]
        if not include_inactive and "status" in roles_all.columns:
            status_lower = roles_all["status"].astype(str).str.lower()
            roles_all = roles_all[status_lower != "inactive"].copy()
        # remove dupes like same player under multiple slots
        roles_all = roles_all.drop_duplicates(subset=["team","player","role"])
        roles_all = roles_all.sort_values(["team","role","player"])

        offense_roles = {"QB1", "RB1", "WR1", "WR2", "WR3", "TE1"}
        offense_positions = {"QB", "RB", "WR", "TE"}
        roles_all = roles_all[roles_all["role"].isin(offense_roles)].copy()
        roles_all = roles_all[roles_all["position"].isin(offense_positions)].copy()

        pos_priority = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
        role_priority = {"QB1": 0, "RB1": 1, "WR1": 2, "WR2": 3, "WR3": 4, "TE1": 5}
        roles_all["_pos_rank"] = roles_all["position"].map(pos_priority).fillna(99)
        roles_all["_role_rank"] = roles_all["role"].map(role_priority).fillna(99)
        roles_all = roles_all.sort_values(
            ["team", "player", "_pos_rank", "_role_rank", "role", "position"]
        )
        # keep strongest role encountered for each player (Player 1 slot first)
        roles_all = roles_all.drop_duplicates(subset=["team", "player"], keep="first")
        roles_all = roles_all.drop(columns=["_pos_rank", "_role_rank"])

    roles_all["player_key"] = roles_all["player"].apply(canonical_player_key)
    cols = ["team","player","role","position","player_key"]
    if "status" in roles_all.columns:
        cols.insert(2, "status")
    output_df = roles_all[cols].copy()
    output_df.to_csv(OUT_ROLES, index=False)
    print(
        f"[ourlads_depth] wrote rows={len(output_df)} (include_inactive={include_inactive}) → {OUT_ROLES}"
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # Accept either a positional SEASON or the --season flag (both ints)
    parser.add_argument(
        "season_positional",
        nargs="?",
        type=int,
        help="Season year (e.g., 2025)",
    )
    parser.add_argument(
        "--season",
        dest="season_flag",
        type=int,
        help="Season year (e.g., 2025)",
    )
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="Keep players marked inactive (red text) in output",
    )
    args = parser.parse_args()
    season = args.season_flag if args.season_flag is not None else args.season_positional
    if season is None:
        raise SystemExit(
            "ourlads_depth.py: missing season. Pass either a positional season (e.g., `python ... 2025`) or `--season 2025`."
        )
    main(season=season, include_inactive=args.include_inactive)
