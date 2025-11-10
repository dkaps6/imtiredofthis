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

import logging
import os, re, time, warnings, sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag

DATA_DIR = "data"
OUT_ROLES = os.path.join(DATA_DIR, "roles_ourlads.csv")

logger = logging.getLogger("ourlads_depth")

# canonical team set
VALID = {
    "ARI","ATL","BAL","BUF","CAR","CHI","CIN","CLE","DAL","DEN","DET","GB","HOU",
    "IND","JAX","KC","LAC","LAR","LV","MIA","MIN","NE","NO","NYG","NYJ",
    "PHI","PIT","SEA","SF","TB","TEN","WAS"
}

TEAM_MAP = {
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    "WSH": "WAS",
    "JAC": "JAX",
    "SD": "LAC",
    "LA": "LAR",
    "STL": "LAR",
}

TEAM_FIXES = {
    "ARZ": "ARI",
    **TEAM_MAP,
}

# Retain historical aliases while layering the new TEAM_FIXES guidance.
TEAM_ALIASES = {
    **TEAM_FIXES,
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

NON_NAME_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v", "vi", "vii"}

SKILL_POSITIONS = ["QB", "RB", "FB", "WR", "TE"]
SKILL_GROUPS = set(SKILL_POSITIONS)
WR_PATTERN = re.compile(r"\b(W|WR|LWR|RWR|SWR|Slot WR|Wide Receiver)\b", re.I)


def clean_name(raw):
    raw = re.sub(r"\([^)]*\)", "", raw)            # remove parenthetical status
    raw = re.sub(r"\d+/?\d*", "", raw)             # remove jersey/injury numbers
    raw = re.sub(r"[^A-Za-z\s\-'\.]+", "", raw)    # keep letters only
    return raw.strip()


def normalize_team(t):
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return ""
    text = str(t).strip()
    if not text:
        return ""
    return TEAM_MAP.get(text.upper(), text.upper())


def make_keys(name):
    full = clean_name(name)
    if not full:
        return "", "", ""
    first, *last = full.split(" ", 1)
    last = last[0] if last else ""
    return full, f"{first[0].lower()}{last.lower()}" if first else "", full.replace(" ", "").lower()


def _strip_name_suffix(raw: str) -> str:
    """Remove trailing numeric/letter suffixes introduced by OurLads scraping artifacts."""

    if not raw:
        return raw
    working = raw.strip()
    match = re.match(r"([A-Za-z\-']+?)(?:\d+[A-Za-z]*)?$", working)
    if match:
        return match.group(1)
    return working


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
    working = re.sub(r"\*.*?\*", " ", working)
    working = re.sub(
        r"\b(?:IR|PUP|NFI|DNR|OUT|INJ|INACTIVE)\b",
        " ",
        working,
        flags=re.I,
    )
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
        piece = _strip_name_suffix(piece)
        clean_piece = re.sub(r"[^A-Za-z\-']", "", piece)
        clean_piece = clean_piece.strip("'")
        if not clean_piece:
            continue
        tokens.append(clean_piece)

    if not tokens:
        return ""

    def _normalize_token_case(token: str) -> str:
        upper_token = token.upper()
        if len(token) <= 3 and token.isupper() and upper_token not in {"MAC"}:
            return upper_token
        if upper_token.startswith("MC") and len(token) > 2:
            return upper_token[:2].title() + upper_token[2:].title()
        return token.title()

    first = _normalize_token_case(tokens[0])
    last = _normalize_token_case(tokens[-1])
    if not last:
        name = first
    else:
        name = f"{first} {last}".strip()
    name = re.sub(r"[\d/]+$", "", name).strip()
    return name


def _position_group(base_pos: str) -> str:
    base = (base_pos or "").upper().strip()
    if not base:
        return ""
    if base in {"HB", "TB"}:
        return "RB"
    if base == "FB":
        return "FB"
    if base.startswith("WR") or WR_PATTERN.search(base_pos or ""):
        return "WR"
    if base in {"TE", "Y"}:
        return "TE"
    if base == "QB":
        return "QB"
    return base


def _slot_depth(slot_label: str) -> Optional[int]:
    if not slot_label:
        return None
    m = re.search(r"(\d+)", slot_label)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


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
ROLE_SLOT_MAPPING = {
    "LWR": {1: "WR1"},
    "RWR": {1: "WR2"},
    "SWR": {1: "WR3"},
    "QB":  {1: "QB1"},
    "RB":  {1: "RB1"},
    "TE":  {1: "TE1"},
    "HB":  {1: "RB1"},
    "TB":  {1: "RB1"},
    "FB":  {1: "FB1"},
    # NEW: when the table uses a single WR row with Player 1..3
    "WR":  {1: "WR1", 2: "WR2", 3: "WR3"},
}

def map_role(base_pos: str, depth_slot: str):
    """
    Maps an Ourlads base position + slot label ('Player N') to a depth role (WR1, WR2, WR3, etc).
    Supports pages that list one 'WR' row with Player 1–3 columns.
    """
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

def _normalized_role(base_pos: str, pos_group: str, depth_idx: Optional[int]) -> str:
    """Return a normalized role label (e.g., WR1, RB2) for downstream merges."""

    base = (base_pos or "").upper().strip()
    group = (pos_group or base).upper().strip()
    if base in {"LWR", "RWR", "SWR"}:
        mapping = {"LWR": "WR1", "RWR": "WR2", "SWR": "WR3"}
        return mapping.get(base, f"WR{depth_idx or 1}")
    if group == "WR" and depth_idx:
        return f"WR{depth_idx}"
    if group in {"QB", "RB", "TE", "FB"}:
        if depth_idx:
            return f"{group}{depth_idx}"
        return f"{group}1"
    return group or base



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


def fetch_team_roles(team: str, soup: BeautifulSoup, include_inactive: bool) -> List[dict]:
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
    canon_team = _canon_team(team)
    team_code = TEAM_MAP.get(canon_team, canon_team)

    for tr in data_rows:
        tds = tr.find_all("td")
        pos_raw = BeautifulSoup(tds[0].decode_contents(), "lxml").get_text(
            " ", strip=True
        ).upper()
        base_pos = pos_raw.strip().upper()
        if not base_pos or base_pos in OL_POSITIONS:
            continue
        pos_group = _position_group(base_pos)
        if pos_group not in SKILL_GROUPS:
            continue

        for idx, slot_label in slot_idxs:
            if idx >= len(tds):
                continue
            cell_text = BeautifulSoup(tds[idx].decode_contents(), "lxml").get_text(
                " ", strip=True
            )
            candidates = _split_candidates(cell_text)
            if not candidates:
                continue

            chosen_name: Optional[str] = None
            player_key: str = ""
            player_clean_key: str = ""
            for candidate in candidates:
                text_lower = candidate.lower().strip()
                if not candidate or "injured" in text_lower:
                    continue
                cleaned_candidate = clean_ourlads_name(candidate)
                full_name, short_key, clean_key = make_keys(cleaned_candidate)
                if full_name and " " in full_name:
                    chosen_name = full_name
                    player_key = short_key
                    player_clean_key = clean_key
                    break
            if not chosen_name:
                continue

            depth_idx = _slot_depth(slot_label)
            depth_role = f"{base_pos}{depth_idx}" if depth_idx else base_pos
            status = _cell_status(tds[idx])
            if status == "inactive" and not include_inactive:
                continue

            normalized_role = map_role(base_pos, slot_label) or _normalized_role(
                base_pos, pos_group, depth_idx
            )

            records.append(
                {
                    "team": team_code,
                    "player": chosen_name,
                    "position": base_pos,
                    "position_group": pos_group,
                    "depth_slot": slot_label,
                    "depth_index": depth_idx,
                    "depth_chart_role": depth_role,
                    "role": normalized_role,
                    "model_role": normalized_role,
                    "status": status,
                    "player_key": player_key,
                    "player_clean_key": player_clean_key,
                }
            )

    return records


def main(*, season: Optional[int] = None, include_inactive: bool = True):
    warnings.simplefilter("ignore")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.setLevel(logging.INFO)
    os.makedirs(DATA_DIR, exist_ok=True)

    all_rows: List[dict] = []

    for tm in sorted(TEAM_URLS.keys()):
        try:
            soup = _get_depth_soup(tm)
            team_rows = fetch_team_roles(tm, soup, include_inactive)
            if not team_rows:
                logger.warning("[OUR-LADS] %s returned zero skill players", tm)
            else:
                all_rows.extend(team_rows)
        except Exception as e:
            logger.warning("[OUR-LADS] %s fetch failed: %s", tm, e)
        time.sleep(0.5)  # don't hammer

    if all_rows:
        roles_all = pd.DataFrame(all_rows)
    else:
        roles_all = pd.DataFrame(columns=["team","player","position","depth_slot","role","status"])

    # Final cleanup & canonicalization
    roles_all = roles_all.dropna(subset=["player", "role"])
    if not roles_all.empty:
        # re-clean in case anything weird leaked
        roles_all["player"] = roles_all["player"].map(clean_ourlads_name)
        roles_all["team"] = (
            roles_all["team"].astype(str).str.upper().str.strip().map(_canon_team)
        )
        roles_all = roles_all[roles_all["team"].isin(VALID)]
        if "position_group" not in roles_all.columns:
            roles_all["position_group"] = roles_all["position"].astype(str)
        roles_all["position_group"] = roles_all["position_group"].astype(str).str.upper().str.strip()
        roles_all = roles_all[roles_all["position_group"].isin(SKILL_GROUPS)].copy()

        if not include_inactive and "status" in roles_all.columns:
            status_lower = roles_all["status"].astype(str).str.lower()
            roles_all = roles_all[status_lower != "inactive"].copy()

        roles_all["role"] = roles_all["role"].astype(str).str.upper().str.strip()
        roles_all["model_role"] = (
            roles_all.get("model_role", roles_all["role"])
            .astype(str)
            .str.upper()
            .str.strip()
        )
        roles_all["model_role"] = roles_all["model_role"].replace({"": pd.NA, "NAN": pd.NA})
        roles_all["depth_chart_role"] = (
            roles_all.get("depth_chart_role", roles_all["role"])
            .astype(str)
            .str.upper()
            .str.strip()
        )
        roles_all["depth_chart_role"] = roles_all["depth_chart_role"].replace({"": pd.NA, "NAN": pd.NA})

        if "depth_index" in roles_all.columns:
            roles_all["depth_index"] = pd.to_numeric(
                roles_all["depth_index"], errors="coerce"
            )
        else:
            roles_all["depth_index"] = pd.NA
        roles_all = roles_all.sort_values(
            [
                "team",
                "position_group",
                "depth_index",
                "role",
                "player",
            ]
        )
        roles_all = roles_all.drop_duplicates(
            subset=["team", "player", "role"], keep="first"
        )

    def _derive_keys(name: str) -> Tuple[str, str, str]:
        full, short, clean = make_keys(name)
        return full or name, short, clean

    key_df = roles_all["player"].apply(_derive_keys).apply(pd.Series)
    key_df.columns = ["player_cleaned", "player_key", "player_clean_key"]
    roles_all["player"] = key_df["player_cleaned"]
    roles_all["player_key"] = key_df["player_key"]
    roles_all["player_clean_key"] = key_df["player_clean_key"]

    cols = [
        "team",
        "player",
        "status" if "status" in roles_all.columns else None,
        "role",
        "model_role",
        "position",
        "position_group",
        "depth_chart_role",
        "depth_slot",
        "depth_index",
        "player_key",
        "player_clean_key",
    ]
    cols = [c for c in cols if c is not None and c in roles_all.columns]
    df = roles_all[cols].copy()
    if not df.empty and "team" in df.columns:
        df["team"] = df["team"].apply(normalize_team)
    if not df.empty:
        mask = pd.Series(False, index=df.index)
        if "position" in df.columns:
            mask = mask | df["position"].astype(str).str.upper().isin(SKILL_POSITIONS)
        if "position_group" in df.columns:
            mask = mask | df["position_group"].astype(str).str.upper().isin(SKILL_POSITIONS)
        df = df[mask].copy()
    total_rows = len(df)
    if df.empty:
        logger.warning("[OUR-LADS] No active offensive players parsed; output will be empty")
    else:
        order = ["WR", "RB", "TE", "QB", "FB"]
        for team_code, group in df.groupby("team"):
            pos_series = group.get("position_group")
            if pos_series is None:
                pos_series = pd.Series(dtype=str)
            pos_counts = Counter(
                str(val).upper().strip()
                for val in pos_series.dropna().astype(str)
                if str(val).strip()
            )
            breakdown = ", ".join(
                f"{pos}:{pos_counts[pos]}" for pos in order if pos_counts.get(pos)
            )
            status_series = group.get("status")
            status_counts: Counter = Counter()
            if status_series is not None:
                for raw_status in status_series.tolist():
                    status_text = "" if pd.isna(raw_status) else str(raw_status).strip().lower()
                    if not status_text or status_text in {"nan", "<na>"}:
                        continue
                    status_counts[status_text] += 1
            status_breakdown = ", ".join(
                f"{status.upper()}:{status_counts[status]}" for status in sorted(status_counts)
            )
            detail_parts = []
            if breakdown:
                detail_parts.append(f"positions: {breakdown}")
            if status_breakdown:
                detail_parts.append(f"status: {status_breakdown}")
            detail = f" ({'; '.join(detail_parts)})" if detail_parts else ""
            print(f"[OURlads] {team_code}: {len(group)} players written{detail}")

    final_df = df.drop(
        columns=[
            col
            for col in df.columns
            if "injury" in col.lower() or "status" in col.lower()
        ],
        errors="ignore",
    )

    # ensure only SKILL_POSITIONS are kept, normalize team via TEAM_MAP, and log totals:
    print(f"[OURlads] Writing {len(final_df)} total rows.")
    if "position" in final_df.columns:
        print(final_df["position"].value_counts())

    final_df.to_csv(OUT_ROLES, index=False)
    logger.info(
        "[OUR-LADS] wrote %d rows → %s (include_inactive=%s)",
        len(final_df),
        OUT_ROLES,
        include_inactive,
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
        dest="include_inactive",
        action="store_true",
        default=True,
        help="Keep players marked inactive (red text) in output",
    )
    parser.add_argument(
        "--exclude-inactive",
        dest="include_inactive",
        action="store_false",
        help="Exclude players marked inactive (red text) from output",
    )
    args = parser.parse_args()
    season = args.season_flag if args.season_flag is not None else args.season_positional
    if season is None:
        raise SystemExit(
            "ourlads_depth.py: missing season. Pass either a positional season (e.g., `python ... 2025`) or `--season 2025`."
        )
    main(season=season, include_inactive=args.include_inactive)
