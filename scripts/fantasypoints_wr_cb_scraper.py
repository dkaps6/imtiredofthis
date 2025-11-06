"""
Auto-updating FantasyPoints WR–CB Matchup Scraper
-------------------------------------------------
Fetches, parses, normalizes, and archives weekly WR–CB matchup data
from https://www.fantasypoints.com/nfl/reports/wr-cb-matchups
Integrates with canonicalize_names.py and stores both current and
historical matchup files for model ingestion and backtesting.
"""

from __future__ import annotations

import datetime
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.utils.canonical_names import canonicalize_player_name

TEAM_NORMALIZATION = {"BLT": "BAL", "CLV": "CLE", "HST": "HOU"}
URL = "https://www.fantasypoints.com/nfl/reports/wr-cb-matchups"


def fetch_wr_cb_html() -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; FPBot/1.0)",
        "Accept-Language": "en-US,en;q=0.9",
    }
    response = requests.get(URL, headers=headers, timeout=30)
    response.raise_for_status()
    return response.text


def _extract_json_from_script(script_text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", script_text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def extract_wr_cb_json(html: str) -> dict[str, Any] | None:
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.find_all("script"):
        if "__NEXT_DATA__" in script.text:
            data = _extract_json_from_script(script.text)
            if data is not None:
                return data
    match = re.search(r"\{.*\"matchup\".*\}", html, re.S)
    return json.loads(match.group(0)) if match else None


def _flatten_wr_cb_payload(data: dict[str, Any]) -> list[dict[str, Any]]:
    page_props = data.get("props") or data.get("pageProps") or {}
    if "pageProps" in data:
        page_props = data.get("pageProps", {})
    if isinstance(page_props, dict):
        page_props = page_props.get("pageProps", page_props)
    matchup_data = None

    if isinstance(page_props, dict):
        for key in ("pageData", "data", "matchups", "matchupData"):
            candidate = page_props.get(key)
            if candidate:
                matchup_data = candidate
                break

    if matchup_data is None and isinstance(data, dict):
        matchup_data = data.get("matchupData") or data.get("data")

    if isinstance(matchup_data, dict):
        for key in ("matchups", "rows", "table"):
            candidate = matchup_data.get(key)
            if candidate:
                matchup_data = candidate
                break

    if isinstance(matchup_data, list):
        return matchup_data

    if isinstance(matchup_data, dict):
        # assume nested by team or grouping
        flattened: list[dict[str, Any]] = []
        for value in matchup_data.values():
            if isinstance(value, list):
                flattened.extend(v for v in value if isinstance(v, dict))
            elif isinstance(value, dict):
                flattened.append(value)
        return flattened

    return []


def normalize_wr_cb_data(data: dict[str, Any], week: int) -> pd.DataFrame:
    rows = _flatten_wr_cb_payload(data)
    if not rows:
        df = pd.json_normalize(data, max_level=5, sep="_")
    else:
        df = pd.json_normalize(rows, max_level=5, sep="_")

    if df.empty:
        return df

    df.columns = [re.sub(r"(?i)(pageprops_|props_|data_|pageData_)", "", c) for c in df.columns]

    keep_patterns = [
        "player",
        "team",
        "corner",
        "align",
        "cover",
        "advant",
        "opp",
        "matchup",
    ]
    keep_cols = [c for c in df.columns if any(p in c.lower() for p in keep_patterns)]
    df = df[keep_cols].copy()

    rename_map: dict[str, str] = {}
    for col in df.columns:
        base = col.split("_")[-1]
        base = base.lower()
        if base in {"name", "player"}:
            rename_map[col] = "player"
        elif base in {"team", "teamabbr", "team_abbr"}:
            rename_map[col] = "team"
        elif base in {"opp", "opponent", "oppteam"}:
            rename_map[col] = "opponent"
        elif base in {"advantage", "adv", "advant"}:
            rename_map[col] = "wr_cb_advantage"
        elif base in {"corner", "cb"}:
            rename_map[col] = "primary_corner"
        elif base in {"coverage", "cover"}:
            rename_map[col] = "primary_coverage"
        elif base in {"slotrate", "slot_align", "slot"}:
            rename_map[col] = "slot_rate"
        elif base in {"left", "leftalign", "left_align"}:
            rename_map[col] = "left_align_rate"
        elif base in {"right", "rightalign", "right_align"}:
            rename_map[col] = "right_align_rate"
    df.rename(columns=rename_map, inplace=True)

    for col in df.columns:
        if "team" in col.lower():
            df[col] = df[col].replace(TEAM_NORMALIZATION)

    if "player" in df.columns:
        df["player"] = df["player"].apply(canonicalize_player_name)

    df["week"] = week
    df["timestamp"] = datetime.datetime.utcnow()

    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def save_wr_cb_data(df: pd.DataFrame, week: int) -> None:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    current_out = data_dir / f"wr_cb_matchups_week_{week}.csv"
    df.to_csv(current_out, index=False)
    print(f"✅ Saved current WR–CB data → {current_out}")

    hist_path = data_dir / "wr_cb_matchups_historical.csv"
    if hist_path.exists():
        hist_df = pd.read_csv(hist_path)
        hist_df = pd.concat([hist_df, df], ignore_index=True)
        if {"player", "week"}.issubset(hist_df.columns):
            hist_df = hist_df.drop_duplicates(subset=["player", "week"], keep="last")
    else:
        hist_df = df.copy()
    hist_df.to_csv(hist_path, index=False)
    print(f"📜 Historical archive updated → {hist_path}")


def get_current_week() -> int:
    today = datetime.date.today()
    nfl_start = datetime.date(2025, 9, 4)
    delta_weeks = ((today - nfl_start).days // 7) + 1
    return max(1, min(18, delta_weeks))


def main() -> None:
    week = get_current_week()
    print(f"🔍 Fetching WR–CB matchups for Week {week}...")
    html = fetch_wr_cb_html()
    data = extract_wr_cb_json(html)
    if not data:
        raise ValueError("Could not extract WR–CB JSON data from FantasyPoints.")
    df = normalize_wr_cb_data(data, week)
    if df.empty:
        raise ValueError("FantasyPoints WR–CB dataset parsed to an empty DataFrame.")
    save_wr_cb_data(df, week)
    print("✅ WR–CB weekly data ingestion complete.")


if __name__ == "__main__":
    main()
