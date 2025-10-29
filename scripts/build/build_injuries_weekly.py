#!/usr/bin/env python3
# build_injuries_weekly.py
# Fetch current week's NFL injuries & practice participation (free sources), auto-detecting week.
# Primary source: NFL.com Injuries index (week selector).
#
# Output CSV schema:
#   player,team,week,status,practice_status,body_part,designation,report_date
#
# Citations you can include in your docs:
# - NFL.com Injuries (current week): https://www.nfl.com/injuries/
# - NFL.com weekly inactives / injuries-to-monitor posts (same week)
#
# Usage:
#   python build_injuries_weekly.py            # writes injuries.csv
#   python build_injuries_weekly.py data/injuries.csv
#
import io
import re
import sys
from datetime import datetime, timezone
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup

HDRS = {"User-Agent": "Mozilla/5.0 (+github.com/your-org/your-repo)"}

TEAM_CODE = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA",
    "San Francisco 49ers": "SF",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}


def fetch_injuries_html(url: str = "https://www.nfl.com/injuries/") -> str:
    response = requests.get(url, headers=HDRS, timeout=45)
    response.raise_for_status()
    return response.text


def detect_current_week_from_html(html: str) -> int:
    match = re.search(r"Week\s+(\d+)\s+of the\s+(\d{4})\s+Season", html, re.IGNORECASE)
    if not match:
        raise RuntimeError("Unable to detect current week from NFL.com injuries page.")
    return int(match.group(1))


def extract_injury_tables(soup: BeautifulSoup) -> List[pd.DataFrame]:
    tables = []
    for table in soup.find_all("table"):
        header_cells = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if not header_cells:
            continue
        if "player" not in header_cells:
            continue
        if "game status" not in header_cells and "status" not in header_cells:
            continue
        try:
            frame = pd.read_html(io.StringIO(str(table)))[0]
        except ValueError:
            continue
        tables.append(frame)
    if not tables:
        raise RuntimeError("Unable to locate injury tables on NFL.com injuries page.")
    return tables


def normalize_injury_dataframe(df: pd.DataFrame, week: int) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if "player" in key:
            rename_map[col] = "player"
        elif "team" in key or "club" in key:
            rename_map[col] = "team_raw"
        elif "game status" in key or key == "status":
            rename_map[col] = "status"
        elif "practice" in key:
            rename_map[col] = "practice_status"
        elif "injury" in key:
            rename_map[col] = "injury"
    normalized = df.rename(columns=rename_map)

    team_series = normalized.get("team_raw")
    if team_series is not None:
        normalized["team"] = team_series.astype("string").str.strip()
    else:
        normalized["team"] = pd.Series(pd.NA, index=normalized.index, dtype="string")

    invalid_team_mask = normalized["team"].notna() & normalized["team"].isin(
        ["", "nan", "<NA>", "None"]
    )
    normalized.loc[invalid_team_mask, "team"] = pd.NA
    normalized["team"] = normalized["team"].replace(TEAM_CODE)
    normalized["team"] = normalized["team"].str.upper()

    if "injury" in normalized.columns:
        normalized["body_part"] = (
            normalized["injury"].astype(str).str.split("[,(/]").str[0].str.strip()
        )
        normalized.loc[normalized["body_part"] == "", "body_part"] = "Unknown"
    else:
        normalized["body_part"] = "Unknown"

    normalized["week"] = week
    normalized["report_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    normalized["designation"] = normalized.get("status")

    required_columns = {"player", "team", "status", "week"}
    missing_columns = required_columns - set(normalized.columns)
    if missing_columns:
        raise RuntimeError(
            f"Missing required columns from injuries table: {', '.join(sorted(missing_columns))}."
        )

    normalized = normalized.dropna(subset=["player"]).copy()
    normalized["player"] = normalized["player"].astype("string").str.strip()
    normalized["team"] = normalized["team"].astype("string").str.strip()
    normalized["status"] = normalized["status"].astype("string").str.strip()

    columns = [
        "player",
        "team",
        "week",
        "status",
        "practice_status",
        "body_part",
        "designation",
        "report_date",
    ]
    for column in columns:
        if column not in normalized.columns:
            normalized[column] = pd.NA

    normalized = normalized[columns]
    normalized = normalized.drop_duplicates(subset=["player", "team"])

    if len(normalized) < 50:
        raise RuntimeError(
            f"NFL.com injuries table returned only {len(normalized)} rows; expected at least 50."
        )

    normalized = normalized.reset_index(drop=True)
    return normalized


def parse_nfl_injuries_week() -> pd.DataFrame:
    html = fetch_injuries_html()
    week = detect_current_week_from_html(html)
    soup = BeautifulSoup(html, "html.parser")
    tables = extract_injury_tables(soup)
    combined = pd.concat(tables, ignore_index=True)
    return normalize_injury_dataframe(combined, week)


def main(out_csv: str = "injuries.csv") -> None:
    df = parse_nfl_injuries_week()
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows for week {df['week'].iloc[0]}.")


if __name__ == "__main__":
    output_path = "injuries.csv" if len(sys.argv) < 2 else sys.argv[1]
    main(output_path)
