#!/usr/bin/env python3
"""Fetch the current NFL injury report and write data/injuries.csv."""
from __future__ import annotations

import io
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HDRS = {"User-Agent": "Mozilla/5.0 (+github.com/dkaps6/imtiredofthis)"}
DEFAULT_OUT = Path("data") / "injuries.csv"

TEAM_CODE = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "Seattle Seahawks": "SEA",
    "San Francisco 49ers": "SF", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
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
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if not headers or "player" not in headers or ("game status" not in headers and "status" not in headers):
            continue
        try:
            tables.append(pd.read_html(io.StringIO(str(table)))[0])
        except ValueError:
            continue
    if not tables:
        raise RuntimeError("Unable to locate injury tables on NFL.com injuries page.")
    return tables


def normalize_injury_dataframe(df: pd.DataFrame, week: int) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if "player" in key:
            rename[col] = "player"
        elif "team" in key or "club" in key:
            rename[col] = "team_raw"
        elif "game status" in key or key == "status":
            rename[col] = "status"
        elif "practice" in key:
            rename[col] = "practice_status"
        elif "injury" in key:
            rename[col] = "injury"
    x = df.rename(columns=rename).copy()
    if "player" not in x.columns or "status" not in x.columns:
        raise RuntimeError("NFL injury table missing player/status columns")
    team = x.get("team_raw", pd.Series(pd.NA, index=x.index, dtype="string")).astype("string").str.strip()
    x["team"] = team.replace(TEAM_CODE).str.upper()
    x["body_part"] = x.get("injury", pd.Series("Unknown", index=x.index)).astype(str).str.split("[,(/]").str[0].str.strip().replace("", "Unknown")
    x["week"] = int(week)
    x["report_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    x["designation"] = x["status"]
    for col in ("practice_status",):
        if col not in x.columns:
            x[col] = pd.NA
    x = x.dropna(subset=["player"]).copy()
    x["player"] = x["player"].astype("string").str.strip()
    x["status"] = x["status"].astype("string").str.strip()
    cols = ["player", "team", "week", "status", "practice_status", "body_part", "designation", "report_date"]
    x = x[cols].drop_duplicates(["player", "team"]).reset_index(drop=True)
    if len(x) < 20:
        raise RuntimeError(f"NFL.com injury table returned only {len(x)} rows; refusing suspiciously small output")
    return x


def parse_nfl_injuries_week() -> pd.DataFrame:
    html = fetch_injuries_html()
    week = detect_current_week_from_html(html)
    tables = extract_injury_tables(BeautifulSoup(html, "html.parser"))
    return normalize_injury_dataframe(pd.concat(tables, ignore_index=True), week)


def main(out_csv: str | Path = DEFAULT_OUT) -> None:
    df = parse_nfl_injuries_week()
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out} with {len(df)} rows for week {df['week'].iloc[0]}.")


if __name__ == "__main__":
    main(DEFAULT_OUT if len(sys.argv) < 2 else Path(sys.argv[1]))
