#!/usr/bin/env python3
import sys
import time
from pathlib import Path

import pandas as pd
import requests

SHARP_URL = "https://www.sharpfootballanalysis.com/stats-nfl/nfl-coverage-schemes/"
HDRS = {"User-Agent": "FullSlate/CI (+github-actions)"}

TEAM_NAME_TO_ABBR = {
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

OUTPUT_COLS = ["team", "man_rate", "zone_rate"]


def fetch_sharp_coverage() -> pd.DataFrame:
    for attempt in range(3):
        try:
            resp = requests.get(SHARP_URL, headers=HDRS, timeout=45)
            resp.raise_for_status()
            tables = pd.read_html(resp.text)
            if not tables:
                raise ValueError("no tables returned")
            target = tables[0]
            for table in tables:
                cols = [str(c).lower() for c in table.columns]
                if any("team" in c for c in cols) and any("man" in c for c in cols) and any("zone" in c for c in cols):
                    target = table
                    break
            df = target.loc[:, ~target.columns.duplicated()].copy()
            colmap = {}
            for c in df.columns:
                lc = str(c).lower()
                if "team" in lc:
                    colmap[c] = "team_name"
                elif "man" in lc:
                    colmap[c] = "man_rate"
                elif "zone" in lc:
                    colmap[c] = "zone_rate"
            df = df.rename(columns=colmap)
            df = df[[c for c in ["team_name", "man_rate", "zone_rate"] if c in df.columns]].copy()
            for col in ["man_rate", "zone_rate"]:
                if col in df.columns:
                    cleaned = (
                        df[col]
                        .astype(str)
                        .str.replace("%", "", regex=False)
                        .str.extract(r"([0-9]+\.?[0-9]*)")[0]
                    )
                    df[col] = pd.to_numeric(cleaned, errors="coerce") / 100.0
            df["team"] = df.get("team_name", "").map(TEAM_NAME_TO_ABBR)
            df = df.dropna(subset=["team"])
            return df[["team", "man_rate", "zone_rate"]]
        except Exception:
            if attempt == 2:
                break
            time.sleep(2 * (attempt + 1))
    return pd.DataFrame(columns=OUTPUT_COLS)


def main(out_path: str | None = None) -> None:
    df = fetch_sharp_coverage()
    if df.empty:
        df = pd.DataFrame(columns=OUTPUT_COLS)
    else:
        df = df.drop_duplicates(subset=["team"]).sort_values("team").reset_index(drop=True)

    target = Path(out_path or "data/cb_coverage_team.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    print(f"[build_cb_coverage_team] wrote {target} with {len(df)} rows.")


if __name__ == "__main__":
    dest = "data/cb_coverage_team.csv" if len(sys.argv) < 2 else sys.argv[1]
    main(dest)
