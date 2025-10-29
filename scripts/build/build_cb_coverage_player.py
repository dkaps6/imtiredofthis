#!/usr/bin/env python3
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

HDRS = {"User-Agent": "FullSlate/CI (+github-actions)"}
ROTOWIRE_URL = "https://www.rotowire.com/football/player-alignment.php"
SHARP_URL = "https://www.sharpfootballanalysis.com/stats-nfl/nfl-coverage-schemes/"
ROTOBALLER_URL = "https://www.rotoballer.com/wr-cb-matchups"
DEBUG_HTML_PATH = Path("data") / "_debug" / "_build_cb_coverage_player.py.html"

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

OUTPUT_COLS = [
    "player",
    "team",
    "slot_pct",
    "wide_pct",
    "man_rate",
    "zone_rate",
    "primary_cb",
    "shadow_flag",
]


def fetch_tables(url: str) -> list[pd.DataFrame]:
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HDRS, timeout=45)
            resp.raise_for_status()
            tables = pd.read_html(resp.text)
            if not tables:
                raise ValueError("no tables")
            return tables
        except Exception:
            if attempt == 2:
                return []
            time.sleep(2 * (attempt + 1))
    return []


def fetch_rotowire_alignment() -> pd.DataFrame:
    last_html: Optional[str] = None
    last_exc: Optional[Exception] = None
    target: Optional[pd.DataFrame] = None
    for attempt in range(3):
        try:
            resp = requests.get(ROTOWIRE_URL, headers=HDRS, timeout=45)
            resp.raise_for_status()
            last_html = resp.text
            tables = pd.read_html(last_html)
            if not tables:
                raise ValueError("no tables found in Rotowire alignment page")
            target = tables[0]
            for table in tables:
                cols = [str(c).lower() for c in table.columns]
                if any("player" in c for c in cols) and any("slot" in c for c in cols) and (
                    any("outside" in c for c in cols) or any("wide" in c for c in cols)
                ) and any("team" in c for c in cols):
                    target = table
                    break
            break
        except Exception as exc:  # pragma: no cover - network/HTML shifts
            last_exc = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))

    if target is None:
        if last_html:
            DEBUG_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
            DEBUG_HTML_PATH.write_text(last_html)
        if last_exc:
            raise last_exc
        return pd.DataFrame(columns=["player", "team", "slot_pct", "wide_pct"])

    df = target.loc[:, ~target.columns.duplicated()].copy()
    colmap = {}
    for c in df.columns:
        lc = str(c).lower()
        if "player" in lc:
            colmap[c] = "player"
        elif "team" in lc:
            colmap[c] = "team_name"
        elif "slot" in lc:
            colmap[c] = "slot_pct"
        elif "outside" in lc or "wide" in lc:
            colmap[c] = "wide_pct"
    df = df.rename(columns=colmap)
    keep = [c for c in ["player", "team_name", "slot_pct", "wide_pct"] if c in df.columns]
    df = df[keep].copy()
    for col in ["slot_pct", "wide_pct"]:
        if col in df.columns:
            cleaned = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.extract(r"([0-9]+\.?[0-9]*)")[0]
            )
            df[col] = pd.to_numeric(cleaned, errors="coerce") / 100.0
    for col in ["slot_pct", "wide_pct"]:
        if col not in df.columns:
            df[col] = pd.NA
    df["team"] = df.get("team_name", "").map(TEAM_NAME_TO_ABBR)
    df = df.dropna(subset=["team"]).copy()
    df = df.drop(columns=[c for c in ["team_name"] if c in df.columns])
    df = df.drop_duplicates(subset=["player", "team"])
    return df[["player", "team", "slot_pct", "wide_pct"]]


def fetch_sharp_team_rates() -> pd.DataFrame:
    tables = fetch_tables(SHARP_URL)
    if not tables:
        return pd.DataFrame(columns=["team", "man_rate", "zone_rate"])

    target = tables[0]
    for table in tables:
        cols = [str(c).lower() for c in table.columns]
        if any("team" in c for c in cols) and any("man" in c for c in cols) and any(
            "zone" in c for c in cols
        ):
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
    return df[["team", "man_rate", "zone_rate"]].drop_duplicates(subset=["team"])


def fetch_rotoballer_notes() -> pd.DataFrame:
    html = None
    for attempt in range(3):
        try:
            resp = requests.get(ROTOBALLER_URL, headers=HDRS, timeout=45)
            resp.raise_for_status()
            html = resp.text
            break
        except Exception:
            if attempt == 2:
                return pd.DataFrame(columns=["player", "primary_cb", "shadow_flag"])
            time.sleep(2 * (attempt + 1))
    if not html:
        return pd.DataFrame(columns=["player", "primary_cb", "shadow_flag"])

    pairs = []
    pattern = re.compile(
        r"([A-Z][a-zA-Z'\.]+(?:\s[A-Z][a-zA-Z'\.]+){0,2})\s+(?:vs\.?|will see|draws)\s+([A-Z][a-zA-Z'\-\.]+(?:\s[A-Z][a-zA-Z'\-\.]+){0,2})"
    )
    for match in pattern.finditer(html):
        wr = match.group(1).strip()
        cb = match.group(2).strip()
        if 1 <= len(wr.split()) <= 3 and 1 <= len(cb.split()) <= 4:
            pairs.append((wr, cb))
    df = pd.DataFrame(pairs, columns=["player", "primary_cb"]).drop_duplicates()
    shadow_flag = "shadow_watch" if re.search(r"shadow", html, re.IGNORECASE) else ""
    df["shadow_flag"] = shadow_flag
    return df


def build_cb_coverage_player() -> pd.DataFrame:
    align = fetch_rotowire_alignment()
    if align.empty:
        return pd.DataFrame(columns=OUTPUT_COLS)

    coverage = fetch_sharp_team_rates()
    merged = align.merge(coverage, on="team", how="left")

    notes = fetch_rotoballer_notes()
    if not notes.empty:
        merged = merged.merge(notes, on="player", how="left")
    if "primary_cb" not in merged.columns:
        merged["primary_cb"] = ""
    if "shadow_flag" not in merged.columns:
        merged["shadow_flag"] = ""

    merged = merged[OUTPUT_COLS].copy()
    merged = merged.drop_duplicates(subset=["player", "team"]).reset_index(drop=True)
    return merged


def main() -> None:
    out_path = Path("data") / "cb_coverage_player.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = build_cb_coverage_player()
    if df.empty:
        raise RuntimeError("Coverage table parsed empty. Inspect DOM selectors.")
    df.to_csv(out_path, index=False)
    print(f"[build_cb_coverage_player] wrote {out_path} with {len(df)} rows.")


if __name__ == "__main__":
    main()
