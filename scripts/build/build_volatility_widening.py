#!/usr/bin/env python3
# build_volatility_widening.py
#
# Output: volatility_widening.csv with columns:
#   team,week,pace_std,pass_rate_std,score_margin_volatility
#
# Free data source:
# - nflverse/nflfastR play-by-play (2025 season): https://github.com/nflverse/nflverse-data/releases/download/pbp/pbp_2025.csv.gz
#   Contains qb_dropback, rush_attempt, posteam, defteam, score_differential, play_id, game_id, game_seconds_remaining.
# - Field dictionary: https://nflreadr.nflverse.com/articles/dictionary_pbp.html
#
# Metric definitions:
# - pace_std: standard deviation of seconds/play for that team's offensive plays in a week (using deltas in game_seconds_remaining between snaps).
# - pass_rate_std: standard deviation of pass rate across drives within the week. For each drive, pass_rate = passes / (passes + runs).
# - score_margin_volatility: standard deviation of score_differential for that team's offensive plays in that week.
#
# Neutral filters are not used here; it's raw volatility. All free data.
#
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import numpy as np

from scripts.utils.nflverse_fetch import get_pbp_2025
from scripts.utils.pbp_threshold import get_dynamic_min_rows


def compute_pace_std(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    d = df.sort_values(["game_id", "play_id"]).copy()
    d["prev_time"] = d.groupby("game_id")["game_seconds_remaining"].shift(1)
    d["prev_team"] = d.groupby("game_id")["posteam"].shift(1)
    d = d[d["prev_team"] == d["posteam"]]
    d["delta"] = d["prev_time"] - d["game_seconds_remaining"]
    d = d[(d["delta"].notna()) & (d["delta"] >= 0) & (d["delta"] < 600)]
    if d.empty:
        return np.nan
    return float(d["delta"].std())


def compute_pass_rate_std(df: pd.DataFrame) -> float:
    if "drive" not in df.columns:
        return np.nan
    # Compute drive-level pass rate for this team's offensive plays
    drives = (
        df.groupby("drive")
        .apply(
            lambda x: (x["qb_dropback"].sum())
            / max(1, (x["qb_dropback"].sum() + x["rush_attempt"].sum()))
        )
        .rename("pass_rate")
    )
    if len(drives) < 2:
        return np.nan
    return float(drives.std())


def compute_score_margin_volatility(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    return float(df["score_differential"].std())


def _maybe_warn(df: pd.DataFrame, pbp: pd.DataFrame, label: str) -> None:
    total_games = 0
    if "game_id" in pbp.columns:
        total_games = int(pbp["game_id"].dropna().nunique())
    elif {"week", "posteam"}.issubset(pbp.columns):
        total_games = int(pbp[["week", "posteam"]].dropna().drop_duplicates().shape[0] // 2)
    min_dynamic = max(2000, total_games * 150)
    if len(df) < min_dynamic:
        print(
            f"[builder WARNING] {label} low sample size ({len(df)} rows < {min_dynamic}), writing partial output anyway"
        )


def main(out_csv: str = str(Path("data") / "volatility_widening.csv")):
    min_rows_target = get_dynamic_min_rows()
    pbp = get_pbp_2025(min_rows=20000)
    print(
        f"[volatility_widening] PBP loaded rows: {len(pbp)} (soft target {min_rows_target})"
    )
    if "season" in pbp.columns:
        pbp = pbp[pbp["season"] == 2025].copy()
    if len(pbp) <= 1000:
        print(
            "[builder WARNING] volatility_widening.csv generated from limited PBP sample (<=1000 rows)"
        )

    # Required fields
    cols = [
        "week",
        "posteam",
        "qb_dropback",
        "rush_attempt",
        "drive",
        "score_differential",
        "game_id",
        "play_id",
        "game_seconds_remaining",
    ]
    for c in cols:
        if c not in pbp.columns:
            raise RuntimeError(f"Missing required column: {c}")
    for col in ["qb_dropback", "rush_attempt"]:
        pbp[col] = pd.to_numeric(pbp[col], errors="coerce").fillna(0).astype(int)
    pbp["score_differential"] = pd.to_numeric(
        pbp["score_differential"], errors="coerce"
    )

    # Offensive plays
    pbp_off = pbp[(pbp["qb_dropback"] == 1) | (pbp["rush_attempt"] == 1)].copy()

    rows = []
    for wk in sorted(pbp_off["week"].dropna().unique()):
        wk_df = pbp_off[pbp_off["week"] == wk]
        for team in sorted(wk_df["posteam"].dropna().unique()):
            team_df = wk_df[wk_df["posteam"] == team]
            pace_std = compute_pace_std(team_df)
            pass_rate_std = compute_pass_rate_std(team_df)
            score_vol = compute_score_margin_volatility(team_df)
            rows.append(
                {
                    "team": team,
                    "week": int(wk),
                    "pace_std": None if pd.isna(pace_std) else round(pace_std, 2),
                    "pass_rate_std": (
                        None if pd.isna(pass_rate_std) else round(pass_rate_std, 4)
                    ),
                    "score_margin_volatility": (
                        None if pd.isna(score_vol) else round(score_vol, 2)
                    ),
                }
            )

    out = pd.DataFrame(
        rows,
        columns=[
            "team",
            "week",
            "pace_std",
            "pass_rate_std",
            "score_margin_volatility",
        ],
    )
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _maybe_warn(out, pbp, out_path.name)
    out.to_csv(out_path, index=False)
    print(f"[builder] wrote {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    out = (
        str(Path("data") / "volatility_widening.csv")
        if len(sys.argv) < 2
        else sys.argv[1]
    )
    main(out)
