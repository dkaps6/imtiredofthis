#!/usr/bin/env python3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import re

import pandas as pd

SRC_PLAYER = Path("data") / "cb_coverage_player.csv"
SRC_TEAM = Path("data") / "cb_coverage_team.csv"
SRC_SCHEDULE = Path("data") / "opponent_map_from_props.csv"
OUT = Path("data") / "wr_cb_exposure.csv"

OUTPUT_COLS = [
    "player",
    "player_pf",
    "team",
    "opponent",
    "week",
    "season",
    "game_timestamp",
    "slot_pct",
    "wide_pct",
    "man_rate",
    "zone_rate",
    "exp_vs_man",
    "exp_vs_zone",
    "primary_cb",
    "shadow_flag",
]


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:  # pragma: no cover - defensive fallback
        return None
    return df if not df.empty and len(df.columns) else None


def normalize_team(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper().str.strip()


def to_datetime_utc(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="datetime64[ns, UTC]")
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    numeric = pd.to_numeric(series, errors="coerce")
    dt_numeric = pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
    dt_series = pd.to_datetime(series, utc=True, errors="coerce")
    return dt_numeric.combine_first(dt_series)


def pf_key(name: str) -> str:
    tokens = re.sub(r"[^A-Za-z\s\-]", "", str(name)).strip().split()
    if not tokens:
        return ""
    first = tokens[0][0].upper()
    last = tokens[-1].capitalize()
    return f"{first}{last}"


def select_upcoming(schedule: pd.DataFrame | None) -> pd.DataFrame:
    if schedule is None or schedule.empty:
        return pd.DataFrame(columns=["team", "opponent", "week", "season", "game_timestamp"])

    df = schedule.copy()
    df["team"] = normalize_team(df.get("team", ""))
    df["opponent"] = normalize_team(df.get("opponent", ""))
    df["week"] = pd.to_numeric(df.get("week"), errors="coerce").astype("Int64")
    df["season"] = pd.to_numeric(df.get("season"), errors="coerce").astype("Int64")
    if "game_timestamp" in df.columns:
        df["game_time"] = to_datetime_utc(df["game_timestamp"])
    else:
        df["game_time"] = pd.NaT

    now = pd.Timestamp.now(tz="UTC")
    selected_rows = []
    for _, grp in df.groupby("team"):
        grp_sorted = grp.sort_values(
            by=["game_time", "season", "week"],
            ascending=[True, False, False],
            na_position="last",
        )
        future = grp_sorted[grp_sorted["game_time"].notna() & (grp_sorted["game_time"] >= now - pd.Timedelta(hours=12))]
        if not future.empty:
            selected_rows.append(future.iloc[0])
            continue
        valid_time = grp_sorted[grp_sorted["game_time"].notna()]
        if not valid_time.empty:
            selected_rows.append(valid_time.iloc[0])
            continue
        grp_week = grp_sorted.sort_values(["season", "week"], ascending=[False, False])
        selected_rows.append(grp_week.iloc[0])

    if not selected_rows:
        return pd.DataFrame(columns=["team", "opponent", "week", "season", "game_timestamp"])

    selected = pd.DataFrame(selected_rows)
    if "game_timestamp" not in selected:
        selected["game_timestamp"] = ""
    return selected[["team", "opponent", "week", "season", "game_timestamp"]]


def build_wr_cb_exposure() -> pd.DataFrame:
    players = safe_read_csv(SRC_PLAYER)
    if players is None:
        return pd.DataFrame(columns=OUTPUT_COLS)

    players = players.copy()
    players["player"] = players.get("player", "").fillna("").astype(str).str.strip()
    players["team"] = normalize_team(players.get("team", ""))
    players["player_pf"] = players["player"].apply(pf_key)

    for col in ["slot_pct", "wide_pct", "man_rate", "zone_rate"]:
        if col in players.columns:
            players[col] = pd.to_numeric(players[col], errors="coerce")
        else:
            players[col] = pd.NA

    players["primary_cb"] = players.get("primary_cb", "").fillna("").astype(str)
    players["shadow_flag"] = players.get("shadow_flag", "").fillna("").astype(str)

    schedule = safe_read_csv(SRC_SCHEDULE)
    opp = select_upcoming(schedule)
    if not opp.empty:
        players = players.merge(opp, on="team", how="left")
    else:
        players["opponent"] = ""
        players["week"] = pd.NA
        players["season"] = pd.NA
        players["game_timestamp"] = ""

    for col, fill in (
        ("opponent", ""),
        ("game_timestamp", ""),
    ):
        if col in players.columns:
            players[col] = players[col].fillna(fill)

    for col in ("week", "season"):
        if col in players.columns:
            players[col] = pd.to_numeric(players[col], errors="coerce").astype("Int64")

    team_rates = safe_read_csv(SRC_TEAM)
    if team_rates is not None:
        team_rates = team_rates.copy()
        team_rates["team"] = normalize_team(team_rates.get("team", ""))
        team_rates = team_rates.drop_duplicates(subset=["team"])
        opp_rates = team_rates.rename(
            columns={"team": "opponent", "man_rate": "exp_vs_man", "zone_rate": "exp_vs_zone"}
        )[["opponent", "exp_vs_man", "exp_vs_zone"]]
        players = players.merge(opp_rates, on="opponent", how="left")
    else:
        players["exp_vs_man"] = pd.NA
        players["exp_vs_zone"] = pd.NA

    for col in ["exp_vs_man", "exp_vs_zone"]:
        if col in players.columns:
            players[col] = pd.to_numeric(players[col], errors="coerce")

    for col in OUTPUT_COLS:
        if col not in players.columns:
            players[col] = pd.NA if col not in {"player", "player_pf", "team", "opponent", "primary_cb", "shadow_flag", "game_timestamp"} else ""

    return players[OUTPUT_COLS].drop_duplicates(subset=["team", "player_pf"])


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = build_wr_cb_exposure()
    if df.empty:
        print("[build_wr_cb_exposure] WARN: source coverage data missing or empty; writing header only")
        df = pd.DataFrame(columns=OUTPUT_COLS)
    df.to_csv(OUT, index=False)
    print(f"[build_wr_cb_exposure] wrote {OUT} with {len(df)} rows.")


if __name__ == "__main__":
    main()
