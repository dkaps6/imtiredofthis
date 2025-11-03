#!/usr/bin/env python3
"""Derive opponent map from sportsbook props and odds feeds."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

from scripts.utils.name_clean import canonical_player
from scripts.utils.team_maps import TEAM_NAME_TO_ABBR

DATA_DIR = Path("data")
OUT_PATH = DATA_DIR / "opponent_map_from_props.csv"

REQUIRED_COLUMNS = [
    "player",
    "team",
    "opponent",
    "season",
    "week",
    "event_id",
    "player_clean_key",
    "team_abbr",
    "opponent_abbr",
    "game_timestamp",
]

TEAM_SOURCE_CANDIDATES = [
    "team",
    "team_abbr",
    "player_team",
    "player_team_abbr",
    "player_team_code",
    "player_team_name",
    "team_name",
]
OPP_SOURCE_CANDIDATES = [
    "opponent",
    "opponent_abbr",
    "opponent_team_abbr",
    "player_opponent",
    "player_opponent_abbr",
]
TIMESTAMP_CANDIDATES = [
    "market_timestamp",
    "timestamp",
    "last_update",
    "updated_at",
    "created_at",
    "commence_time",
]


def _read_first_existing(paths: Iterable[str]) -> Tuple[pd.DataFrame, str]:
    for raw in paths:
        path = Path(raw)
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        except Exception as err:  # pragma: no cover - best-effort logging only
            print(f"[opponent_map] WARN: failed to read {path}: {err}")
            continue
        return df, str(path)
    return pd.DataFrame(), ""


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[])
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    return out


def _team_lookup(value) -> str | pd.NA:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    direct = TEAM_NAME_TO_ABBR.get(text)
    if direct:
        return direct
    direct = TEAM_NAME_TO_ABBR.get(text.upper())
    if direct:
        return direct
    direct = TEAM_NAME_TO_ABBR.get(text.lower())
    if direct:
        return direct
    upper = text.upper()
    if upper in TEAM_NAME_TO_ABBR.values():
        return upper
    return pd.NA


def _map_team_series(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(pd.NA, index=index, dtype="string")
    mapped = series.map(_team_lookup)
    return mapped.astype("string")


def _ensure_player_column(df: pd.DataFrame) -> pd.DataFrame:
    if "player" in df.columns:
        return df
    for cand in ("player_name", "player_full_name", "name"):
        if cand in df.columns:
            df = df.rename(columns={cand: "player"})
            return df
    df["player"] = pd.NA
    return df


def _ensure_event_id(df: pd.DataFrame) -> pd.DataFrame:
    if "event_id" not in df.columns:
        df["event_id"] = pd.NA
    df["event_id"] = df["event_id"].astype("string").str.strip()
    return df


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.Series(dtype="string") if col not in {"season", "week"} else pd.Series(dtype="Int64")
    return df[REQUIRED_COLUMNS]


def _empty_output(out_path: Path) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    empty = pd.DataFrame({col: [] for col in REQUIRED_COLUMNS})
    empty.to_csv(out_path, index=False)
    return empty


def build_opponent_map(
    props_path: str = "outputs/props_raw.csv",
    odds_path: str = "outputs/odds_game.csv",
    out_path: str | Path = OUT_PATH,
) -> pd.DataFrame:
    """Create opponent map CSV by joining props to odds on event_id."""
    out_path = Path(out_path)

    props_candidates = [props_path]
    for fallback in ("outputs/props_raw.csv", "data/props_raw.csv"):
        if fallback not in props_candidates:
            props_candidates.append(fallback)
    props_raw, props_source = _read_first_existing(props_candidates)
    props = _normalize_columns(props_raw)
    props = _ensure_player_column(props)
    props = _ensure_event_id(props)

    odds_candidates = [odds_path]
    for fallback in ("outputs/odds_game.csv", "data/odds_game.csv"):
        if fallback not in odds_candidates:
            odds_candidates.append(fallback)
    games_raw, games_source = _read_first_existing(odds_candidates)
    games = _normalize_columns(games_raw)
    if not games.empty:
        games = _ensure_event_id(games)

    if props.empty:
        print("[opponent_map] WARN: no props data available; writing empty opponent map")
        _empty_output(out_path)
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    props = props[props["player"].astype("string").str.strip() != ""].copy()
    props = props[props["event_id"].astype("string").str.strip() != ""].copy()
    if props.empty:
        print("[opponent_map] WARN: props lack player/event_id identifiers; writing empty opponent map")
        _empty_output(out_path)
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    props["player"] = props["player"].astype("string").str.strip()
    props["player_clean_key"] = props["player"].map(canonical_player).astype("string")

    keep_cols = {"event_id", "player", "player_clean_key", "season", "week"}
    keep_cols.update(TEAM_SOURCE_CANDIDATES)
    keep_cols.update(OPP_SOURCE_CANDIDATES)
    existing_cols = [c for c in props.columns if c in keep_cols]
    props = props.loc[:, existing_cols].copy()

    if games.empty:
        merged = props.copy()
        merged["home_team"] = pd.NA
        merged["away_team"] = pd.NA
        merged["commence_time"] = pd.NA
        merged["season"] = merged.get("season", pd.Series(pd.NA, index=merged.index, dtype="Int64"))
        merged["week"] = merged.get("week", pd.Series(pd.NA, index=merged.index, dtype="Int64"))
    else:
        keep_game_cols = [
            c
            for c in (
                "event_id",
                "home_team",
                "away_team",
                "season",
                "week",
                "commence_time",
            )
            if c in games.columns
        ]
        games_subset = games.loc[:, keep_game_cols].drop_duplicates(subset=["event_id"], keep="last")
        merged = props.merge(games_subset, on="event_id", how="left")

    merged["home_team_abbr"] = _map_team_series(merged.get("home_team"), merged.index)
    merged["away_team_abbr"] = _map_team_series(merged.get("away_team"), merged.index)

    team_series = pd.Series(pd.NA, index=merged.index, dtype="string")
    for col in TEAM_SOURCE_CANDIDATES:
        if col in merged.columns:
            team_series = team_series.combine_first(_map_team_series(merged[col], merged.index))

    opp_series = pd.Series(pd.NA, index=merged.index, dtype="string")
    for col in OPP_SOURCE_CANDIDATES:
        if col in merged.columns:
            opp_series = opp_series.combine_first(_map_team_series(merged[col], merged.index))

    # Derive missing opponent/team values from home/away sides when possible
    if not merged.empty:
        mask = team_series.isna() & opp_series.notna() & merged.get("home_team_abbr").notna()
        if mask.any():
            opp_home = opp_series[mask] == merged.loc[mask, "home_team_abbr"]
            opp_away = opp_series[mask] == merged.loc[mask, "away_team_abbr"]
            team_series.loc[mask & opp_home] = merged.loc[mask & opp_home, "away_team_abbr"]
            team_series.loc[mask & opp_away] = merged.loc[mask & opp_away, "home_team_abbr"]

        mask = team_series.notna() & opp_series.isna()
        if mask.any():
            team_is_home = team_series[mask] == merged.loc[mask, "home_team_abbr"]
            team_is_away = team_series[mask] == merged.loc[mask, "away_team_abbr"]
            opp_series.loc[mask & team_is_home] = merged.loc[mask & team_is_home, "away_team_abbr"]
            opp_series.loc[mask & team_is_away] = merged.loc[mask & team_is_away, "home_team_abbr"]

    # Normalize season/week types
    for col in ("season", "week"):
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("Int64")
        else:
            merged[col] = pd.Series(pd.NA, index=merged.index, dtype="Int64")

    timestamp_col = next((c for c in TIMESTAMP_CANDIDATES if c in merged.columns), None)
    if timestamp_col:
        ts = pd.to_datetime(merged[timestamp_col], errors="coerce", utc=True)
        game_timestamp = ts.astype("string")
    else:
        game_timestamp = pd.Series(pd.NA, index=merged.index, dtype="string")

    out = pd.DataFrame(index=merged.index)
    out["player"] = merged["player"].astype("string").str.strip()
    out["event_id"] = merged["event_id"].astype("string").str.strip()
    out["player_clean_key"] = merged["player_clean_key"].astype("string")
    out["team_abbr"] = team_series
    out["opponent_abbr"] = opp_series
    out["team"] = out["team_abbr"]
    out["opponent"] = out["opponent_abbr"]
    out["season"] = merged["season"]
    out["week"] = merged["week"]
    out["game_timestamp"] = game_timestamp

    out = out.dropna(subset=["event_id", "player_clean_key"], how="any")
    out = out.drop_duplicates(subset=["event_id", "player_clean_key"])
    out = _ensure_required_columns(out)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    joined_rows = len(out)
    print(
        "[opponent_map] wrote"
        f" {joined_rows:,} rows → {out_path}"
        + (f" (props={props_source})" if props_source else "")
        + (f" (odds={games_source})" if games_source else ""),
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--props-path", default="outputs/props_raw.csv")
    parser.add_argument("--odds-path", default="outputs/odds_game.csv")
    parser.add_argument("--out-path", default=str(OUT_PATH))
    args = parser.parse_args()

    build_opponent_map(
        props_path=args.props_path,
        odds_path=args.odds_path,
        out_path=args.out_path,
    )


if __name__ == "__main__":
    main()
