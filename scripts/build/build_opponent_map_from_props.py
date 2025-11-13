#!/usr/bin/env python3
"""Derive live opponent mapping from sportsbook props and odds."""
from __future__ import annotations

import argparse
import os
from datetime import datetime, date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from scripts.utils.name_canon import make_player_key, most_common_full_name
from scripts.utils.name_clean import (
    build_roster_lookup,
    canonical_key,
    canonical_player,
    initials_last_to_full,
    normalize_team,
)
from scripts._opponent_map import canon_team
from scripts.utils.df_keys import coerce_merge_keys


DATA_DIR = Path("data")
OUT_PATH = DATA_DIR / "opponent_map_from_props.csv"
DEFAULT_OUT = OUT_PATH
UNRESOLVED_OUT = DATA_DIR / "opponent_map_unresolved.csv"
PLAYER_NAME_MAP_PATH = DATA_DIR / "player_name_map_from_props.csv"
MISSING_SAMPLE_PATH = DATA_DIR / "opponent_map_missing_sample.csv"
TEAM_WEEK_MAP_PATH = DATA_DIR / "team_week_map.csv"
ODDS_GAME_PATH = Path("outputs/odds_game.csv")
UNRESOLVED_COLUMNS = [
    "reason",
    "player_raw",
    "player_name_clean",
    "player_clean_key",
    "team",
    "opponent",
    "season",
    "week",
    "event_id",
]


def _read_first(paths: Iterable[Path]) -> pd.DataFrame:
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        except Exception as err:  # pragma: no cover - defensive logging
            print(f"[opponent_map] WARN: failed to read {path}: {err}")
            continue
        if not df.empty:
            return df
    return pd.DataFrame()


def _load_csv_safe(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def _to_epoch_seconds(series: pd.Series) -> pd.Series:
    """Coerce arbitrary datetime-like inputs into UTC epoch seconds."""

    if series is None:
        return pd.Series(dtype="Int64")

    if not isinstance(series, pd.Series):
        series = pd.Series(series)

    out = pd.Series(pd.NA, index=series.index, dtype="Int64")

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_mask = numeric.notna()
    if numeric_mask.any():
        out.loc[numeric_mask] = (
            numeric.loc[numeric_mask].round().astype("Int64")
        )

    remaining = out.isna()
    if remaining.any():
        dt = pd.to_datetime(series.loc[remaining], utc=True, errors="coerce")
        if not dt.empty:
            epoch = (dt.dropna().view("int64") // 1_000_000_000)
            epoch_series = pd.Series(epoch, index=dt.dropna().index, dtype="Int64")
            out.loc[epoch_series.index] = epoch_series

    return out


def _add_game_timestamp_from_odds(df: pd.DataFrame) -> pd.DataFrame:
    """Attach UTC kickoff timestamp for each event when possible."""

    if df is None or df.empty:
        return df

    if "event_id" not in df.columns:
        df = df.copy()
        df["game_timestamp"] = pd.NaT
        return df

    if not ODDS_GAME_PATH.exists():
        df = df.copy()
        df["game_timestamp"] = pd.NaT
        return df

    try:
        odds_games = pd.read_csv(ODDS_GAME_PATH)
    except Exception:
        df = df.copy()
        df["game_timestamp"] = pd.NaT
        return df

    if odds_games.empty or "event_id" not in odds_games.columns:
        df = df.copy()
        df["game_timestamp"] = pd.NaT
        return df

    working = odds_games.copy()
    if "commence_time" in working.columns:
        working["game_timestamp"] = _to_epoch_seconds(working["commence_time"])
    elif "kickoff_utc" in working.columns:
        working["game_timestamp"] = _to_epoch_seconds(working["kickoff_utc"])
    elif "game_timestamp" in working.columns:
        working["game_timestamp"] = _to_epoch_seconds(working["game_timestamp"])
    else:
        working["game_timestamp"] = pd.Series(pd.NA, index=working.index, dtype="Int64")

    keep = working[["event_id", "game_timestamp"]].drop_duplicates()
    merged = df.merge(keep, on="event_id", how="left")
    if "game_timestamp_x" in merged.columns:
        merged["game_timestamp"] = merged["game_timestamp_x"].combine_first(
            merged.get("game_timestamp_y")
        )
        merged.drop(columns=["game_timestamp_x", "game_timestamp_y"], inplace=True)

    if "game_timestamp" not in merged.columns:
        merged["game_timestamp"] = pd.Series(
            pd.NA, index=merged.index, dtype="Int64"
        )

    return merged


def _load_roles_dataframe() -> pd.DataFrame:
    candidates = [Path("data/roles_ourlads.csv"), Path("data/roles.csv")]
    frames: list[pd.DataFrame] = []

    for path in candidates:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        except Exception as err:  # pragma: no cover - defensive logging
            print(f"[opponent_map] WARN: failed to read {path}: {err}")
            continue
        if df.empty or "player" not in df.columns:
            continue
        frames.append(df)

    if frames:
        merged = pd.concat(frames, ignore_index=True, sort=False)
        return merged
    return pd.DataFrame()


def _load_team_week_map() -> pd.DataFrame:
    path = TEAM_WEEK_MAP_PATH
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as err:  # pragma: no cover - defensive logging
        print(f"[opponent_map] WARN: failed to read {path}: {err}")
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    working = df.copy()
    working.columns = [str(c).lower() for c in working.columns]

    for col in ("season", "week"):
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce").astype("Int64")

    for col in ("team", "opponent"):
        if col in working.columns:
            working[col] = _norm_team(working[col])

    if "opponent" in working.columns:
        opp_series = working["opponent"].astype("string")
        bye_mask = opp_series.str.upper().eq("BYE")
        working.loc[bye_mask.fillna(False), "opponent"] = pd.NA

    for col in ("game_id", "event_id"):
        if col in working.columns:
            working[col] = working[col].astype("string").str.strip()

    for col in ("kickoff_local", "kickoff_utc"):
        if col in working.columns:
            working[col] = pd.to_datetime(working[col], errors="coerce")

    return working


def _infer_week_from_schedule(
    schedule: pd.DataFrame, season_hint: int | None
) -> int | None:
    if schedule is None or schedule.empty:
        return None

    working = schedule.copy()

    if "season" in working.columns:
        working["season"] = pd.to_numeric(
            working["season"], errors="coerce"
        ).astype("Int64")
        if season_hint is not None:
            working = working.loc[working["season"] == season_hint]
        elif working["season"].notna().any():
            season_hint = int(working["season"].dropna().max())
            working = working.loc[working["season"] == season_hint]

    if "week" not in working.columns:
        return None

    week_values = pd.to_numeric(working["week"], errors="coerce").dropna()
    if week_values.empty:
        return None

    return int(week_values.max())


def _write_unresolved(records: list[dict[str, object]]) -> pd.DataFrame:
    UNRESOLVED_OUT.parent.mkdir(parents=True, exist_ok=True)
    if records:
        df = pd.DataFrame(records)
    else:
        df = pd.DataFrame(columns=UNRESOLVED_COLUMNS)
    missing_cols = [c for c in UNRESOLVED_COLUMNS if c not in df.columns]
    for col in missing_cols:
        df[col] = pd.NA
    df = df[UNRESOLVED_COLUMNS]
    df.to_csv(UNRESOLVED_OUT, index=False)
    return df


def _write_player_name_map(props: pd.DataFrame) -> None:
    if props is None or props.empty:
        return
    if "player_clean_key" not in props.columns or "player" not in props.columns:
        return

    try:
        grouped = (
            props.loc[:, ["player_clean_key", "player"]]
            .dropna(subset=["player_clean_key", "player"])
            .groupby("player_clean_key", dropna=False)["player"]
            .agg(list)
            .reset_index()
        )
    except Exception:
        return

    if grouped.empty:
        return

    grouped["player_name_full"] = grouped["player"].map(most_common_full_name)
    keep_cols = ["player_clean_key", "player_name_full"]
    name_map = grouped.loc[:, keep_cols].copy()
    name_map["player_clean_key"] = name_map["player_clean_key"].astype("string")
    name_map["player_name_full"] = name_map["player_name_full"].astype("string")

    PLAYER_NAME_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    name_map.to_csv(PLAYER_NAME_MAP_PATH, index=False)


def _norm_team(series: pd.Series) -> pd.Series:
    series = series.fillna("").astype("string")
    normalized = series.map(normalize_team).astype("object")

    def _canon(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        text = str(value).strip()
        if not text:
            return pd.NA
        canon = canon_team(text)
        return canon if canon else pd.NA

    return pd.Series(normalized.apply(_canon), index=series.index, dtype="string")


def _parse_commence(series: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, utc=True, errors="coerce")
    except Exception:
        parsed = pd.Series(pd.NaT, index=series.index if isinstance(series, pd.Series) else None)
    return parsed


def _resolve_slate_date(raw: str | None = None) -> datetime.date | None:
    """Resolve desired slate date from CLI override or environment."""

    if raw is None:
        raw = os.getenv("SLATE_DATE", "")

    raw = (raw or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw).date()
    except Exception:
        return None


def build_opponent_map(
    props_path: Path | None = None,
    odds_path: Path | None = None,
    out_path: Path = DEFAULT_OUT,
    slate_date: date | datetime | str | None = None,
    week: int | None = None,
) -> pd.DataFrame:
    props_candidates = [p for p in [props_path, Path("outputs/props_raw.csv"), Path("data/props_raw.csv")] if p]
    odds_candidates = [p for p in [odds_path, Path("outputs/odds_game.csv"), Path("data/odds_game.csv")] if p]

    props = _read_first(props_candidates)
    odds = _read_first(odds_candidates)

    if isinstance(slate_date, datetime):
        slate_dt: date | None = slate_date.date()
    elif isinstance(slate_date, date):
        slate_dt = slate_date
    elif isinstance(slate_date, str):
        slate_dt = _resolve_slate_date(slate_date)
    else:
        slate_dt = _resolve_slate_date()

    week_value: int | None = None
    if week is not None:
        try:
            week_value = int(week)
        except (TypeError, ValueError):
            week_value = None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    required_cols = [
        "player",
        "player_name_clean",
        "player_clean_key",
        "player_canonical",
        "team",
        "opponent",
        "team_abbr",
        "opponent_abbr",
        "season",
        "week",
        "event_id",
        "game_timestamp",
    ]

    unresolved_records: list[dict[str, object]] = []

    def _record_unresolved_rows(df_subset: pd.DataFrame, reason: str) -> None:
        if df_subset is None or df_subset.empty:
            return
        for _, rec in df_subset.iterrows():
            unresolved_records.append(
                {
                    "reason": reason,
                    "player_raw": rec.get("player"),
                    "player_name_clean": rec.get("player_name_clean"),
                    "player_clean_key": rec.get("player_clean_key"),
                    "team": rec.get("team"),
                    "opponent": rec.get("opponent", pd.NA),
                    "season": rec.get("season"),
                    "week": rec.get("week"),
                    "event_id": rec.get("event_id"),
                }
            )

    def _return_empty(message: str) -> pd.DataFrame:
        empty = pd.DataFrame(columns=required_cols)
        empty.to_csv(out_path, index=False)
        unresolved_df = _write_unresolved(unresolved_records)
        print(message)
        print(f"[opponent_map] unresolved rows: {len(unresolved_df):,} -> {UNRESOLVED_OUT}")
        return empty

    if props.empty or odds.empty:
        return _return_empty("[opponent_map] WARN: missing props or odds source; wrote empty map")

    def _extract_season(frame: pd.DataFrame) -> int | None:
        if frame is None or frame.empty or "season" not in frame.columns:
            return None
        try:
            series = pd.to_numeric(frame["season"], errors="coerce").dropna()
        except Exception:
            return None
        if series.empty:
            return None
        return int(series.max())

    season_hint = _extract_season(props)
    if season_hint is None:
        season_hint = _extract_season(odds)

    schedule_df = _load_team_week_map()
    if (
        season_hint is None
        and schedule_df is not None
        and not schedule_df.empty
        and "season" in schedule_df.columns
    ):
        season_vals = pd.to_numeric(schedule_df["season"], errors="coerce").dropna()
        if not season_vals.empty:
            season_hint = int(season_vals.max())

    if season_hint is None:
        env_season = os.getenv("SEASON", "").strip()
        try:
            season_hint = int(env_season)
        except (TypeError, ValueError):
            season_hint = None

    target_week = week_value
    if target_week is None and slate_dt is None:
        target_week = _infer_week_from_schedule(schedule_df, season_hint)

    roster_lookup = build_roster_lookup(_load_roles_dataframe())

    props = props.copy()
    props.columns = [c.lower() for c in props.columns]
    if "commence_time" in props.columns:
        props["commence_time"] = pd.to_datetime(
            props["commence_time"], errors="coerce", utc=True
        )
    if "player" not in props.columns:
        alt = next((c for c in ("player_name", "name") if c in props.columns), None)
        if alt:
            props.rename(columns={alt: "player"}, inplace=True)
        else:
            props["player"] = pd.NA
    props["player"] = props["player"].astype("string").str.strip()
    props = props[props["player"].str.len() > 0]

    if "player" in props.columns:
        props["_player_key_fallback"] = props["player"].astype("string").map(
            make_player_key
        )
    else:
        props["_player_key_fallback"] = ""

    if "event_id" not in props.columns:
        props["event_id"] = pd.NA
    props["event_id"] = props["event_id"].astype("string").str.strip()
    props = props[props["event_id"].str.len() > 0]
    if props.empty:
        return _return_empty("[opponent_map] WARN: props missing event_id; wrote empty map")

    def _clean_player(row: pd.Series) -> pd.Series:
        raw_name = row.get("player")
        team_val = row.get("team_abbr")
        if not team_val or str(team_val).strip() == "":
            team_val = row.get("team")
        team_norm = normalize_team(team_val) if isinstance(team_val, str) else team_val
        full = canonical_player(raw_name)
        if (not full) or full.count(" ") == 0:
            alt = initials_last_to_full(raw_name, team_norm, roster_lookup)
            if alt:
                full = alt
        clean_key = canonical_key(full)
        return pd.Series({"player_name_clean": full, "player_clean_key": clean_key})

    cleaned = props.apply(_clean_player, axis=1)
    props["player_name_clean"] = cleaned["player_name_clean"].astype("string")
    props["player_clean_key"] = cleaned["player_clean_key"].astype("string")
    props["_player_key_fallback"] = props["_player_key_fallback"].astype("string")
    props["player_canonical"] = (
        props["player_name_clean"].where(
            props["player_name_clean"].astype("string").str.strip().str.len() > 0,
            props["player"],
        )
    ).astype("string")

    missing_key_mask = props["player_clean_key"].fillna("").str.len() == 0
    if missing_key_mask.any():
        props.loc[missing_key_mask, "player_clean_key"] = props.loc[
            missing_key_mask, "_player_key_fallback"
        ]
    missing_key_mask = props["player_clean_key"].fillna("").str.len() == 0
    if missing_key_mask.any():
        _record_unresolved_rows(props.loc[missing_key_mask], "missing_player_key")
        props = props.loc[~missing_key_mask].copy()

    if props.empty:
        return _return_empty("[opponent_map] WARN: no props with resolvable players; wrote empty map")

    _write_player_name_map(props)

    if "_player_key_fallback" in props.columns:
        props.drop(columns=["_player_key_fallback"], inplace=True)

    for col in ("team", "season", "week"):
        if col not in props.columns:
            props[col] = pd.NA
    if "team" in props.columns:
        props["team"] = _norm_team(props["team"])
    if "team_abbr" in props.columns:
        props["team_abbr"] = _norm_team(props["team_abbr"])
    for col in ("season", "week"):
        props[col] = pd.to_numeric(props[col], errors="coerce").astype("Int64")
    if season_hint is not None and "season" in props.columns:
        props["season"] = props["season"].fillna(int(season_hint)).astype("Int64")
    if target_week is not None and "week" in props.columns:
        props["week"] = props["week"].fillna(int(target_week)).astype("Int64")

    odds = odds.copy()
    odds.columns = [c.lower() for c in odds.columns]
    rename = {"home": "home_team", "away": "away_team"}
    for src, dst in rename.items():
        if src in odds.columns and dst not in odds.columns:
            odds.rename(columns={src: dst}, inplace=True)
    for col in ("home_team", "away_team"):
        if col in odds.columns:
            odds[col] = _norm_team(odds[col])

    if "commence_time" in odds.columns:
        odds["commence_time"] = _parse_commence(odds["commence_time"])

    live_mask = pd.Series(True, index=odds.index)
    if "status" in odds.columns:
        live_mask &= odds["status"].astype("string").str.lower().isin(["pre", "inprogress", "open"])
    if slate_dt is not None and "commence_time" in odds.columns:
        live_mask &= odds["commence_time"].dt.date.eq(slate_dt)
    odds = odds.loc[live_mask].copy()

    if odds.empty:
        empty = pd.DataFrame(columns=required_cols)
        empty.to_csv(out_path, index=False)
        print("[opponent_map] WARN: odds filter produced no live games; wrote empty map")
        return empty

    if "event_id" not in odds.columns:
        if "game_id" in odds.columns:
            odds.rename(columns={"game_id": "event_id"}, inplace=True)
        else:
            odds["event_id"] = pd.NA
    odds["event_id"] = odds["event_id"].astype("string").str.strip()
    odds = odds[odds["event_id"].str.len() > 0]
    if odds.empty:
        return _return_empty("[opponent_map] WARN: odds missing event_id; wrote empty map")

    keep_odds_cols = [
        c
        for c in (
            "event_id",
            "home_team",
            "away_team",
            "season",
            "week",
        )
        if c in odds.columns
    ]
    odds = odds.loc[:, keep_odds_cols].drop_duplicates(subset=["event_id"], keep="last")

    merged = props.merge(odds, on="event_id", how="left")
    if merged.empty:
        return _return_empty("[opponent_map] WARN: no props matched live odds; wrote empty map")

    merged["player_canonical"] = merged.get("player_canonical", pd.Series(dtype="string"))
    merged["player_canonical"] = merged["player_canonical"].astype("string").str.strip()

    if "home_team" in merged.columns:
        merged["home_team"] = _norm_team(merged["home_team"])
    if "away_team" in merged.columns:
        merged["away_team"] = _norm_team(merged["away_team"])

    if "home_team" in merged.columns and "home" not in merged.columns:
        merged["home"] = merged["home_team"]
    if "away_team" in merged.columns and "away" not in merged.columns:
        merged["away"] = merged["away_team"]

    player_upper = merged["player_canonical"].astype("string").str.upper()
    home_vals = merged.get("home_team", pd.Series(index=merged.index, dtype="string")).astype("string").str.upper()
    away_vals = merged.get("away_team", pd.Series(index=merged.index, dtype="string")).astype("string").str.upper()

    home_match = pd.Series(
        [
            bool(player) and bool(home) and home in player
            for player, home in zip(player_upper.fillna(""), home_vals.fillna(""))
        ],
        index=merged.index,
    )

    team_guess = pd.Series(
        np.where(home_match, home_vals.fillna(""), away_vals.fillna("")),
        index=merged.index,
        dtype="string",
    ).replace("", pd.NA)
    opponent_guess = pd.Series(
        np.where(home_match, away_vals.fillna(""), home_vals.fillna("")),
        index=merged.index,
        dtype="string",
    ).replace("", pd.NA)

    if "team_abbr" in merged.columns:
        merged["team_abbr"] = _norm_team(merged["team_abbr"])
    else:
        merged["team_abbr"] = pd.Series(pd.NA, index=merged.index, dtype="string")
    merged["team_abbr"] = merged["team_abbr"].replace({"": pd.NA})
    merged["team_abbr"] = merged["team_abbr"].combine_first(team_guess)

    if "opponent_abbr" in merged.columns:
        merged["opponent_abbr"] = _norm_team(merged["opponent_abbr"])
    else:
        merged["opponent_abbr"] = pd.Series(pd.NA, index=merged.index, dtype="string")
    merged["opponent_abbr"] = merged["opponent_abbr"].replace({"": pd.NA})
    merged["opponent_abbr"] = merged["opponent_abbr"].combine_first(opponent_guess)

    if "team" in merged.columns:
        merged["team"] = _norm_team(merged["team"])
        merged["team"] = merged["team"].combine_first(merged["team_abbr"])
    else:
        merged["team"] = merged["team_abbr"]

    if "opponent" in merged.columns:
        merged["opponent"] = _norm_team(merged["opponent"])
        merged["opponent"] = merged["opponent"].combine_first(merged["opponent_abbr"])
    else:
        merged["opponent"] = merged["opponent_abbr"]

    if "commence_time" in merged.columns:
        merged["game_timestamp"] = _to_epoch_seconds(merged["commence_time"])
    elif "game_timestamp" not in merged.columns:
        merged["game_timestamp"] = pd.Series(pd.NA, index=merged.index, dtype="Int64")

    if "team" in merged.columns and "home_team" in merged.columns and "away_team" in merged.columns:
        merged["opponent"] = merged.apply(
            lambda row: (
                row["away_team"]
                if row.get("team") == row.get("home_team")
                else row["home_team"]
                if row.get("team") == row.get("away_team")
                else pd.NA
            ),
            axis=1,
        ).astype("string")
    else:
        merged["opponent"] = pd.NA

    keep_cols = [
        c
        for c in [
            "player",
            "player_name_clean",
            "player_clean_key",
            "player_canonical",
            "team",
            "opponent",
            "season",
            "week",
            "event_id",
            "home",
            "away",
            "home_team",
            "away_team",
            "team_abbr",
            "opponent_abbr",
            "game_timestamp",
        ]
        if c in merged.columns
    ]
    out = merged.loc[:, keep_cols].copy()
    for col in required_cols:
        if col not in out.columns:
            out[col] = pd.NA

    out["team"] = _norm_team(out["team"])
    out["opponent"] = _norm_team(out["opponent"])
    out["player"] = out["player"].astype("string")
    out["player_name_clean"] = out["player_name_clean"].astype("string")
    out["player_clean_key"] = out["player_clean_key"].astype("string")
    if "player_canonical" in out.columns:
        out["player_canonical"] = out["player_canonical"].astype("string")
    for col in ("team_abbr", "opponent_abbr"):
        if col in out.columns:
            out[col] = _norm_team(out[col])
    for col in ("season", "week"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    out["event_id"] = out["event_id"].astype("string")
    if season_hint is not None and "season" in out.columns:
        out["season"] = out["season"].fillna(int(season_hint)).astype("Int64")
    if target_week is not None and "week" in out.columns:
        out["week"] = out["week"].fillna(int(target_week)).astype("Int64")

    schedule_subset = schedule_df if isinstance(schedule_df, pd.DataFrame) else pd.DataFrame()
    if not schedule_subset.empty:
        schedule_subset = schedule_subset.copy()
        if (
            season_hint is not None
            and "season" in schedule_subset.columns
        ):
            schedule_subset = schedule_subset.loc[
                schedule_subset["season"] == season_hint
            ]
        if (
            slate_dt is None
            and target_week is not None
            and "week" in schedule_subset.columns
        ):
            schedule_subset = schedule_subset.loc[
                schedule_subset["week"] == target_week
            ]

    if not schedule_subset.empty:
        rename = {}
        if "opponent" in schedule_subset.columns:
            rename["opponent"] = "schedule_opponent"
        if "game_id" in schedule_subset.columns:
            rename["game_id"] = "schedule_game_id"
        if "event_id" in schedule_subset.columns and "schedule_game_id" not in rename.values():
            rename["event_id"] = "schedule_game_id"
        if "kickoff_local" in schedule_subset.columns:
            rename["kickoff_local"] = "schedule_kickoff_local"
        if "kickoff_utc" in schedule_subset.columns:
            rename["kickoff_utc"] = "schedule_kickoff_utc"

        schedule_subset = schedule_subset.rename(columns=rename)
        join_cols = [
            col
            for col in ("season", "week", "team")
            if col in schedule_subset.columns and col in out.columns
        ]
        schedule_keep = join_cols + [
            col
            for col in (
                "schedule_opponent",
                "schedule_game_id",
                "schedule_kickoff_local",
                "schedule_kickoff_utc",
            )
            if col in schedule_subset.columns
        ]
        schedule_subset = schedule_subset.loc[:, schedule_keep].drop_duplicates(
            subset=join_cols, keep="last"
        )

        if join_cols and not schedule_subset.empty:
            left = out.copy()
            right = schedule_subset.copy()

            numeric = [c for c in ("season", "week") if c in join_cols]
            text = [c for c in join_cols if c not in numeric]
            if numeric:
                left = coerce_merge_keys(left, numeric, as_str=False)
                right = coerce_merge_keys(right, numeric, as_str=False)
            if text:
                left = coerce_merge_keys(left, text, as_str=True)
                right = coerce_merge_keys(right, text, as_str=True)

            merged = left.merge(right, on=join_cols, how="left", suffixes=("", "_sched"))

            if "opponent" in merged.columns and "opponent_props" not in merged.columns:
                merged["opponent_props"] = merged["opponent"]

            if "schedule_opponent" in merged.columns:
                sched_opp = merged["schedule_opponent"].astype("string")
                merged["schedule_opponent"] = sched_opp.replace("", pd.NA)
                opp_series = merged["opponent"].astype("string")
                fill_mask = opp_series.isna() | opp_series.str.strip().eq("")
                merged.loc[fill_mask, "opponent"] = merged.loc[fill_mask, "schedule_opponent"]

            if "schedule_game_id" in merged.columns:
                sched_event = merged["schedule_game_id"].astype("string").str.strip()
                merged["schedule_game_id"] = sched_event
                event_series = merged["event_id"].astype("string")
                missing_event = event_series.isna() | event_series.str.strip().eq("")
                merged.loc[missing_event, "event_id"] = merged.loc[missing_event, "schedule_game_id"]

            if "game_timestamp" in merged.columns:
                merged["game_timestamp"] = _to_epoch_seconds(merged["game_timestamp"])
            else:
                merged["game_timestamp"] = pd.Series(
                    pd.NA, index=merged.index, dtype="Int64"
                )

            if "schedule_kickoff_utc" in merged.columns:
                sched_ts = _to_epoch_seconds(merged["schedule_kickoff_utc"])
                merged["game_timestamp"] = merged["game_timestamp"].combine_first(sched_ts)

            out = merged

    drop_sched_cols = [c for c in out.columns if c.startswith("schedule_")]
    if drop_sched_cols:
        out.drop(columns=drop_sched_cols, inplace=True)

    opponent_str = out["opponent"].astype("string")
    missing_opponent_mask = opponent_str.fillna("").str.strip() == ""
    total_rows = len(out)
    missing_count = int(missing_opponent_mask.sum())
    if missing_opponent_mask.any():
        missing_rows = out.loc[missing_opponent_mask].copy()
        _record_unresolved_rows(missing_rows, "missing_opponent")
        for row in missing_rows.itertuples(index=False):
            name = getattr(row, "player_name_clean", "") or getattr(row, "player", "")
            team_val = getattr(row, "team", pd.NA)
            event_val = getattr(row, "event_id", pd.NA)
            print(f"[opponent_map] missing mapping for {name} ({team_val}, event {event_val})")
        sample_cols = [c for c in ["player", "event_id"] if c in missing_rows.columns]
        if sample_cols:
            try:
                MISSING_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
                missing_rows.loc[:, sample_cols].head(200).to_csv(
                    MISSING_SAMPLE_PATH, index=False
                )
            except Exception as err:
                print(
                    f"[opponent_map] WARN: failed writing missing sample to {MISSING_SAMPLE_PATH}: {err}"
                )
        out = out.loc[~missing_opponent_mask].copy()

    mapped_count = total_rows - missing_count
    print(
        "[opponent_map] summary: total=%s mapped=%s missing=%s"
        % (
            f"{total_rows:,}",
            f"{mapped_count:,}",
            f"{missing_count:,}",
        )
    )

    opponent_map = out.drop_duplicates(
        subset=["player_clean_key", "team", "opponent", "event_id"], keep="last"
    )
    opponent_map = coerce_merge_keys(
        opponent_map, ["player_clean_key", "team", "event_id"], as_str=True
    )
    opponent_map = _add_game_timestamp_from_odds(opponent_map)
    if "game_timestamp" in opponent_map.columns:
        opponent_map["game_timestamp"] = _to_epoch_seconds(
            opponent_map["game_timestamp"]
        )
    else:
        opponent_map["game_timestamp"] = pd.Series(
            pd.NA, index=opponent_map.index, dtype="Int64"
        )
    for col in [
        "season",
        "week",
        "player_clean_key",
        "team_abbr",
        "opponent_abbr",
        "event_id",
        "game_timestamp",
    ]:
        if col not in opponent_map.columns:
            opponent_map[col] = pd.NA
    if "season" in opponent_map.columns:
        opponent_map["season"] = pd.to_numeric(
            opponent_map["season"], errors="coerce"
        ).astype("Int64")
        if season_hint is not None:
            opponent_map["season"] = opponent_map["season"].fillna(
                int(season_hint)
            ).astype("Int64")
    if "week" in opponent_map.columns:
        opponent_map["week"] = pd.to_numeric(
            opponent_map["week"], errors="coerce"
        ).astype("Int64")
        if target_week is not None:
            opponent_map["week"] = opponent_map["week"].fillna(
                int(target_week)
            ).astype("Int64")
    for text_col in ("player_clean_key", "opponent", "event_id"):
        if text_col in opponent_map.columns:
            opponent_map[text_col] = opponent_map[text_col].astype("string")
    for team_col in ("team", "opponent", "home_team", "away_team", "team_abbr", "opponent_abbr"):
        if team_col in opponent_map.columns:
            opponent_map[team_col] = (
                opponent_map[team_col]
                .astype("string")
                .map(lambda val: canon_team(val) if val else val)
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    opponent_map.to_csv(out_path, index=False)
    print(f"[oddsapi] wrote {len(opponent_map)} rows → {out_path}")
    unresolved_df = _write_unresolved(unresolved_records)
    size_bytes = out_path.stat().st_size if out_path.exists() else 0
    missing_home = (
        opponent_map["home_team"].isna().sum()
        if "home_team" in opponent_map.columns
        else 0
    )
    missing_away = (
        opponent_map["away_team"].isna().sum()
        if "away_team" in opponent_map.columns
        else 0
    )
    print(
        "[opponent_map] wrote %s rows -> %s (%s bytes); missing_home=%s missing_away=%s"
        % (
            f"{len(opponent_map):,}",
            out_path,
            f"{size_bytes:,}",
            int(missing_home),
            int(missing_away),
        )
    )
    print(f"[opponent_map] unresolved rows: {len(unresolved_df):,} -> {UNRESOLVED_OUT}")
    if PLAYER_NAME_MAP_PATH.exists():
        print(
            f"[opponent_map] wrote player name map with {PLAYER_NAME_MAP_PATH.stat().st_size:,} bytes -> {PLAYER_NAME_MAP_PATH}"
        )
    if MISSING_SAMPLE_PATH.exists():
        print(
            f"[opponent_map] missing sample preview -> {MISSING_SAMPLE_PATH}"
        )
    return opponent_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--props-path", type=Path, default=Path("outputs/props_raw.csv"))
    parser.add_argument("--odds-path", type=Path, default=Path("outputs/odds_game.csv"))
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Slate date like YYYY-MM-DD (leave blank to infer from schedule)",
    )
    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help="Target week number when schedule inference is desired",
    )
    args = parser.parse_args()

    build_opponent_map(
        props_path=args.props_path,
        odds_path=args.odds_path,
        out_path=args.out_path,
        slate_date=args.date,
        week=args.week,
    )


if __name__ == "__main__":
    main()
