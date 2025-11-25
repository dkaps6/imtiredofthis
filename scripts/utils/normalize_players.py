"""Helpers to normalize player-level tables before persistence/joins."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence, Union

import pandas as pd

try:
    from scripts.utils.name_clean import canonical_key, canonical_player
except Exception:  # pragma: no cover - fallback for stripped environments

    def canonical_player(raw) -> str:
        if raw is None:
            return ""
        value = str(raw).strip()
        if not value:
            return ""
        value = value.replace(".", " ").replace("-", " ").replace("'", " ")
        value = " ".join(value.split())
        return value.title()

    def canonical_key(raw) -> str:
        base = canonical_player(raw)
        return base.lower().replace(" ", "") if base else ""


FrameLike = Union[str, Path, pd.DataFrame]


logger = logging.getLogger(__name__)


def _to_frame(source: FrameLike | None) -> pd.DataFrame:
    if source is None:
        return pd.DataFrame()
    if isinstance(source, pd.DataFrame):
        return source.copy()
    path = Path(source)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _as_string(series, *, index: pd.Index | None = None) -> pd.Series:
    """
    Normalize an arbitrary input to a Pandas string Series:
    - If `series` is not a Series (e.g., a scalar ''), coerce to a Series
      aligned to `index` (or a length-1 Series if index is None).
    - Always return dtype 'string' with trimmed text and NA as empty string.
    """
    if not isinstance(series, pd.Series):
        if index is None:
            series = pd.Series([series])
        else:
            series = pd.Series([series] * len(index), index=index)
    return series.astype("string").fillna("").str.strip()


def _as_int(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.astype("Int64")


def _ensure_player_clean_key(df: pd.DataFrame) -> pd.DataFrame:
    if "player_clean_key" in df.columns:
        df["player_clean_key"] = _as_string(df["player_clean_key"])
        if df["player_clean_key"].str.len().gt(0).any():
            return df

    candidate_cols: Sequence[str] = (
        "player_clean_key",
        "player_key",
        "player",
        "display_name",
    )
    for col in candidate_cols:
        if col in df.columns:
            series = _as_string(df[col])
            if series.str.len().gt(0).any():
                df["player_clean_key"] = series.map(canonical_key)
                break
    if "player_clean_key" not in df.columns:
        df["player_clean_key"] = ""
    df["player_clean_key"] = _as_string(df["player_clean_key"])
    return df


def _normalize_schedule_join(df: pd.DataFrame, team_week: FrameLike | None) -> pd.DataFrame:
    schedule = _to_frame(team_week)
    if schedule.empty:
        return df

    working = schedule.copy()
    working.columns = [str(c).lower() for c in working.columns]
    required = {"season", "week", "team"}
    if not required.issubset(working.columns):
        return df

    for col in ("season", "week"):
        working[col] = _as_int(working[col])
    for col in ("team", "opponent", "event_id"):
        if col in working.columns:
            working[col] = _as_string(working[col]).str.upper()

    rename_map: Mapping[str, str] = {}
    if "opponent" in working.columns:
        rename_map["opponent"] = "_schedule_opponent"
    if "event_id" in working.columns:
        rename_map["event_id"] = "_schedule_event_id"
    if "bye" in working.columns:
        rename_map["bye"] = "_schedule_bye"
    if "game_id" in working.columns:
        rename_map["game_id"] = "_schedule_game_id"
    if rename_map:
        working = working.rename(columns=rename_map)

    join_cols = [col for col in ("season", "week", "team") if col in df.columns]
    if len(join_cols) != 3:
        return df

    left = df.copy()
    for col in ("season", "week"):
        if col in left.columns:
            left[col] = _as_int(left[col])
    for col in ("team", "opponent"):
        if col in left.columns:
            left[col] = _as_string(left[col]).str.upper()

    subset_cols = join_cols + [c for c in working.columns if c.startswith("_schedule_")]
    subset = working[subset_cols].drop_duplicates(join_cols)
    pre_merge_rows = len(left)

    merged = left.merge(subset, on=join_cols, how="left", validate="m:1")

    if "_schedule_opponent" in merged.columns:
        merged["opponent"] = _as_string(
            merged["opponent"] if "opponent" in merged.columns else pd.Series(pd.NA, index=merged.index),
            index=merged.index,
        )
        merged["opponent"] = merged["opponent"].where(
            merged["opponent"].ne(""), merged["_schedule_opponent"]
        )
        merged.drop(columns=["_schedule_opponent"], inplace=True)
    if "_schedule_event_id" in merged.columns:
        merged["event_id"] = _as_string(
            merged["event_id"] if "event_id" in merged.columns else pd.Series(pd.NA, index=merged.index),
            index=merged.index,
        )
        merged["event_id"] = merged["event_id"].where(
            merged["event_id"].ne(""), merged["_schedule_event_id"]
        )
        merged.drop(columns=["_schedule_event_id"], inplace=True)
    if "_schedule_bye" in merged.columns:
        merged["bye"] = merged.get("bye")
        merged["bye"] = merged["bye"].combine_first(merged["_schedule_bye"])
        merged.drop(columns=["_schedule_bye"], inplace=True)
    if "_schedule_game_id" in merged.columns:
        merged["game_id"] = _as_string(
            merged.get("game_id", pd.Series(pd.NA, index=merged.index)),
            index=merged.index,
        )
        merged["game_id"] = merged["game_id"].where(
            merged["game_id"].ne(""), merged["_schedule_game_id"]
        )
        merged.drop(columns=["_schedule_game_id"], inplace=True)

    if "game_id" in merged.columns:
        non_null = merged.dropna(subset=["game_id"])
        if non_null.empty:
            raise RuntimeError(
                "[FATAL] normalize_game_logs removed all rows after team_week_map join. "
                "Team abbreviations do not match (case/format issue)."
            )
        if len(non_null) < pre_merge_rows * 0.5:
            logger.error(
                "[DIAG] normalize_game_logs lost more than 50%% of rows during join."
            )

    return merged


def _overlay_props(df: pd.DataFrame, props_map: FrameLike | None) -> pd.DataFrame:
    mapping = _to_frame(props_map)
    if mapping.empty:
        return df

    working = mapping.copy()
    for col in ("season", "week"):
        if col in working.columns:
            working[col] = _as_int(working[col])
    for col in ("player_clean_key", "event_id", "opponent"):
        if col in working.columns:
            working[col] = _as_string(working[col]).str.upper()

    working = _ensure_player_clean_key(working)

    if "player_clean_key" not in df.columns:
        return df

    left = df.copy()
    left["player_clean_key"] = _as_string(left["player_clean_key"]).str.upper()
    if "event_id" in left.columns:
        left["event_id"] = _as_string(left["event_id"]).str.upper()

    join_cols: list[str]
    if {"event_id", "player_clean_key"}.issubset(working.columns) and "event_id" in left.columns:
        join_cols = ["event_id", "player_clean_key"]
    else:
        join_cols = ["player_clean_key"]

    subset_cols = [col for col in ["opponent", "event_id"] if col in working.columns]
    if not subset_cols:
        return left

    suffix_map = {"opponent": "_props_opponent", "event_id": "_props_event_id"}
    subset = working[join_cols + subset_cols].drop_duplicates(join_cols)
    subset = subset.rename(columns={col: suffix_map[col] for col in subset_cols})

    merged = left.merge(subset, on=join_cols, how="left", validate="m:1")

    if "_props_opponent" in merged.columns:
        merged["opponent"] = _as_string(merged.get("opponent", ""))
        merged["opponent"] = merged["opponent"].where(
            merged["opponent"].ne(""), merged["_props_opponent"]
        )
        merged.drop(columns=["_props_opponent"], inplace=True)
    if "_props_event_id" in merged.columns:
        merged["event_id"] = _as_string(merged.get("event_id", ""))
        merged["event_id"] = merged["event_id"].where(
            merged["event_id"].ne(""), merged["_props_event_id"]
        )
        merged.drop(columns=["_props_event_id"], inplace=True)

    return merged


def normalize_game_logs(
    frame: FrameLike,
    *,
    team_week_map: FrameLike | None = None,
    props_map: FrameLike | None = None,
) -> pd.DataFrame:
    df = _to_frame(frame)
    if df.empty:
        return df

    working = df.copy()

    for col in ("season", "week"):
        if col in working.columns:
            working[col] = _as_int(working[col])
        else:
            working[col] = pd.Series(pd.NA, index=working.index, dtype="Int64")

    str_cols = [
        "team",
        "opponent",
        "event_id",
        "game_id",
        "player_key",
        "display_name",
        "player",
    ]
    for col in str_cols:
        if col in working.columns:
            working[col] = _as_string(working[col])

    working = _ensure_player_clean_key(working)
    working = _normalize_schedule_join(working, team_week_map)
    working = _overlay_props(working, props_map)

    return working


def normalize_season_totals(frame: FrameLike) -> pd.DataFrame:
    df = _to_frame(frame)
    if df.empty:
        return df

    working = df.copy()
    if "season" in working.columns:
        working["season"] = _as_int(working["season"])
    else:
        working["season"] = pd.Series(pd.NA, index=working.index, dtype="Int64")

    for col in ("team", "player_key", "display_name", "player"):
        if col in working.columns:
            working[col] = _as_string(working[col])

    working = _ensure_player_clean_key(working)
    return working

