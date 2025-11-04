#!/usr/bin/env python3
"""Derive live opponent mapping from sportsbook props and odds."""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from scripts.utils.name_clean import (
    build_roster_lookup,
    canonical_key,
    canonical_player,
    initials_last_to_full,
    normalize_team,
)


DEFAULT_OUT = Path("data/opponent_map_from_props.csv")
UNRESOLVED_OUT = Path("data/opponent_map_unresolved.csv")
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


def _norm_team(series: pd.Series) -> pd.Series:
    series = series.fillna("").astype("string")
    return series.map(normalize_team).astype("string")


def _parse_commence(series: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, utc=True, errors="coerce")
    except Exception:
        parsed = pd.Series(pd.NaT, index=series.index if isinstance(series, pd.Series) else None)
    return parsed


def _resolve_slate_date() -> datetime.date | None:
    raw = os.getenv("SLATE_DATE", "").strip()
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
) -> pd.DataFrame:
    props_candidates = [p for p in [props_path, Path("outputs/props_raw.csv"), Path("data/props_raw.csv")] if p]
    odds_candidates = [p for p in [odds_path, Path("outputs/odds_game.csv"), Path("data/odds_game.csv")] if p]

    props = _read_first(props_candidates)
    odds = _read_first(odds_candidates)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    required_cols = [
        "player",
        "player_name_clean",
        "player_clean_key",
        "team",
        "opponent",
        "season",
        "week",
        "event_id",
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

    roster_lookup = build_roster_lookup(_load_roles_dataframe())

    props = props.copy()
    props.columns = [c.lower() for c in props.columns]
    if "player" not in props.columns:
        alt = next((c for c in ("player_name", "name") if c in props.columns), None)
        if alt:
            props.rename(columns={alt: "player"}, inplace=True)
        else:
            props["player"] = pd.NA
    props["player"] = props["player"].astype("string").str.strip()
    props = props[props["player"].str.len() > 0]

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

    missing_key_mask = props["player_clean_key"].fillna("").str.len() == 0
    if missing_key_mask.any():
        _record_unresolved_rows(props.loc[missing_key_mask], "missing_player_key")
        props = props.loc[~missing_key_mask].copy()

    if props.empty:
        return _return_empty("[opponent_map] WARN: no props with resolvable players; wrote empty map")

    for col in ("team", "season", "week"):
        if col not in props.columns:
            props[col] = pd.NA
    if "team" in props.columns:
        props["team"] = _norm_team(props["team"])
    if "team_abbr" in props.columns:
        props["team_abbr"] = _norm_team(props["team_abbr"])
    for col in ("season", "week"):
        props[col] = pd.to_numeric(props[col], errors="coerce").astype("Int64")

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

    slate_date = _resolve_slate_date()
    live_mask = pd.Series(True, index=odds.index)
    if "status" in odds.columns:
        live_mask &= odds["status"].astype("string").str.lower().isin(["pre", "inprogress", "open"])
    if slate_date is not None and "commence_time" in odds.columns:
        live_mask &= odds["commence_time"].dt.date.eq(slate_date)
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

    merged = props.merge(odds, on="event_id", how="inner")
    if merged.empty:
        return _return_empty("[opponent_map] WARN: no props matched live odds; wrote empty map")

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

    keep_cols = [c for c in ["player", "player_name_clean", "player_clean_key", "team", "opponent", "season", "week", "event_id"] if c in merged.columns]
    out = merged.loc[:, keep_cols].copy()
    for col in required_cols:
        if col not in out.columns:
            out[col] = pd.NA

    out["team"] = _norm_team(out["team"])
    out["opponent"] = _norm_team(out["opponent"])
    out["player"] = out["player"].astype("string")
    out["player_name_clean"] = out["player_name_clean"].astype("string")
    out["player_clean_key"] = out["player_clean_key"].astype("string")
    for col in ("season", "week"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    out["event_id"] = out["event_id"].astype("string")

    opponent_str = out["opponent"].astype("string")
    missing_opponent_mask = opponent_str.fillna("").str.strip() == ""
    if missing_opponent_mask.any():
        missing_rows = out.loc[missing_opponent_mask].copy()
        _record_unresolved_rows(missing_rows, "missing_opponent")
        for row in missing_rows.itertuples(index=False):
            name = getattr(row, "player_name_clean", "") or getattr(row, "player", "")
            team_val = getattr(row, "team", pd.NA)
            event_val = getattr(row, "event_id", pd.NA)
            print(f"[opponent_map] missing mapping for {name} ({team_val}, event {event_val})")
        out = out.loc[~missing_opponent_mask].copy()

    out = out.drop_duplicates(subset=["player_clean_key", "team", "opponent", "event_id"], keep="last")
    out.to_csv(out_path, index=False)
    unresolved_df = _write_unresolved(unresolved_records)
    size_bytes = out_path.stat().st_size if out_path.exists() else 0
    print(f"[opponent_map] wrote {len(out):,} rows -> {out_path} ({size_bytes:,} bytes)")
    print(f"[opponent_map] unresolved rows: {len(unresolved_df):,} -> {UNRESOLVED_OUT}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--props-path", type=Path, default=Path("outputs/props_raw.csv"))
    parser.add_argument("--odds-path", type=Path, default=Path("outputs/odds_game.csv"))
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    build_opponent_map(props_path=args.props_path, odds_path=args.odds_path, out_path=args.out_path)


if __name__ == "__main__":
    main()
