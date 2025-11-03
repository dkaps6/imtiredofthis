#!/usr/bin/env python3
"""Derive live opponent mapping from sportsbook props and odds."""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from scripts.utils.name_clean import canonical_player, normalize_team


DEFAULT_OUT = Path("data/opponent_map_from_props.csv")


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
        "player_clean_key",
        "team",
        "opponent",
        "season",
        "week",
        "event_id",
    ]

    if props.empty or odds.empty:
        empty = pd.DataFrame(columns=required_cols)
        empty.to_csv(out_path, index=False)
        print("[opponent_map] WARN: missing props or odds source; wrote empty map")
        return empty

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
        empty = pd.DataFrame(columns=required_cols)
        empty.to_csv(out_path, index=False)
        print("[opponent_map] WARN: props missing event_id; wrote empty map")
        return empty

    props["player_clean_key"] = props["player"].map(canonical_player).astype("string")

    for col in ("team", "season", "week"):
        if col not in props.columns:
            props[col] = pd.NA
    if "team" in props.columns:
        props["team"] = _norm_team(props["team"])
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
        empty = pd.DataFrame(columns=required_cols)
        empty.to_csv(out_path, index=False)
        print("[opponent_map] WARN: odds missing event_id; wrote empty map")
        return empty

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
        empty = pd.DataFrame(columns=required_cols)
        empty.to_csv(out_path, index=False)
        print("[opponent_map] WARN: no props matched live odds; wrote empty map")
        return empty

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

    out = merged[[c for c in required_cols if c in merged.columns]].copy()
    for col in required_cols:
        if col not in out.columns:
            out[col] = pd.NA

    out["team"] = _norm_team(out["team"])
    out["opponent"] = _norm_team(out["opponent"])
    out["player"] = out["player"].astype("string")
    out["player_clean_key"] = out["player_clean_key"].astype("string")
    for col in ("season", "week"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    out["event_id"] = out["event_id"].astype("string")

    out = out.drop_duplicates(subset=["player_clean_key", "team", "opponent", "event_id"], keep="last")
    out.to_csv(out_path, index=False)
    print(f"[opponent_map] wrote {len(out):,} rows -> {out_path}")
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
