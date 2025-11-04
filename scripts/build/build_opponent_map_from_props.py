#!/usr/bin/env python3
"""Derive live opponent mapping from sportsbook props and odds."""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from zoneinfo import ZoneInfo

from scripts.utils.name_clean import (
    build_roster_lookup,
    canonical_key,
    canonical_player,
    initials_last_to_full,
    normalize_team,
)
from scripts.utils.df_keys import coerce_merge_keys


DEFAULT_OUT = Path("data/opponent_map_from_props.csv")
UNRESOLVED_OUT = Path("data/opponent_map_unresolved.csv")
TEAM_WEEK_MAP_PATH = Path("data/team_week_map.csv")
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


def _attach_game_timestamp(op_map: pd.DataFrame) -> pd.DataFrame:
    """Add game_timestamp (UTC) by first joining odds_game, else use game_lines (or leave NaT)."""
    if op_map is None or op_map.empty:
        return op_map

    df = op_map.copy()

    if "home" not in df.columns and "home_team" in df.columns:
        df["home"] = df["home_team"]
    if "away" not in df.columns and "away_team" in df.columns:
        df["away"] = df["away_team"]

    # 1) Try odds_game first (preferred)
    odds_game = _load_csv_safe("outputs/odds_game.csv")
    rename_og = {"commence_time": "kickoff_utc", "kickoff": "kickoff_utc", "game_timestamp": "kickoff_utc"}
    for k, v in rename_og.items():
        if k in odds_game.columns and "kickoff_utc" not in odds_game.columns:
            odds_game = odds_game.rename(columns={k: v})
    og_keep = [c for c in ["event_id", "home", "away", "kickoff_utc"] if c in odds_game.columns]
    odds_game = odds_game[og_keep].drop_duplicates() if og_keep else pd.DataFrame()

    if "event_id" in df.columns and not odds_game.empty:
        df = df.merge(odds_game, on="event_id", how="left", suffixes=("", "_og"))
    else:
        if "kickoff_utc" not in df.columns:
            df["kickoff_utc"] = pd.NaT

    # 2) Fallback to slate (game_lines) by (home,away) match if kickoff still missing
    needs_kl = df["kickoff_utc"].isna() if "kickoff_utc" in df.columns else pd.Series([], dtype=bool)
    if needs_kl.any():
        slate = _load_csv_safe("data/game_lines.csv")
        if not slate.empty:
            rename_gl = {"kickoff": "kickoff_local"}
            for k, v in rename_gl.items():
                if k in slate.columns and "kickoff_local" not in slate.columns:
                    slate = slate.rename(columns={k: v})
            gl_keep = [c for c in ["home", "away", "local_tz", "kickoff_local", "kickoff_utc"] if c in slate.columns]
            slate = slate[gl_keep].drop_duplicates() if gl_keep else pd.DataFrame()
            if not slate.empty:
                df = df.merge(slate, on=["home", "away"], how="left", suffixes=("", "_slate"))
                if "kickoff_utc_slate" in df.columns:
                    df["kickoff_utc"] = df["kickoff_utc"].fillna(df["kickoff_utc_slate"])
                # convert local times to UTC where needed
                local_cols = [
                    col
                    for col in ["kickoff_local", "kickoff_local_slate"]
                    if col in df.columns
                ]
                tz_cols = [col for col in ["local_tz", "local_tz_slate"] if col in df.columns]
                for idx in df.index[df["kickoff_utc"].isna()]:
                    local_val = None
                    for col in local_cols:
                        val = df.at[idx, col]
                        if pd.notna(val):
                            local_val = val
                            break
                    if local_val is None:
                        continue
                    tz_name = None
                    for col in tz_cols:
                        val = df.at[idx, col]
                        if isinstance(val, str) and val:
                            tz_name = val
                            break
                    try:
                        local_dt = pd.to_datetime(local_val)
                    except Exception:
                        local_dt = pd.NaT
                    if pd.isna(local_dt):
                        continue
                    if getattr(local_dt, "tzinfo", None) is None:
                        if tz_name:
                            try:
                                local_dt = local_dt.tz_localize(ZoneInfo(tz_name))
                            except (TypeError, ValueError):
                                local_dt = local_dt.tz_localize("UTC")
                        else:
                            local_dt = local_dt.tz_localize("UTC")
                    try:
                        df.at[idx, "kickoff_utc"] = local_dt.astimezone(ZoneInfo("UTC"))
                    except Exception:
                        continue

    # 3) Fallback to team_week_map if still missing
    needs_tw = df["kickoff_utc"].isna() if "kickoff_utc" in df.columns else pd.Series([], dtype=bool)
    if needs_tw.any():
        team_week = _load_csv_safe("data/team_week_map.csv")
        if not team_week.empty:
            working = team_week.copy()
            working.columns = [c.lower() for c in working.columns]
            if {"team", "opponent", "is_home"}.issubset(working.columns):
                home_mask = working["is_home"].astype(str).str.lower().isin(
                    ["true", "1", "yes", "t"]
                )
                home_df = working.loc[home_mask].copy()
                if not home_df.empty:
                    home_df["home"] = _norm_team(home_df["team"])
                    home_df["away"] = _norm_team(home_df["opponent"])
                    if "local_tz" not in home_df.columns:
                        home_df["local_tz"] = pd.NA
                    tw_keep = [
                        c
                        for c in ["home", "away", "local_tz", "kickoff_local", "kickoff_utc"]
                        if c in home_df.columns
                    ]
                    slate_tw = home_df[tw_keep].drop_duplicates(subset=["home", "away"])
                    df = df.merge(slate_tw, on=["home", "away"], how="left", suffixes=("", "_tw"))
                    fill_cols = ["kickoff_utc_tw"]
                    for col in fill_cols:
                        if col in df.columns:
                            df["kickoff_utc"] = df["kickoff_utc"].fillna(df[col])
                    local_cols = [
                        col
                        for col in ["kickoff_local", "kickoff_local_tw"]
                        if col in df.columns
                    ]
                    tz_cols = [
                        col for col in ["local_tz", "local_tz_tw"] if col in df.columns
                    ]
                    for idx in df.index[df["kickoff_utc"].isna()]:
                        local_val = None
                        for col in local_cols:
                            val = df.at[idx, col]
                            if pd.notna(val):
                                local_val = val
                                break
                        if local_val is None:
                            continue
                        tz_name = None
                        for col in tz_cols:
                            val = df.at[idx, col]
                            if isinstance(val, str) and val:
                                tz_name = val
                                break
                        try:
                            local_dt = pd.to_datetime(local_val)
                        except Exception:
                            local_dt = pd.NaT
                        if pd.isna(local_dt):
                            continue
                        if getattr(local_dt, "tzinfo", None) is None:
                            if tz_name:
                                try:
                                    local_dt = local_dt.tz_localize(ZoneInfo(tz_name))
                                except (TypeError, ValueError):
                                    local_dt = local_dt.tz_localize("UTC")
                            else:
                                local_dt = local_dt.tz_localize("UTC")
                        try:
                            df.at[idx, "kickoff_utc"] = local_dt.astimezone(ZoneInfo("UTC"))
                        except Exception:
                            continue

    if "kickoff_utc" not in df.columns:
        df["kickoff_utc"] = pd.NaT
    df["game_timestamp"] = pd.to_datetime(df["kickoff_utc"], utc=True, errors="coerce")

    drop_cols = [c for c in df.columns if c.endswith("_og") or c.endswith("_slate") or c.endswith("_tw")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


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

    if "home_team" in merged.columns and "home" not in merged.columns:
        merged["home"] = merged["home_team"]
    if "away_team" in merged.columns and "away" not in merged.columns:
        merged["away"] = merged["away_team"]

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
    for col in ("season", "week"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    out["event_id"] = out["event_id"].astype("string")

    schedule_df = _load_team_week_map()
    if not schedule_df.empty:
        rename = {}
        if "opponent" in schedule_df.columns:
            rename["opponent"] = "schedule_opponent"
        if "game_id" in schedule_df.columns:
            rename["game_id"] = "schedule_game_id"
        if "event_id" in schedule_df.columns and "schedule_game_id" not in rename.values():
            rename["event_id"] = "schedule_game_id"
        if "kickoff_local" in schedule_df.columns:
            rename["kickoff_local"] = "schedule_kickoff_local"
        if "kickoff_utc" in schedule_df.columns:
            rename["kickoff_utc"] = "schedule_kickoff_utc"

        schedule_subset = schedule_df.rename(columns=rename)
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

            out = merged

    drop_sched_cols = [c for c in out.columns if c.startswith("schedule_")]
    if drop_sched_cols:
        out.drop(columns=drop_sched_cols, inplace=True)

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

    opponent_map = out.drop_duplicates(
        subset=["player_clean_key", "team", "opponent", "event_id"], keep="last"
    )
    opponent_map = coerce_merge_keys(
        opponent_map, ["player_clean_key", "team", "event_id"], as_str=True
    )
    opponent_map = _attach_game_timestamp(opponent_map)
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    opponent_map.to_csv(out_path, index=False)
    try:
        op_rows = len(pd.read_csv(out_path))
    except Exception:
        op_rows = len(opponent_map)
    print(f"[oddsapi] wrote data/opponent_map_from_props.csv rows={op_rows}")
    unresolved_df = _write_unresolved(unresolved_records)
    size_bytes = out_path.stat().st_size if out_path.exists() else 0
    print(
        f"[opponent_map] wrote {len(opponent_map):,} rows -> {out_path} ({size_bytes:,} bytes)"
    )
    print(f"[opponent_map] unresolved rows: {len(unresolved_df):,} -> {UNRESOLVED_OUT}")
    return opponent_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--props-path", type=Path, default=Path("outputs/props_raw.csv"))
    parser.add_argument("--odds-path", type=Path, default=Path("outputs/odds_game.csv"))
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    build_opponent_map(props_path=args.props_path, odds_path=args.odds_path, out_path=args.out_path)


if __name__ == "__main__":
    main()
