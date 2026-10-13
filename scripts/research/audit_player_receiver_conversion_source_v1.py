#!/usr/bin/env python3
"""Audit receiver-level FTN drop/catchability source viability for WR/TE research.

Source audit only. No predictive model, no sportsbook, no production change.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

SEASONS = (2023, 2024, 2025)
REG_WEEKS = set(range(1, 19))
MIN_JOIN = 0.95
MIN_RECEIVER_ID = 0.95
MIN_ROSTER_MAP = 0.95
MIN_FIELD_COVERAGE = 0.80
MIN_PLAYERS_4 = 100
MIN_PLAYERS_8 = 75
WRTE = {"WR", "TE"}


def lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def to_pandas(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return pd.DataFrame(obj)


def first(frame: pd.DataFrame, names: tuple[str, ...], default="") -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series(default, index=frame.index)


def position(value) -> str:
    text = "" if value is None or pd.isna(value) else str(value).upper().strip()
    if text in {"LWR", "RWR", "SWR"} or text.startswith("WR"):
        return "WR"
    if text.startswith("TE"):
        return "TE"
    if text in {"HB", "TB"} or text.startswith("RB"):
        return "RB"
    if text.startswith("FB"):
        return "FB"
    return text


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_url_parquet(url: str) -> tuple[pd.DataFrame, dict]:
    req = Request(url, headers={"User-Agent": "receiver-conversion-source-audit-v1"})
    with urlopen(req, timeout=180) as response:
        raw = response.read()
        final = response.geturl()
    return pd.read_parquet(io.BytesIO(raw)), {
        "url": final,
        "bytes": int(len(raw)),
        "sha256": sha256_bytes(raw),
    }


def bool_num(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out.loc[text.isin(["1", "true", "t", "yes", "y"])] = 1.0
    out.loc[text.isin(["0", "false", "f", "no", "n"])] = 0.0
    return out


def present(series: pd.Series) -> pd.Series:
    text = series.astype("string").fillna("").str.strip()
    return text.ne("") & text.str.lower().ne("nan")


def load_roster(season: int) -> pd.DataFrame:
    import nflreadpy as nfl

    raw = lower(to_pandas(nfl.load_rosters_weekly(int(season))))
    if raw.empty:
        raise RuntimeError(f"weekly roster source empty season={season}")
    raw["season"] = pd.to_numeric(first(raw, ("season",), season), errors="coerce").fillna(season).astype(int)
    raw["week"] = pd.to_numeric(first(raw, ("week",)), errors="coerce")
    raw["player_id"] = first(raw, ("gsis_id", "player_id")).astype("string").fillna("").str.strip()
    raw["position"] = first(raw, ("position", "pos")).map(position)
    raw = raw.loc[
        raw["season"].eq(season)
        & raw["week"].between(1, 18, inclusive="both")
        & raw["player_id"].ne("")
    ].copy()
    return raw[["season", "week", "player_id", "position"]].drop_duplicates(
        ["season", "week", "player_id"], keep="last"
    )


def audit_season(season: int) -> tuple[dict, pd.DataFrame, list[dict]]:
    ftn_url = (
        "https://github.com/nflverse/nflverse-data/releases/download/"
        f"ftn_charting/ftn_charting_{season}.parquet"
    )
    pbp_url = (
        "https://github.com/nflverse/nflverse-data/releases/download/"
        f"pbp/play_by_play_{season}.parquet"
    )
    ftn, fmeta = read_url_parquet(ftn_url)
    pbp, pmeta = read_url_parquet(pbp_url)
    ftn, pbp = lower(ftn), lower(pbp)

    required_ftn = {
        "nflverse_game_id", "nflverse_play_id", "season", "week",
        "is_drop", "is_catchable_ball",
    }
    required_pbp = {
        "game_id", "play_id", "season", "week", "season_type", "posteam",
        "defteam", "pass_attempt", "complete_pass", "receiver_player_id",
        "receiver_player_name",
    }
    miss_f = sorted(required_ftn - set(ftn.columns))
    miss_p = sorted(required_pbp - set(pbp.columns))
    if miss_f or miss_p:
        raise RuntimeError(
            f"source schema missing season={season} ftn={miss_f} pbp={miss_p}"
        )

    ftn["week"] = pd.to_numeric(ftn["week"], errors="coerce")
    pbp["week"] = pd.to_numeric(pbp["week"], errors="coerce")
    ftn = ftn.loc[ftn["week"].between(1, 18, inclusive="both")].copy()
    pbp = pbp.loc[
        pbp["season_type"].astype(str).str.upper().eq("REG")
        & pbp["week"].between(1, 18, inclusive="both")
    ].copy()

    fw = set(ftn["week"].dropna().astype(int))
    pw = set(pbp["week"].dropna().astype(int))

    ftn["join_game"] = ftn["nflverse_game_id"].astype(str)
    ftn["join_play"] = pd.to_numeric(ftn["nflverse_play_id"], errors="coerce")
    keep = [
        "game_id", "play_id", "season", "week", "posteam", "defteam",
        "pass_attempt", "complete_pass", "receiver_player_id",
        "receiver_player_name",
    ]
    p = pbp[keep].copy()
    p["join_game"] = p["game_id"].astype(str)
    p["join_play"] = pd.to_numeric(p["play_id"], errors="coerce")
    merged = ftn.merge(
        p.drop(columns=["game_id", "play_id", "season", "week"]),
        on=["join_game", "join_play"],
        how="left",
        validate="one_to_one",
    )
    join_rate = float(merged["posteam"].notna().mean())

    merged["season"] = int(season)
    merged["week"] = pd.to_numeric(merged["week"], errors="coerce").astype(int)
    merged["game_id"] = merged["join_game"].astype(str)
    merged["pass_attempt"] = pd.to_numeric(merged["pass_attempt"], errors="coerce")
    merged["complete_pass"] = pd.to_numeric(merged["complete_pass"], errors="coerce")
    merged["is_drop_num"] = bool_num(merged["is_drop"])
    merged["is_catchable_num"] = bool_num(merged["is_catchable_ball"])

    attempts = merged.loc[merged["pass_attempt"].eq(1)].copy()
    named = present(attempts["receiver_player_name"])
    stable_id = present(attempts["receiver_player_id"])
    receiver_named_rows = int(named.sum())
    receiver_id_rate = float(stable_id[named].mean()) if receiver_named_rows else 0.0

    recv = attempts.loc[stable_id].copy()
    recv["player_id"] = recv["receiver_player_id"].astype("string").fillna("").str.strip()
    roster = load_roster(season)
    recv = recv.merge(
        roster,
        on=["season", "week", "player_id"],
        how="left",
        validate="many_to_one",
    )
    roster_map_rate = float(recv["position"].notna().mean()) if len(recv) else 0.0
    drop_coverage = float(recv["is_drop_num"].notna().mean()) if len(recv) else 0.0
    catchable_coverage = float(recv["is_catchable_num"].notna().mean()) if len(recv) else 0.0

    wrte = recv.loc[recv["position"].isin(WRTE)].copy()
    game_support = (
        wrte.groupby(["player_id", "game_id"], as_index=False)
        .agg(targeted_throws=("join_play", "size"))
        .groupby("player_id", as_index=False)
        .agg(games=("game_id", "nunique"), targeted_throws=("targeted_throws", "sum"))
    )
    players_4 = int(game_support["games"].ge(4).sum())
    players_8 = int(game_support["games"].ge(8).sum())

    gates = {
        "ftn_weeks_1_18_complete": REG_WEEKS.issubset(fw),
        "pbp_weeks_1_18_complete": REG_WEEKS.issubset(pw),
        "ftn_pbp_join_rate_ge_095": join_rate >= MIN_JOIN,
        "receiver_id_rate_ge_095": receiver_id_rate >= MIN_RECEIVER_ID,
        "roster_position_map_rate_ge_095": roster_map_rate >= MIN_ROSTER_MAP,
        "drop_field_coverage_ge_080": drop_coverage >= MIN_FIELD_COVERAGE,
        "catchable_field_coverage_ge_080": catchable_coverage >= MIN_FIELD_COVERAGE,
        "wrte_players_4plus_games_ge_100": players_4 >= MIN_PLAYERS_4,
        "wrte_players_8plus_games_ge_75": players_8 >= MIN_PLAYERS_8,
    }

    summary = {
        "season": int(season),
        "ftn_rows": int(len(ftn)),
        "pbp_rows": int(len(pbp)),
        "ftn_pbp_join_rate": join_rate,
        "receiver_named_pass_attempt_rows": receiver_named_rows,
        "stable_receiver_id_rows": int(len(recv)),
        "receiver_id_rate": receiver_id_rate,
        "roster_position_map_rate": roster_map_rate,
        "drop_field_coverage": drop_coverage,
        "catchable_field_coverage": catchable_coverage,
        "wrte_receiver_rows": int(len(wrte)),
        "wrte_distinct_receivers": int(wrte["player_id"].nunique()),
        "wrte_players_4plus_games": players_4,
        "wrte_players_8plus_games": players_8,
        "gates": gates,
        "season_pass": bool(all(gates.values())),
    }
    sources = [
        {"season": season, "source": "FTN", **fmeta},
        {"season": season, "source": "PBP", **pmeta},
    ]
    support = game_support.assign(season=int(season))
    return summary, support, sources


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries, supports, sources = [], [], []
    for season in SEASONS:
        summary, support, source = audit_season(season)
        summaries.append(summary)
        supports.append(support)
        sources.extend(source)

    all_pass = all(row["season_pass"] for row in summaries)
    result = {
        "version": "PLAYER_RECEIVER_CONVERSION_STATE_SOURCE_AUDIT_V1",
        "disposition": (
            "PLAYER_RECEIVER_CONVERSION_SOURCE_READY"
            if all_pass
            else "PLAYER_RECEIVER_CONVERSION_SOURCE_BLOCKED"
        ),
        "seasons": list(SEASONS),
        "thresholds": {
            "min_ftn_pbp_join_rate": MIN_JOIN,
            "min_receiver_id_rate": MIN_RECEIVER_ID,
            "min_roster_position_map_rate": MIN_ROSTER_MAP,
            "min_charted_field_coverage": MIN_FIELD_COVERAGE,
            "min_wrte_players_4plus_games": MIN_PLAYERS_4,
            "min_wrte_players_8plus_games": MIN_PLAYERS_8,
        },
        "season_results": summaries,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "predictive_models_fit": 0,
        "candidate_mean_changes": 0,
        "production_changed": False,
    }

    (args.out_dir / "player_receiver_conversion_source_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.concat(supports, ignore_index=True).to_csv(
        args.out_dir / "player_receiver_conversion_support.csv", index=False
    )
    pd.DataFrame(sources).to_csv(
        args.out_dir / "player_receiver_conversion_source_snapshot.csv", index=False
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
