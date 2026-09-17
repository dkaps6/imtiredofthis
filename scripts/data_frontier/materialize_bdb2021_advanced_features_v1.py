"""Materialize BDB 2021 route/proximity features under Feature Dictionary V1."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data_frontier.advanced_feature_materializer_common_v1 import (
    assert_strict_prior,
    corpus_sha256,
    load_dictionary,
    print_summary,
    sanitized_report,
    source_contract,
    write_private_table,
)

SOURCE_KEY = "BDB2021_ROUTE_GEOMETRY"


def _event_frame(events: pd.DataFrame, label: str) -> pd.Series:
    return (
        events.loc[events["event"].eq(label)]
        .groupby(["gameId", "playId"])["frameId"]
        .min()
        .rename(label)
    )


def _event_geometry(
    off: pd.DataFrame,
    player: pd.DataFrame,
    defenders: pd.DataFrame,
    events: pd.DataFrame,
    event_col: str,
    prefix: str,
) -> pd.DataFrame:
    ef = _event_frame(events, event_col).reset_index().rename(columns={event_col: "eventFrame"})
    rr = off.merge(ef, on=["gameId", "playId"], how="inner")
    rp = rr.merge(
        player,
        left_on=["gameId", "playId", "nflId", "team", "eventFrame"],
        right_on=["gameId", "playId", "nflId", "team", "frameId"],
        how="inner",
    )
    dp = defenders.merge(ef, on=["gameId", "playId"], how="inner")
    dp = dp.loc[dp["frameId"].eq(dp["eventFrame"]), ["gameId", "playId", "eventFrame", "nflId", "x", "y"]]
    dp = dp.rename(columns={"nflId": "defenderNflId", "x": "dx", "y": "dy"})

    cand = rp[["gameId", "playId", "nflId", "eventFrame", "x", "y"]].merge(
        dp, on=["gameId", "playId", "eventFrame"], how="inner"
    )
    if cand.empty:
        return pd.DataFrame(columns=["gameId", "playId", "nflId"])
    cand["distance"] = np.hypot(cand["x"] - cand["dx"], cand["y"] - cand["dy"])
    cand = cand.sort_values(["gameId", "playId", "nflId", "distance", "defenderNflId"], kind="mergesort")
    cand["rank"] = cand.groupby(["gameId", "playId", "nflId"]).cumcount() + 1

    near = (
        cand.loc[cand["rank"].le(2)]
        .pivot_table(
            index=["gameId", "playId", "nflId", "eventFrame"],
            columns="rank",
            values="distance",
            aggfunc="first",
        )
        .reset_index()
        .rename(
            columns={
                1: f"{prefix}_nearest_defender_distance_yards",
                2: f"{prefix}_second_nearest_defender_distance_yards",
                "eventFrame": f"{prefix}_frame_id",
            }
        )
    )
    return near


def _history_snapshots(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    work = raw.copy()
    work["throw_spacing_gap"] = (
        work["throw_second_nearest_defender_distance_yards"]
        - work["throw_nearest_defender_distance_yards"]
    )
    for target_week in range(1, 19):
        hist = work.loc[work["week"].lt(target_week)].copy()
        if hist.empty:
            continue
        for (nfl_id, route), g in hist.groupby(["nfl_id", "route_label"], dropna=False, sort=False):
            nearest = pd.to_numeric(g["throw_nearest_defender_distance_yards"], errors="coerce").dropna()
            second = pd.to_numeric(g["throw_second_nearest_defender_distance_yards"], errors="coerce").dropna()
            gap = pd.to_numeric(g["throw_spacing_gap"], errors="coerce").dropna()
            n = int(len(nearest))
            row = {
                "source_season": 2018,
                "target_week": int(target_week),
                "nfl_id": nfl_id,
                "route_label": route,
                "history_max_source_week": int(g["week"].max()),
                "hist_player_route_throw_geometry_sample_count": n,
                "second_defender_sample_count": int(len(second)),
                "spacing_gap_sample_count": int(len(gap)),
                "hist_player_route_throw_nearest_defender_median_yards": float(nearest.median()) if n >= 5 else np.nan,
                "hist_player_route_throw_second_defender_median_yards": float(second.median()) if len(second) >= 5 else np.nan,
                "hist_player_route_throw_spacing_gap_median_yards": float(gap.median()) if len(gap) >= 5 else np.nan,
                "strict_prior_only": True,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def materialize(corpus_dir: Path, out_root: Path) -> dict:
    dictionary = load_dictionary()
    source, _ = source_contract(SOURCE_KEY, dictionary)
    expected_files = ['games.csv', 'players.csv', 'plays.csv'] + [f'week{i}.csv' for i in range(1, 18)]
    observed_hash, manifest = corpus_sha256(corpus_dir, expected_files, relative_names=False)
    if observed_hash != source["source_hash_sha256"]:
        raise SystemExit(f"BDB2021 source hash mismatch: {observed_hash} != {source['source_hash_sha256']}")

    games = pd.read_csv(next(corpus_dir.rglob("games.csv")))[
        ["gameId", "homeTeamAbbr", "visitorTeamAbbr", "week"]
    ]
    plays = pd.read_csv(next(corpus_dir.rglob("plays.csv")))[
        ["gameId", "playId", "possessionTeam"]
    ]
    meta = plays.merge(games, on="gameId", how="left")
    meta["offenseSide"] = np.where(
        meta["possessionTeam"].eq(meta["homeTeamAbbr"]),
        "home",
        np.where(meta["possessionTeam"].eq(meta["visitorTeamAbbr"]), "away", "unresolved"),
    )
    if int(meta["offenseSide"].eq("unresolved").sum()):
        raise SystemExit("BDB2021 possession side unresolved")

    outputs = []
    structural = {
        "route_playerplays": 0,
        "multi_route_label_playerplays": 0,
        "wrong_side_route_rows": 0,
        "weeks": {},
        "source_file_count": len(manifest),
    }

    for week in range(1, 18):
        tr = pd.read_csv(next(corpus_dir.rglob(f"week{week}.csv")))
        mw = meta.loc[meta["week"].eq(week), ["gameId", "playId", "offenseSide"]]
        route_pp = (
            tr.loc[
                tr["route"].notna() & tr["nflId"].notna(),
                ["gameId", "playId", "nflId", "position", "team", "route"],
            ]
            .drop_duplicates()
        )
        per = route_pp.groupby(["gameId", "playId", "nflId"])["route"].nunique()
        multi = int((per > 1).sum())
        structural["multi_route_label_playerplays"] += multi
        if multi:
            raise SystemExit(f"BDB2021 route label instability in week {week}: {multi}")

        route_pp = route_pp.merge(mw, on=["gameId", "playId"], how="left")
        wrong = int((~route_pp["team"].eq(route_pp["offenseSide"])).sum())
        structural["wrong_side_route_rows"] += wrong
        off = route_pp.loc[route_pp["team"].eq(route_pp["offenseSide"])].copy()
        structural["route_playerplays"] += int(len(off))

        events = tr[["gameId", "playId", "frameId", "event"]].dropna().drop_duplicates()
        player = tr.loc[
            tr["nflId"].notna(), ["gameId", "playId", "nflId", "frameId", "team", "x", "y"]
        ]
        dbase = player.merge(mw, on=["gameId", "playId"], how="inner")
        defenders = dbase.loc[
            dbase["team"].isin(["home", "away"]) & ~dbase["team"].eq(dbase["offenseSide"])
        ]

        base = off[["gameId", "playId", "nflId", "route", "position"]].drop_duplicates()
        snap = _event_geometry(off, player, defenders, events, "ball_snap", "snap")
        throw = _event_geometry(off, player, defenders, events, "pass_forward", "throw")
        arrival = _event_geometry(off, player, defenders, events, "pass_arrived", "arrival")

        wide = base.merge(snap, on=["gameId", "playId", "nflId"], how="left")
        wide = wide.merge(throw, on=["gameId", "playId", "nflId"], how="left")
        wide = wide.merge(arrival, on=["gameId", "playId", "nflId"], how="left")
        wide.insert(0, "source_season", 2018)
        wide.insert(1, "week", week)
        wide = wide.rename(
            columns={
                "gameId": "game_id",
                "playId": "play_id",
                "nflId": "nfl_id",
                "route": "route_label",
            }
        )
        wide["snap_to_throw_nearest_defender_delta_yards"] = (
            wide["throw_nearest_defender_distance_yards"]
            - wide["snap_nearest_defender_distance_yards"]
        )
        outputs.append(wide)

        structural["weeks"][str(week)] = {
            "route_playerplays": int(len(wide)),
            "snap_valid": int(wide["snap_nearest_defender_distance_yards"].notna().sum()),
            "throw_valid": int(wide["throw_nearest_defender_distance_yards"].notna().sum()),
            "arrival_valid": int(wide["arrival_nearest_defender_distance_yards"].notna().sum()),
        }

    raw = pd.concat(outputs, ignore_index=True)
    if structural["wrong_side_route_rows"] > 20:
        raise SystemExit("BDB2021 excessive route labels on non-offense side")

    history = _history_snapshots(raw)
    temporal = {
        "route_history": assert_strict_prior(history),
        "same_game_partial_history_forbidden": True,
        "target_game_rows_used": 0,
    }

    private = out_root / "private"
    manifests = {
        "route_playerplays": write_private_table(
            raw,
            private / "bdb2021_route_playerplays_v1.csv",
            ["week", "game_id", "play_id", "nfl_id"],
        ),
        "route_history_snapshots": write_private_table(
            history,
            private / "bdb2021_route_history_snapshots_v1.csv",
            ["target_week", "nfl_id", "route_label"],
        ),
    }

    report = sanitized_report(
        source_key=SOURCE_KEY,
        observed_source_hash=observed_hash,
        tables={"route_playerplays": raw, "route_history_snapshots": history},
        private_manifests=manifests,
        temporal_audit=temporal,
        structural_audit=structural,
        out_dir=out_root / "sanitized",
    )
    print_summary(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    materialize(args.corpus_dir, args.out_dir)


if __name__ == "__main__":
    main()
