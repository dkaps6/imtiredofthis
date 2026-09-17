"""Materialize BDB 2023 protection-interaction features under Feature Dictionary V1."""
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

SOURCE_KEY = "BDB2023_PROTECTION_GEOMETRY"


def _first_event(ev: pd.DataFrame, name: str) -> pd.Series:
    return (
        ev.loc[ev["event"].eq(name)]
        .groupby(["gameId", "playId"])["frameId"]
        .min()
        .rename(name)
    )


def _resolve_frames(tr: pd.DataFrame) -> pd.DataFrame:
    ev = tr[["gameId", "playId", "frameId", "event"]].dropna().drop_duplicates()
    ft = (
        tr.groupby(["gameId", "playId"])["frameId"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "firstFrame", "max": "lastFrame"})
    )
    for name in [
        "ball_snap",
        "autoevent_ballsnap",
        "pass_forward",
        "autoevent_passforward",
        "qb_sack",
        "qb_strip_sack",
    ]:
        ft = ft.set_index(["gameId", "playId"]).join(_first_event(ev, name), how="left").reset_index()

    manual = ft["ball_snap"]
    auto = ft["autoevent_ballsnap"]
    both = manual.notna() & auto.notna()
    close = both & ((manual - auto).abs() <= 3)
    divergent = both & ((manual - auto).abs() > 3)
    ft["snapFrame"] = np.nan
    ft["snapSource"] = "missing"
    ft.loc[close, "snapFrame"] = manual[close]
    ft.loc[close, "snapSource"] = "manual_close_agreement"
    ft.loc[divergent, "snapFrame"] = np.minimum(manual[divergent], auto[divergent])
    ft.loc[divergent, "snapSource"] = "earlier_of_divergent_labels"
    only_manual = manual.notna() & auto.isna()
    only_auto = auto.notna() & manual.isna()
    ft.loc[only_manual, "snapFrame"] = manual[only_manual]
    ft.loc[only_manual, "snapSource"] = "manual_only"
    ft.loc[only_auto, "snapFrame"] = auto[only_auto]
    ft.loc[only_auto, "snapSource"] = "auto_only"

    pm = ft["pass_forward"]
    pa = ft["autoevent_passforward"]
    pboth = pm.notna() & pa.notna()
    pclose = pboth & ((pm - pa).abs() <= 3)
    pdiv = pboth & ((pm - pa).abs() > 3)
    ft["terminalFrame"] = np.nan
    ft["terminalSource"] = "missing"
    ft.loc[pclose, "terminalFrame"] = pm[pclose]
    ft.loc[pclose, "terminalSource"] = "pass_manual_close_agreement"
    ft.loc[pdiv, "terminalFrame"] = np.minimum(pm[pdiv], pa[pdiv])
    ft.loc[pdiv, "terminalSource"] = "pass_earlier_of_divergent_labels"
    only_pm = pm.notna() & pa.isna()
    only_pa = pa.notna() & pm.isna()
    ft.loc[only_pm, "terminalFrame"] = pm[only_pm]
    ft.loc[only_pm, "terminalSource"] = "pass_manual_only"
    ft.loc[only_pa, "terminalFrame"] = pa[only_pa]
    ft.loc[only_pa, "terminalSource"] = "pass_auto_only"
    sack = ft["terminalFrame"].isna() & ft["qb_sack"].notna()
    strip = ft["terminalFrame"].isna() & ft["qb_strip_sack"].notna()
    ft.loc[sack, "terminalFrame"] = ft.loc[sack, "qb_sack"]
    ft.loc[sack, "terminalSource"] = "qb_sack"
    ft.loc[strip, "terminalFrame"] = ft.loc[strip, "qb_strip_sack"]
    ft.loc[strip, "terminalSource"] = "qb_strip_sack"
    fallback = ft["terminalFrame"].isna()
    ft.loc[fallback, "terminalFrame"] = ft.loc[fallback, "lastFrame"]
    ft.loc[fallback, "terminalSource"] = "last_frame_fallback"
    return ft


def _history_snapshots(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for target_week in range(1, 10):
        hist = raw.loc[raw["week"].lt(target_week)].copy()
        if hist.empty:
            continue
        for blocker, g in hist.groupby("blocker_nfl_id", sort=False):
            snap = pd.to_numeric(g["blocker_target_snap_distance_yards"], errors="coerce").dropna()
            mind = pd.to_numeric(g["blocker_target_min_distance_yards"], errors="coerce").dropna()
            tmin = pd.to_numeric(g["blocker_target_time_to_min_distance_seconds"], errors="coerce").dropna()
            rows.append(
                {
                    "source_season": 2021,
                    "target_week": int(target_week),
                    "blocker_nfl_id": blocker,
                    "history_max_source_week": int(g["week"].max()),
                    "snap_sample_count": int(len(snap)),
                    "min_distance_sample_count": int(len(mind)),
                    "time_to_min_sample_count": int(len(tmin)),
                    "hist_blocker_snap_distance_median_yards": float(snap.median()) if len(snap) >= 10 else np.nan,
                    "hist_blocker_min_distance_median_yards": float(mind.median()) if len(mind) >= 10 else np.nan,
                    "hist_blocker_time_to_min_distance_median_seconds": float(tmin.median()) if len(tmin) >= 10 else np.nan,
                    "strict_prior_only": True,
                }
            )
    return pd.DataFrame(rows)


def materialize(corpus_dir: Path, out_root: Path) -> dict:
    dictionary = load_dictionary()
    source, _ = source_contract(SOURCE_KEY, dictionary)
    expected_files = ['games.csv', 'players.csv', 'plays.csv', 'pffScoutingData.csv'] + [f'week{i}.csv' for i in range(1, 9)]
    observed_hash, manifest = corpus_sha256(corpus_dir, expected_files, relative_names=False)
    if observed_hash != source["source_hash_sha256"]:
        raise SystemExit(f"BDB2023 source hash mismatch: {observed_hash} != {source['source_hash_sha256']}")

    games = pd.read_csv(next(corpus_dir.rglob("games.csv")))[["gameId", "week"]]
    pff = pd.read_csv(next(corpus_dir.rglob("pffScoutingData.csv")))
    pff["blockedNflId"] = pd.to_numeric(pff["pff_nflIdBlockedPlayer"], errors="coerce").astype("Int64")
    inter = pff.loc[pff["blockedNflId"].notna()].copy()
    lookup = pff[["gameId", "playId", "nflId", "pff_role"]].rename(
        columns={"nflId": "blockedNflId", "pff_role": "blockedPlayerRole"}
    )
    inter = inter.merge(lookup, on=["gameId", "playId", "blockedNflId"], how="left")
    unresolved_refs = int(inter["blockedPlayerRole"].isna().sum())
    inter = inter.loc[inter["blockedPlayerRole"].notna()].rename(columns={"nflId": "blockerNflId"})
    keys = ["gameId", "playId", "blockerNflId", "blockedNflId"]
    inter = (
        inter[keys + ["pff_role", "blockedPlayerRole", "pff_blockType"]]
        .drop_duplicates()
        .merge(games, on="gameId", how="left")
    )
    duplicate_pairs = int(inter.duplicated(keys).sum())
    if duplicate_pairs:
        raise SystemExit(f"BDB2023 duplicate blocker-target pairs: {duplicate_pairs}")

    all_metrics = []
    structural = {
        "source_file_count": len(manifest),
        "resolved_source_interactions": int(len(inter)),
        "unresolved_blocked_player_refs": unresolved_refs,
        "duplicate_blocker_target_pairs": duplicate_pairs,
        "weeks": {},
    }

    for week in range(1, 9):
        tr = pd.read_csv(next(corpus_dir.rglob(f"week{week}.csv")))
        iw = inter.loc[inter["week"].eq(week)].drop(columns=["week"]).copy()
        expected = int(len(iw))
        ft = _resolve_frames(tr)
        base = tr.loc[tr["nflId"].notna(), ["gameId", "playId", "nflId", "frameId", "x", "y"]]

        bt = (
            base.merge(
                iw,
                left_on=["gameId", "playId", "nflId"],
                right_on=["gameId", "playId", "blockerNflId"],
                how="inner",
            )
            .drop(columns=["nflId"])
            .rename(columns={"x": "bx", "y": "by"})
        )
        dt = (
            base.merge(
                iw,
                left_on=["gameId", "playId", "nflId"],
                right_on=["gameId", "playId", "blockedNflId"],
                how="inner",
            )
            .drop(columns=["nflId", "pff_role", "blockedPlayerRole", "pff_blockType"])
            .rename(columns={"x": "dx", "y": "dy"})
        )
        play_frames = ft[
            ["gameId", "playId", "snapFrame", "terminalFrame", "snapSource", "terminalSource"]
        ]
        pairs = bt.merge(dt, on=keys + ["frameId"], how="inner").merge(
            play_frames, on=["gameId", "playId"], how="left"
        )
        pairs = pairs.loc[
            pairs["snapFrame"].notna()
            & pairs["terminalFrame"].notna()
            & pairs["frameId"].ge(pairs["snapFrame"])
            & pairs["frameId"].le(pairs["terminalFrame"])
        ].copy()
        pairs["distance"] = np.hypot(pairs["bx"] - pairs["dx"], pairs["by"] - pairs["dy"])
        pairs["elapsedSeconds"] = (pairs["frameId"] - pairs["snapFrame"]) / 10.0

        agg = (
            pairs.groupby(keys)
            .agg(
                blocker_target_shared_protection_frames=("frameId", "size"),
                blocker_target_min_distance_yards=("distance", "min"),
            )
            .reset_index()
        )
        sr = pairs.loc[pairs["frameId"].eq(pairs["snapFrame"]), keys + ["distance"]].rename(
            columns={"distance": "blocker_target_snap_distance_yards"}
        )
        er = pairs.loc[pairs["frameId"].eq(pairs["terminalFrame"]), keys + ["distance"]].rename(
            columns={"distance": "blocker_target_terminal_distance_yards"}
        )
        if len(pairs):
            idx = pairs.groupby(keys)["distance"].idxmin()
            mi = pairs.loc[idx, keys + ["elapsedSeconds"]].rename(
                columns={"elapsedSeconds": "blocker_target_time_to_min_distance_seconds"}
            )
        else:
            mi = pd.DataFrame(columns=keys + ["blocker_target_time_to_min_distance_seconds"])

        agg = agg.merge(sr, on=keys, how="left").merge(er, on=keys, how="left").merge(mi, on=keys, how="left")
        agg = agg.merge(iw, on=keys, how="left")
        agg = agg.merge(play_frames, on=["gameId", "playId"], how="left")
        agg["source_season"] = 2021
        agg["week"] = week
        agg["protection_window_length_seconds"] = (
            pd.to_numeric(agg["terminalFrame"], errors="coerce")
            - pd.to_numeric(agg["snapFrame"], errors="coerce")
        ) / 10.0
        agg["block_interaction_role"] = agg["pff_role"]
        agg["blocked_defender_source_role"] = agg["blockedPlayerRole"]
        agg["chip_release_interaction_flag"] = np.where(
            agg["pff_role"].eq("Pass Route"),
            1.0,
            np.where(agg["pff_role"].eq("Pass Block"), 0.0, np.nan),
        )
        agg["terminal_frame_fallback_flag"] = agg["terminalSource"].eq("last_frame_fallback").astype(int)
        agg = agg.rename(
            columns={
                "gameId": "game_id",
                "playId": "play_id",
                "blockerNflId": "blocker_nfl_id",
                "blockedNflId": "blocked_nfl_id",
                "snapSource": "snap_source",
                "terminalSource": "terminal_source",
            }
        )
        all_metrics.append(agg)

        structural["weeks"][str(week)] = {
            "expected_interactions": expected,
            "reconstructed_interactions": int(len(agg)),
            "geometry_coverage": float(len(agg) / expected) if expected else None,
            "snap_missing_plays": int(ft["snapFrame"].isna().sum()),
            "terminal_last_frame_fallback_plays": int(ft["terminalSource"].eq("last_frame_fallback").sum()),
        }

    raw = pd.concat(all_metrics, ignore_index=True)
    pooled_expected = sum(x["expected_interactions"] for x in structural["weeks"].values())
    structural["pooled_geometry_coverage"] = float(len(raw) / pooled_expected) if pooled_expected else None
    if structural["pooled_geometry_coverage"] is None or structural["pooled_geometry_coverage"] < 0.995:
        raise SystemExit("BDB2023 pooled interaction geometry coverage below 99.5%")
    if any((x["geometry_coverage"] or 0) < 0.99 for x in structural["weeks"].values()):
        raise SystemExit("BDB2023 at least one week interaction geometry coverage below 99%")

    history = _history_snapshots(raw)
    temporal = {
        "blocker_history": assert_strict_prior(history),
        "same_game_partial_history_forbidden": True,
        "target_game_rows_used": 0,
    }

    private = out_root / "private"
    manifests = {
        "protection_interactions": write_private_table(
            raw,
            private / "bdb2023_protection_interactions_v1.csv",
            ["week", "game_id", "play_id", "blocker_nfl_id", "blocked_nfl_id"],
        ),
        "blocker_history_snapshots": write_private_table(
            history,
            private / "bdb2023_blocker_history_snapshots_v1.csv",
            ["target_week", "blocker_nfl_id"],
        ),
    }

    report = sanitized_report(
        source_key=SOURCE_KEY,
        observed_source_hash=observed_hash,
        tables={"protection_interactions": raw, "blocker_history_snapshots": history},
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
