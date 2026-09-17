"""Materialize BDB 2026 Analytics throw-window features under Feature Dictionary V1."""
from __future__ import annotations

import argparse
import math
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

SOURCE_KEY = "BDB2026_THROW_WINDOW"


def _release_geometry(frame: pd.DataFrame, target_id: int | float, land_x: float, land_y: float) -> dict | None:
    target = frame.loc[frame["nfl_id"].eq(target_id)]
    defense = frame.loc[frame["player_role"].eq("Defensive Coverage")]
    if len(target) != 1 or defense.empty:
        return None
    tr = target.iloc[0]
    tx, ty = float(tr["x"]), float(tr["y"])
    dist = np.hypot(
        defense["x"].to_numpy(dtype=float) - tx,
        defense["y"].to_numpy(dtype=float) - ty,
    )
    land_dist = np.hypot(
        defense["x"].to_numpy(dtype=float) - float(land_x),
        defense["y"].to_numpy(dtype=float) - float(land_y),
    )
    sdist = np.sort(dist)
    return {
        "receiver_release_nearest_defender_distance_yards": float(sdist[0]),
        "receiver_release_second_defender_distance_yards": float(sdist[1]) if len(sdist) > 1 else np.nan,
        "receiver_release_defenders_within_2yd_count": int((dist <= 2.0).sum()),
        "receiver_release_defenders_within_3yd_count": int((dist <= 3.0).sum()),
        "receiver_release_target_to_land_distance_yards": float(math.hypot(tx - float(land_x), ty - float(land_y))),
        "release_nearest_defender_to_landing_zone_yards": float(np.min(land_dist)),
        "defender_count_release": int(len(defense)),
    }


def _terminal_geometry(
    output: pd.DataFrame,
    role_meta: pd.DataFrame,
    target_id: int | float,
    land_x: float,
    land_y: float,
) -> dict | None:
    if output.empty:
        return None
    out = output.merge(
        role_meta[["nfl_id", "player_role"]].drop_duplicates(),
        on="nfl_id",
        how="left",
    )
    target_out = out.loc[out["nfl_id"].eq(target_id)]
    if target_out.empty:
        return None

    terminal_frame = int(out["frame_id"].max())
    ff = out.loc[out["frame_id"].eq(terminal_frame)]
    target = ff.loc[ff["nfl_id"].eq(target_id)]
    if len(target) != 1:
        return None
    tr = target.iloc[0]
    tx, ty = float(tr["x"]), float(tr["y"])
    defense = ff.loc[ff["player_role"].eq("Defensive Coverage")]

    result = {
        "terminal_frame_id": terminal_frame,
        "terminal_receiver_to_land_distance_yards": float(math.hypot(tx - float(land_x), ty - float(land_y))),
        "predicted_defender_count_terminal": int(len(defense)),
        "terminal_nearest_predicted_defender_to_target_yards": np.nan,
        "terminal_nearest_predicted_defender_to_land_yards": np.nan,
        "postrelease_min_nearest_predicted_defender_distance_yards": np.nan,
    }

    if not defense.empty:
        result["terminal_nearest_predicted_defender_to_target_yards"] = float(
            np.hypot(
                defense["x"].to_numpy(dtype=float) - tx,
                defense["y"].to_numpy(dtype=float) - ty,
            ).min()
        )
        result["terminal_nearest_predicted_defender_to_land_yards"] = float(
            np.hypot(
                defense["x"].to_numpy(dtype=float) - float(land_x),
                defense["y"].to_numpy(dtype=float) - float(land_y),
            ).min()
        )

    mins: list[float] = []
    for _, gf in out.groupby("frame_id", sort=False):
        t = gf.loc[gf["nfl_id"].eq(target_id)]
        d = gf.loc[gf["player_role"].eq("Defensive Coverage")]
        if len(t) == 1 and not d.empty:
            trow = t.iloc[0]
            mins.append(
                float(
                    np.hypot(
                        d["x"].to_numpy(dtype=float) - float(trow["x"]),
                        d["y"].to_numpy(dtype=float) - float(trow["y"]),
                    ).min()
                )
            )
    if mins:
        result["postrelease_min_nearest_predicted_defender_distance_yards"] = float(min(mins))
    return result


def _receiver_history(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for target_week in range(1, 20):
        hist = raw.loc[raw["week"].lt(target_week)].copy()
        if hist.empty:
            continue
        for nfl_id, g in hist.groupby("nfl_id", sort=False):
            nearest = pd.to_numeric(g["receiver_release_nearest_defender_distance_yards"], errors="coerce").dropna()
            second = pd.to_numeric(g["receiver_release_second_defender_distance_yards"], errors="coerce").dropna()
            valid = g.loc[g["receiver_release_nearest_defender_distance_yards"].notna()].copy()
            n = int(len(nearest))
            crowd2 = (pd.to_numeric(valid["receiver_release_defenders_within_2yd_count"], errors="coerce") >= 1).astype(float)
            crowd3 = (pd.to_numeric(valid["receiver_release_defenders_within_3yd_count"], errors="coerce") >= 1).astype(float)
            rows.append(
                {
                    "source_season": 2023,
                    "target_week": int(target_week),
                    "nfl_id": nfl_id,
                    "history_max_source_week": int(g["week"].max()),
                    "hist_receiver_release_geometry_sample_count": n,
                    "second_defender_sample_count": int(len(second)),
                    "hist_receiver_release_nearest_defender_median_yards": float(nearest.median()) if n >= 8 else np.nan,
                    "hist_receiver_release_second_defender_median_yards": float(second.median()) if len(second) >= 8 else np.nan,
                    "hist_receiver_release_crowding_2yd_rate": float(crowd2.mean()) if n >= 8 else np.nan,
                    "hist_receiver_release_crowding_3yd_rate": float(crowd3.mean()) if n >= 8 else np.nan,
                    "strict_prior_only": True,
                }
            )
    return pd.DataFrame(rows)


def _receiver_route_history(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for target_week in range(1, 20):
        hist = raw.loc[raw["week"].lt(target_week)].copy()
        hist = hist.loc[hist["source_target_route_label"].notna()]
        if hist.empty:
            continue
        for (nfl_id, route), g in hist.groupby(["nfl_id", "source_target_route_label"], sort=False):
            nearest = pd.to_numeric(g["receiver_release_nearest_defender_distance_yards"], errors="coerce").dropna()
            rows.append(
                {
                    "source_season": 2023,
                    "target_week": int(target_week),
                    "nfl_id": nfl_id,
                    "source_target_route_label": route,
                    "history_max_source_week": int(g["week"].max()),
                    "route_history_sample_count": int(len(nearest)),
                    "hist_receiver_route_release_nearest_defender_median_yards": float(nearest.median()) if len(nearest) >= 5 else np.nan,
                    "strict_prior_only": True,
                }
            )
    return pd.DataFrame(rows)


def _benchmarks(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = raw.loc[raw["receiver_release_nearest_defender_distance_yards"].notna()].copy()

    man = (
        base.loc[base["source_target_route_label"].notna() & base["source_team_coverage_man_zone"].notna()]
        .groupby(["source_target_route_label", "source_team_coverage_man_zone"], dropna=False)
        .agg(
            route_x_man_zone_release_nearest_defender_median_yards=(
                "receiver_release_nearest_defender_distance_yards",
                "median",
            ),
            sample_count=("receiver_release_nearest_defender_distance_yards", "size"),
        )
        .reset_index()
    )

    cov = (
        base.loc[base["source_target_route_label"].notna() & base["source_team_coverage_type"].notna()]
        .groupby(["source_target_route_label", "source_team_coverage_type"], dropna=False)
        .agg(
            route_x_coverage_family_release_nearest_defender_median_yards=(
                "receiver_release_nearest_defender_distance_yards",
                "median",
            ),
            sample_count=("receiver_release_nearest_defender_distance_yards", "size"),
        )
        .reset_index()
    )
    return man, cov


def materialize(corpus_dir: Path, out_root: Path) -> dict:
    dictionary = load_dictionary()
    source, _ = source_contract(SOURCE_KEY, dictionary)
    observed_hash, manifest = corpus_sha256(corpus_dir)
    if observed_hash != source["source_hash_sha256"]:
        raise SystemExit(f"BDB2026 source hash mismatch: {observed_hash} != {source['source_hash_sha256']}")

    supp = pd.read_csv(next(corpus_dir.rglob("supplementary_data.csv")))
    supp = supp.loc[supp["season"].eq(2023)].copy()

    records: list[dict] = []
    structural = {
        "source_file_count": len(manifest),
        "plays_total": 0,
        "plays_target_role_not_unique": 0,
        "plays_target_not_player_to_predict": 0,
        "plays_missing_release_geometry": 0,
        "plays_missing_target_output": 0,
        "plays_missing_predicted_defender_output": 0,
        "weeks": {},
    }

    for week in range(1, 19):
        inp = pd.read_csv(next(corpus_dir.rglob(f"input_2023_w{week:02d}.csv")))
        out = pd.read_csv(next(corpus_dir.rglob(f"output_2023_w{week:02d}.csv")))
        role_meta = inp[["game_id", "play_id", "nfl_id", "player_role", "player_to_predict"]].drop_duplicates()
        wk_supp = supp.loc[supp["week"].eq(week)].copy()
        week_rows = 0

        for (game_id, play_id), play in inp.groupby(["game_id", "play_id"], sort=False):
            structural["plays_total"] += 1
            target_ids = play.loc[play["player_role"].eq("Targeted Receiver"), "nfl_id"].dropna().unique()
            if len(target_ids) != 1:
                structural["plays_target_role_not_unique"] += 1
                continue
            target_id = target_ids[0]
            pmeta = role_meta.loc[role_meta["game_id"].eq(game_id) & role_meta["play_id"].eq(play_id)]
            pred = pmeta.loc[pmeta["nfl_id"].eq(target_id), "player_to_predict"]
            if pred.empty or not bool(pred.iloc[0]):
                structural["plays_target_not_player_to_predict"] += 1

            release_frame_id = int(play["frame_id"].max())
            release = play.loc[play["frame_id"].eq(release_frame_id)]
            land_x_values = pd.to_numeric(play["ball_land_x"], errors="coerce").dropna().unique()
            land_y_values = pd.to_numeric(play["ball_land_y"], errors="coerce").dropna().unique()
            if len(land_x_values) != 1 or len(land_y_values) != 1:
                raise SystemExit(f"BDB2026 non-unique/missing landing coordinate game={game_id} play={play_id}")
            land_x = float(land_x_values[0])
            land_y = float(land_y_values[0])

            rg = _release_geometry(release, target_id, land_x, land_y)
            if rg is None:
                structural["plays_missing_release_geometry"] += 1
                continue

            pout = out.loc[out["game_id"].eq(game_id) & out["play_id"].eq(play_id)]
            tg = _terminal_geometry(pout, pmeta, target_id, land_x, land_y)
            if tg is None:
                structural["plays_missing_target_output"] += 1
            elif tg["predicted_defender_count_terminal"] == 0:
                structural["plays_missing_predicted_defender_output"] += 1

            srow = wk_supp.loc[wk_supp["game_id"].eq(game_id) & wk_supp["play_id"].eq(play_id)]
            route = srow["route_of_targeted_receiver"].iloc[0] if len(srow) else None
            man_zone = srow["team_coverage_man_zone"].iloc[0] if len(srow) else None
            coverage_type = srow["team_coverage_type"].iloc[0] if len(srow) else None

            rec = {
                "source_season": 2023,
                "week": week,
                "game_id": game_id,
                "play_id": play_id,
                "nfl_id": target_id,
                "release_frame_id": release_frame_id,
                "source_target_route_label": route,
                "source_team_coverage_man_zone": man_zone,
                "source_team_coverage_type": coverage_type,
            }
            rec.update(rg)
            if tg is not None:
                rec.update(tg)
            else:
                rec.update(
                    {
                        "terminal_frame_id": np.nan,
                        "terminal_receiver_to_land_distance_yards": np.nan,
                        "predicted_defender_count_terminal": 0,
                        "terminal_nearest_predicted_defender_to_target_yards": np.nan,
                        "terminal_nearest_predicted_defender_to_land_yards": np.nan,
                        "postrelease_min_nearest_predicted_defender_distance_yards": np.nan,
                    }
                )
            rec["postrelease_closing_delta_yards"] = (
                rec["receiver_release_nearest_defender_distance_yards"]
                - rec["postrelease_min_nearest_predicted_defender_distance_yards"]
                if pd.notna(rec["postrelease_min_nearest_predicted_defender_distance_yards"])
                else np.nan
            )
            records.append(rec)
            week_rows += 1

        structural["weeks"][str(week)] = {"release_geometry_rows": int(week_rows)}

    raw = pd.DataFrame(records)
    if structural["plays_target_role_not_unique"]:
        raise SystemExit("BDB2026 targeted receiver role non-unique")
    if structural["plays_target_not_player_to_predict"]:
        raise SystemExit("BDB2026 targeted receiver not player_to_predict")
    release_rate = len(raw) / max(1, structural["plays_total"])
    structural["release_geometry_rate"] = float(release_rate)
    if release_rate < 0.995:
        raise SystemExit(f"BDB2026 release geometry coverage below 99.5%: {release_rate:.6f}")
    structural["terminal_target_geometry_rows"] = int(raw["terminal_receiver_to_land_distance_yards"].notna().sum())
    terminal_rate = float(raw["terminal_receiver_to_land_distance_yards"].notna().mean()) if len(raw) else 0.0
    structural["terminal_target_geometry_rate"] = terminal_rate
    if terminal_rate < 0.995:
        raise SystemExit(f"BDB2026 terminal target geometry coverage below 99.5%: {terminal_rate:.6f}")

    receiver_history = _receiver_history(raw)
    route_history = _receiver_route_history(raw)
    man_benchmark, coverage_benchmark = _benchmarks(raw)

    temporal = {
        "receiver_history": assert_strict_prior(receiver_history),
        "receiver_route_history": assert_strict_prior(route_history),
        "same_game_partial_history_forbidden": True,
        "target_game_rows_used": 0,
        "landing_and_postrelease_fields_pregame_used": 0,
    }

    private = out_root / "private"
    manifests = {
        "targeted_receiver_plays": write_private_table(
            raw,
            private / "bdb2026_targeted_receiver_throw_window_v1.csv",
            ["week", "game_id", "play_id", "nfl_id"],
        ),
        "receiver_history_snapshots": write_private_table(
            receiver_history,
            private / "bdb2026_receiver_history_snapshots_v1.csv",
            ["target_week", "nfl_id"],
        ),
        "receiver_route_history_snapshots": write_private_table(
            route_history,
            private / "bdb2026_receiver_route_history_snapshots_v1.csv",
            ["target_week", "nfl_id", "source_target_route_label"],
        ),
        "route_man_zone_benchmark": write_private_table(
            man_benchmark,
            private / "bdb2026_route_man_zone_benchmark_v1.csv",
            ["source_target_route_label", "source_team_coverage_man_zone"],
        ),
        "route_coverage_family_benchmark": write_private_table(
            coverage_benchmark,
            private / "bdb2026_route_coverage_family_benchmark_v1.csv",
            ["source_target_route_label", "source_team_coverage_type"],
        ),
    }

    report = sanitized_report(
        source_key=SOURCE_KEY,
        observed_source_hash=observed_hash,
        tables={
            "targeted_receiver_plays": raw,
            "receiver_history_snapshots": receiver_history,
            "receiver_route_history_snapshots": route_history,
            "route_man_zone_benchmark": man_benchmark,
            "route_coverage_family_benchmark": coverage_benchmark,
        },
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
