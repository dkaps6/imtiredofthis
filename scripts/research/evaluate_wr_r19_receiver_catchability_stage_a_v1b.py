#!/usr/bin/env python3
"""WR-R19 Stage A v1b mechanical corrections, pre-result.

This wrapper preserves every frozen R19 football definition/gate from v1 while fixing
only two source/holdout mechanics discovered by the first real Stage-A attempt:
1) FTN/PBP both expose season/week, so PBP metadata is explicitly prefixed before the
   exact game/play merge and matched season/week parity is asserted;
2) the authority CSV is streamed so 2024 rows are counted/key-audited for exact artifact
   parity but 2024 outcome/projection fields are never parsed or materialized.

No football result was produced by the failed v1 attempt.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research import evaluate_wr_r19_receiver_catchability_stage_a_v1 as base


def load_authority_development_only(path: Path) -> pd.DataFrame:
    """Stream exact authority; materialize only 2023 development outcome rows."""
    required = {
        "variant", "team", "player_clean_key", "player", "wr_rank", "pred_targets",
        "entitlement_tgt_share", "mc_rec_yards", "season", "week", "actual_rec_yards",
    }
    counts: dict[int, int] = {}
    seen: set[tuple[int, int, str, str]] = set()
    dev_rows: list[dict] = []

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        missing = sorted(required - fields)
        if missing:
            raise RuntimeError(f"WR-R19 authority missing columns: {missing}")
        for raw in reader:
            if str(raw.get("variant", "")) != base.AUTHORITY_VARIANT:
                continue
            season = int(float(raw["season"]))
            week = int(float(raw["week"]))
            team = base.canon_team(raw["team"])
            key = str(raw["player_clean_key"])
            counts[season] = counts.get(season, 0) + 1
            ident = (season, week, team, key)
            if ident in seen:
                raise RuntimeError(f"WR-R19 duplicate authority identity: {ident}")
            seen.add(ident)

            # Holdout protection: do not parse/materialize any 2024 model/outcome value.
            if season != base.DEV_SEASON:
                continue
            dev_rows.append({
                "variant": raw["variant"],
                "team": team,
                "player_clean_key": key,
                "player": str(raw["player"]),
                "wr_rank": int(float(raw["wr_rank"])),
                "pred_targets": float(raw["pred_targets"]),
                "entitlement_tgt_share": float(raw["entitlement_tgt_share"]),
                "mc_rec_yards": float(raw["mc_rec_yards"]),
                "season": season,
                "week": week,
                "actual_rec_yards": float(raw["actual_rec_yards"]),
            })

    if counts != base.EXPECTED_ROWS:
        raise RuntimeError(f"WR-R19 authority row-count parity failed: {counts} != {base.EXPECTED_ROWS}")
    x = pd.DataFrame(dev_rows)
    if len(x) != base.EXPECTED_ROWS[base.DEV_SEASON]:
        raise RuntimeError(f"WR-R19 development materialization count drift: {len(x)}")
    x["yard_residual"] = base._num(x["actual_rec_yards"]) - base._num(x["mc_rec_yards"])
    return x.sort_values(["season", "week", "team", "player_clean_key"]).reset_index(drop=True)


def _merge_ftn_pbp_targets(ftn: pd.DataFrame, pbp: pd.DataFrame, season: int) -> tuple[pd.DataFrame, float]:
    """Pure exact-play merge used by real loader and synthetic regression tests."""
    ftn = ftn.copy()
    pbp = pbp.copy()
    ftn.columns = [str(c).strip().lower() for c in ftn.columns]
    pbp.columns = [str(c).strip().lower() for c in pbp.columns]

    req_f = {"nflverse_game_id", "nflverse_play_id", "season", "week", "is_catchable_ball"}
    req_p = {
        "game_id", "play_id", "season", "week", "season_type", "posteam",
        "pass_attempt", "sack", "two_point_attempt", "receiver_player_id",
        "receiver_player_name", "air_yards", "complete_pass",
    }
    miss_f = sorted(req_f - set(ftn.columns))
    miss_p = sorted(req_p - set(pbp.columns))
    if miss_f or miss_p:
        raise RuntimeError(f"WR-R19 source schema missing season={season}: ftn={miss_f}, pbp={miss_p}")

    ftn = ftn.loc[base._num(ftn["week"]).between(1, 18)].copy()
    pbp = pbp.loc[
        pbp["season_type"].astype(str).str.upper().eq("REG")
        & base._num(pbp["week"]).between(1, 18)
    ].copy()
    ftn["join_game"] = ftn["nflverse_game_id"].astype(str)
    ftn["join_play"] = base._num(ftn["nflverse_play_id"])
    pbp["join_game"] = pbp["game_id"].astype(str)
    pbp["join_play"] = base._num(pbp["play_id"])
    if ftn.duplicated(["join_game", "join_play"]).any():
        raise RuntimeError(f"WR-R19 duplicate FTN game/play key season={season}")
    if pbp.duplicated(["join_game", "join_play"]).any():
        raise RuntimeError(f"WR-R19 duplicate PBP game/play key season={season}")

    pcols = [
        "join_game", "join_play", "game_id", "season", "week", "posteam",
        "pass_attempt", "sack", "two_point_attempt", "receiver_player_id",
        "receiver_player_name", "air_yards", "complete_pass",
    ]
    p = pbp[pcols].rename(columns={
        "game_id": "pbp_game_id", "season": "pbp_season", "week": "pbp_week",
        "posteam": "pbp_posteam", "pass_attempt": "pbp_pass_attempt",
        "sack": "pbp_sack", "two_point_attempt": "pbp_two_point_attempt",
        "receiver_player_id": "pbp_receiver_player_id",
        "receiver_player_name": "pbp_receiver_player_name",
        "air_yards": "pbp_air_yards", "complete_pass": "pbp_complete_pass",
    })
    m = ftn.merge(p, on=["join_game", "join_play"], how="left", validate="one_to_one", indicator=True)
    matched = m["_merge"].eq("both")
    join_rate = float(matched.mean()) if len(m) else 0.0
    if join_rate < 0.95:
        raise RuntimeError(f"WR-R19 FTN/PBP exact join below source contract season={season}: {join_rate}")

    if matched.any():
        f_season = base._num(m.loc[matched, "season"])
        p_season = base._num(m.loc[matched, "pbp_season"])
        f_week = base._num(m.loc[matched, "week"])
        p_week = base._num(m.loc[matched, "pbp_week"])
        parity = f_season.eq(p_season) & f_week.eq(p_week)
        if not bool(parity.all()):
            bad = m.loc[matched].loc[~parity, ["join_game", "join_play", "season", "week", "pbp_season", "pbp_week"]].head(10)
            raise RuntimeError(f"WR-R19 FTN/PBP season-week parity failed season={season}: {bad.to_dict('records')}")

    official = base._num(m["pbp_pass_attempt"]).fillna(0).eq(1)
    official &= ~base._num(m["pbp_sack"]).fillna(0).eq(1)
    official &= ~base._num(m["pbp_two_point_attempt"]).fillna(0).eq(1)
    rid = m["pbp_receiver_player_id"].map(base._clean_id)
    t = m.loc[matched & official & rid.ne("")].copy()
    t["game_id"] = t["pbp_game_id"].astype(str)
    t["receiver_id"] = t["pbp_receiver_player_id"].map(base._clean_id)
    t["receiver_name_key"] = t["pbp_receiver_player_name"].map(base._name_key)
    t["team"] = t["pbp_posteam"].map(base.canon_team)
    t["season"] = base._num(t["season"]).astype(int)
    t["week"] = base._num(t["week"]).astype(int)
    t["catchable"] = base._bool_num(t["is_catchable_ball"])
    t["air"] = base._num(t["pbp_air_yards"])
    t["complete_pass_num"] = base._num(t["pbp_complete_pass"]).fillna(0)
    t["target_event_seq"] = np.arange(len(t), dtype=np.int64)
    return t[[
        "season", "week", "game_id", "team", "receiver_id", "receiver_name_key",
        "catchable", "air", "complete_pass_num", "target_event_seq",
    ]].reset_index(drop=True), join_rate


def load_ftn_pbp_targets_v1b(seasons) -> tuple[pd.DataFrame, list[dict]]:
    frames: list[pd.DataFrame] = []
    source_meta: list[dict] = []
    for season in sorted({int(s) for s in seasons}):
        ftn_url = f"https://github.com/nflverse/nflverse-data/releases/download/ftn_charting/ftn_charting_{season}.parquet"
        pbp_url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
        ftn, fm = base._read_parquet(ftn_url)
        pbp, pm = base._read_parquet(pbp_url)
        t, join_rate = _merge_ftn_pbp_targets(ftn, pbp, season)
        frames.append(t)
        source_meta.append({
            "season": season,
            "ftn_sha256": fm["sha256"], "ftn_bytes": fm["bytes"],
            "pbp_sha256": pm["sha256"], "pbp_bytes": pm["bytes"],
            "exact_join_rate": join_rate,
            "season_week_parity": True,
            "receiver_target_rows": int(len(t)),
            "catchable_coverage": float(t["catchable"].notna().mean()) if len(t) else 0.0,
            "completed_target_rows": int(t["complete_pass_num"].eq(1).sum()),
            "incomplete_target_rows": int(t["complete_pass_num"].eq(0).sum()),
        })
    if not frames:
        raise RuntimeError("WR-R19 FTN/PBP target source returned zero rows")
    return pd.concat(frames, ignore_index=True, sort=False), source_meta


def main() -> int:
    # Monkey-patch only corrected mechanical loaders; all football calculations/gates stay in frozen v1.
    base.load_authority = load_authority_development_only
    base.load_ftn_pbp_targets = load_ftn_pbp_targets_v1b
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
