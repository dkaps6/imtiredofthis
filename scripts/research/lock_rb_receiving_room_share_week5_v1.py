#!/usr/bin/env python3
"""Freeze Week-5 pregame RB receiving-room share shadow V1.

Identity authority is the already-frozen Week-5 RB player-state lock plus its
certified live player-state parent. The receiving rule is unchanged from the
successful W1-W4 retrospective impact replay: normalize strict-prior
prior_rb_room_share within each frozen RB room.

No Week-5 outcomes or sportsbook inputs are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.modeling.rb_receiving_identity_runtime_v1 import (
    _snapshot_queries,
    identity_atlas,
)

SEASON = 2026
WEEK = 5
TOL = 1e-10

PARENT_LOCK_RUN = 37560824479
PARENT_LOCK_ARTIFACT = 11456916566
PARENT_LOCK_DIGEST = "sha256:edf51bbd93920ef0af580a0396422af3288cf511785be59deeea95c117062af4"
PARENT_STATE_RUN = 37560311001
PARENT_STATE_ARTIFACT = 11456556226
PARENT_STATE_DIGEST = "sha256:a39b958e492a781e310de0f14d34153e21ca589a76cd226479b3b39e10f9328e"


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"true", "1", "yes"})


def _attach_prior_room_share(lock: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Attach raw strict-prior receiving-room state to exact frozen identities."""
    required_lock = {
        "season", "week", "team", "opponent", "state_key", "player",
        "gsis_id", "pfr_id", "room_size", "control_recent_carry_share",
        "shadow_player_state_share",
    }
    miss = required_lock - set(lock.columns)
    if miss:
        raise RuntimeError(f"Week-5 RB lock missing columns: {sorted(miss)}")

    required_live = {
        "state_key", "team", "player", "name_key", "position_group",
        "chronology_valid", "last3_source_max_week", "snap_source_max_week",
        "sportsbook_inputs_used" if "sportsbook_inputs_used" in live.columns else "state_key",
    }
    # The conditional item above avoids inventing a required sportsbook column
    # when the certified parent stores that guarantee only in its JSON.
    required_live.discard("sportsbook_inputs_used")
    miss = required_live - set(live.columns)
    if miss:
        raise RuntimeError(f"live player-state parent missing columns: {sorted(miss)}")

    if len(lock) != 98 or int(lock["team"].nunique()) != 30:
        raise RuntimeError(
            f"frozen Week-5 RB identity drift rows={len(lock)} teams={lock['team'].nunique()}"
        )
    if not pd.to_numeric(lock["week"], errors="coerce").eq(WEEK).all():
        raise RuntimeError("frozen RB lock contains non-Week5 rows")
    if lock["state_key"].astype(str).duplicated().any():
        raise RuntimeError("frozen RB lock has duplicate state_key")

    live = live.copy()
    live["team"] = live["team"].map(canon_team)
    if live["state_key"].astype(str).duplicated().any():
        d = live.loc[live["state_key"].astype(str).duplicated(keep=False), ["state_key", "team", "player"]]
        raise RuntimeError(f"live parent duplicate state_key: {d.head(20).to_dict('records')}")

    parent = lock.merge(
        live[
            [
                "state_key", "team", "name_key", "position_group",
                "chronology_valid", "last3_source_max_week", "snap_source_max_week",
                "roster_source_week", "current_games", "same_team_current_games",
                "last3_targets",
            ]
        ],
        on=["state_key", "team"],
        how="left",
        validate="one_to_one",
    )
    if parent["name_key"].isna().any():
        bad = parent.loc[parent["name_key"].isna(), ["team", "player", "state_key"]]
        raise RuntimeError(
            f"frozen RB identities missing from certified live parent: {bad.to_dict('records')}"
        )
    if not parent["position_group"].astype(str).str.upper().eq("RB").all():
        bad = parent.loc[
            ~parent["position_group"].astype(str).str.upper().eq("RB"),
            ["team", "player", "position_group"],
        ]
        raise RuntimeError(f"frozen lock contains non-RB live parent rows: {bad.to_dict('records')}")
    if not _bool_series(parent["chronology_valid"]).all():
        bad = parent.loc[~_bool_series(parent["chronology_valid"]), ["team", "player", "state_key"]]
        raise RuntimeError(f"live parent chronology failure: {bad.to_dict('records')}")
    for c in ("last3_source_max_week", "snap_source_max_week"):
        vals = pd.to_numeric(parent[c], errors="coerce")
        if vals.notna().any() and not vals.dropna().lt(WEEK).all():
            bad = parent.loc[vals.ge(WEEK), ["team", "player", c]]
            raise RuntimeError(f"target/future chronology in {c}: {bad.to_dict('records')}")

    # name_key is the certified parent alias used to bridge into the same
    # canonical weekly-stat name key produced by player_form_v2.
    queries = pd.DataFrame(
        {
            "player_clean_key": parent["name_key"].astype(str),
            "team": parent["team"].astype(str),
            "season": SEASON,
            "week": WEEK,
        }
    )
    states, prev = identity_atlas(2013, SEASON)
    feat = _snapshot_queries(queries, states, prev)
    keep = ["player_clean_key", "team", "season", "week"]
    for c in ("prior_rb_room_share", "prior_games", "same_team_prior_rb_room_share"):
        if c in feat.columns:
            keep.append(c)
    feat = feat[keep].copy()
    if "prior_rb_room_share" not in feat.columns:
        feat["prior_rb_room_share"] = np.nan
    if "prior_games" not in feat.columns:
        feat["prior_games"] = np.nan
    if "same_team_prior_rb_room_share" not in feat.columns:
        feat["same_team_prior_rb_room_share"] = np.nan

    feat = feat.rename(columns={"player_clean_key": "name_key"})
    out = parent.merge(
        feat[
            [
                "name_key", "team", "prior_rb_room_share", "prior_games",
                "same_team_prior_rb_room_share",
            ]
        ],
        on=["name_key", "team"],
        how="left",
        validate="one_to_one",
    )
    for c in ("prior_rb_room_share", "prior_games", "same_team_prior_rb_room_share"):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # This prospective lock intentionally fails closed if a frozen live RB lacks
    # genuine strict-prior receiving-room history. Do not silently substitute the
    # historical-roster universe or a fitted fallback.
    missing = out["prior_rb_room_share"].isna() | ~np.isfinite(out["prior_rb_room_share"])
    if missing.any():
        bad = out.loc[
            missing,
            [
                "team", "player", "state_key", "name_key", "gsis_id",
                "current_games", "last3_targets",
            ],
        ]
        raise RuntimeError(
            "frozen Week-5 RB identities missing strict-prior prior_rb_room_share: "
            f"{bad.to_dict('records')}"
        )
    if out["prior_rb_room_share"].lt(0).any():
        raise RuntimeError("negative prior_rb_room_share in Week-5 lock")
    return out


def build_candidate(lock: pd.DataFrame, live: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    x = _attach_prior_room_share(lock.copy(), live.copy())
    x["candidate_rb_receiving_room_share"] = np.nan

    team_rows = []
    for team, idx in x.groupby("team", sort=True).groups.items():
        g = x.loc[idx].copy()
        if len(g) < 2:
            raise RuntimeError(f"frozen Week-5 RB room has <2 players team={team}")
        state = pd.to_numeric(g["prior_rb_room_share"], errors="coerce")
        total = float(state.sum())
        if not np.isfinite(total) or total <= 0:
            raise RuntimeError(f"nonpositive Week-5 prior RB receiving-room mass team={team}")
        candidate = state / total
        x.loc[idx, "candidate_rb_receiving_room_share"] = candidate.to_numpy(float)
        team_rows.append(
            {
                "team": team,
                "players": int(len(g)),
                "prior_state_sum": total,
                "candidate_sum": float(candidate.sum()),
                "candidate_max": float(candidate.max()),
                "candidate_min": float(candidate.min()),
                "candidate_hhi": float(np.square(candidate.to_numpy(float)).sum()),
                "carry_reference_hhi": float(
                    np.square(
                        pd.to_numeric(g["shadow_player_state_share"], errors="coerce")
                        .fillna(0.0)
                        .to_numpy(float)
                    ).sum()
                ),
            }
        )

    teams = pd.DataFrame(team_rows)
    if x["candidate_rb_receiving_room_share"].isna().any():
        raise RuntimeError("candidate Week-5 receiving-room share contains missing values")
    if float((teams["candidate_sum"] - 1.0).abs().max()) > TOL:
        raise RuntimeError("candidate Week-5 receiving-room shares fail conservation")

    out = pd.DataFrame(
        {
            "season": SEASON,
            "week": WEEK,
            "team": x["team"],
            "opponent": x["opponent"],
            "state_key": x["state_key"],
            "player": x["player"],
            "name_key": x["name_key"],
            "gsis_id": x["gsis_id"],
            "pfr_id": x["pfr_id"],
            "room_size": x["room_size"],
            "prior_games": x["prior_games"],
            "prior_rb_room_share": x["prior_rb_room_share"],
            "same_team_prior_rb_room_share": x["same_team_prior_rb_room_share"],
            "candidate_rb_receiving_room_share": x["candidate_rb_receiving_room_share"],
            "reference_control_recent_carry_share": x["control_recent_carry_share"],
            "reference_carry_snap_shadow_share": x["shadow_player_state_share"],
            "current_games": x["current_games"],
            "same_team_current_games": x["same_team_current_games"],
            "last3_targets": x["last3_targets"],
            "last3_source_max_week": x["last3_source_max_week"],
            "snap_source_max_week": x["snap_source_max_week"],
            "chronology_valid": x["chronology_valid"],
            "roster_source_week": x["roster_source_week"],
        }
    ).sort_values(["team", "state_key"], kind="mergesort").reset_index(drop=True)

    result = {
        "version": "RB_RECEIVING_ROOM_SHARE_SHADOW_V1_WEEK5_LOCK",
        "status": "WEEK5_PREGAME_LOCK_FROZEN",
        "season": SEASON,
        "week": WEEK,
        "players": int(len(out)),
        "teams": int(out["team"].nunique()),
        "history_available_players": int(out["prior_rb_room_share"].notna().sum()),
        "history_available_rate": float(out["prior_rb_room_share"].notna().mean()),
        "rooms_locked": int(len(teams)),
        "max_candidate_conservation_gap": float((teams["candidate_sum"] - 1.0).abs().max()),
        "median_candidate_hhi": float(teams["candidate_hhi"].median()),
        "parameters_fit": 0,
        "sportsbook_inputs_used": 0,
        "week5_outcomes_read": 0,
        "production_changed": False,
        "retrospective_w1_w4_impact_run": 37703522415,
        "retrospective_result_used_to_change_rule": False,
        "identity_authority": "FROZEN_WEEK5_RB_PLAYER_STATE_LOCK",
        "receiving_state": "prior_rb_room_share",
        "redistribution_rule": "normalize strict-prior prior_rb_room_share within exact frozen RB room",
        "parent_rb_player_state_run": PARENT_LOCK_RUN,
        "parent_rb_player_state_artifact": PARENT_LOCK_ARTIFACT,
        "parent_rb_player_state_digest": PARENT_LOCK_DIGEST,
        "parent_live_player_state_run": PARENT_STATE_RUN,
        "parent_live_player_state_artifact": PARENT_STATE_ARTIFACT,
        "parent_live_player_state_digest": PARENT_STATE_DIGEST,
    }
    return out, teams, result


def run(*, parent_rb_lock_path: Path, parent_live_rows_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = _read(parent_rb_lock_path, "frozen Week-5 RB player-state lock")
    live = _read(parent_live_rows_path, "certified Week-5 live player-state rows")
    out, teams, result = build_candidate(lock, live)

    csv_path = out_dir / "rb_receiving_room_share_week5_lock.csv"
    team_path = out_dir / "rb_receiving_room_share_week5_team_summary.csv"
    out.to_csv(csv_path, index=False, float_format="%.12f")
    teams.to_csv(team_path, index=False, float_format="%.12f")
    result["row_csv_sha256"] = "sha256:" + _sha256(csv_path)
    result["team_csv_sha256"] = "sha256:" + _sha256(team_path)
    result["generated_at"] = datetime.now(timezone.utc).isoformat()

    (out_dir / "rb_receiving_room_share_week5_lock.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--parent-rb-lock", type=Path, required=True)
    p.add_argument("--parent-live-rows", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    run(
        parent_rb_lock_path=a.parent_rb_lock,
        parent_live_rows_path=a.parent_live_rows,
        out_dir=a.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
