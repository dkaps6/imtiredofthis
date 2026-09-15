#!/usr/bin/env python3
"""WR Phase 4A: source-only full WR-room actual-target reconciliation.

This audit does NOT load receiving-yard projection/outcome fields. It validates
whether nflverse weekly player stats can be trusted as the independent source
for actual full WR-room target mass before Phase-4 error attribution begins.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.player_form_v2 import _normalize_weekly, _to_pandas

CANDIDATE = "WR_R15_WR1_ANCHORED_PARTICIPATION"
EXPECTED_CANDIDATE_ROWS = 4193
EXPECTED_TEAM_GAMES = 1088
EXPECTED_WR1_PRESENT = 1026
EXPECTED_COMPLETE_ROOMS = 71
WR_POS = {"WR", "LWR", "RWR", "SWR"}
TOL = 1e-9


def _load_authority(pred_path: Path, feat_path: Path):
    pred_cols = [
        "variant", "season", "week", "team", "player_clean_key", "player",
        "wr_rank", "actual_targets",
    ]
    pred = pd.read_csv(pred_path, usecols=pred_cols)
    pred = pred.loc[pred["variant"].astype(str).eq(CANDIDATE)].copy()
    if len(pred) != EXPECTED_CANDIDATE_ROWS:
        raise RuntimeError(f"candidate row count drift: {len(pred)} != {EXPECTED_CANDIDATE_ROWS}")
    pred["season"] = pd.to_numeric(pred["season"], errors="raise").astype(int)
    pred["week"] = pd.to_numeric(pred["week"], errors="raise").astype(int)
    pred["team"] = pred["team"].map(canon_team)
    pred["player_clean_key"] = pred["player_clean_key"].astype(str)
    pred["wr_rank"] = pd.to_numeric(pred["wr_rank"], errors="raise").astype(int)
    pred["actual_targets"] = pd.to_numeric(pred["actual_targets"], errors="raise").astype(float)

    key = ["season", "week", "team", "player_clean_key"]
    if pred.duplicated(key).any():
        bad = pred.loc[pred.duplicated(key, keep=False), key].head(10)
        raise RuntimeError(f"duplicate candidate authority identity: {bad.to_dict('records')}")

    feat_cols = ["season", "week", "team", "player_clean_key", "baseline_wr_rank"]
    feat = pd.read_csv(feat_path, usecols=feat_cols)
    feat["season"] = pd.to_numeric(feat["season"], errors="raise").astype(int)
    feat["week"] = pd.to_numeric(feat["week"], errors="raise").astype(int)
    feat["team"] = feat["team"].map(canon_team)
    feat["player_clean_key"] = feat["player_clean_key"].astype(str)
    feat["baseline_wr_rank"] = pd.to_numeric(feat["baseline_wr_rank"], errors="raise").astype(int)
    if int(feat["baseline_wr_rank"].min()) < 2:
        raise RuntimeError("feature file unexpectedly contains WR1 rows")
    if feat.duplicated(key).any():
        bad = feat.loc[feat.duplicated(key, keep=False), key].head(10)
        raise RuntimeError(f"duplicate feature identity: {bad.to_dict('records')}")

    tg = ["season", "week", "team"]
    team_games = pred[tg].drop_duplicates().sort_values(tg).reset_index(drop=True)
    if len(team_games) != EXPECTED_TEAM_GAMES:
        raise RuntimeError(f"authority team-game count drift: {len(team_games)} != {EXPECTED_TEAM_GAMES}")

    anchors = pred.loc[pred["wr_rank"].eq(1), tg + ["player_clean_key"]].copy()
    if anchors.duplicated(tg).any():
        raise RuntimeError("more than one WR1 anchor in candidate authority team-game")
    if len(anchors) != EXPECTED_WR1_PRESENT:
        raise RuntimeError(f"WR1-present count drift: {len(anchors)} != {EXPECTED_WR1_PRESENT}")

    cand_sets = pred.groupby(tg)["player_clean_key"].agg(lambda s: frozenset(str(x) for x in s))
    feat_sets = feat.groupby(tg)["player_clean_key"].agg(lambda s: frozenset(str(x) for x in s))
    anchor_map = anchors.set_index(tg)["player_clean_key"].to_dict()

    room_rows = []
    complete_keys = []
    for r in team_games.itertuples(index=False):
        k = (int(r.season), int(r.week), str(r.team))
        candidate_set = set(cand_sets.get(k, frozenset()))
        secondary_set = set(feat_sets.get(k, frozenset()))
        anchor_key = str(anchor_map.get(k, ""))
        expected_set = set(secondary_set)
        if anchor_key:
            expected_set.add(anchor_key)
        missing = sorted(expected_set - candidate_set)
        extra = sorted(candidate_set - expected_set)
        complete = bool(anchor_key and candidate_set == expected_set)
        if complete:
            complete_keys.append(k)
        room_rows.append({
            "season": k[0], "week": k[1], "team": k[2],
            "anchor_key": anchor_key,
            "candidate_rows": len(candidate_set),
            "expected_room_rows": len(expected_set),
            "missing_expected_rows": len(missing),
            "extra_candidate_rows": len(extra),
            "missing_expected_keys": "|".join(missing),
            "extra_candidate_keys": "|".join(extra),
            "complete_canonical_room": complete,
        })
    membership = pd.DataFrame(room_rows)
    if len(complete_keys) != EXPECTED_COMPLETE_ROOMS:
        raise RuntimeError(
            f"complete canonical room count drift: {len(complete_keys)} != {EXPECTED_COMPLETE_ROOMS}"
        )
    return pred, feat, membership, set(complete_keys)


def _load_weekly_actuals(seasons=(2023, 2024)) -> pd.DataFrame:
    import nflreadpy as nfl

    frames = []
    for season in seasons:
        raw = nfl.load_player_stats(seasons=[int(season)], summary_level="week")
        x = _normalize_weekly(_to_pandas(raw), int(season))
        x = x.loc[pd.to_numeric(x["week"], errors="coerce").between(1, 18)].copy()
        x["season"] = int(season)
        x["week"] = pd.to_numeric(x["week"], errors="raise").astype(int)
        x["team"] = x["team"].map(canon_team)
        x["player_clean_key"] = x["player_clean_key"].astype(str)
        x["position"] = x["position"].astype("string").fillna("").str.upper().str.strip()
        x["targets"] = pd.to_numeric(x["targets"], errors="raise").astype(float)
        frames.append(x[[
            "season", "week", "team", "player_clean_key", "player", "position", "targets"
        ]])
    out = pd.concat(frames, ignore_index=True, sort=False)
    key = ["season", "week", "team", "player_clean_key"]
    if out.duplicated(key).any():
        bad = out.loc[out.duplicated(key, keep=False), key].head(10)
        raise RuntimeError(f"raw weekly duplicate identity: {bad.to_dict('records')}")
    return out


def _complete_key_frame(complete_keys: set[tuple[int, int, str]]) -> pd.DataFrame:
    return pd.DataFrame(sorted(complete_keys), columns=["season", "week", "team"])


def _reconcile(pred: pd.DataFrame, raw: pd.DataFrame, complete_keys: set[tuple[int, int, str]]):
    tg = ["season", "week", "team"]
    ident = tg + ["player_clean_key"]
    ck = _complete_key_frame(complete_keys)

    cp = pred.merge(ck, on=tg, how="inner", validate="many_to_one").copy()
    if cp.empty:
        raise RuntimeError("complete authority subset is empty")

    raw_subset = raw.merge(ck, on=tg, how="inner", validate="many_to_one").copy()
    raw_wr = raw_subset.loc[raw_subset["position"].isin(WR_POS)].copy()

    raw_ident = raw[[*ident, "targets", "position", "player"]].rename(columns={
        "targets": "raw_targets", "position": "raw_position", "player": "raw_player"
    })
    player = cp.merge(raw_ident, on=ident, how="left", validate="one_to_one", indicator=True)
    player["raw_identity_found"] = player["_merge"].eq("both")
    player["raw_targets_effective"] = player["raw_targets"]
    missing_zero = (~player["raw_identity_found"]) & player["actual_targets"].abs().le(TOL)
    player.loc[missing_zero, "raw_targets_effective"] = 0.0
    player["target_delta"] = player["raw_targets_effective"] - player["actual_targets"]
    player["player_parity"] = player["target_delta"].abs().le(TOL)
    positive_missing = (~player["raw_identity_found"]) & player["actual_targets"].gt(TOL)

    expected_sets = cp.groupby(tg)["player_clean_key"].agg(lambda s: set(str(x) for x in s)).to_dict()
    extra_rows = []
    for r in raw_wr.itertuples(index=False):
        k = (int(r.season), int(r.week), str(r.team))
        expected = expected_sets.get(k, set())
        if str(r.player_clean_key) not in expected and float(r.targets) > TOL:
            extra_rows.append({
                "season": int(r.season), "week": int(r.week), "team": str(r.team),
                "player_clean_key": str(r.player_clean_key), "player": str(r.player),
                "position": str(r.position), "targets": float(r.targets),
            })
    extras = pd.DataFrame(extra_rows, columns=[
        "season", "week", "team", "player_clean_key", "player", "position", "targets"
    ])

    auth_room = cp.groupby(tg, as_index=False).agg(
        authority_room_targets=("actual_targets", "sum"),
        authority_room_rows=("player_clean_key", "size"),
    )
    raw_room = raw_wr.groupby(tg, as_index=False).agg(
        raw_all_wr_targets=("targets", "sum"),
        raw_target_bearing_wr_rows=("player_clean_key", "size"),
    )
    room = auth_room.merge(raw_room, on=tg, how="left", validate="one_to_one")
    room["raw_all_wr_targets"] = room["raw_all_wr_targets"].fillna(0.0)
    room["raw_target_bearing_wr_rows"] = room["raw_target_bearing_wr_rows"].fillna(0).astype(int)
    room["room_target_delta"] = room["raw_all_wr_targets"] - room["authority_room_targets"]
    room["room_total_parity"] = room["room_target_delta"].abs().le(TOL)

    audit = {
        "authority_complete_team_games": int(len(complete_keys)),
        "authority_complete_player_rows": int(len(cp)),
        "raw_wr_rows_on_complete_games": int(len(raw_wr)),
        "player_exact_or_zero_missing_parity_rows": int(player["player_parity"].sum()),
        "player_parity_fail_rows": int((~player["player_parity"]).sum()),
        "positive_target_authority_rows_missing_raw_identity": int(positive_missing.sum()),
        "room_total_parity_team_games": int(room["room_total_parity"].sum()),
        "room_total_parity_fail_team_games": int((~room["room_total_parity"]).sum()),
        "extra_raw_wr_positive_target_rows": int(len(extras)),
        "extra_raw_wr_positive_targets": float(extras["targets"].sum()) if len(extras) else 0.0,
        "max_abs_player_target_delta": float(player["target_delta"].abs().max()) if len(player) else np.nan,
        "max_abs_room_target_delta": float(room["room_target_delta"].abs().max()) if len(room) else np.nan,
        "sportsbook_inputs": 0,
        "receiving_yard_fields_loaded": False,
    }
    passed = bool(
        len(complete_keys) == EXPECTED_COMPLETE_ROOMS
        and audit["player_parity_fail_rows"] == 0
        and audit["positive_target_authority_rows_missing_raw_identity"] == 0
        and audit["room_total_parity_fail_team_games"] == 0
    )
    result = {
        "specification": "WR_PHASE4A_FULL_ROOM_TARGET_SOURCE_RECONCILIATION_V1",
        "disposition": (
            "PHASE4A_FULL_ROOM_TARGET_SOURCE_RECONCILED"
            if passed else "PHASE4A_FULL_ROOM_TARGET_SOURCE_NOT_RECONCILED"
        ),
        "source_reconciliation_passed": passed,
        "phase4_attribution_authorized": passed,
        "challenger_model_authorized": False,
        "production_change": False,
        "sportsbook_inputs": 0,
    }
    return player, room, extras, audit, result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    pred, feat, membership, complete_keys = _load_authority(args.predictions, args.features)
    raw = _load_weekly_actuals((2023, 2024))
    player, room, extras, audit, result = _reconcile(pred, raw, complete_keys)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    membership.to_csv(out / "phase4a_room_membership_audit.csv", index=False)
    player.to_csv(out / "phase4a_player_target_parity.csv", index=False)
    room.to_csv(out / "phase4a_room_target_parity.csv", index=False)
    extras.to_csv(out / "phase4a_extra_raw_wr_contributors.csv", index=False)
    (out / "phase4a_source_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=True) + "\n")
    (out / "phase4a_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=True))
    if not result["source_reconciliation_passed"]:
        print("Phase 4 attribution remains BLOCKED: source reconciliation did not pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
