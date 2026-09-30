#!/usr/bin/env python3
"""Strict identity audit for the one verified 2024-W1 archived WR/CB snapshot.

Input is the sanitized exact-archive verification JSON. No archive acquisition,
provider-ID bridge, sportsbook, outcomes, grading, or model fitting. Only exact
weekly roster identity on the WR team and CB opponent team may pass.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

from scripts._opponent_map import canon_team
from scripts.research.audit_fantasyalarm_wr_cb_source_quality_v1 import (
    DEF_POSITIONS, WR_POSITIONS, _load_rosters, _load_schedule,
)
from scripts.utils.canonical_names import canonicalize_player_name_safe


def name_key(value) -> str:
    return str(canonicalize_player_name_safe(value)[1] or "").strip()


def exact_week_lookup(rosters: pd.DataFrame, positions: set[str]) -> dict[tuple[int,int,str,str], tuple[str,str]]:
    r = rosters.loc[rosters["position"].isin(positions)].copy()
    grouped = r.groupby(["season","week","team","name_key"])["player_id"].agg(
        lambda s: tuple(sorted(set(str(x) for x in s if str(x))))
    )
    out = {}
    for key, ids in grouped.items():
        if len(ids) == 1:
            out[key] = (ids[0], "WEEK_EXACT")
        elif len(ids) > 1:
            out[key] = ("", "COLLISION")
    return out


def audit(snapshot_json: Path, out_dir: Path) -> dict:
    src = json.loads(snapshot_json.read_text(encoding="utf-8"))
    if not src.get("archived_body_digest_verified"):
        raise RuntimeError("archived body digest not verified")
    if src.get("status") != "ARCHIVE_BODY_HASH_MATCH_WITH_STRICT_PREGAME_ROW_CANDIDATES":
        raise RuntimeError("snapshot has no strict pregame factual candidates")
    rows = pd.DataFrame(src.get("factual_pregame_rows") or [])
    if rows.empty:
        raise RuntimeError("zero sanitized pregame factual rows")
    if len(rows) != int(src.get("verified_pregame_factual_rows", -1)):
        raise RuntimeError("snapshot count mismatch")
    for col in ["season","week","wr_raw","wr_team","cb_raw","opponent","alignment_bucket"]:
        if col not in rows:
            raise RuntimeError(f"missing snapshot field {col}")
    rows["season"] = pd.to_numeric(rows["season"],errors="raise").astype(int)
    rows["week"] = pd.to_numeric(rows["week"],errors="raise").astype(int)
    rows["wr_team"] = rows["wr_team"].map(canon_team)
    rows["opponent"] = rows["opponent"].map(canon_team)
    rows["wr_key"] = rows["wr_raw"].map(name_key)
    rows["cb_key"] = rows["cb_raw"].map(name_key)

    rosters = _load_rosters([2024])
    wr_lookup = exact_week_lookup(rosters, WR_POSITIONS)
    cb_lookup = exact_week_lookup(rosters, DEF_POSITIONS)
    schedule = _load_schedule([2024]).rename(columns={"team":"wr_team"})
    rows = rows.merge(
        schedule[["season","week","wr_team","scheduled_opponent"]],
        on=["season","week","wr_team"],how="left",validate="many_to_one",
    )
    rows["schedule_match"] = rows["opponent"].eq(rows["scheduled_opponent"])

    def resolve(row, lookup, team_col, key_col):
        return lookup.get(
            (int(row["season"]),int(row["week"]),str(row[team_col]),str(row[key_col])),
            ("","NOT_FOUND"),
        )
    wr_res = rows.apply(lambda r: resolve(r,wr_lookup,"wr_team","wr_key"),axis=1)
    cb_res = rows.apply(lambda r: resolve(r,cb_lookup,"opponent","cb_key"),axis=1)
    rows["wr_gsis_id"] = [x[0] for x in wr_res]
    rows["wr_roster_status"] = [x[1] for x in wr_res]
    rows["cb_gsis_id"] = [x[0] for x in cb_res]
    rows["cb_roster_status"] = [x[1] for x in cb_res]
    rows["strict_snapshot_source_ready"] = (
        rows["schedule_match"]
        & rows["wr_roster_status"].eq("WEEK_EXACT")
        & rows["cb_roster_status"].eq("WEEK_EXACT")
        & rows["alignment_bucket"].isin(["LWR_VS_RCB","RWR_VS_LCB","SWR_VS_SCB"])
    )
    # No fallback/bridge. Missing or typo identities remain quarantined.
    rows["provider_bridge_used"] = False
    rows["model_feature_eligible"] = False

    dedup = rows.drop_duplicates(
        ["season","week","wr_team","wr_key","opponent","cb_key","alignment_bucket"]
    )
    if len(dedup) != len(rows):
        raise RuntimeError("duplicate factual row keys in archived snapshot")

    out_dir.mkdir(parents=True,exist_ok=True)
    rows.to_csv(out_dir/"verified_snapshot_identity_audit.csv",index=False)
    summary = {
        "contract":"WR_CB_2024W1_VERIFIED_SNAPSHOT_IDENTITY_V1",
        "archive_capture_utc":src.get("archive_index_timestamp_utc"),
        "source_publication_utc":src.get("source_publication_utc"),
        "archived_body_sha256":src.get("body_sha256"),
        "article_body_sha256":(
            (src.get("archived_article_body_structure") or {}).get("candidates") or [{}]
        )[0].get("sha256"),
        "input_pregame_factual_rows":int(len(rows)),
        "schedule_match_rows":int(rows["schedule_match"].sum()),
        "wr_week_exact_rows":int(rows["wr_roster_status"].eq("WEEK_EXACT").sum()),
        "cb_opponent_week_exact_rows":int(rows["cb_roster_status"].eq("WEEK_EXACT").sum()),
        "strict_snapshot_source_ready_rows":int(rows["strict_snapshot_source_ready"].sum()),
        "quarantined_rows":int((~rows["strict_snapshot_source_ready"]).sum()),
        "quarantine_reason_counts":{
            "wr_not_exact":int(rows["wr_roster_status"].ne("WEEK_EXACT").sum()),
            "cb_not_exact_opponent":int(rows["cb_roster_status"].ne("WEEK_EXACT").sum()),
            "schedule_mismatch":int((~rows["schedule_match"]).sum()),
        },
        "provider_bridge_used":False,
        "editorial_grade_used":False,
        "sportsbook_inputs":False,
        "target_game_outcomes":False,
        "parameters_fit":0,
        "source_model_gate_cleared":False,
        "note":"Strict one-snapshot source evidence only; no completeness or predictive validity claim.",
    }
    (out_dir/"verified_snapshot_identity_summary.json").write_text(
        json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8"
    )
    print(json.dumps(summary,indent=2,sort_keys=True))
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--snapshot-json",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args()
    audit(a.snapshot_json,a.out_dir)
