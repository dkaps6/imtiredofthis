#!/usr/bin/env python3
"""Prepare football-only primary-QB identity for the real-slate QB-C2 shadow.

The historical C2 helper expects qb_projection_eligible/qb_role_score fields.
The production 469-player universe currently carries Ourlads depth_role instead.
This adapter derives exactly one primary QB per team from *football-only*
depth_role, writes a temporary universe, and delegates to the frozen V2 shadow.
No sportsbook identity is used to choose a quarterback and production pricing is
not modified.

Before delegation, the downstream paid-snapshot pass-yard offer identity is
compared against the football-only QB1 authority. A mismatch is classified as a
stale downstream-offer blocker rather than a C2 scientific failure. This keeps
sportsbook identity downstream and prevents an old offer row from defining the
quarterback being tested.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from scripts.utils.player_identity_v3 import player_name_key


def _role_rank(value: object) -> int:
    text = str(value or "").upper().replace(" ", "")
    if text in {"QB1", "QB01"} or "START" in text or "FIRST" in text:
        return 1
    if text in {"QB2", "QB02"} or "SECOND" in text:
        return 2
    if text in {"QB3", "QB03"} or "THIRD" in text:
        return 3
    return 99


def _key(value: object) -> str:
    try:
        return str(player_name_key(value, strip_suffix=True) or "").strip()
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", required=True)
    ap.add_argument("--priced", required=True)
    ap.add_argument("--state-context", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--result", required=True)
    args = ap.parse_args()

    universe = pd.read_csv(args.universe, low_memory=False)
    priced = pd.read_csv(args.priced, low_memory=False)
    universe.columns = [str(c).lower() for c in universe.columns]
    priced.columns = [str(c).lower() for c in priced.columns]
    if "depth_role" not in universe.columns or "position" not in universe.columns:
        raise RuntimeError("football universe missing Ourlads depth_role/position")

    universe["qb_projection_eligible"] = 0
    universe["qb_role_score"] = np.nan
    universe["qb_role_source"] = "not_qb"

    qb_mask = universe["position"].astype(str).str.upper().eq("QB")
    audit_rows = []
    for team, part in universe.loc[qb_mask].groupby("team", sort=True):
        ranked = part.copy()
        ranked["_depth_rank"] = ranked["depth_role"].map(_role_rank)
        ranked = ranked.sort_values(["_depth_rank", "player_clean_key"], kind="mergesort")
        if ranked.empty or int(ranked.iloc[0]["_depth_rank"]) != 1:
            raise RuntimeError(f"team={team} has no football-only QB1 depth role")
        if int((ranked["_depth_rank"] == 1).sum()) != 1:
            sample = ranked.loc[ranked["_depth_rank"].eq(1), ["player","depth_role"]].to_dict("records")
            raise RuntimeError(f"team={team} ambiguous football-only QB1: {sample}")
        primary_idx = ranked.index[0]
        universe.loc[ranked.index, "qb_role_score"] = -ranked["_depth_rank"].astype(float).to_numpy()
        universe.loc[ranked.index, "qb_role_source"] = "ourlads_depth_role"
        universe.at[primary_idx, "qb_projection_eligible"] = 1
        audit_rows.append({
            "team": str(team),
            "primary_player": str(universe.at[primary_idx,"player"]),
            "primary_player_key": _key(universe.at[primary_idx,"player"]),
            "depth_role": str(universe.at[primary_idx,"depth_role"]),
            "sportsbook_inputs_used": 0,
        })

    audit = pd.DataFrame(audit_rows)
    if len(audit) != 32 or audit["team"].nunique() != 32:
        raise RuntimeError(f"football-only QB1 authority must cover 32 teams, got {len(audit)}")
    if not audit["sportsbook_inputs_used"].eq(0).all():
        raise RuntimeError("football-only QB1 authority leakage flag")

    # Downstream identity audit only. The paid artifact is intentionally old and
    # may contain a pass-yard offer for a player who is no longer the football QB1.
    pass_rows = priced.loc[priced["market"].astype(str).str.lower().eq("pass_yards")].copy()
    pass_rows["priced_player_key"] = pass_rows["player"].map(_key)
    pass_rows = pass_rows.sort_values(["team","player","side"], kind="mergesort").drop_duplicates(
        ["team","priced_player_key"], keep="first"
    )
    offered = pass_rows.groupby("team", sort=True).agg(
        priced_qb_count=("priced_player_key","nunique"),
        priced_primary_player=("player","first"),
        priced_primary_player_key=("priced_player_key","first"),
    ).reset_index()
    audit = audit.merge(offered, on="team", how="left", validate="one_to_one")
    audit["priced_qb_count"] = pd.to_numeric(audit["priced_qb_count"], errors="coerce").fillna(0).astype(int)
    audit["downstream_identity_match"] = (
        audit["priced_qb_count"].eq(1)
        & audit["priced_primary_player_key"].fillna("").eq(audit["primary_player_key"])
    )
    audit.to_csv("data/qb_c2_shadow_primary_qb_audit.csv", index=False)

    mismatches = audit.loc[~audit["downstream_identity_match"], [
        "team","primary_player","priced_primary_player","priced_qb_count"
    ]].to_dict("records")
    if mismatches:
        result = {
            "disposition": "QB_DISTRIBUTION_FULL_ROSTER_SHADOW_BLOCKED_STALE_DOWNSTREAM_IDENTITY",
            "football_only_qb1_teams": 32,
            "downstream_identity_mismatch_teams": int(len(mismatches)),
            "mismatches": mismatches,
            "sportsbook_inputs_to_qb_selection": 0,
            "production_pricing_modified": 0,
            "scientific_candidate_rejected": False,
            "reason": "paid replay artifact pass-yard player identity does not match current football-only QB1 authority",
        }
        Path(args.result).parent.mkdir(parents=True, exist_ok=True)
        Path(args.result).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(3)

    tmp = Path("data/qb_c2_shadow_universe_with_primary_identity.csv")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(tmp, index=False)

    cmd = [
        sys.executable,
        "scripts/run_qb_distribution_shadow_full_roster_v2.py",
        "--universe", str(tmp),
        "--priced", args.priced,
        "--state-context", args.state_context,
        "--out", args.out,
        "--result", args.result,
    ]
    print("[qb_c2_primary_identity] football_only_qb1_teams=32 sportsbook_inputs_used=0 downstream_identity_match=32")
    return int(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
