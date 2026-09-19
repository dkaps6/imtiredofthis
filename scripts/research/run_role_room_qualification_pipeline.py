#!/usr/bin/env python3
"""Run the outcome-free role/room qualification pipeline on canonical history.

This orchestrator deliberately stops before predictive science. It materializes strict-prior
player usage, room continuity, their canonical join, transition diagnostics, stability evidence,
and qualification-ready candidate profiles. Target-game outcomes and sportsbook data are never
read by this script.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("[role_room_pipeline]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--historical", type=Path, required=True, help="canonical historical player-game CSV")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--python", default=sys.executable)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    s = root / "scripts" / "research"
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)

    usage = out / "usage_regime_context.csv"
    room = out / "room_continuity_context.csv"
    joined = out / "role_room_context.csv"
    transition = out / "role_room_transition_detail.csv"
    transition_summary = out / "role_room_transition_summary.csv"
    stability = out / "role_room_stability.csv"
    profile = out / "role_room_candidate_profile.csv"

    run([a.python, str(s / "build_usage_regime_context.py"), "--input", str(a.historical), "--out", str(usage)])
    run([a.python, str(s / "build_room_continuity_context.py"), "--input", str(a.historical), "--out", str(room)])
    run([a.python, str(s / "build_role_room_context.py"), "--usage", str(usage), "--room", str(room), "--out", str(joined)])
    run([a.python, str(s / "build_role_room_transition_diagnostics.py"), "--input", str(joined), "--detail-out", str(transition), "--summary-out", str(transition_summary)])

    features = ",".join([
        "prior_tgt_share_game", "rolling3_tgt_share", "rolling5_tgt_share",
        "prior_rush_share_game", "rolling3_rush_share", "rolling5_rush_share",
        "prior_tgt_share_game_top1", "prior_tgt_share_game_top2",
        "prior_rush_share_game_top1", "prior_rush_share_game_top2",
        "returning_target_opportunity_overlap", "returning_rush_opportunity_overlap",
    ])
    run([a.python, str(s / "build_context_stability_evidence.py"), "--input", str(joined),
         "--entity-key", "player_identity_key", "--features", features, "--out", str(stability)])
    run([a.python, str(s / "build_context_candidate_profile.py"), "--input", str(joined),
         "--features", features, "--family", "ROLE_ROOM_CONTEXT_V1", "--grain", "player_game",
         "--key-cols", "season,week,team,player_identity_key", "--eligible-col", "pregame_context_eligible_flag",
         "--stable-id-col", "stable_identity_flag", "--unknown-col", "any_context_unknown_flag",
         "--prior-support-col", "strict_prior_support_games", "--stability", str(stability),
         "--intended-component", "player opportunity entitlement and regime uncertainty",
         "--mechanism-note", "strict-prior player usage plus position-room continuity can identify stale-history regimes",
         "--redundancy-notes", "requires downstream comparison with canonical production opportunity inputs before experiment",
         "--out", str(profile)])

    print(f"[role_room_pipeline] complete -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
