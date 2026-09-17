"""Phase-0 artifact persistence and source-window QA for BDB 2024.

This module is deliberately data-engineering only. It does not alter the frozen
contact detector and computes no predictive/model/sportsbook metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from scripts.data_frontier.bdb_2024_contact import (
        FEATURE_VERSION,
        build_qa,
        derive_contact_features,
        load_inputs,
    )
except ModuleNotFoundError:  # direct script execution
    from bdb_2024_contact import FEATURE_VERSION, build_qa, derive_contact_features, load_inputs


def event_window_diagnostics(tracking: pd.DataFrame) -> dict:
    """Describe source event/window coverage without inventing missing events."""
    play_keys = tracking[["gameId", "playId"]].drop_duplicates()
    n_plays = int(len(play_keys))
    if "event" not in tracking.columns:
        return {
            "tracking_plays": n_plays,
            "event_column_present": False,
            "event_counts": {},
            "play_event_presence": {},
        }

    x = tracking[["gameId", "playId", "frameId", "event"]].copy()
    x["event_norm"] = x["event"].astype("string").str.strip().str.lower()
    x = x.loc[x["event_norm"].notna() & x["event_norm"].ne("") & x["event_norm"].ne("<na>")]
    unique_events = x.drop_duplicates(["gameId", "playId", "frameId", "event_norm"])
    event_counts = unique_events["event_norm"].value_counts().sort_index()

    families = {
        "ball_snap": {"ball_snap"},
        "handoff": {"handoff"},
        "tackle": {"tackle"},
        "out_of_bounds": {"out_of_bounds"},
        "touchdown": {"touchdown"},
        "fumble": {"fumble"},
        "qb_slide": {"qb_slide"},
    }
    presence = {}
    for name, labels in families.items():
        keys = unique_events.loc[unique_events["event_norm"].isin(labels), ["gameId", "playId"]].drop_duplicates()
        count = int(len(keys))
        presence[name] = {
            "plays": count,
            "rate": (count / n_plays) if n_plays else None,
        }

    frames = tracking.groupby(["gameId", "playId"], as_index=False)["frameId"].agg(["min", "max", "nunique"]).reset_index()
    frame_counts = frames["nunique"] if len(frames) else pd.Series(dtype=float)
    return {
        "tracking_plays": n_plays,
        "event_column_present": True,
        "event_counts": {str(k): int(v) for k, v in event_counts.items()},
        "play_event_presence": presence,
        "frames_per_play": {
            "min": int(frame_counts.min()) if len(frame_counts) else None,
            "median": float(frame_counts.median()) if len(frame_counts) else None,
            "max": int(frame_counts.max()) if len(frame_counts) else None,
        },
        "interpretation": "BDB 2024 is event-window filtered; absence of an event from a play is not evidence that the football event did not occur outside the published window.",
    }


def persist_normalized(out_dir: Path, tracking: pd.DataFrame, plays: pd.DataFrame, tackles: pd.DataFrame) -> dict:
    """Persist normalized source tables as auditable CSVs with stable sorting."""
    norm_dir = out_dir / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "tracking": tracking.sort_values(["gameId", "playId", "frameId", "club", "nflId"], na_position="last"),
        "plays": plays.sort_values(["gameId", "playId"]),
        "tackles": tackles.sort_values(["gameId", "playId", "nflId"]),
    }
    manifest = {}
    for name, df in tables.items():
        path = norm_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        manifest[name] = {"path": str(path), "rows": int(len(df)), "columns": list(map(str, df.columns))}
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Persist BDB 2024 normalized tables and Phase-0 QA artifacts.")
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    tracking, plays, tackles, source_manifest = load_inputs(args.input_dir)
    features, dispositions = derive_contact_features(tracking, plays, tackles)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    normalized_manifest = persist_normalized(args.out_dir, tracking, plays, tackles)
    features.to_csv(args.out_dir / "rb_contact_features.csv", index=False)
    dispositions.to_csv(args.out_dir / "benchmark_dispositions.csv", index=False)

    qa = build_qa(features, dispositions, source_manifest)
    qa["source_window_diagnostics"] = event_window_diagnostics(tracking)
    qa["normalized_artifacts"] = normalized_manifest
    qa["contact_detector_changed"] = False
    qa["feature_version"] = FEATURE_VERSION
    (args.out_dir / "qa_summary.json").write_text(json.dumps(qa, indent=2, sort_keys=True), encoding="utf-8")
    (args.out_dir / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"feature_version": FEATURE_VERSION, "attempted_plays": len(dispositions), "scoreable_plays": int((dispositions["benchmark_disposition"] == "SCOREABLE").sum()), "normalized": normalized_manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
