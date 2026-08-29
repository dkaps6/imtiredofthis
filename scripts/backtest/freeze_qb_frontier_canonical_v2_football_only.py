#!/usr/bin/env python3
"""Freeze the original pre-market QB projection layer into canonical v2.

Why v2 exists
-------------
The immutable `qb_frontier_canonical_v1` froze the later BOTH-RAW research
frontier. Review subsequently established that BOTH-RAW attempt and contextual
YPA residual models used sportsbook game-market variables. That makes v1 useful
for historical research provenance, but not an acceptable baseline for a
football-only projection experiment.

The authoritative M69 artifact also retained, inside each season's
`qb_both_raw_walkforward_trace.csv`, the *unmodified pre-market* projection
columns that existed before those residual corrections were applied:

- `mc_proj`       -> original historical Monte Carlo passing-yards projection
- `pred_attempts` -> original MC expected official QB pass attempts
- `pred_ypa`      -> original rules YPA

Those columns were produced by the historical football context pipeline before
M50 attached schedule market fields. This one-time freezer extracts only those
football-side columns plus actual outcomes and identifiers. No market column is
written to v2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SNAPSHOT_ID = "qb_frontier_canonical_v2_football_only"
SOURCE_ARTIFACT_ID = 9717868133
SOURCE_ARTIFACT_DIGEST = "sha256:fbe7500d04e971c0d83ac468cecdf022407c3179e3904334521e380afa4070f4"
EXPECTED_SOURCE_SHA = {
    2024: "7a7fa9ff9c3ee392055ea827fcfc81842d45df60722ce305d2ee3ab1d18c24a3",
    2025: "b0d10478d3be7b333433dbe8a0d3805bc1e3819062a3a9934584052445eeac5b",
}
EXPECTED_SNAPSHOT_SHA256 = "87735c78fb29ef30b3a0acaba54aad5c7e0e61df1c473d84de37529a594c7644"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def met(a, p):
    z = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "p": pd.to_numeric(p, errors="coerce")}).dropna()
    e = z.p - z.a
    return {
        "n": int(len(z)),
        "mae": float(e.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(e)))),
        "bias": float(e.mean()),
        "corr": float(z.a.corr(z.p)) if len(z) > 2 else np.nan,
        "tail100": int(e.abs().ge(100).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    frames = []
    source_meta = {}
    required = {
        "week", "team", "opponent", "player_clean_key", "mc_proj",
        "pred_attempts", "pred_ypa", "actual_pass_att",
        "actual_pass_yards_raw", "actual_ypa",
    }
    for season in (2024, 2025):
        p = a.artifact_root / str(season) / "qb_both_raw" / "qb_both_raw_walkforward_trace.csv"
        if not p.exists():
            raise RuntimeError(f"missing authoritative source trace: {p}")
        got = sha256(p)
        if got != EXPECTED_SOURCE_SHA[season]:
            raise RuntimeError(f"source trace hash drift for {season}: {got}")
        x = pd.read_csv(p, low_memory=False)
        missing = sorted(required - set(x.columns))
        if missing:
            raise RuntimeError(f"{season} trace missing required columns: {missing}")
        x["season"] = season
        frames.append(x)
        source_meta[str(season)] = {
            "relative_path": f"{season}/qb_both_raw/qb_both_raw_walkforward_trace.csv",
            "rows": int(len(x)),
            "sha256": got,
        }

    x = pd.concat(frames, ignore_index=True, sort=False)
    snap = pd.DataFrame({
        "season": pd.to_numeric(x.season, errors="coerce").astype(int),
        "week": pd.to_numeric(x.week, errors="coerce").astype(int),
        "team": x.team.astype(str),
        "opponent": x.opponent.astype(str),
        "player_clean_key": x.player_clean_key.astype(str),
        "actual_pass_yards": pd.to_numeric(x.actual_pass_yards_raw, errors="coerce"),
        "actual_attempts": pd.to_numeric(x.actual_pass_att, errors="coerce"),
        "actual_ypa": pd.to_numeric(x.actual_ypa, errors="coerce"),
        "pred_pass_yards": pd.to_numeric(x.mc_proj, errors="coerce"),
        "pred_attempts": pd.to_numeric(x.pred_attempts, errors="coerce"),
        "pred_ypa": pd.to_numeric(x.pred_ypa, errors="coerce"),
    })
    snap["implied_pred_ypa"] = snap.pred_pass_yards / snap.pred_attempts.replace(0, np.nan)
    snap["det_pass_yards"] = snap.pred_attempts * snap.pred_ypa
    snap = snap.sort_values(["season", "week", "team", "player_clean_key"]).reset_index(drop=True)

    if len(snap) != 643:
        raise RuntimeError(f"v2 expected 643 stable-QB rows, got {len(snap)}")
    if snap.duplicated(["season", "week", "team", "player_clean_key"]).any():
        raise RuntimeError("v2 key uniqueness failed")
    if snap.season.value_counts().to_dict() != {2024: 332, 2025: 311}:
        raise RuntimeError(f"v2 season invariant failed: {snap.season.value_counts().to_dict()}")

    # No market-derived field is permitted in this snapshot by construction.
    prohibited = [c for c in snap.columns if any(k in c.lower() for k in ["market", "spread", "moneyline", "implied_total"])]
    if prohibited:
        raise RuntimeError(f"market-derived fields leaked into v2: {prohibited}")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = a.out_dir / f"{SNAPSHOT_ID}.csv"
    snap.to_csv(csv_path, index=False, float_format="%.10g")
    got_sha = sha256(csv_path)
    if got_sha != EXPECTED_SNAPSHOT_SHA256:
        raise RuntimeError(f"v2 snapshot hash drift: {got_sha}")

    # Quantify the football-only baseline and the two simple hindsight oracles.
    actual_yards = snap.actual_pass_yards
    implied_y = snap.implied_pred_ypa
    actual_y = snap.actual_ypa
    oracle_attempts = snap.actual_attempts * implied_y
    oracle_ypa = snap.pred_attempts * actual_y
    current = met(actual_yards, snap.pred_pass_yards)
    oa = met(actual_yards, oracle_attempts)
    oy = met(actual_yards, oracle_ypa)
    metrics = pd.DataFrame([
        {"candidate": "football_only_current", **current, "mae_gain_vs_current": 0.0},
        {"candidate": "oracle_actual_attempts", **oa, "mae_gain_vs_current": current["mae"] - oa["mae"]},
        {"candidate": "oracle_actual_ypa", **oy, "mae_gain_vs_current": current["mae"] - oy["mae"]},
    ])
    metrics.to_csv(a.out_dir / "football_only_oracle_summary.csv", index=False)

    manifest = {
        "snapshot_id": SNAPSHOT_ID,
        "schema_version": 2,
        "row_count": int(len(snap)),
        "seasons": [2024, 2025],
        "snapshot_file": csv_path.name,
        "snapshot_sha256": got_sha,
        "source_artifact_id": SOURCE_ARTIFACT_ID,
        "source_artifact_digest": SOURCE_ARTIFACT_DIGEST,
        "source_traces": source_meta,
        "projection_definition": {
            "pred_pass_yards": "original pre-market mc_proj from historical component_predictions/predict_week",
            "pred_attempts": "original pre-market mc_expected_pass_attempts carried as pred_attempts",
            "pred_ypa": "original pre-market mc_rules_ypa carried as pred_ypa",
        },
        "market_boundary": "No sportsbook/game-market field or market-trained residual correction is included in canonical v2.",
        "purpose": "Football-only QB research foundation after discovering that canonical v1 BOTH-RAW corrections used game-market inputs.",
    }
    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"[{SNAPSHOT_ID}] rows={len(snap)} sha256={got_sha}")
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
