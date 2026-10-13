#!/usr/bin/env python3
"""Archive immutable market snapshots from an already-produced Full Slate artifact.

This script never fetches odds and never changes the existing latest-row market
ledger. It preserves each paid/live Full Slate market state append-only so
later same-book pregame movement / CLV can be measured honestly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.artifact_contracts import get_contract

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "data" / "market_track_record" / "snapshots"
PRIMARY_KEY = ["event_id", "player_clean_key", "market", "book", "side"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc(series: pd.Series) -> pd.Series:
    x = pd.to_datetime(series, utc=True, errors="coerce")
    return x


def _load_priced(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"priced board missing/empty: {path}")
    df = pd.read_csv(path)
    contract = get_contract("props_priced")
    missing = sorted(set(contract.required_columns) - set(df.columns))
    if missing:
        raise RuntimeError(f"priced board missing required columns: {missing}")
    missing_key = sorted(set(PRIMARY_KEY) - set(df.columns))
    if missing_key:
        raise RuntimeError(f"priced board missing snapshot key columns: {missing_key}")
    if df.duplicated(PRIMARY_KEY).any():
        sample = df.loc[df.duplicated(PRIMARY_KEY, keep=False), PRIMARY_KEY + ["vegas_line"]].head(20)
        raise RuntimeError(f"priced board has duplicate primary snapshot keys: {sample.to_dict('records')}")
    return df


def _event_metadata(raw_path: Path) -> pd.DataFrame:
    if not raw_path.exists() or not raw_path.stat().st_size:
        raise RuntimeError(f"raw props missing/empty: {raw_path}")
    raw = pd.read_csv(raw_path, usecols=lambda c: c in {
        "event_id", "fetched_at", "commence_time"
    })
    required = {"event_id", "fetched_at", "commence_time"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"raw props missing timestamp columns: {missing}")
    if raw.empty:
        raise RuntimeError("raw props contains no rows")

    raw["odds_fetched_at_utc"] = _utc(raw["fetched_at"])
    raw["commence_time_utc"] = _utc(raw["commence_time"])
    if raw["odds_fetched_at_utc"].isna().any() or raw["commence_time_utc"].isna().any():
        raise RuntimeError("raw props contains invalid fetched_at / commence_time timestamps")

    rows = []
    for event_id, g in raw.groupby("event_id", dropna=False):
        f = g["odds_fetched_at_utc"].drop_duplicates()
        k = g["commence_time_utc"].drop_duplicates()
        if len(f) != 1:
            raise RuntimeError(f"event {event_id} has {len(f)} fetched_at timestamps")
        if len(k) != 1:
            raise RuntimeError(f"event {event_id} has {len(k)} commence_time timestamps")
        rows.append({
            "event_id": event_id,
            "odds_fetched_at_utc": f.iloc[0],
            "commence_time_utc": k.iloc[0],
        })
    return pd.DataFrame(rows)


def build_snapshot(
    priced_path: Path,
    raw_path: Path,
    *,
    source_run_id: str,
    source_git_sha: str,
) -> pd.DataFrame:
    priced = _load_priced(priced_path)
    meta = _event_metadata(raw_path)
    out = priced.merge(meta, on="event_id", how="left", validate="many_to_one")
    if out["odds_fetched_at_utc"].isna().any() or out["commence_time_utc"].isna().any():
        sample = out.loc[
            out["odds_fetched_at_utc"].isna() | out["commence_time_utc"].isna(),
            PRIMARY_KEY,
        ].head(20).to_dict("records")
        raise RuntimeError(f"priced event ids missing raw event metadata: {sample}")

    out["minutes_to_kickoff"] = (
        (out["commence_time_utc"] - out["odds_fetched_at_utc"]).dt.total_seconds() / 60.0
    )
    out["pregame_snapshot_valid"] = out["minutes_to_kickoff"].gt(0.0)
    out["source_run_id"] = str(source_run_id)
    out["source_git_sha"] = str(source_git_sha)
    out["snapshot_id"] = (
        "run_" + str(source_run_id) + "__" +
        out["odds_fetched_at_utc"].dt.strftime("%Y%m%dT%H%M%SZ")
    )
    for c in ("odds_fetched_at_utc", "commence_time_utc"):
        out[c] = out[c].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


def archive_snapshot(
    *,
    priced_path: Path,
    raw_path: Path,
    source_run_id: str,
    source_git_sha: str,
    snapshot_root: Path,
    season: int | None = None,
    week: int | None = None,
) -> dict:
    snap = build_snapshot(
        priced_path, raw_path,
        source_run_id=source_run_id,
        source_git_sha=source_git_sha,
    )

    season_values = pd.to_numeric(snap.get("season"), errors="coerce").dropna().unique()
    week_values = pd.to_numeric(snap.get("week"), errors="coerce").dropna().unique()
    if len(season_values) != 1 or len(week_values) != 1:
        raise RuntimeError(
            f"snapshot board must contain exactly one season/week: seasons={season_values}, weeks={week_values}"
        )
    board_season = int(season_values[0])
    board_week = int(week_values[0])
    if season is not None and int(season) != board_season:
        raise RuntimeError(f"requested season {season} != board season {board_season}")
    if week is not None and int(week) != board_week:
        raise RuntimeError(f"requested week {week} != board week {board_week}")

    out_dir = snapshot_root / f"{board_season}_wk{board_week:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"run_{source_run_id}.csv"
    json_path = out_dir / f"run_{source_run_id}.json"

    tmp = out_dir / f".run_{source_run_id}.tmp.csv"
    snap.to_csv(tmp, index=False)
    new_hash = _sha256(tmp)

    if csv_path.exists():
        old_hash = _sha256(csv_path)
        tmp.unlink()
        if old_hash != new_hash:
            raise RuntimeError(
                f"immutable snapshot collision for run {source_run_id}: {old_hash} != {new_hash}"
            )
        status = "already_archived_identical"
    else:
        tmp.replace(csv_path)
        status = "archived"

    valid = snap.loc[snap["pregame_snapshot_valid"]]
    # The on-disk manifest describes immutable source bytes, not the
    # idempotent caller outcome. Re-archiving identical bytes may return
    # "already_archived_identical", but must reproduce the exact same manifest.
    manifest = {
        "contract": "MARKET_SNAPSHOT_HISTORY_CLV_CAPTURE_V1",
        "archive_record_state": "IMMUTABLE_ARCHIVED_SOURCE",
        "season": board_season,
        "week": board_week,
        "source_run_id": str(source_run_id),
        "source_git_sha": str(source_git_sha),
        "source_priced_sha256": _sha256(priced_path),
        "source_raw_sha256": _sha256(raw_path),
        "snapshot_csv_sha256": _sha256(csv_path),
        "rows": int(len(snap)),
        "events": int(snap["event_id"].nunique()),
        "pregame_valid_rows": int(snap["pregame_snapshot_valid"].sum()),
        "postkickoff_or_invalid_rows": int((~snap["pregame_snapshot_valid"]).sum()),
        "capture_times_utc": sorted(snap["odds_fetched_at_utc"].drop_duplicates().tolist()),
        "min_minutes_to_kickoff": float(valid["minutes_to_kickoff"].min()) if len(valid) else None,
        "max_minutes_to_kickoff": float(valid["minutes_to_kickoff"].max()) if len(valid) else None,
        "odds_fetch_triggered_by_archiver": False,
    }
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if json_path.exists() and json_path.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"immutable manifest collision for run {source_run_id}")
    json_path.write_text(payload, encoding="utf-8")
    return {
        **manifest,
        "snapshot_state": "IMMUTABLE_CAPTURE",
        "operation_status": status,
        "csv_path": str(csv_path),
        "manifest_path": str(json_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--priced-file", type=Path, required=True)
    ap.add_argument("--raw-props-file", type=Path, required=True)
    ap.add_argument("--source-run-id", required=True)
    ap.add_argument("--source-git-sha", required=True)
    ap.add_argument("--snapshot-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    a = ap.parse_args()
    result = archive_snapshot(
        priced_path=a.priced_file,
        raw_path=a.raw_props_file,
        source_run_id=a.source_run_id,
        source_git_sha=a.source_git_sha,
        snapshot_root=a.snapshot_root,
        season=a.season,
        week=a.week,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
