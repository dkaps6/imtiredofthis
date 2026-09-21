#!/usr/bin/env python3
"""Assemble immutable RB-PD2 forward pregame locks from an accepted §7 session."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build._schedule_utils import get_nfl_schedule
from scripts.research import rb_pd2_shadow_capture_v1 as capture
from scripts.research.rb_pd2_forward_shadow_v1 import (
    history_state_digest,
    lock_row,
)

LOCK_MANIFEST = "rb_pd2_forward_locks.jsonl"
LOCK_ARRAYS = "rb_pd2_forward_lock_arrays.npz"
LOCK_RECEIPT = "rb_pd2_forward_lock_receipt.json"


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _array_key(prefix: str, football_key: str) -> str:
    digest = hashlib.sha256(football_key.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}_{digest}"


def _strict_json(value) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _load_schedule(records: list[dict], schedule_csv: Path | None) -> pd.DataFrame:
    if schedule_csv is not None:
        if not schedule_csv.exists() or schedule_csv.stat().st_size == 0:
            raise RuntimeError(f"missing schedule CSV: {schedule_csv}")
        return pd.read_csv(schedule_csv)

    seasons = sorted({int(r["season"]) for r in records})
    frames = [get_nfl_schedule(season) for season in seasons]
    return pd.concat(frames, ignore_index=True)


def _assert_history_current_for_capture(
    history: pd.DataFrame,
    history_manifest: dict,
    records: list[dict],
) -> dict:
    """Require completed predictor history through exactly target_week - 1."""
    targets = {
        (int(r["season"]), int(r["week"]))
        for r in records
        if r.get("baseline_lock_eligible")
    }
    if not targets:
        return {"target_season": None, "target_week": None, "history_completed_through_week": None}
    if len(targets) != 1:
        raise RuntimeError(f"capture session spans multiple target season/weeks: {sorted(targets)}")

    target_season, target_week = next(iter(targets))
    weeks = sorted(
        pd.to_numeric(
            history.loc[
                pd.to_numeric(history["season"], errors="coerce").eq(target_season),
                "week",
            ],
            errors="coerce",
        ).dropna().astype(int).unique().tolist()
    )
    through = max(weeks) if weeks else 0
    if weeks and weeks != list(range(1, through + 1)):
        raise RuntimeError(
            f"current-season history is not contiguous through target: weeks={weeks}"
        )

    manifest_field = (
        "completed_2026_through_week"
        if target_season == 2026
        else f"completed_{target_season}_through_week"
    )
    declared = history_manifest.get(manifest_field)
    if declared is None:
        raise RuntimeError(f"history manifest missing {manifest_field}")
    try:
        declared_int = int(declared)
    except Exception as exc:
        raise RuntimeError(f"history manifest has malformed {manifest_field}: {declared!r}") from exc
    if declared_int != through:
        raise RuntimeError(
            f"history manifest/state completed-through mismatch: manifest={declared_int} state={through}"
        )

    required = max(0, int(target_week) - 1)
    if through != required:
        raise RuntimeError(
            f"stale or future predictor history for target Week {target_week}: "
            f"requires completed through Week {required}, found Week {through}"
        )
    return {
        "target_season": int(target_season),
        "target_week": int(target_week),
        "history_completed_through_week": int(through),
    }


def assemble(
    *,
    session_dir: Path,
    history_state_path: Path,
    history_manifest_path: Path,
    prospective_start_utc: str,
    out_dir: Path,
    schedule_csv: Path | None = None,
    lock_timestamp_utc: str | None = None,
) -> dict:
    records, receipt = capture.load_session(session_dir)
    capture.assert_session_valid(receipt)

    history = pd.read_csv(history_state_path, low_memory=False)
    history_manifest = _read_json(history_manifest_path)
    actual_history_digest = history_state_digest(history)
    if actual_history_digest != str(history_manifest.get("history_state_sha256") or ""):
        raise RuntimeError("history-state file does not match manifest digest")
    history_alignment = _assert_history_current_for_capture(
        history, history_manifest, records
    )

    schedule = _load_schedule(records, schedule_csv)
    lock_ts = (
        pd.to_datetime(lock_timestamp_utc, utc=True, errors="raise")
        if lock_timestamp_utc is not None
        else pd.Timestamp(datetime.now(timezone.utc))
    )

    locked_rows: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    population_exclusions: list[dict] = []
    integrity_failures: list[dict] = []

    for record in sorted(records, key=lambda r: str(r.get("football_key") or "")):
        football_key = str(record.get("football_key") or "")
        if not record.get("baseline_lock_eligible"):
            population_exclusions.append({
                "football_key": football_key,
                "reason": str(record.get("baseline_lock_ineligible_reason") or "baseline_lock_ineligible"),
            })
            continue

        try:
            baseline = capture.load_draws(session_dir, record)
            lock, candidate = lock_row(
                capture_record=record,
                capture_receipt=receipt,
                baseline_draws=baseline,
                history_state=history,
                history_manifest=history_manifest,
                schedule=schedule,
                prospective_start_utc=prospective_start_utc,
                lock_timestamp_utc=lock_ts,
            )
        except RuntimeError as exc:
            message = str(exc)
            if message.startswith("target is outside scientific population"):
                population_exclusions.append({
                    "football_key": football_key,
                    "reason": message,
                })
                continue
            if "predates frozen prospective start boundary" in message:
                population_exclusions.append({
                    "football_key": football_key,
                    "reason": "PRE_PROSPECTIVE_START",
                })
                continue
            integrity_failures.append({
                "football_key": football_key,
                "reason": message,
            })
            continue

        base_key = _array_key("baseline", football_key)
        cand_key = _array_key("candidate", football_key)
        arrays[base_key] = np.asarray(baseline, dtype=np.float64).copy()
        arrays[cand_key] = np.asarray(candidate, dtype=np.float64).copy()
        lock["baseline_array_file"] = LOCK_ARRAYS
        lock["baseline_array_key"] = base_key
        lock["candidate_array_file"] = LOCK_ARRAYS
        lock["candidate_array_key"] = cand_key
        locked_rows.append(lock)

    # Any mechanical/integrity defect invalidates the session-level lock
    # artifact. Population exclusions are expected and do not invalidate it.
    valid = len(integrity_failures) == 0
    if not valid:
        locked_rows = []
        arrays = {}

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / LOCK_ARRAYS, **arrays)
    with (out_dir / LOCK_MANIFEST).open("w", encoding="utf-8") as fh:
        for row in locked_rows:
            fh.write(_strict_json(row) + "\n")

    result = {
        "version": "RB_PD2_FORWARD_LOCK_SESSION_V1",
        "valid": valid,
        "source_capture_session": str(receipt.get("session_id") or ""),
        "source_capture_dir": str(session_dir),
        "history_state_path": str(history_state_path),
        "history_manifest_path": str(history_manifest_path),
        "history_state_sha256": actual_history_digest,
        **history_alignment,
        "prospective_start_utc": pd.to_datetime(
            prospective_start_utc, utc=True, errors="raise"
        ).isoformat(),
        "lock_timestamp_utc": pd.Timestamp(lock_ts).isoformat(),
        "capture_rows": len(records),
        "locked_rows": len(locked_rows),
        "population_exclusions": population_exclusions,
        "integrity_failures": integrity_failures,
        "production_changed": False,
        "sportsbook_inputs_used_in_candidate": False,
        "outcome_present_at_lock": False,
    }
    (out_dir / LOCK_RECEIPT).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--session-dir", type=Path, required=True)
    p.add_argument("--history-state", type=Path, required=True)
    p.add_argument("--history-manifest", type=Path, required=True)
    p.add_argument("--prospective-start-utc", required=True)
    p.add_argument("--schedule-csv", type=Path, default=None)
    p.add_argument("--lock-timestamp-utc", default=None)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()

    result = assemble(
        session_dir=a.session_dir,
        history_state_path=a.history_state,
        history_manifest_path=a.history_manifest,
        prospective_start_utc=a.prospective_start_utc,
        schedule_csv=a.schedule_csv,
        lock_timestamp_utc=a.lock_timestamp_utc,
        out_dir=a.out_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
