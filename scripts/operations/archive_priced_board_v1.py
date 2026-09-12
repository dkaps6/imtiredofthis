#!/usr/bin/env python3
"""Archive a Full Slate priced board into the durable market track record.

This closes the gap identified while auditing the parked historical
market-certification lane: the repository had extensive machinery for
grading a football projection against Vegas, but no live/forward track
record, because paid-live-odds Full Slate runs discarded ``outputs/props_priced_clean.csv``
instead of keeping a permanent copy.

This script is purely downstream and read-only with respect to the football
pipeline: it only copies rows that a Full Slate run already produced and
already paid for (no new sportsbook fetch, no new API credits) into a durable,
git-tracked ledger under ``data/market_track_record/``. It never feeds
anything back into pricing, projection, or model selection.

One board row is written per (season, week, game/event id, player, market,
side, book) key. Running this again for the same week (e.g. a later capture
closer to kickoff) replaces the prior row for the same key rather than
duplicating it, so the archived row always reflects the latest available
capture for that slate -- the closest approximation of a closing line this
project can produce without a dedicated market-monitoring feed.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts.artifact_contracts import get_contract

ROOT = Path(__file__).resolve().parents[2]
LEDGER_DIR = ROOT / "data" / "market_track_record" / "boards"

NATURAL_KEY = ["season", "week", "player", "market", "side"]
OPTIONAL_KEY_EXTRAS = ["event_id", "book"]

PROVENANCE_COLUMNS = [
    "archived_at_utc",
    "source_run_id",
    "source_git_sha",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ledger_path(season: int, week: int) -> Path:
    return LEDGER_DIR / f"{int(season)}_wk{int(week):02d}.csv"


def load_priced_board(path: Path) -> pd.DataFrame:
    contract = get_contract("props_priced")
    if not path.exists() or not path.stat().st_size:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    missing = sorted(set(contract.required_columns) - set(df.columns))
    if missing:
        raise RuntimeError(f"priced board missing required columns: {missing}")
    return df


def stamp_provenance(df: pd.DataFrame, *, run_id: str, git_sha: str) -> pd.DataFrame:
    out = df.copy()
    out["archived_at_utc"] = _now_iso()
    out["source_run_id"] = str(run_id)
    out["source_git_sha"] = str(git_sha)
    return out


def key_columns(df: pd.DataFrame) -> list[str]:
    return NATURAL_KEY + [c for c in OPTIONAL_KEY_EXTRAS if c in df.columns]


def merge_into_ledger(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Replace-by-key merge: the newest capture for a key wins."""
    if existing.empty:
        return incoming.copy()
    if incoming.empty:
        return existing.copy()
    keys = key_columns(incoming)
    keys = [k for k in keys if k in existing.columns]
    combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    combined = combined.sort_values("archived_at_utc")
    combined = combined.drop_duplicates(subset=keys, keep="last")
    return combined.reset_index(drop=True)


def archive(*, priced_path: Path, season: int, week: int, run_id: str, git_sha: str) -> dict:
    board = load_priced_board(priced_path)
    if board.empty:
        return {
            "status": "nothing_to_archive",
            "reason": "priced board missing/empty (expected in no-live-odds mode)",
            "rows_archived": 0,
        }

    board = board.copy()
    board["season"] = season
    board["week"] = week
    stamped = stamp_provenance(board, run_id=run_id, git_sha=git_sha)

    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path = _ledger_path(season, week)
    existing = pd.read_csv(ledger_path) if ledger_path.exists() else pd.DataFrame()
    merged = merge_into_ledger(existing, stamped)
    merged.to_csv(ledger_path, index=False)

    return {
        "status": "archived",
        "ledger_path": str(ledger_path),
        "rows_in_this_capture": int(len(stamped)),
        "rows_in_ledger_after_merge": int(len(merged)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--priced-file", type=Path, default=ROOT / "outputs" / "props_priced_clean.csv")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--run-id", default=os.getenv("GITHUB_RUN_ID", "local"))
    ap.add_argument("--git-sha", default=os.getenv("GITHUB_SHA", "local"))
    args = ap.parse_args()

    result = archive(
        priced_path=args.priced_file,
        season=args.season,
        week=args.week,
        run_id=args.run_id,
        git_sha=args.git_sha,
    )
    print("=== MARKET TRACK RECORD ARCHIVE ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
