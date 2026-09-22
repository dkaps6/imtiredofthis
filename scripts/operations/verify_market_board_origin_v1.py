#!/usr/bin/env python3
"""Verify durable live boards against frozen origin/recovery content digests.

The archive ledger adds three downstream provenance columns and may reorder
rows, so byte-for-byte CSV hashing is not the right invariant. Instead we
canonicalize every non-provenance cell, sort rows, and hash the canonical
table. Week 2 is anchored directly to the still-accessible originating Full
Slate artifact. Week 1's originating artifact expired; its content anchor is
the surviving Week-1 recovery artifact whose replay log proves it downloaded
the exact originating paid Full Slate artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "data" / "market_track_record" / "ORIGIN_PROVENANCE_V1.json"
PROVENANCE_COLUMNS = {"archived_at_utc", "source_run_id", "source_git_sha"}


def _norm_numeric(v) -> str:
    if pd.isna(v):
        return ""
    return format(float(v), ".12g")


def canonical_content_digest(path: Path) -> tuple[int, int, str]:
    df = pd.read_csv(path, low_memory=False)
    cols = [c for c in df.columns if c not in PROVENANCE_COLUMNS]
    frame = df[cols].copy()

    normalized_cols: list[list[str]] = []
    for col in cols:
        s = frame[col]
        if pd.api.types.is_numeric_dtype(s):
            normalized_cols.append([_norm_numeric(v) for v in s])
        else:
            normalized_cols.append(["" if pd.isna(v) else str(v) for v in s])

    rows = sorted(zip(*normalized_cols))
    h = hashlib.sha256()
    h.update(("\x1f".join(cols) + "\n").encode("utf-8"))
    for row in rows:
        h.update(("\x1f".join(row) + "\n").encode("utf-8"))
    return int(len(df)), int(len(cols)), h.hexdigest()


def verify(manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = {}
    for week_text, spec in manifest["weeks"].items():
        week = int(week_text)
        path = ROOT / spec["board_path"]
        rows, content_cols, digest = canonical_content_digest(path)

        if rows != int(spec["rows"]):
            raise RuntimeError(f"Week {week} row-count drift: {rows} != {spec['rows']}")
        if content_cols != int(spec["content_columns"]):
            raise RuntimeError(
                f"Week {week} content-column drift: {content_cols} != {spec['content_columns']}"
            )
        if digest != str(spec["canonical_content_sha256"]):
            raise RuntimeError(
                f"Week {week} content digest drift: {digest} != "
                f"{spec['canonical_content_sha256']}"
            )

        board = pd.read_csv(path, low_memory=False)
        run_ids = set(board["source_run_id"].astype(str))
        shas = set(board["source_git_sha"].astype(str))
        if run_ids != {str(spec["source_run_id"])}:
            raise RuntimeError(f"Week {week} source_run_id drift: {sorted(run_ids)}")
        if shas != {str(spec["source_git_sha"])}:
            raise RuntimeError(f"Week {week} source_git_sha drift: {sorted(shas)}")

        results[str(week)] = {
            "rows": rows,
            "content_columns": content_cols,
            "canonical_content_sha256": digest,
            "source_run_id": str(spec["source_run_id"]),
            "source_git_sha": str(spec["source_git_sha"]),
            "origin_evidence": spec["origin_evidence"],
        }
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = ap.parse_args()
    out = verify(args.manifest)
    print("=== MARKET BOARD ORIGIN VERIFICATION PASS ===")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
