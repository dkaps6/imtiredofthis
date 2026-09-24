#!/usr/bin/env python3
"""Verify durable live boards against immutable origin/recovery anchors.

The archive ledger adds downstream provenance columns and may reorder rows, so
byte-for-byte hashing of the committed CSV is not the right invariant.  We
canonicalize non-provenance content and compare it to fingerprints recovered
from the original/recovery GitHub Actions artifacts.

Crucially, those fingerprints are read with `git show` from the exact commit
that first pinned them.  Editing a board and the mutable manifest together can
therefore no longer make the provenance gate pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "data" / "market_track_record" / "ORIGIN_PROVENANCE_V1.json"
IMMUTABLE_ANCHOR_COMMIT = "8839a34e549c04a3d688fc6437a032e386a216fb"
IMMUTABLE_ANCHOR_PATH = "data/market_track_record/IMMUTABLE_ORIGIN_ANCHORS_V1.json"
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


def load_immutable_anchors() -> dict:
    proc = subprocess.run(
        ["git", "show", f"{IMMUTABLE_ANCHOR_COMMIT}:{IMMUTABLE_ANCHOR_PATH}"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        detail = proc.stderr.strip()
        raise RuntimeError(
            "unable to load immutable market-board origin anchors from "
            f"{IMMUTABLE_ANCHOR_COMMIT}: {detail}"
        )
    anchors = json.loads(proc.stdout)
    if str(anchors.get("version", "")) != "MARKET_BOARD_IMMUTABLE_ORIGIN_ANCHORS_V1":
        raise RuntimeError("unexpected immutable market-board anchor version")
    return anchors


def _assert_manifest_matches_anchor(week: int, spec: dict, anchor: dict) -> None:
    fields = (
        "rows",
        "content_columns",
        "canonical_content_sha256",
        "source_run_id",
        "source_git_sha",
    )
    for field in fields:
        if str(spec.get(field)) != str(anchor.get(field)):
            raise RuntimeError(
                f"Week {week} editable manifest {field} drifted from immutable "
                f"anchor commit {IMMUTABLE_ANCHOR_COMMIT}"
            )


def verify(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    immutable_anchors: dict | None = None,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    anchors = immutable_anchors if immutable_anchors is not None else load_immutable_anchors()
    results = {}

    manifest_weeks = set(manifest.get("weeks", {}))
    anchor_weeks = set(anchors.get("weeks", {}))
    if manifest_weeks != anchor_weeks:
        raise RuntimeError(
            f"manifest/immutable-anchor week set mismatch: {manifest_weeks} != {anchor_weeks}"
        )

    for week_text, spec in manifest["weeks"].items():
        week = int(week_text)
        anchor = anchors["weeks"][week_text]
        _assert_manifest_matches_anchor(week, spec, anchor)

        path = ROOT / spec["board_path"]
        rows, content_cols, digest = canonical_content_digest(path)

        if rows != int(anchor["rows"]):
            raise RuntimeError(
                f"Week {week} row-count drift from immutable origin: "
                f"{rows} != {anchor['rows']}"
            )
        if content_cols != int(anchor["content_columns"]):
            raise RuntimeError(
                f"Week {week} content-column drift from immutable origin: "
                f"{content_cols} != {anchor['content_columns']}"
            )
        if digest != str(anchor["canonical_content_sha256"]):
            raise RuntimeError(
                f"Week {week} content digest drift from immutable origin: {digest} != "
                f"{anchor['canonical_content_sha256']}"
            )

        board = pd.read_csv(path, low_memory=False)
        required = {"source_run_id", "source_git_sha"}
        missing = required - set(board.columns)
        if missing:
            raise RuntimeError(
                f"Week {week} board missing provenance columns: {sorted(missing)}"
            )
        run_ids = set(board["source_run_id"].astype(str))
        shas = set(board["source_git_sha"].astype(str))
        if run_ids != {str(anchor["source_run_id"])}:
            raise RuntimeError(
                f"Week {week} source_run_id drift from immutable origin: {sorted(run_ids)}"
            )
        if shas != {str(anchor["source_git_sha"])}:
            raise RuntimeError(
                f"Week {week} source_git_sha drift from immutable origin: {sorted(shas)}"
            )

        results[str(week)] = {
            "rows": rows,
            "content_columns": content_cols,
            "canonical_content_sha256": digest,
            "source_run_id": str(anchor["source_run_id"]),
            "source_git_sha": str(anchor["source_git_sha"]),
            "immutable_anchor_commit": IMMUTABLE_ANCHOR_COMMIT,
            "origin_artifact_id": str(anchor.get("origin_artifact_id", "")),
            "recovery_artifact_id": str(anchor.get("recovery_artifact_id", "")),
        }
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = ap.parse_args()
    out = verify(args.manifest)
    print("=== MARKET BOARD IMMUTABLE ORIGIN VERIFICATION PASS ===")
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
