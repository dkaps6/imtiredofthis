#!/usr/bin/env python3
"""Audit clean historical Vegas benchmark artifacts and fail on identity drift."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.backtest.benchmark_identity_v1 import assert_benchmark_identity


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing {label}: {path}")
    out = pd.read_csv(path, low_memory=False)
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projection", type=Path, required=True)
    ap.add_argument("--props", type=Path, required=True)
    ap.add_argument("--detail", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    projection = _read(a.projection, "projection trace")
    props = _read(a.props, "historical props")
    detail = _read(a.detail, "graded detail")

    projection_identity = assert_benchmark_identity(
        projection,
        label="projection trace",
        require_team=True,
        require_opponent=True,
    )
    props_identity = assert_benchmark_identity(
        props,
        label="historical props",
        require_team=False,
        require_opponent=False,
    )
    detail_identity = assert_benchmark_identity(
        detail,
        label="graded detail",
        require_team=True,
        require_opponent=("opponent" in detail.columns),
    )

    if detail.empty:
        raise RuntimeError("clean benchmark produced zero graded rows")
    if not {2024, 2025}.issubset(set(pd.to_numeric(detail["season"], errors="coerce").dropna().astype(int))):
        raise RuntimeError("clean benchmark detail does not contain both 2024 and 2025")

    audit = {
        "status": "PASS",
        "contract": "HISTORICAL_BENCHMARK_IDENTITY_CLEAN_V1",
        "projection_identity": projection_identity,
        "props_identity": props_identity,
        "detail_identity": detail_identity,
        "projection_rows": int(len(projection)),
        "props_rows": int(len(props)),
        "graded_rows": int(len(detail)),
        "graded_by_season_market": (
            detail.groupby(["season", "market"]).size().rename("rows").reset_index().to_dict(orient="records")
        ),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
