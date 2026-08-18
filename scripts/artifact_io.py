"""Small helpers for validating runtime CSV artifacts before consuming them.

A tracked placeholder file can exist and have non-zero size while still being an
invalid runtime artifact. Production callers should validate parseability,
required columns, and row count rather than relying on Path.exists().
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from pandas.errors import EmptyDataError, ParserError


def read_valid_csv(
    path: str | Path,
    *,
    required_columns: Iterable[str] = (),
    min_rows: int = 1,
    required: bool = False,
    label: str | None = None,
) -> pd.DataFrame | None:
    """Return a validated CSV dataframe, or ``None`` for an optional invalid artifact.

    ``required=True`` converts every invalid state into a RuntimeError. This lets
    live/production modes hard-fail while offseason/no-market modes can explicitly
    ignore stale placeholders.
    """
    p = Path(path)
    name = label or str(p)

    def invalid(reason: str) -> None:
        if required:
            raise RuntimeError(f"Required artifact {name} is invalid: {reason}")
        print(f"[artifact_io] ignoring optional artifact {name}: {reason}")

    if not p.exists():
        invalid("file does not exist")
        return None
    if p.stat().st_size <= 0:
        invalid("file is empty")
        return None

    try:
        df = pd.read_csv(p)
    except (EmptyDataError, ParserError, UnicodeDecodeError, OSError, ValueError) as exc:
        invalid(f"CSV parse failed: {exc}")
        return None

    if len(df) < int(min_rows):
        invalid(f"rows={len(df)} < min_rows={min_rows}")
        return None

    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = sorted(set(str(c).lower() for c in required_columns) - set(df.columns))
    if missing:
        invalid(f"missing required columns {missing}")
        return None

    return df
