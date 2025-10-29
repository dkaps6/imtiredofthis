#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.config import FILES, ROOT

DEFAULT_REQUIRED_KEYS: Sequence[str] = (
    "team_form",
    "player_form",
    "metrics_ready",
)


def _pretty_path(path: Path) -> str:
    """Return the path relative to the repo root when possible."""

    try:
        return str(path.resolve(strict=False).relative_to(ROOT))
    except ValueError:
        return str(path.resolve(strict=False))


def _resolve_item(item: str) -> Path:
    """Resolve a config key or filesystem path to an absolute Path."""

    if item in FILES:
        return FILES[item]

    candidate = Path(item)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    seen: set[Path] = set()
    ordered: List[Path] = []

    for pth in paths:
        resolved = pth.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(pth)

    return ordered


def _check_csv(path: Path) -> str | None:
    if not path.exists():
        return f"{_pretty_path(path)} (missing)"

    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return f"{_pretty_path(path)} (empty)"
    except Exception as exc:  # pragma: no cover - guard rails for runtime issues
        return f"{_pretty_path(path)} (error reading: {exc})"

    if frame.empty:
        return f"{_pretty_path(path)} (empty)"

    return None


def check_required_inputs(required: Iterable[str | Path] | None = None) -> None:
    """Validate that each required CSV exists and has at least one row."""

    items = list(required) if required is not None else list(DEFAULT_REQUIRED_KEYS)
    resolved_paths = _dedupe_paths(_resolve_item(str(item)) for item in items)

    failures: List[str] = []

    for csv_path in resolved_paths:
        error = _check_csv(csv_path)
        if error:
            failures.append(error)
        else:
            print(f"[metrics_ready] ✓ {_pretty_path(csv_path)}")

    if failures:
        raise Exception(f"Missing inputs: {', '.join(failures)}")

    print("[metrics_ready] ✅ all required inputs present")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ensure critical data/*.csv inputs exist and contain at least one row. "
            "Accepts either config keys (e.g. 'team_form') or explicit paths."
        )
    )
    parser.add_argument(
        "--require",
        dest="required",
        nargs="+",
        metavar="KEY_OR_PATH",
        action="append",
        help="Additional config keys or paths to validate (defaults to core metrics inputs).",
    )
    parser.add_argument(
        "--extra",
        dest="extra",
        nargs="+",
        metavar="PATH",
        action="append",
        help="Explicit filesystem paths to include in the readiness check.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    required_items: List[str | Path] = list(DEFAULT_REQUIRED_KEYS)
    if args.required:
        for group in args.required:
            required_items.extend(group)
    if args.extra:
        for group in args.extra:
            required_items.extend(Path(item) for item in group)

    check_required_inputs(required_items)


if __name__ == "__main__":
    main()
