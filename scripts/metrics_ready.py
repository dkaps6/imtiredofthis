#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.config import FILES, ROOT


def _check(path: str, cols: list[str]):
    if not os.path.exists(path):
        raise RuntimeError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"{path} exists but is empty")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path} missing required columns: {missing}")
    return df


# Required core inputs
pf = _check("data/player_form_consensus.csv", ["player", "team", "week", "opponent"])
om = _check("data/opponent_map_from_props.csv", ["team", "opponent", "week"])

# Optional: assert minimum unique weeks for a weekly slate (uncomment if desired)
# if om['week'].nunique() != 1:
#     raise RuntimeError(f"Expected a single-week slate in opponent_map; found weeks: {sorted(om['week'].unique().tolist())}")

print("[metrics_ready] ✅ core inputs present and valid")

REQUIRED: dict[str, Sequence[str]] = {
    os.path.join("data", "player_form_consensus.csv"): (
        "player",
        "team",
        "opponent",
        "season",
        "position",
        "role",
        "tgt_share",
        "route_rate",
        "rush_share",
        "yprr",
        "ypt",
        "ypc",
        "ypa",
        "receptions_per_target",
        "rz_share",
        "rz_tgt_share",
        "rz_rush_share",
        "week",
    ),
    os.path.join("data", "team_form.csv"): (
        "team",
        "season",
        "def_pass_epa",
        "def_rush_epa",
        "def_sack_rate",
        "pace",
        "proe",
        "light_box_rate",
        "heavy_box_rate",
        "ay_per_att",
    ),
    os.path.join("data", "opponent_map_from_props.csv"): (
        "player",
        "team",
        "opponent",
        "week",
        "season",
        "game_timestamp",
    ),
    os.path.join("data", "qb_designed_runs.csv"): (
        "player",
        "week",
        "designed_run_rate",
        "designed_runs",
        "snaps",
    ),
    os.path.join("data", "qb_scramble_rates.csv"): (
        "player",
        "week",
        "scramble_rate",
        "scrambles",
        "dropbacks",
    ),
    os.path.join("data", "weather_week.csv"): (
        "team",
        "opponent",
        "week",
        "stadium",
        "roof",
        "forecast_summary",
        "temp_f",
        "wind_mph",
        "precip_prob",
        "forecast_datetime_utc",
    ),
}

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

    schema_failures: List[str] = []
    for item, expected_cols in REQUIRED.items():
        csv_path = _resolve_item(str(item))
        pretty = _pretty_path(csv_path)
        if not csv_path.exists():
            schema_failures.append(f"{pretty} (missing)")
            continue
        try:
            sample = pd.read_csv(csv_path, nrows=5)
        except pd.errors.EmptyDataError:
            schema_failures.append(f"{pretty} (empty)")
            continue
        except Exception as exc:  # pragma: no cover - guard rails for runtime issues
            schema_failures.append(f"{pretty} (error reading: {exc})")
            continue

        missing_cols = [col for col in expected_cols if col not in sample.columns]
        if missing_cols:
            joined = ", ".join(missing_cols)
            schema_failures.append(f"{pretty} missing columns: {joined}")

    items = list(required) if required is not None else list(DEFAULT_REQUIRED_KEYS)
    resolved_paths = _dedupe_paths(_resolve_item(str(item)) for item in items)

    failures: List[str] = []

    for csv_path in resolved_paths:
        error = _check_csv(csv_path)
        if error:
            failures.append(error)
        else:
            print(f"[metrics_ready] ✓ {_pretty_path(csv_path)}")

    all_failures = schema_failures + failures
    if all_failures:
        details = "\n".join(f"  - {msg}" for msg in all_failures)
        raise RuntimeError(f"Missing or incomplete inputs:\n{details}")

    print("[metrics_ready] ✅ All required inputs present.")


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
