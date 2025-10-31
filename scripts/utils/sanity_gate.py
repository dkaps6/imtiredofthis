"""Hard stop validations for the Full Slate workflow.

This module ensures the downstream pricing model only runs when
critical upstream datasets exist, are non-empty, and expose the
expected schema.  It should remain lightweight so the GitHub Actions
step fails fast with a clear error message when something is off.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

TEAM_FORM_PATH = ROOT / "data" / "team_form.csv"
METRICS_READY_PATH = ROOT / "data" / "metrics_ready.csv"
WEATHER_PATH = ROOT / "data" / "weather_week.csv"
OPPONENT_MAP_PATH = ROOT / "data" / "opponent_map_from_props.csv"

TEAM_FORM_REQUIRED = [
    "team",
    "season",
    "neutral_pace",
    "coverage_man_rate",
    "coverage_zone_rate",
    "pass_rate_over_expected",
]

TEAM_FORM_NONEMPTY_COLS = [
    "neutral_pace",
    "coverage_man_rate",
    "coverage_zone_rate",
    "pass_rate_over_expected",
]

METRICS_READY_REQUIRED = [
    "player",
    "team",
    "opponent",
    "market",
    "line",
    "over_odds",
    "under_odds",
]

OPPONENT_MAP_REQUIRED = ["player", "team", "opponent", "week"]


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Missing required file: {path.relative_to(ROOT)}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(f"{path.relative_to(ROOT)} is empty") from exc
    except Exception as exc:  # pragma: no cover - runtime guard
        raise RuntimeError(f"Unable to read {path.relative_to(ROOT)}: {exc}") from exc

    if df.empty:
        raise RuntimeError(f"{path.relative_to(ROOT)} is empty")
    return df


def _require_columns(df: pd.DataFrame, path: Path, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"{path.relative_to(ROOT)} missing required columns: {joined}"
        )


def _require_nonempty_values(
    df: pd.DataFrame, path: Path, columns: list[str]
) -> None:
    for col in columns:
        if df[col].dropna().empty:
            raise RuntimeError(
                f"{path.relative_to(ROOT)} column '{col}' has no non-null values"
            )


def _sanity_check_team_form() -> None:
    df = _load_csv(TEAM_FORM_PATH)
    _require_columns(df, TEAM_FORM_PATH, TEAM_FORM_REQUIRED)
    _require_nonempty_values(df, TEAM_FORM_PATH, TEAM_FORM_NONEMPTY_COLS)


def _sanity_check_metrics_ready() -> None:
    df = _load_csv(METRICS_READY_PATH)
    _require_columns(df, METRICS_READY_PATH, METRICS_READY_REQUIRED)
    _require_nonempty_values(df, METRICS_READY_PATH, ["market", "line"])


def _sanity_check_weather() -> None:
    df = _load_csv(WEATHER_PATH)
    # ensure we have at least one matchup worth of weather
    if df.shape[0] == 0:
        raise RuntimeError(f"{WEATHER_PATH.relative_to(ROOT)} contains no rows")


def _sanity_check_opponent_map() -> None:
    df = _load_csv(OPPONENT_MAP_PATH)
    _require_columns(df, OPPONENT_MAP_PATH, OPPONENT_MAP_REQUIRED)
    if df.shape[0] == 0:
        raise RuntimeError(
            f"{OPPONENT_MAP_PATH.relative_to(ROOT)} contains no rows"
        )


def main() -> None:
    try:
        _sanity_check_team_form()
        _sanity_check_metrics_ready()
        _sanity_check_weather()
        _sanity_check_opponent_map()
    except RuntimeError as exc:
        print(f"[sanity_gate] ❌ {exc}")
        sys.exit(1)

    print("[sanity_gate] ✅ All critical inputs present.")


if __name__ == "__main__":
    main()
