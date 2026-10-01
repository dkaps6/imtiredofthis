#!/usr/bin/env python3
"""Compute same-book pregame movement / CLV from immutable market snapshots.

No outcome fields are consumed. Closing-line value is reported only for a later
same-book quote captured within 30 minutes of kickoff. Older later quotes are
labeled market movement, not CLV.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PRIMARY_KEY = ["event_id", "player_clean_key", "market", "book", "side"]
TOL = 1e-12


def american_to_decimal(odds: float) -> float:
    x = float(odds)
    if not np.isfinite(x) or x == 0:
        return np.nan
    return 1.0 + (x / 100.0 if x > 0 else 100.0 / abs(x))


def _read(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path)
    missing = sorted(set(PRIMARY_KEY + [
        "vegas_line", "vegas_odds", "model_proj",
        "odds_fetched_at_utc", "commence_time_utc", "minutes_to_kickoff",
    ]) - set(x.columns))
    if missing:
        raise RuntimeError(f"snapshot {path} missing columns: {missing}")
    if x.duplicated(PRIMARY_KEY).any():
        raise RuntimeError(f"snapshot {path} has duplicate primary keys")
    x["odds_fetched_at_utc"] = pd.to_datetime(x["odds_fetched_at_utc"], utc=True, errors="coerce")
    x["commence_time_utc"] = pd.to_datetime(x["commence_time_utc"], utc=True, errors="coerce")
    if x["odds_fetched_at_utc"].isna().any() or x["commence_time_utc"].isna().any():
        raise RuntimeError(f"snapshot {path} has invalid timestamps")
    x["source_snapshot_file"] = path.name
    return x


def timing_class(minutes: float) -> str:
    if not np.isfinite(minutes) or minutes <= 0:
        return "INVALID_FOR_PREGAME_MOVEMENT"
    if minutes <= 30:
        return "VALID_T30_CLOSE"
    if minutes <= 60:
        return "VALID_T60_LATE_MARKET"
    return "PREGAME_MOVEMENT_ONLY"


def compare(entry: pd.DataFrame, later: pd.DataFrame) -> pd.DataFrame:
    rows = []
    later_groups = {k: g for k, g in later.groupby(PRIMARY_KEY, dropna=False)}
    for r in entry.itertuples(index=False):
        key = tuple(getattr(r, c) for c in PRIMARY_KEY)
        entry_time = getattr(r, "odds_fetched_at_utc")
        kickoff = getattr(r, "commence_time_utc")
        candidates = later_groups.get(key)
        if candidates is None:
            rows.append({**{c: getattr(r, c) for c in PRIMARY_KEY},
                         "comparison_status": "NO_LATER_PREGAME_CAPTURE"})
            continue
        q = candidates.loc[
            (candidates["odds_fetched_at_utc"] > entry_time)
            & (candidates["odds_fetched_at_utc"] < kickoff)
        ].copy()
        if q.empty:
            rows.append({**{c: getattr(r, c) for c in PRIMARY_KEY},
                         "comparison_status": "NO_LATER_PREGAME_CAPTURE"})
            continue
        close = q.sort_values("odds_fetched_at_utc").iloc[-1]
        mins = float((kickoff - close["odds_fetched_at_utc"]).total_seconds() / 60.0)
        klass = timing_class(mins)

        side = str(getattr(r, "side")).upper()
        sign = 1.0 if side == "OVER" else (-1.0 if side == "UNDER" else np.nan)
        entry_line = float(getattr(r, "vegas_line"))
        close_line = float(close["vegas_line"])
        line_clv = sign * (close_line - entry_line) if np.isfinite(sign) else np.nan

        entry_odds = float(getattr(r, "vegas_odds"))
        close_odds = float(close["vegas_odds"])
        same_line = abs(close_line - entry_line) <= TOL
        price_clv = np.nan
        price_status = "NOT_COMPARABLE_LINE_CHANGED"
        if same_line:
            de = american_to_decimal(entry_odds)
            dc = american_to_decimal(close_odds)
            if np.isfinite(de) and np.isfinite(dc) and dc > 0:
                price_clv = (de / dc - 1.0) * 100.0
                price_status = "COMPARABLE_SAME_LINE"
            else:
                price_status = "INVALID_ODDS"

        model_proj = float(getattr(r, "model_proj"))
        model_gap_close = (
            sign * (model_proj - close_line) if np.isfinite(sign) else np.nan
        )

        rows.append({
            **{c: getattr(r, c) for c in PRIMARY_KEY},
            "comparison_status": klass,
            "entry_snapshot_file": getattr(r, "source_snapshot_file"),
            "close_snapshot_file": str(close["source_snapshot_file"]),
            "entry_odds_fetched_at_utc": entry_time.isoformat(),
            "close_odds_fetched_at_utc": close["odds_fetched_at_utc"].isoformat(),
            "commence_time_utc": kickoff.isoformat(),
            "close_minutes_to_kickoff": mins,
            "entry_line": entry_line,
            "close_line": close_line,
            "side_aligned_line_clv": line_clv,
            "entry_selected_side_odds": entry_odds,
            "close_selected_side_odds": close_odds,
            "price_clv_status": price_status,
            "same_line_price_clv_pct": price_clv,
            "entry_model_proj": model_proj,
            "side_aligned_entry_model_gap_to_close": model_gap_close,
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entry-snapshot", type=Path, required=True)
    ap.add_argument("--snapshot-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    entry = _read(a.entry_snapshot)
    files = sorted(
        p for p in a.snapshot_dir.glob("*.csv")
        if p.resolve() != a.entry_snapshot.resolve()
    )
    later = pd.concat([_read(p) for p in files], ignore_index=True) if files else pd.DataFrame(
        columns=entry.columns
    )

    result = compare(entry, later)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(a.out_dir / "market_snapshot_movement_detail.csv", index=False)

    counts = result["comparison_status"].value_counts(dropna=False).to_dict()
    t30 = result.loc[result["comparison_status"].eq("VALID_T30_CLOSE")].copy()
    summary = {
        "contract": "MARKET_SNAPSHOT_HISTORY_CLV_CAPTURE_V1",
        "entry_rows": int(len(entry)),
        "later_snapshot_files": int(len(files)),
        "status_counts": {str(k): int(v) for k, v in counts.items()},
        "t30_close_rows": int(len(t30)),
        "t30_positive_line_clv_rate": (
            float(t30["side_aligned_line_clv"].gt(0).mean()) if len(t30) else None
        ),
        "t30_zero_line_clv_rate": (
            float(t30["side_aligned_line_clv"].abs().le(TOL).mean()) if len(t30) else None
        ),
        "t30_mean_side_aligned_line_clv": (
            float(t30["side_aligned_line_clv"].mean()) if len(t30) else None
        ),
        "outcomes_used": False,
        "sportsbook_used_as_football_input": False,
    }
    (a.out_dir / "market_snapshot_movement_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
