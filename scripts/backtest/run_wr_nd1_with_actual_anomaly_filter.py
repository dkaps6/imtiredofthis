#!/usr/bin/env python3
"""Mechanical WR-ND1 wrapper for impossible target-result factorization rows.

The canonical historical player logs remain untouched for every pregame bundle.
Only the target-week *evaluation view* passed to actual_week() excludes rows with
nonzero receiving yards and zero recorded targets, because such rows cannot be
represented by targets x YPT.  The excluded rows are written to an audit file.
No WR-ND1 factor, gate, projection input, or model parameter is changed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from scripts.backtest import evaluate_wr_nd1_post_m38_decomposition as nd1

_ORIGINAL_ACTUAL_WEEK = nd1.actual_week
_ANOMALIES: list[pd.DataFrame] = []


def _filtered_actual_week(logs: pd.DataFrame, season: int, week: int):
    x = logs.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    if not {"season", "week", "targets", "rec_yards"}.issubset(x.columns):
        return _ORIGINAL_ACTUAL_WEEK(logs, season, week)
    s = pd.to_numeric(x["season"], errors="coerce")
    w = pd.to_numeric(x["week"], errors="coerce")
    targets = pd.to_numeric(x["targets"], errors="coerce").fillna(0.0)
    yards = pd.to_numeric(x["rec_yards"], errors="coerce").fillna(0.0)
    mask = s.eq(int(season)) & w.eq(int(week)) & targets.le(0) & yards.abs().gt(1e-9)
    bad = x.loc[mask].copy()
    if not bad.empty:
        bad["wr_nd1_exclusion_reason"] = "NONZERO_REC_YARDS_WITH_ZERO_RECORDED_TARGETS"
        _ANOMALIES.append(bad)
        who = bad[[c for c in ["season", "week", "team", "player", "position", "targets", "rec_yards"] if c in bad.columns]]
        print("[wr-nd1] factorization anomaly excluded from target-result evaluation only:")
        print(who.to_string(index=False))
        filtered = logs.loc[~mask.to_numpy()].copy()
    else:
        filtered = logs
    return _ORIGINAL_ACTUAL_WEEK(filtered, season, week)


def _arg_value(flag: str, default: str) -> str:
    try:
        i = sys.argv.index(flag)
        return sys.argv[i + 1]
    except (ValueError, IndexError):
        return default


def main() -> int:
    nd1.actual_week = _filtered_actual_week
    code = int(nd1.main())
    out_dir = Path(_arg_value("--out-dir", "data/backtests/wr_nd1_post_m38"))
    audit_path = out_dir / "wr_nd1_factorization_anomalies.csv"
    if _ANOMALIES:
        audit = pd.concat(_ANOMALIES, ignore_index=True)
    else:
        audit = pd.DataFrame(columns=["season", "week", "team", "player", "position", "targets", "rec_yards", "wr_nd1_exclusion_reason"])
    out_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)

    summary_path = out_dir / "wr_nd1_summary.json"
    if code == 0 and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["factorization_anomalies_excluded_from_target_result_view"] = int(len(audit))
        summary["factorization_anomaly_rule"] = "nonzero receiving yards with zero recorded targets; historical pregame logs remain untouched"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
