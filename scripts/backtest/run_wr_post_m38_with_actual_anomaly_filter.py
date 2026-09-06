#!/usr/bin/env python3
"""Mechanical wrapper for non-factorizable WR target-result rows.

This carries forward the WR-ND1 integrity treatment without changing the frozen
post-M38 scientific experiment. Canonical historical player logs and the PBP
prior universe remain untouched. Only the target-game evaluation prediction
view passed to prepare_casebook() excludes WR rows with nonzero receiving yards
and zero recorded targets, because the frozen targets x catch-rate x YPR
factorization cannot mathematically reproduce those provider/stat anomalies.

Excluded rows are written to an audit CSV. No component definition, prior,
threshold, slice, M38 reconstruction, projection input, or routing gate changes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from scripts.backtest import evaluate_wr_post_m38_error_decomposition as diag

_ORIGINAL_PREPARE_CASEBOOK = diag.prepare_casebook
_ANOMALIES: list[pd.DataFrame] = []


def _filtered_prepare_casebook(
    cp: pd.DataFrame,
    logs: pd.DataFrame,
    pbpg: pd.DataFrame,
) -> pd.DataFrame:
    """Filter only matching target-game prediction rows; preserve all priors."""
    x = logs.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {"season", "week", "team", "player_clean_key", "targets", "rec_yards"}
    if not required.issubset(x.columns):
        return _ORIGINAL_PREPARE_CASEBOOK(cp, logs, pbpg)

    targets = pd.to_numeric(x["targets"], errors="coerce").fillna(0.0)
    yards = pd.to_numeric(x["rec_yards"], errors="coerce").fillna(0.0)
    mask = targets.le(0) & yards.abs().gt(1e-9)

    # This diagnostic evaluates WRs only. If position is available, keep the
    # mechanical exclusion scoped to rows that can actually enter that view.
    if "position" in x.columns:
        pos = x["position"].fillna("").astype(str).str.upper().str.strip()
        mask &= pos.isin(diag.WR_POSITIONS)

    bad = x.loc[mask].copy()
    if bad.empty:
        return _ORIGINAL_PREPARE_CASEBOOK(cp, logs, pbpg)

    bad["wr_post_m38_exclusion_reason"] = (
        "NONZERO_REC_YARDS_WITH_ZERO_RECORDED_TARGETS"
    )
    _ANOMALIES.append(bad)
    cols = [
        c
        for c in [
            "season",
            "week",
            "team",
            "player",
            "player_id",
            "position",
            "targets",
            "receptions",
            "rec_yards",
        ]
        if c in bad.columns
    ]
    print(
        "[wr-post-m38] factorization anomaly excluded from target-result "
        "evaluation only:"
    )
    print(bad[cols].to_string(index=False))

    # Exclude the anomaly from the M38 target-game evaluation population, not
    # from historical logs. This mirrors ND1's evaluation-only treatment while
    # keeping build_pbp_games() and every strict-prior lookup on full history.
    keys = set()
    for _, row in bad.iterrows():
        season = pd.to_numeric(pd.Series([row["season"]]), errors="coerce").iloc[0]
        week = pd.to_numeric(pd.Series([row["week"]]), errors="coerce").iloc[0]
        if pd.isna(season) or pd.isna(week):
            continue
        keys.add(
            (
                int(season),
                int(week),
                diag.canon_team(row["team"]),
                str(row["player_clean_key"]),
            )
        )

    cp_eval = cp.copy()
    cp_season = pd.to_numeric(cp_eval["season"], errors="coerce")
    cp_week = pd.to_numeric(cp_eval["week"], errors="coerce")
    cp_team = cp_eval["team"].map(diag.canon_team)
    cp_player = cp_eval["player_clean_key"].astype(str)
    cp_market = cp_eval["market"].fillna("").astype(str).str.lower()
    exclude = pd.Series(False, index=cp_eval.index)
    for season, week, team, player_key in keys:
        exclude |= (
            cp_season.eq(season)
            & cp_week.eq(week)
            & cp_team.eq(team)
            & cp_player.eq(player_key)
            & cp_market.eq("rec_yards")
        )

    matched = int(exclude.sum())
    if matched != len(keys):
        raise RuntimeError(
            "mechanical anomaly wrapper did not match exactly one rec_yards "
            f"evaluation row per anomaly: keys={len(keys)} matched={matched}"
        )

    return _ORIGINAL_PREPARE_CASEBOOK(cp_eval.loc[~exclude].copy(), logs, pbpg)


def _arg_value(flag: str, default: str) -> str:
    try:
        i = sys.argv.index(flag)
        return sys.argv[i + 1]
    except (ValueError, IndexError):
        return default


def _write_audit(out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "wr_post_m38_factorization_anomalies.csv"
    if _ANOMALIES:
        audit = pd.concat(_ANOMALIES, ignore_index=True)
    else:
        audit = pd.DataFrame(
            columns=[
                "season",
                "week",
                "team",
                "player",
                "player_id",
                "position",
                "targets",
                "receptions",
                "rec_yards",
                "wr_post_m38_exclusion_reason",
            ]
        )
    audit.to_csv(audit_path, index=False)
    return audit


def main() -> int:
    # diag.main() builds pbpg from the untouched logs before invoking this
    # patched prepare_casebook(), so strict-prior construction is unchanged.
    diag.prepare_casebook = _filtered_prepare_casebook
    out_dir = Path(
        _arg_value(
            "--out-dir",
            "data/backtests/wr_post_m38_error_decomposition",
        )
    )

    try:
        code = int(diag.main())
    except Exception:
        _write_audit(out_dir)
        raise

    audit = _write_audit(out_dir)
    result_path = out_dir / "wr_post_m38_result.json"
    if code == 0 and result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["mechanical_factorization_wrapper"] = {
            "excluded_rows": int(len(audit)),
            "rule": (
                "WR target-game evaluation only: nonzero receiving yards with "
                "zero recorded targets; historical logs and strict priors untouched"
            ),
            "scientific_protocol_changed": False,
        }
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
