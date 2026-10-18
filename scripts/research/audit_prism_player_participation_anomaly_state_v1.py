#!/usr/bin/env python3
"""PRISM Player Participation-Anomaly State V1 source audit.

Builds a player-specific participation-collapse state from the immutable
historical observation-health-state authority. Source/state construction only:
no performance outcomes, projection errors, sportsbook data, or production
mutation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "PRISM_PLAYER_PARTICIPATION_ANOMALY_STATE_V1_SOURCE_AUDIT"
FREEZE_COMMIT = "fd787e681aa5c7b6b55c5f8d0dc1d67ec4651b32"
SEASONS = (2024, 2025)
FAMILIES = ("QB", "RB", "WR", "TE")
MAX_PRIOR = 4
MIN_PRIOR = 3
RETENTION_MAX = 0.65
ABS_DROP_MIN = 0.20

STATE_NOT_COLLAPSED = "PARTICIPATION_NOT_COLLAPSED"
STATE_COLLAPSE_CORROBORATED = "PARTICIPATION_COLLAPSE_INJURY_CORROBORATED"
STATE_COLLAPSE_UNRESOLVED = "PARTICIPATION_COLLAPSE_UNRESOLVED"
STATE_BASELINE_INSUFFICIENT = "PARTICIPATION_BASELINE_INSUFFICIENT"

REQ = {
    "season", "week", "team", "history_key", "family",
    "positive_participation", "offense_pct", "injury_reported",
    "observation_state",
}


def _as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return (
        s.astype("string")
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def load_parent(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty parent: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    miss = REQ - set(x.columns)
    if miss:
        raise RuntimeError(f"parent missing required fields: {sorted(miss)}")

    x["season"] = pd.to_numeric(x["season"], errors="coerce")
    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x["offense_pct"] = pd.to_numeric(x["offense_pct"], errors="coerce")
    x["positive_participation"] = _as_bool(x["positive_participation"])
    x["injury_reported"] = _as_bool(x["injury_reported"])
    x["team"] = x["team"].astype("string").fillna("").str.strip()
    x["history_key"] = x["history_key"].astype("string").fillna("").str.strip()
    x["family"] = x["family"].astype("string").fillna("").str.upper().str.strip()

    x = x.loc[
        x["season"].isin(SEASONS)
        & x["family"].isin(FAMILIES)
        & x["team"].ne("")
        & x["history_key"].ne("")
    ].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)

    keys = ["season", "week", "team", "history_key"]
    if x.duplicated(keys).any():
        bad = x.loc[x.duplicated(keys, keep=False), keys].head(20).to_dict("records")
        raise RuntimeError(f"duplicate parent identities: {bad}")

    return x.sort_values(
        ["season", "team", "history_key", "week"], kind="mergesort"
    ).reset_index(drop=True)


def construct_states(x: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    chronology_violations = 0
    next_corroboration_violations = 0

    for (season, team, history_key), g in x.groupby(
        ["season", "team", "history_key"], sort=False
    ):
        g = g.sort_values("week", kind="mergesort").reset_index(drop=True)
        positive_history: list[dict] = []
        records = g.to_dict("records")

        for i, r in enumerate(records):
            if not bool(r["positive_participation"]):
                continue

            cur_pct = r["offense_pct"]
            if pd.isna(cur_pct):
                # Persist source row for coverage accounting; not scoreable.
                rec = dict(r)
                rec.update({
                    "prior_positive_rows": 0,
                    "player_prior_snap_pct": np.nan,
                    "absolute_snap_drop": np.nan,
                    "snap_retention_ratio": np.nan,
                    "major_participation_collapse": False,
                    "current_injury_corroborated": bool(r["injury_reported"]),
                    "next_game_injury_corroborated": False,
                    "next_scheduled_week": np.nan,
                    "corroboration_week": np.nan,
                    "corroboration_source": "NONE",
                    "participation_anomaly_state": STATE_BASELINE_INSUFFICIENT,
                    "baseline_supported": False,
                    "chronology_valid": True,
                })
                rows.append(rec)
                positive_history.append(r)
                continue

            prior = positive_history[-MAX_PRIOR:]
            rec = dict(r)
            rec["prior_positive_rows"] = int(len(prior))
            rec["baseline_supported"] = bool(len(prior) >= MIN_PRIOR)

            next_row = records[i + 1] if i + 1 < len(records) else None
            next_week = int(next_row["week"]) if next_row is not None else None
            next_injury = bool(next_row["injury_reported"]) if next_row is not None else False

            if next_week is not None and next_week <= int(r["week"]):
                next_corroboration_violations += 1

            if len(prior) < MIN_PRIOR:
                rec.update({
                    "player_prior_snap_pct": np.nan,
                    "absolute_snap_drop": np.nan,
                    "snap_retention_ratio": np.nan,
                    "major_participation_collapse": False,
                    "current_injury_corroborated": bool(r["injury_reported"]),
                    "next_game_injury_corroborated": next_injury,
                    "next_scheduled_week": next_week if next_week is not None else np.nan,
                    "corroboration_week": (
                        int(r["week"]) if bool(r["injury_reported"])
                        else (next_week if next_injury and next_week is not None else np.nan)
                    ),
                    "corroboration_source": (
                        "CURRENT_WEEK_INJURY" if bool(r["injury_reported"])
                        else ("NEXT_SCHEDULED_GAME_INJURY" if next_injury else "NONE")
                    ),
                    "participation_anomaly_state": STATE_BASELINE_INSUFFICIENT,
                    "chronology_valid": True,
                })
                rows.append(rec)
                positive_history.append(r)
                continue

            prior_weeks = [int(p["week"]) for p in prior]
            if not all(w < int(r["week"]) for w in prior_weeks):
                chronology_violations += 1
            prior_pcts = [float(p["offense_pct"]) for p in prior if pd.notna(p["offense_pct"])]
            if len(prior_pcts) != len(prior):
                raise RuntimeError("supported prior history contains nonfinite offense_pct")

            baseline = float(np.median(prior_pcts))
            if not np.isfinite(baseline) or baseline <= 0:
                raise RuntimeError("nonpositive player-specific snap baseline")

            cur = float(cur_pct)
            drop = float(baseline - cur)
            retention = float(cur / baseline)
            collapse = bool(retention <= RETENTION_MAX and drop >= ABS_DROP_MIN)
            current_injury = bool(r["injury_reported"])
            corroborated = bool(current_injury or next_injury)

            if collapse and corroborated:
                state = STATE_COLLAPSE_CORROBORATED
            elif collapse:
                state = STATE_COLLAPSE_UNRESOLVED
            else:
                state = STATE_NOT_COLLAPSED

            if current_injury:
                corr_week = int(r["week"])
                corr_source = "CURRENT_WEEK_INJURY"
            elif next_injury and next_week is not None:
                corr_week = int(next_week)
                corr_source = "NEXT_SCHEDULED_GAME_INJURY"
            else:
                corr_week = np.nan
                corr_source = "NONE"

            rec.update({
                "player_prior_snap_pct": baseline,
                "absolute_snap_drop": drop,
                "snap_retention_ratio": retention,
                "major_participation_collapse": collapse,
                "current_injury_corroborated": current_injury,
                "next_game_injury_corroborated": next_injury,
                "next_scheduled_week": next_week if next_week is not None else np.nan,
                "corroboration_week": corr_week,
                "corroboration_source": corr_source,
                "participation_anomaly_state": state,
                "chronology_valid": bool(all(w < int(r["week"]) for w in prior_weeks)),
            })
            rows.append(rec)
            positive_history.append(r)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("zero positive-participation rows constructed")

    return out, {
        "chronology_violations": int(chronology_violations),
        "next_corroboration_order_violations": int(next_corroboration_violations),
    }


def summarize(out: pd.DataFrame, season: int) -> dict:
    s = out.loc[out["season"].eq(season)].copy()
    supported = s.loc[s["baseline_supported"]].copy()
    finite_pos = s["offense_pct"].notna()
    collapse = supported.loc[supported["major_participation_collapse"]].copy()
    corroborated = collapse.loc[
        collapse["participation_anomaly_state"].eq(STATE_COLLAPSE_CORROBORATED)
    ]
    unresolved = collapse.loc[
        collapse["participation_anomaly_state"].eq(STATE_COLLAPSE_UNRESOLVED)
    ]

    by_family = []
    for fam in FAMILIES:
        f = s.loc[s["family"].eq(fam)]
        fs = f.loc[f["baseline_supported"]]
        fc = fs.loc[fs["major_participation_collapse"]]
        by_family.append({
            "family": fam,
            "positive_participation_rows": int(len(f)),
            "baseline_supported_rows": int(len(fs)),
            "baseline_insufficient_rows": int((~f["baseline_supported"]).sum()),
            "participation_not_collapsed_rows": int(
                fs["participation_anomaly_state"].eq(STATE_NOT_COLLAPSED).sum()
            ),
            "injury_corroborated_collapse_rows": int(
                fs["participation_anomaly_state"].eq(STATE_COLLAPSE_CORROBORATED).sum()
            ),
            "unresolved_collapse_rows": int(
                fs["participation_anomaly_state"].eq(STATE_COLLAPSE_UNRESOLVED).sum()
            ),
            "current_week_corroboration_rows": int(
                fc["corroboration_source"].eq("CURRENT_WEEK_INJURY").sum()
            ),
            "next_game_corroboration_rows": int(
                fc["corroboration_source"].eq("NEXT_SCHEDULED_GAME_INJURY").sum()
            ),
        })

    gates = {
        "positive_participation_offense_pct_coverage_ge_099": bool(
            finite_pos.mean() >= 0.99 if len(s) else False
        ),
        "duplicate_identity_rows_zero": bool(
            not s.duplicated(["season","week","team","history_key"]).any()
        ),
        "baseline_supported_rows_nonzero": bool(len(supported) > 0),
        "prior_baseline_positive_all": bool(
            (supported["player_prior_snap_pct"] > 0).all()
        ),
        "retention_finite_nonnegative_all": bool(
            np.isfinite(supported["snap_retention_ratio"]).all()
            and (supported["snap_retention_ratio"] >= 0).all()
        ),
        "major_participation_collapse_rows_nonzero": bool(len(collapse) > 0),
        "injury_corroborated_collapse_rows_nonzero": bool(len(corroborated) > 0),
        "baseline_chronology_valid_all": bool(supported["chronology_valid"].all()),
        "corroboration_week_not_before_observation": bool(
            (
                s["corroboration_week"].isna()
                | (pd.to_numeric(s["corroboration_week"], errors="coerce") >= s["week"])
            ).all()
        ),
    }

    return {
        "season": int(season),
        "positive_participation_rows": int(len(s)),
        "offense_pct_finite_coverage": float(finite_pos.mean()) if len(s) else 0.0,
        "baseline_supported_rows": int(len(supported)),
        "baseline_insufficient_rows": int((~s["baseline_supported"]).sum()),
        "participation_not_collapsed_rows": int(
            supported["participation_anomaly_state"].eq(STATE_NOT_COLLAPSED).sum()
        ),
        "injury_corroborated_collapse_rows": int(len(corroborated)),
        "unresolved_collapse_rows": int(len(unresolved)),
        "current_week_corroboration_rows": int(
            collapse["corroboration_source"].eq("CURRENT_WEEK_INJURY").sum()
        ),
        "next_game_corroboration_rows": int(
            collapse["corroboration_source"].eq("NEXT_SCHEDULED_GAME_INJURY").sum()
        ),
        "median_prior_snap_pct_supported": (
            float(supported["player_prior_snap_pct"].median()) if len(supported) else np.nan
        ),
        "median_current_snap_pct_collapse": (
            float(collapse["offense_pct"].median()) if len(collapse) else np.nan
        ),
        "median_retention_ratio_collapse": (
            float(collapse["snap_retention_ratio"].median()) if len(collapse) else np.nan
        ),
        "family_summary": by_family,
        "gates": gates,
        "source_ready": bool(all(gates.values())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    parent = load_parent(args.parent)
    out, audit = construct_states(parent)
    summaries = [summarize(out, s) for s in SEASONS]

    chronology_ok = (
        audit["chronology_violations"] == 0
        and audit["next_corroboration_order_violations"] == 0
    )
    ready = bool(all(s["source_ready"] for s in summaries) and chronology_ok)

    result = {
        "version": VERSION,
        "freeze_commit": FREEZE_COMMIT,
        "parent_run": 39154145666,
        "parent_artifact": 12146625382,
        "parent_digest": "sha256:d74f2d3c37a88aa639527aedf719e7e08001d464c722613b46291cda0f74cbeb",
        "disposition": (
            "PRISM_PLAYER_PARTICIPATION_ANOMALY_STATE_V1_SOURCE_READY"
            if ready else
            "PRISM_PLAYER_PARTICIPATION_ANOMALY_STATE_V1_SOURCE_BLOCKED"
        ),
        "prior_window_max": MAX_PRIOR,
        "prior_window_min": MIN_PRIOR,
        "snap_retention_max": RETENTION_MAX,
        "absolute_snap_drop_min": ABS_DROP_MIN,
        "audit": audit,
        "season_summaries": summaries,
        "performance_outcomes_loaded": 0,
        "predictive_relationships_scored": 0,
        "projection_errors_scored": 0,
        "coefficients_fit": 0,
        "thresholds_searched": 0,
        "outcomes_2026_used": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "production_changed": False,
    }

    out.to_csv(
        args.out_dir / "player_participation_anomaly_state_rows.csv",
        index=False,
        float_format="%.12f",
    )
    pd.DataFrame(
        [dict(season=s["season"], **f) for s in summaries for f in s["family_summary"]]
    ).to_csv(
        args.out_dir / "player_participation_anomaly_state_family_summary.csv",
        index=False,
    )
    (args.out_dir / "player_participation_anomaly_state_source_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
